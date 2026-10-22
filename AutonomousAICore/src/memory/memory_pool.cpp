// src/memory/memory_pool.cpp
#include "memory/memory_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <new>

#include "core/logging.hpp"

#if defined(_WIN32)
#include <malloc.h>
#endif

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#define AIC_CUDA_CHECK_POOL(call)                                                          \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            AIC_THROW(CudaError, "CUDA error at ", __FILE__, ":", __LINE__, ": ",          \
                      cudaGetErrorString(err__));                                          \
        }                                                                                  \
    } while (false)
#endif

namespace aic {

namespace {

void* raw_aligned_alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) bytes = alignment;
#if defined(_WIN32)
    void* p = _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) {
        p = nullptr;
    }
#endif
    if (!p) {
        AIC_THROW(OutOfMemory, "Failed to allocate ", bytes, " bytes with alignment ", alignment);
    }
    return p;
}

void raw_aligned_free(void* ptr) noexcept {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

}  // namespace

// ============================================================================
// HostMemoryPool
// ============================================================================

HostMemoryPool::HostMemoryPool(bool pinned) : is_pinned_(pinned) {}

HostMemoryPool::~HostMemoryPool() {
    try {
        trim();
    } catch (...) {
        // Suppress destructor exceptions
    }
}

size_t HostMemoryPool::round_up_size(size_t bytes, size_t alignment) noexcept {
    if (bytes == 0) return alignment;
    size_t rem = bytes % alignment;
    if (rem == 0) return bytes;
    return bytes + (alignment - rem);
}

int HostMemoryPool::size_to_bin(size_t bytes) noexcept {
    // Quantize into power-of-two size classes above 256 bytes
    if (bytes <= 256) return 256;
    size_t v = bytes - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return static_cast<int>(v + 1);
}

void* HostMemoryPool::allocate(size_t bytes, size_t alignment) {
    if (alignment < 64) alignment = 64;
    size_t bin_size = static_cast<size_t>(size_to_bin(round_up_size(bytes, alignment)));

    std::lock_guard<std::mutex> lock(mu_);
    auto it = free_bins_.find(bin_size);
    if (it != free_bins_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        stats_.record_alloc(bin_size, 0);
        return ptr;
    }

    void* ptr = nullptr;
#if AIC_WITH_CUDA
    if (is_pinned_) {
        cudaError_t err = cudaHostAlloc(&ptr, bin_size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            AIC_THROW(OutOfMemory, "cudaHostAlloc failed for ", bin_size, " bytes: ", cudaGetErrorString(err));
        }
    } else {
        ptr = raw_aligned_alloc(bin_size, alignment);
    }
#else
    ptr = raw_aligned_alloc(bin_size, alignment);
#endif

    stats_.record_alloc(bin_size, bin_size);
    return ptr;
}

void HostMemoryPool::deallocate(void* ptr, size_t bytes) noexcept {
    if (!ptr) return;
    size_t bin_size = static_cast<size_t>(size_to_bin(round_up_size(bytes, 64)));

    std::lock_guard<std::mutex> lock(mu_);
    free_bins_[bin_size].push_back(ptr);
    stats_.record_free(bin_size, 0);
}

void HostMemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [bin_size, ptrs] : free_bins_) {
        for (void* p : ptrs) {
#if AIC_WITH_CUDA
            if (is_pinned_) {
                cudaFreeHost(p);
            } else {
                raw_aligned_free(p);
            }
#else
            raw_aligned_free(p);
#endif
            stats_.reserved_bytes.fetch_sub(bin_size, std::memory_order_relaxed);
        }
        ptrs.clear();
    }
    free_bins_.clear();
}

HostMemoryPool& HostMemoryPool::default_pool() {
    static HostMemoryPool pool(false);
    return pool;
}

HostMemoryPool& HostMemoryPool::default_pinned_pool() {
    static HostMemoryPool pool(true);
    return pool;
}

// ============================================================================
// DeviceMemoryPool
// ============================================================================

DeviceMemoryPool::DeviceMemoryPool(int device_index) : device_index_(device_index) {}

DeviceMemoryPool::~DeviceMemoryPool() {
    try {
        trim();
    } catch (...) {
    }
}

void* DeviceMemoryPool::allocate(size_t bytes, size_t alignment) {
    return allocate_on_stream(bytes, 0, alignment);
}

void DeviceMemoryPool::deallocate(void* ptr, size_t bytes) noexcept {
    deallocate_on_stream(ptr, bytes, 0);
}

void* DeviceMemoryPool::allocate_on_stream(size_t bytes, uintptr_t stream, size_t alignment) {
#if !AIC_WITH_CUDA
    (void)bytes; (void)stream; (void)alignment;
    AIC_THROW(BackendUnavailable, "DeviceMemoryPool requested on device ", device_index_,
              " but CUDA backend was not compiled in this build.");
#else
    if (bytes == 0) bytes = alignment;
    size_t rounded_size = (bytes + alignment - 1) & ~(alignment - 1);

    std::lock_guard<std::mutex> lock(mu_);

    // Search free blocks of suitable size
    auto range = free_blocks_.lower_bound(rounded_size);
    for (auto it = range; it != free_blocks_.end(); ++it) {
        DeviceBlock block = it->second;
        bool can_use = false;

        if (block.stream == stream) {
            can_use = true;
        } else if (block.event == nullptr) {
            can_use = true;
        } else {
            cudaError_t q = cudaEventQuery(static_cast<cudaEvent_t>(block.event));
            if (q == cudaSuccess) {
                can_use = true;
            } else if (stream != 0) {
                // Synchronize stream with completion event of previous stream
                cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                                    static_cast<cudaEvent_t>(block.event), 0);
                can_use = true;
            }
        }

        if (can_use) {
            free_blocks_.erase(it);
            block.stream = stream;
            active_blocks_[block.ptr] = block;
            stats_.record_alloc(block.size, 0);
            return block.ptr;
        }
    }

    // No reusable block found - allocate from device
    AIC_CUDA_CHECK_POOL(cudaSetDevice(device_index_));
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, rounded_size);
    if (err != cudaSuccess) {
        // Try trimming free blocks before failing
        for (auto& [size, block] : free_blocks_) {
            if (block.event) {
                cudaEventDestroy(static_cast<cudaEvent_t>(block.event));
            }
            cudaFree(block.ptr);
            stats_.reserved_bytes.fetch_sub(block.size, std::memory_order_relaxed);
        }
        free_blocks_.clear();

        err = cudaMalloc(&dev_ptr, rounded_size);
        if (err != cudaSuccess) {
            AIC_THROW(OutOfMemory, "cudaMalloc failed for ", rounded_size,
                      " bytes on device ", device_index_, ": ", cudaGetErrorString(err));
        }
    }

    DeviceBlock block;
    block.ptr = dev_ptr;
    block.size = rounded_size;
    block.stream = stream;
    block.event = nullptr;

    active_blocks_[dev_ptr] = block;
    stats_.record_alloc(rounded_size, rounded_size);
    return dev_ptr;
#endif
}

void DeviceMemoryPool::deallocate_on_stream(void* ptr, size_t /*bytes*/, uintptr_t stream) noexcept {
#if !AIC_WITH_CUDA
    (void)ptr;
    (void)stream;
    return;
#else
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_blocks_.find(ptr);
    if (it == active_blocks_.end()) {
        return;
    }

    DeviceBlock block = it->second;
    active_blocks_.erase(it);

    // Record an event on the releasing stream to prevent premature reuse on other streams
    if (stream != 0) {
        if (!block.event) {
            cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(&block.event), cudaEventDisableTiming);
        }
        cudaEventRecord(static_cast<cudaEvent_t>(block.event), reinterpret_cast<cudaStream_t>(stream));
    }
    block.stream = stream;

    stats_.record_free(block.size, 0);
    free_blocks_.insert({block.size, block});
#endif
}

void DeviceMemoryPool::trim() {
#if AIC_WITH_CUDA
    std::lock_guard<std::mutex> lock(mu_);
    AIC_CUDA_CHECK_POOL(cudaSetDevice(device_index_));
    for (auto& [size, block] : free_blocks_) {
        if (block.event) {
            cudaEventDestroy(static_cast<cudaEvent_t>(block.event));
        }
        cudaFree(block.ptr);
        stats_.reserved_bytes.fetch_sub(block.size, std::memory_order_relaxed);
    }
    free_blocks_.clear();
#endif
}

DeviceMemoryPool& DeviceMemoryPool::instance(int device_index) {
    static std::map<int, std::unique_ptr<DeviceMemoryPool>> pools;
    static std::mutex pools_mu;
    std::lock_guard<std::mutex> lock(pools_mu);
    auto it = pools.find(device_index);
    if (it == pools.end()) {
        it = pools.emplace(device_index, std::make_unique<DeviceMemoryPool>(device_index)).first;
    }
    return *it->second;
}

}  // namespace aic
