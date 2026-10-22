// include/memory/memory_pool.hpp
// High-performance CPU and CUDA memory pools with RAII lifetime management and stream awareness.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "memory/allocator.hpp"

namespace aic {

// Block descriptor for pooled allocations.
struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    Device device = Device::cpu();
    bool is_pinned = false;
    uintptr_t stream = 0;
    void* ready_event = nullptr;  // cudaEvent_t if CUDA enabled
    bool in_use = false;
};

// RAII Handle to an allocated memory block.
class AllocationHandle {
public:
    AllocationHandle() = default;
    AllocationHandle(void* ptr, size_t bytes, IAllocator* allocator) noexcept
        : ptr_(ptr), bytes_(bytes), allocator_(allocator) {}

    ~AllocationHandle() { reset(); }

    AllocationHandle(const AllocationHandle&) = delete;
    AllocationHandle& operator=(const AllocationHandle&) = delete;

    AllocationHandle(AllocationHandle&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_), allocator_(other.allocator_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
        other.allocator_ = nullptr;
    }

    AllocationHandle& operator=(AllocationHandle&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            allocator_ = other.allocator_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
            other.allocator_ = nullptr;
        }
        return *this;
    }

    void* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return bytes_; }
    IAllocator* allocator() const noexcept { return allocator_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    void reset() noexcept {
        if (ptr_ && allocator_) {
            allocator_->deallocate(ptr_, bytes_);
            ptr_ = nullptr;
            bytes_ = 0;
            allocator_ = nullptr;
        }
    }

    void* release() noexcept {
        void* p = ptr_;
        ptr_ = nullptr;
        bytes_ = 0;
        allocator_ = nullptr;
        return p;
    }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    IAllocator* allocator_ = nullptr;
};

// CPU Aligned Memory Pool with segregated size classes.
class HostMemoryPool : public IAllocator {
public:
    explicit HostMemoryPool(bool pinned = false);
    ~HostMemoryPool() override;

    void* allocate(size_t bytes, size_t alignment = 64) override;
    void deallocate(void* ptr, size_t bytes) noexcept override;

    Device device() const noexcept override { return Device::cpu(); }
    const MemoryStats& stats() const noexcept override { return stats_; }
    void reset_peak_stats() noexcept override { stats_.reset_peak(); }
    void trim() override;

    bool is_pinned() const noexcept { return is_pinned_; }

    static HostMemoryPool& default_pool();
    static HostMemoryPool& default_pinned_pool();

private:
    static size_t round_up_size(size_t bytes, size_t alignment) noexcept;
    static int size_to_bin(size_t bytes) noexcept;

    bool is_pinned_;
    MemoryStats stats_;
    mutable std::mutex mu_;
    // Segregated free lists: bin index -> list of free MemoryBlock
    std::map<size_t, std::vector<void*>> free_bins_;
};

// CUDA Device Memory Pool with stream awareness and event-based recycling.
class DeviceMemoryPool : public IAllocator {
public:
    explicit DeviceMemoryPool(int device_index = 0);
    ~DeviceMemoryPool() override;

    void* allocate(size_t bytes, size_t alignment = 256) override;
    void deallocate(void* ptr, size_t bytes) noexcept override;

    // Stream-aware allocation and deallocation
    void* allocate_on_stream(size_t bytes, uintptr_t stream, size_t alignment = 256);
    void deallocate_on_stream(void* ptr, size_t bytes, uintptr_t stream) noexcept;

    Device device() const noexcept override { return Device::cuda(device_index_); }
    const MemoryStats& stats() const noexcept override { return stats_; }
    void reset_peak_stats() noexcept override { stats_.reset_peak(); }
    void trim() override;

    static DeviceMemoryPool& instance(int device_index = 0);

private:
    int device_index_;
    MemoryStats stats_;
    mutable std::mutex mu_;

    struct DeviceBlock {
        void* ptr = nullptr;
        size_t size = 0;
        uintptr_t stream = 0;
        void* event = nullptr;  // cudaEvent_t
    };

    // Free blocks sorted by size
    std::multimap<size_t, DeviceBlock> free_blocks_;
    // Active allocations: ptr -> DeviceBlock
    std::map<void*, DeviceBlock> active_blocks_;
};

}  // namespace aic
