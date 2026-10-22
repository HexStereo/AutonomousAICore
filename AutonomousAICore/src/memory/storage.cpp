// src/memory/storage.cpp
#include "memory/storage.hpp"

#include <cstring>

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#define AIC_CUDA_CHECK_STORAGE(call)                                                       \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            AIC_THROW(CudaError, "CUDA error in Storage: ", cudaGetErrorString(err__));   \
        }                                                                                  \
    } while (false)
#endif

namespace aic {

Storage::Storage(size_t nbytes, Device device, IAllocator* allocator)
    : nbytes_(nbytes), device_(device), allocator_(allocator) {
    if (nbytes_ == 0) {
        data_ = nullptr;
        return;
    }

    if (!allocator_) {
        if (device_.is_cpu()) {
            allocator_ = &HostMemoryPool::default_pool();
        } else {
            allocator_ = &DeviceMemoryPool::instance(device_.index);
        }
    }

    void* p = allocator_->allocate(nbytes_);
    handle_ = AllocationHandle(p, nbytes_, allocator_);
    data_ = p;
}

Storage::Storage(void* ptr, size_t nbytes, Device device, bool owns_memory)
    : data_(ptr), nbytes_(nbytes), device_(device) {
    if (owns_memory && ptr && nbytes > 0) {
        if (device_.is_cpu()) {
            allocator_ = &HostMemoryPool::default_pool();
        } else {
            allocator_ = &DeviceMemoryPool::instance(device_.index);
        }
        handle_ = AllocationHandle(ptr, nbytes, allocator_);
    }
}

std::shared_ptr<Storage> Storage::clone() const {
    auto copy = std::make_shared<Storage>(nbytes_, device_, allocator_);
    if (nbytes_ == 0 || !data_) {
        return copy;
    }

    if (device_.is_cpu()) {
        std::memcpy(copy->data(), data_, nbytes_);
    } else {
#if AIC_WITH_CUDA
        AIC_CUDA_CHECK_STORAGE(cudaMemcpy(copy->data(), data_, nbytes_, cudaMemcpyDeviceToDevice));
#else
        AIC_THROW(BackendUnavailable, "Cannot clone CUDA storage: CUDA not compiled in this build.");
#endif
    }
    return copy;
}

}  // namespace aic
