// include/memory/storage.hpp
// Reference-counted buffer owning allocated device or host memory.
#pragma once

#include <cstddef>
#include <memory>

#include "core/device.hpp"
#include "memory/allocator.hpp"
#include "memory/memory_pool.hpp"

namespace aic {

class Storage {
public:
    // Allocate new storage through the appropriate pool
    Storage(size_t nbytes, Device device, IAllocator* allocator = nullptr);

    // Wrap pre-allocated raw memory (with optional custom deleter / non-owning)
    Storage(void* ptr, size_t nbytes, Device device, bool owns_memory = false);

    ~Storage() = default;

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) noexcept = default;
    Storage& operator=(Storage&&) noexcept = default;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    template <typename T>
    T* data_as() noexcept {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    const T* data_as() const noexcept {
        return reinterpret_cast<const T*>(data_);
    }

    size_t nbytes() const noexcept { return nbytes_; }
    Device device() const noexcept { return device_; }
    IAllocator* allocator() const noexcept { return allocator_; }

    // Creates a new independent storage with copied data
    std::shared_ptr<Storage> clone() const;

private:
    void* data_ = nullptr;
    size_t nbytes_ = 0;
    Device device_;
    IAllocator* allocator_ = nullptr;
    AllocationHandle handle_;
};

}  // namespace aic
