// include/memory/allocator.hpp
// Abstract allocator interface and allocation metadata.
#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <string>

#include "core/device.hpp"

namespace aic {

struct MemoryStats {
    std::atomic<size_t> allocated_bytes{0};
    std::atomic<size_t> peak_allocated_bytes{0};
    std::atomic<size_t> reserved_bytes{0};
    std::atomic<size_t> active_allocations{0};
    std::atomic<size_t> total_alloc_count{0};
    std::atomic<size_t> total_free_count{0};

    void record_alloc(size_t bytes, size_t reserved_delta = 0) noexcept {
        allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
        active_allocations.fetch_add(1, std::memory_order_relaxed);
        total_alloc_count.fetch_add(1, std::memory_order_relaxed);
        if (reserved_delta > 0) {
            reserved_bytes.fetch_add(reserved_delta, std::memory_order_relaxed);
        }
        size_t current = allocated_bytes.load(std::memory_order_relaxed);
        size_t prev_peak = peak_allocated_bytes.load(std::memory_order_relaxed);
        while (current > prev_peak &&
               !peak_allocated_bytes.compare_exchange_weak(prev_peak, current, std::memory_order_relaxed)) {
        }
    }

    void record_free(size_t bytes, size_t reserved_delta = 0) noexcept {
        allocated_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        active_allocations.fetch_sub(1, std::memory_order_relaxed);
        total_free_count.fetch_add(1, std::memory_order_relaxed);
        if (reserved_delta > 0) {
            reserved_bytes.fetch_sub(reserved_delta, std::memory_order_relaxed);
        }
    }

    void reset_peak() noexcept {
        peak_allocated_bytes.store(allocated_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    std::string str() const;
};

class IAllocator {
public:
    virtual ~IAllocator() = default;

    virtual void* allocate(size_t bytes, size_t alignment = 64) = 0;
    virtual void deallocate(void* ptr, size_t bytes) noexcept = 0;

    virtual Device device() const noexcept = 0;
    virtual const MemoryStats& stats() const noexcept = 0;
    virtual void reset_peak_stats() noexcept = 0;
    virtual void trim() = 0;
};

}  // namespace aic
