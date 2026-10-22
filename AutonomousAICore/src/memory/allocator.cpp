// src/memory/allocator.cpp
#include "memory/allocator.hpp"

#include <iomanip>
#include <sstream>

namespace aic {

std::string MemoryStats::str() const {
    auto to_mb = [](size_t bytes) -> double {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };

    std::ostringstream os;
    os << std::fixed << std::setprecision(2);
    os << "MemoryStats[allocated=" << to_mb(allocated_bytes.load(std::memory_order_relaxed)) << " MB, "
       << "peak=" << to_mb(peak_allocated_bytes.load(std::memory_order_relaxed)) << " MB, "
       << "reserved=" << to_mb(reserved_bytes.load(std::memory_order_relaxed)) << " MB, "
       << "active_allocs=" << active_allocations.load(std::memory_order_relaxed) << ", "
       << "total_allocs=" << total_alloc_count.load(std::memory_order_relaxed) << ", "
       << "total_frees=" << total_free_count.load(std::memory_order_relaxed) << "]";
    return os.str();
}

}  // namespace aic
