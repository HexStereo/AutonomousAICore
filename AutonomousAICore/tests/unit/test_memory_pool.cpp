// tests/unit/test_memory_pool.cpp
#include "memory/memory_pool.hpp"
#include "tests/test_framework.hpp"

#include <thread>
#include <vector>

using namespace aic;

AIC_TEST(MemoryPoolTest, AlignmentAndAllocation) {
    HostMemoryPool pool;
    void* p1 = pool.allocate(128, 64);
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p1) % 64, 0);

    const auto& stats = pool.stats();
    ASSERT_TRUE(stats.allocated_bytes.load() >= 128);
    ASSERT_EQ(stats.active_allocations.load(), 1);

    pool.deallocate(p1, 128);
    ASSERT_EQ(stats.active_allocations.load(), 0);
}

AIC_TEST(MemoryPoolTest, SegregatedRecycling) {
    HostMemoryPool pool;
    void* p1 = pool.allocate(256, 64);
    pool.deallocate(p1, 256);

    // Reallocation should return cached block without new OS call
    void* p2 = pool.allocate(256, 64);
    ASSERT_EQ(p1, p2);

    pool.deallocate(p2, 256);
}

AIC_TEST(MemoryPoolTest, AllocationHandleRAII) {
    HostMemoryPool pool;
    {
        AllocationHandle handle(pool.allocate(512), 512, &pool);
        ASSERT_TRUE(handle.get() != nullptr);
        ASSERT_EQ(pool.stats().active_allocations.load(), 1);
    }
    // Auto-deallocated on scope exit
    ASSERT_EQ(pool.stats().active_allocations.load(), 0);
}

AIC_TEST(MemoryPoolTest, MultiThreadedStress) {
    HostMemoryPool pool;
    constexpr int num_threads = 4;
    constexpr int iters = 200;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool]() {
            for (int i = 0; i < iters; ++i) {
                size_t sz = 64 * ((i % 8) + 1);
                void* p = pool.allocate(sz, 64);
                // Touch memory to verify accessibility
                std::memset(p, 0xAB, sz);
                pool.deallocate(p, sz);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    ASSERT_EQ(pool.stats().active_allocations.load(), 0);
    ASSERT_EQ(pool.stats().total_alloc_count.load(), num_threads * iters);
    ASSERT_EQ(pool.stats().total_free_count.load(), num_threads * iters);

    // Trim all cached blocks
    pool.trim();
    ASSERT_EQ(pool.stats().reserved_bytes.load(), 0);
}
