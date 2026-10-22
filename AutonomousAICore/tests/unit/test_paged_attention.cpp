// tests/unit/test_paged_attention.cpp
// Unit and Parity tests for Paged Attention & Block Memory Management.
#include "memory/block_manager.hpp"
#include "ops/paged_attention.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace aic;
using namespace aic::memory;
using namespace aic::ops;

AIC_TEST(BlockManagerTest, AllocationAndFreeList) {
    BlockManagerConfig cfg;
    cfg.num_blocks = 16;
    cfg.block_size = 16;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.device = Device::cpu();

    BlockManager mgr(cfg);
    ASSERT_EQ(mgr.num_total_blocks(), 16);
    ASSERT_EQ(mgr.num_free_blocks(), 16);

    // Allocate for 35 tokens -> requires ceil(35/16) = 3 blocks
    BlockTable table = mgr.allocate(35);
    ASSERT_EQ(table.size(), 3);
    ASSERT_EQ(mgr.num_free_blocks(), 13);
    ASSERT_EQ(mgr.num_allocated_blocks(), 3);

    // Append 10 tokens: 35 + 10 = 45 <= 48 (capacity of 3 blocks) -> no new block needed
    mgr.append_slots(table, 35, 10);
    ASSERT_EQ(table.size(), 3);
    ASSERT_EQ(mgr.num_free_blocks(), 13);

    // Append 5 tokens: 45 + 5 = 50 > 48 -> 1 new block needed
    mgr.append_slots(table, 45, 5);
    ASSERT_EQ(table.size(), 4);
    ASSERT_EQ(mgr.num_free_blocks(), 12);

    // Free table
    mgr.free(table);
    ASSERT_EQ(mgr.num_free_blocks(), 16);
    ASSERT_EQ(mgr.num_allocated_blocks(), 0);
}

AIC_TEST(BlockManagerTest, ForkCopyOnWrite) {
    BlockManagerConfig cfg;
    cfg.num_blocks = 8;
    cfg.block_size = 16;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.device = Device::cpu();

    BlockManager mgr(cfg);

    // Sequence A has 2 blocks
    BlockTable table_a = mgr.allocate(30);
    ASSERT_EQ(table_a.size(), 2);
    ASSERT_EQ(mgr.get_ref_count(table_a[0]), 1);
    ASSERT_EQ(mgr.get_ref_count(table_a[1]), 1);
    ASSERT_EQ(mgr.num_free_blocks(), 6);

    // Fork sequence B (e.g. MCTS alternative reasoning branch)
    BlockTable table_b = mgr.fork(table_a);
    ASSERT_EQ(table_b.size(), 2);
    ASSERT_EQ(mgr.get_ref_count(table_a[0]), 2);
    ASSERT_EQ(mgr.get_ref_count(table_a[1]), 2);
    ASSERT_EQ(mgr.num_free_blocks(), 6); // Zero additional blocks allocated!

    // Free sequence A
    mgr.free(table_a);
    ASSERT_EQ(mgr.get_ref_count(table_b[0]), 1);
    ASSERT_EQ(mgr.get_ref_count(table_b[1]), 1);
    ASSERT_EQ(mgr.num_free_blocks(), 6); // Not freed yet because B still holds reference

    // Free sequence B
    mgr.free(table_b);
    ASSERT_EQ(mgr.num_free_blocks(), 8); // All blocks recycled
}

AIC_TEST(PagedAttentionTest, NumericalParityWithStandardAttention) {
    int num_seqs = 1;
    (void)num_seqs;
    int num_heads = 1;
    int head_dim = 4;
    int block_size = 4;
    int context_len = 6;  // Spans across 2 physical blocks (4 tokens in blk0, 2 in blk1)

    BlockManagerConfig cfg;
    cfg.num_blocks = 4;
    cfg.block_size = block_size;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.device = Device::cpu();

    BlockManager mgr(cfg);
    BlockTable table = mgr.allocate(context_len);

    // Prepare Query [1, 1, 4]
    Tensor query = Tensor::from_blob(std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f}.data(),
                                     Shape({1, 1, 4}), DType::Float32, Device::cpu()).clone();

    // Prepare block table tensor [1, 2]
    std::vector<int32_t> b_table_data = {table[0], table[1]};
    Tensor block_table = Tensor::from_blob(b_table_data.data(), Shape({1, 2}), DType::Int32, Device::cpu()).clone();

    // Write 6 tokens into physical cache
    std::vector<std::vector<float>> keys = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {0.5f, 0.5f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f, 1.0f}
    };
    std::vector<std::vector<float>> values = {
        {10.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 20.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 30.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 40.0f},
        {5.0f, 5.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f, 1.0f}
    };

    for (int t = 0; t < context_len; ++t) {
        Tensor k_t = Tensor::from_blob(keys[t].data(), Shape({1, 1, 4}), DType::Float32, Device::cpu()).clone();
        Tensor v_t = Tensor::from_blob(values[t].data(), Shape({1, 1, 4}), DType::Float32, Device::cpu()).clone();
        int32_t slot = t;
        Tensor slot_t = Tensor::from_blob(&slot, Shape({1}), DType::Int32, Device::cpu()).clone();

        write_paged_kv_cache(mgr.key_cache(), mgr.value_cache(), k_t, v_t, block_table, slot_t);
    }

    int32_t ctx_len_val = context_len;
    Tensor context_lens = Tensor::from_blob(&ctx_len_val, Shape({1}), DType::Int32, Device::cpu()).clone();

    float scale = 0.5f; // 1 / sqrt(4) = 0.5
    Tensor out = paged_attention(query, mgr.key_cache(), mgr.value_cache(), block_table, context_lens, scale);

    // Compute manual reference attention
    std::vector<float> logits(context_len);
    float max_logit = -1e20f;
    const float* q = query.data_ptr<float>();
    for (int t = 0; t < context_len; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * keys[t][d];
        }
        dot *= scale;
        logits[t] = dot;
        if (dot > max_logit) max_logit = dot;
    }

    float sum_exp = 0.0f;
    for (int t = 0; t < context_len; ++t) {
        logits[t] = std::exp(logits[t] - max_logit);
        sum_exp += logits[t];
    }

    std::vector<float> ref_out(head_dim, 0.0f);
    for (int d = 0; d < head_dim; ++d) {
        for (int t = 0; t < context_len; ++t) {
            float p = logits[t] / sum_exp;
            ref_out[d] += p * values[t][d];
        }
    }

    const float* paged_out = out.data_ptr<float>();
    for (int d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(paged_out[d], ref_out[d], 1e-4f);
    }
}

#if AIC_WITH_CUDA
AIC_TEST(PagedAttentionTest, CudaExecutionAndParity) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available for PagedAttention GPU tests\n";
        return;
    }

    Device gpu = Device::cuda(0);
    int num_seqs = 2;
    int num_heads = 2;
    int head_dim = 16;
    int block_size = 16;
    int context_len = 24; // 2 blocks per sequence

    BlockManagerConfig cfg_cpu;
    cfg_cpu.num_blocks = 8;
    cfg_cpu.block_size = block_size;
    cfg_cpu.num_heads = num_heads;
    cfg_cpu.head_dim = head_dim;
    cfg_cpu.device = Device::cpu();
    BlockManager mgr_cpu(cfg_cpu);

    BlockManagerConfig cfg_gpu = cfg_cpu;
    cfg_gpu.device = gpu;
    BlockManager mgr_gpu(cfg_gpu);

    // Prepare block table [2, 2]
    std::vector<int32_t> b_table = {0, 1, 2, 3};
    Tensor bt_cpu = Tensor::from_blob(b_table.data(), Shape({2, 2}), DType::Int32, Device::cpu()).clone();
    Tensor bt_gpu = bt_cpu.to(gpu);

    // Prepare queries
    std::vector<float> q_data(num_seqs * num_heads * head_dim);
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 7) * 0.1f;
    }
    Tensor q_cpu = Tensor::from_blob(q_data.data(), Shape({num_seqs, num_heads, head_dim}), DType::Float32, Device::cpu()).clone();
    Tensor q_gpu = q_cpu.to(gpu);

    // Populate KV caches with pseudo-random values on CPU and copy to GPU
    mgr_cpu.key_cache() = Tensor::uniform(mgr_cpu.key_cache().shape(), -0.5f, 0.5f, DType::Float32, Device::cpu());
    mgr_cpu.value_cache() = Tensor::uniform(mgr_cpu.value_cache().shape(), -0.5f, 0.5f, DType::Float32, Device::cpu());

    mgr_gpu.key_cache() = mgr_cpu.key_cache().to(gpu);
    mgr_gpu.value_cache() = mgr_cpu.value_cache().to(gpu);

    std::vector<int32_t> lens_data = {context_len, context_len - 4};
    Tensor lens_cpu = Tensor::from_blob(lens_data.data(), Shape({2}), DType::Int32, Device::cpu()).clone();
    Tensor lens_gpu = lens_cpu.to(gpu);

    Tensor out_cpu = paged_attention(q_cpu, mgr_cpu.key_cache(), mgr_cpu.value_cache(), bt_cpu, lens_cpu);
    Tensor out_gpu = paged_attention(q_gpu, mgr_gpu.key_cache(), mgr_gpu.value_cache(), bt_gpu, lens_gpu);

    Tensor out_gpu_cpu = out_gpu.cpu();
    const float* cpu_p = out_cpu.data_ptr<float>();
    const float* gpu_p = out_gpu_cpu.data_ptr<float>();

    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_NEAR(cpu_p[i], gpu_p[i], 1e-3f);
    }
}
#endif
