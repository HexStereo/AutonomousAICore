// src/ops/paged_attention.cpp
// Paged Attention CPU implementation and CUDA dispatcher.
#include "ops/paged_attention.hpp"
#include "core/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
namespace aic::cuda {
void launch_write_paged_kv_cache_gpu(
    float* key_cache,
    float* value_cache,
    const float* key,
    const float* value,
    const int32_t* block_table,
    const int32_t* slot_indices,
    int num_seqs,
    int num_kv_heads,
    int block_size,
    int head_dim,
    int max_blocks,
    cudaStream_t stream = nullptr
);

void launch_paged_attention_gpu(
    float* out,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int32_t* block_table,
    const int32_t* context_lens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int head_dim,
    int max_blocks,
    int max_context_len,
    float scale,
    cudaStream_t stream = nullptr
);
}
#endif

namespace aic::ops {

void write_paged_kv_cache(
    Tensor& key_cache,
    Tensor& value_cache,
    const Tensor& key,
    const Tensor& value,
    const Tensor& block_table,
    const Tensor& slot_indices
) {
    AIC_CHECK(key.defined() && value.defined(), InvalidArgument, "Key and Value tensors must be defined");
    AIC_CHECK(key_cache.defined() && value_cache.defined(), InvalidArgument, "KV cache tensors must be defined");
    AIC_CHECK(key.rank() == 3, InvalidArgument, "key must be 3D [num_seqs, num_kv_heads, head_dim]");
    AIC_CHECK(value.rank() == 3, InvalidArgument, "value must be 3D [num_seqs, num_kv_heads, head_dim]");
    AIC_CHECK(block_table.rank() == 2, InvalidArgument, "block_table must be 2D [num_seqs, max_blocks]");

    int num_seqs = static_cast<int>(key.dim(0));
    int num_kv_heads = static_cast<int>(key.dim(1));
    int head_dim = static_cast<int>(key.dim(2));
    int block_size = static_cast<int>(key_cache.dim(2));
    int max_blocks = static_cast<int>(block_table.dim(1));

#if AIC_WITH_CUDA
    if (key.is_cuda()) {
        cuda::launch_write_paged_kv_cache_gpu(
            key_cache.data_ptr<float>(),
            value_cache.data_ptr<float>(),
            key.contiguous().data_ptr<float>(),
            value.contiguous().data_ptr<float>(),
            block_table.contiguous().data_ptr<int32_t>(),
            slot_indices.contiguous().data_ptr<int32_t>(),
            num_seqs, num_kv_heads, block_size, head_dim, max_blocks
        );
        return;
    }
#endif

    // CPU Implementation
    float* k_cache_ptr = key_cache.data_ptr<float>();
    float* v_cache_ptr = value_cache.data_ptr<float>();
    const float* k_ptr = key.contiguous().data_ptr<float>();
    const float* v_ptr = value.contiguous().data_ptr<float>();
    const int32_t* b_table_ptr = block_table.contiguous().data_ptr<int32_t>();
    const int32_t* slots_ptr = slot_indices.contiguous().data_ptr<int32_t>();

    for (int seq = 0; seq < num_seqs; ++seq) {
        int32_t slot = slots_ptr[seq];
        if (slot < 0) continue;

        int32_t b_idx = slot / block_size;
        int32_t b_offset = slot % block_size;
        int32_t phys_block = b_table_ptr[seq * max_blocks + b_idx];

        for (int h = 0; h < num_kv_heads; ++h) {
            int64_t src_base = (static_cast<int64_t>(seq) * num_kv_heads + h) * head_dim;
            int64_t dst_base = ((static_cast<int64_t>(phys_block) * num_kv_heads + h) * block_size + b_offset) * head_dim;

            std::memcpy(k_cache_ptr + dst_base, k_ptr + src_base, head_dim * sizeof(float));
            std::memcpy(v_cache_ptr + dst_base, v_ptr + src_base, head_dim * sizeof(float));
        }
    }
}

Tensor paged_attention(
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& block_table,
    const Tensor& context_lens,
    float scale
) {
    AIC_CHECK(query.defined(), InvalidArgument, "query must be defined");
    AIC_CHECK(key_cache.defined() && value_cache.defined(), InvalidArgument, "KV caches must be defined");
    AIC_CHECK(query.rank() == 3, InvalidArgument, "query must be 3D [num_seqs, num_heads, head_dim]");
    AIC_CHECK(block_table.rank() == 2, InvalidArgument, "block_table must be 2D [num_seqs, max_blocks]");

    int num_seqs = static_cast<int>(query.dim(0));
    int num_heads = static_cast<int>(query.dim(1));
    int head_dim = static_cast<int>(query.dim(2));
    int num_kv_heads = static_cast<int>(key_cache.dim(1));
    int block_size = static_cast<int>(key_cache.dim(2));
    int max_blocks = static_cast<int>(block_table.dim(1));

    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    Tensor out = Tensor::empty(query.shape(), DType::Float32, query.device());

#if AIC_WITH_CUDA
    if (query.is_cuda()) {
        Tensor lens_cpu = context_lens.cpu().contiguous();
        const int32_t* lens_ptr = lens_cpu.data_ptr<int32_t>();
        int max_len = 1;
        for (int i = 0; i < num_seqs; ++i) {
            if (lens_ptr[i] > max_len) max_len = lens_ptr[i];
        }

        cuda::launch_paged_attention_gpu(
            out.data_ptr<float>(),
            query.contiguous().data_ptr<float>(),
            key_cache.contiguous().data_ptr<float>(),
            value_cache.contiguous().data_ptr<float>(),
            block_table.contiguous().data_ptr<int32_t>(),
            context_lens.contiguous().data_ptr<int32_t>(),
            num_seqs, num_heads, num_kv_heads, block_size, head_dim, max_blocks, max_len, scale
        );
        return out;
    }
#endif

    // CPU Implementation
    const float* q_ptr = query.contiguous().data_ptr<float>();
    const float* k_cache_ptr = key_cache.contiguous().data_ptr<float>();
    const float* v_cache_ptr = value_cache.contiguous().data_ptr<float>();
    const int32_t* b_table_ptr = block_table.contiguous().data_ptr<int32_t>();
    const int32_t* lens_ptr = context_lens.contiguous().data_ptr<int32_t>();
    float* out_ptr = out.data_ptr<float>();

    int gqa_ratio = num_heads / num_kv_heads;

    for (int seq = 0; seq < num_seqs; ++seq) {
        int32_t c_len = lens_ptr[seq];
        if (c_len <= 0) {
            std::memset(out_ptr + seq * num_heads * head_dim, 0, num_heads * head_dim * sizeof(float));
            continue;
        }

        std::vector<float> logits(static_cast<size_t>(c_len));

        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / gqa_ratio;
            const float* q_vec = q_ptr + (static_cast<int64_t>(seq) * num_heads + h) * head_dim;

            // 1. Calculate logits Q * K_t^T * scale
            float max_logit = -1e20f;
            for (int t = 0; t < c_len; ++t) {
                int32_t b_idx = t / block_size;
                int32_t b_offset = t % block_size;
                int32_t phys_block = b_table_ptr[seq * max_blocks + b_idx];

                const float* k_vec = k_cache_ptr +
                    ((static_cast<int64_t>(phys_block) * num_kv_heads + kv_h) * block_size + b_offset) * head_dim;

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                dot *= scale;
                logits[static_cast<size_t>(t)] = dot;
                if (dot > max_logit) max_logit = dot;
            }

            // 2. Softmax normalization
            double sum_exp = 0.0;
            for (int t = 0; t < c_len; ++t) {
                float exp_val = std::exp(logits[static_cast<size_t>(t)] - max_logit);
                logits[static_cast<size_t>(t)] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-12));

            // 3. Weighted sum over values
            float* out_vec = out_ptr + (static_cast<int64_t>(seq) * num_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                float sum_v = 0.0f;
                for (int t = 0; t < c_len; ++t) {
                    float p = logits[static_cast<size_t>(t)] * inv_sum;
                    int32_t b_idx = t / block_size;
                    int32_t b_offset = t % block_size;
                    int32_t phys_block = b_table_ptr[seq * max_blocks + b_idx];

                    const float* v_vec = v_cache_ptr +
                        ((static_cast<int64_t>(phys_block) * num_kv_heads + kv_h) * block_size + b_offset) * head_dim;

                    sum_v += p * v_vec[d];
                }
                out_vec[d] = sum_v;
            }
        }
    }

    return out;
}

}  // namespace aic::ops
