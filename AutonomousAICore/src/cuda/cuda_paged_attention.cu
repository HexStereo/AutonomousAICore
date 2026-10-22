// src/cuda/cuda_paged_attention.cu
// High-performance CUDA Kernels for Paged Attention on NVIDIA Blackwell sm_120.
#include "core/dtype.hpp"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace aic::cuda {

__global__ void write_paged_kv_kernel(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ slot_indices,
    int num_seqs,
    int num_kv_heads,
    int block_size,
    int head_dim,
    int max_blocks
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int d = threadIdx.x;

    if (seq_idx >= num_seqs || head_idx >= num_kv_heads || d >= head_dim) return;

    int32_t slot = slot_indices[seq_idx];
    if (slot < 0) return;

    int32_t block_idx = slot / block_size;
    int32_t block_offset = slot % block_size;
    int32_t phys_block = block_table[seq_idx * max_blocks + block_idx];

    int64_t src_idx = (static_cast<int64_t>(seq_idx) * num_kv_heads + head_idx) * head_dim + d;
    int64_t dst_idx = ((static_cast<int64_t>(phys_block) * num_kv_heads + head_idx) * block_size + block_offset) * head_dim + d;

    key_cache[dst_idx] = key[src_idx];
    value_cache[dst_idx] = value[src_idx];
}

__global__ void paged_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int head_dim,
    int max_blocks,
    float scale
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    int32_t context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    // GQA/MQA mapping
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    extern __shared__ float s_mem[];
    float* s_query = s_mem;                               // size: head_dim
    float* s_logits = s_mem + head_dim;                   // size: context_len

    // 1. Cooperatively load Query vector into shared memory
    if (tid < head_dim) {
        int64_t q_idx = (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_dim + tid;
        s_query[tid] = query[q_idx];
    }
    __syncthreads();

    // 2. Each thread computes attention dot products for a subset of historical tokens
    float local_max = -1e20f;
    for (int t = tid; t < context_len; t += blockDim.x) {
        int32_t b_idx = t / block_size;
        int32_t b_offset = t % block_size;
        int32_t phys_block = block_table[seq_idx * max_blocks + b_idx];

        const float* k_ptr = key_cache +
            ((static_cast<int64_t>(phys_block) * num_kv_heads + kv_head_idx) * block_size + b_offset) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += s_query[d] * k_ptr[d];
        }
        dot *= scale;
        s_logits[t] = dot;
        if (dot > local_max) local_max = dot;
    }

    // Warp-level and block-level reduction to find global maximum logit
    __shared__ float s_max_val;
    __shared__ float s_sum_exp;

    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    if ((tid & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&s_max_val), __float_as_int(local_max));
    }
    __syncthreads();

    // 3. Compute Softmax denominators and exponentiated weights
    float local_sum = 0.0f;
    float max_logit = s_max_val;

    for (int t = tid; t < context_len; t += blockDim.x) {
        float exp_v = expf(s_logits[t] - max_logit);
        s_logits[t] = exp_v;
        local_sum += exp_v;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if ((tid & 31) == 0) {
        atomicAdd(&s_sum_exp, local_sum);
    }
    __syncthreads();

    float inv_sum = 1.0f / fmaxf(s_sum_exp, 1e-12f);

    // 4. Compute Weighted Sum of Values: Out = sum_t (P_t * V_t)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int t = 0; t < context_len; ++t) {
            float p = s_logits[t] * inv_sum;
            int32_t b_idx = t / block_size;
            int32_t b_offset = t % block_size;
            int32_t phys_block = block_table[seq_idx * max_blocks + b_idx];

            const float* v_ptr = value_cache +
                ((static_cast<int64_t>(phys_block) * num_kv_heads + kv_head_idx) * block_size + b_offset) * head_dim;

            out_val += p * v_ptr[d];
        }

        int64_t out_idx = (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_dim + d;
        out[out_idx] = out_val;
    }
}

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
) {
    dim3 grid(num_seqs, num_kv_heads);
    int block = head_dim > 256 ? 256 : head_dim;
    write_paged_kv_kernel<<<grid, block, 0, stream>>>(
        key_cache, value_cache, key, value, block_table, slot_indices,
        num_seqs, num_kv_heads, block_size, head_dim, max_blocks
    );
}

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
) {
    dim3 grid(num_seqs, num_heads);
    int block = 128;
    size_t shared_mem = (static_cast<size_t>(head_dim) + static_cast<size_t>(max_context_len)) * sizeof(float);

    paged_attention_kernel<<<grid, block, shared_mem, stream>>>(
        out, query, key_cache, value_cache, block_table, context_lens,
        num_seqs, num_heads, num_kv_heads, block_size, head_dim, max_blocks, scale
    );
}

}  // namespace aic::cuda

#endif
