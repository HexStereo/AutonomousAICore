// src/cuda/cuda_flash_attention.cu
// High-performance CUDA Kernels for FlashAttention-2 on NVIDIA Blackwell sm_120.
#include "cuda/cuda_flash_attention.cuh"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace aic::cuda {

namespace {

constexpr int BLOCK_THREADS = 64;
constexpr int MAX_HEAD_DIM = 128;

__global__ void flash_attention_forward_kernel(
    float* __restrict__ out,
    float* __restrict__ logsumexp,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    bool is_causal,
    float scale
) {
    int r = blockIdx.x;  // Query token row
    int h = blockIdx.y;  // Head
    int b = blockIdx.z;  // Batch
    int tid = threadIdx.x;

    if (b >= batch_size || h >= num_heads || r >= seq_len_q) return;

    __shared__ float s_q[MAX_HEAD_DIM];
    __shared__ float s_acc[MAX_HEAD_DIM];
    __shared__ float s_scores[BLOCK_THREADS];
    __shared__ float s_m;
    __shared__ float s_l;

    int64_t bh_offset_q = ((static_cast<int64_t>(b) * num_heads + h) * seq_len_q + r) * head_dim;
    int64_t bh_base_k = (static_cast<int64_t>(b) * num_heads + h) * seq_len_k * head_dim;

    // 1. Cooperatively load Query row and initialize accumulator
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[bh_offset_q + d];
        s_acc[d] = 0.0f;
    }
    if (tid == 0) {
        s_m = -1e20f;
        s_l = 0.0f;
    }
    __syncthreads();

    // 2. Iterate over Key & Value tokens in tiles of size BLOCK_THREADS
    for (int j_b = 0; j_b < seq_len_k; j_b += BLOCK_THREADS) {
        if (is_causal && j_b > r) {
            break;
        }

        int c = j_b + tid;
        float dot = -1e20f;

        if (c < seq_len_k && (!is_causal || c <= r)) {
            int64_t k_idx = bh_base_k + static_cast<int64_t>(c) * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += s_q[d] * k[k_idx + d];
            }
            dot = sum * scale;
        }

        s_scores[tid] = dot;
        __syncthreads();

        // Find block max reduction
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                if (s_scores[tid + offset] > s_scores[tid]) {
                    s_scores[tid] = s_scores[tid + offset];
                }
            }
            __syncthreads();
        }

        float block_max = s_scores[0];
        __syncthreads();

        if (block_max > -1e19f) {
            float old_m = s_m;
            float new_m = fmaxf(old_m, block_max);
            float alpha = (old_m <= -1e19f) ? 0.0f : expf(old_m - new_m);

            // Rescale accumulator
            for (int d = tid; d < head_dim; d += blockDim.x) {
                s_acc[d] *= alpha;
            }

            // Compute unnormalized probability P
            float p = 0.0f;
            if (c < seq_len_k && (!is_causal || c <= r)) {
                p = expf(dot - new_m);
            }
            s_scores[tid] = p;
            __syncthreads();

            // Accumulate P * V into shared accumulator
            if (tid < head_dim) {
                float sum_pv = 0.0f;
                for (int step = 0; step < BLOCK_THREADS; ++step) {
                    int c_curr = j_b + step;
                    if (c_curr < seq_len_k && (!is_causal || c_curr <= r)) {
                        float p_val = s_scores[step];
                        int64_t v_idx = bh_base_k + static_cast<int64_t>(c_curr) * head_dim + tid;
                        sum_pv += p_val * v[v_idx];
                    }
                }
                s_acc[tid] += sum_pv;
            }
            __syncthreads();

            // Sum reduction for normalizer l
            for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
                if (tid < offset) {
                    s_scores[tid] += s_scores[tid + offset];
                }
                __syncthreads();
            }

            if (tid == 0) {
                s_l = s_l * alpha + s_scores[0];
                s_m = new_m;
            }
            __syncthreads();
        }
    }

    // 3. Final normalization and store output & logsumexp
    float final_l = s_l;
    float final_m = s_m;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        out[bh_offset_q + d] = s_acc[d] * inv_l;
    }

    if (tid == 0) {
        int64_t lse_idx = (static_cast<int64_t>(b) * num_heads + h) * seq_len_q + r;
        logsumexp[lse_idx] = (final_l > 0.0f) ? (final_m + logf(final_l)) : 0.0f;
    }
}

__global__ void flash_attention_precompute_d_kernel(
    float* __restrict__ D,
    const float* __restrict__ grad_out,
    const float* __restrict__ out,
    int total_rows,
    int head_dim
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total_rows) return;

    __shared__ float s_red[MAX_HEAD_DIM];

    float val = 0.0f;
    int64_t offset = static_cast<int64_t>(row) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        val += grad_out[offset + d] * out[offset + d];
    }
    s_red[tid] = val;
    __syncthreads();

    for (int offset_r = blockDim.x / 2; offset_r > 0; offset_r /= 2) {
        if (tid < offset_r) {
            s_red[tid] += s_red[tid + offset_r];
        }
        __syncthreads();
    }

    if (tid == 0) {
        D[row] = s_red[0];
    }
}

__global__ void flash_attention_backward_kernel(
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    const float* __restrict__ D,
    const float* __restrict__ grad_out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ logsumexp,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    bool is_causal,
    float scale
) {
    int r = blockIdx.x;  // Query row
    int h = blockIdx.y;  // Head
    int b = blockIdx.z;  // Batch
    int tid = threadIdx.x;

    if (b >= batch_size || h >= num_heads || r >= seq_len_q) return;

    __shared__ float s_q[MAX_HEAD_DIM];
    __shared__ float s_dO[MAX_HEAD_DIM];
    __shared__ float s_dQ[MAX_HEAD_DIM];

    int64_t bh_offset_q = ((static_cast<int64_t>(b) * num_heads + h) * seq_len_q + r) * head_dim;
    int64_t bh_base_k = (static_cast<int64_t>(b) * num_heads + h) * seq_len_k * head_dim;
    int64_t row_idx = (static_cast<int64_t>(b) * num_heads + h) * seq_len_q + r;

    float D_val = D[row_idx];
    float l_val = logsumexp[row_idx];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[bh_offset_q + d];
        s_dO[d] = grad_out[bh_offset_q + d];
        s_dQ[d] = 0.0f;
    }
    __syncthreads();

    int max_c = is_causal ? (r + 1 < seq_len_k ? r + 1 : seq_len_k) : seq_len_k;

    // Loop over keys
    for (int c = tid; c < max_c; c += blockDim.x) {
        int64_t k_idx = bh_base_k + static_cast<int64_t>(c) * head_dim;

        // 1. Dot product S_rc
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += s_q[d] * k[k_idx + d];
        }
        dot *= scale;

        // 2. P_rc = exp(S_rc - L_r)
        float p_rc = expf(dot - l_val);

        // 3. dP_rc = dO_r . V_c
        float dp_rc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dp_rc += s_dO[d] * v[k_idx + d];
        }

        // 4. dS_rc = P_rc * (dP_rc - D_val) * scale
        float ds_rc = p_rc * (dp_rc - D_val) * scale;

        // 5. Accumulate into dQ in shared memory (warp reduction or atomics)
        for (int d = 0; d < head_dim; ++d) {
            atomicAdd(&s_dQ[d], ds_rc * k[k_idx + d]);
            atomicAdd(&grad_k[k_idx + d], ds_rc * s_q[d]);
            atomicAdd(&grad_v[k_idx + d], p_rc * s_dO[d]);
        }
    }
    __syncthreads();

    // Store dQ
    for (int d = tid; d < head_dim; d += blockDim.x) {
        grad_q[bh_offset_q + d] = s_dQ[d];
    }
}

}  // namespace

void launch_flash_attention_forward_gpu(
    float* out,
    float* logsumexp,
    const float* q,
    const float* k,
    const float* v,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    bool is_causal,
    float scale,
    cudaStream_t stream
) {
    dim3 grid(seq_len_q, num_heads, batch_size);
    dim3 block(BLOCK_THREADS);

    flash_attention_forward_kernel<<<grid, block, 0, stream>>>(
        out, logsumexp, q, k, v,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        is_causal, scale
    );
}

void launch_flash_attention_backward_gpu(
    float* grad_q,
    float* grad_k,
    float* grad_v,
    const float* grad_out,
    const float* q,
    const float* k,
    const float* v,
    const float* out,
    const float* logsumexp,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    bool is_causal,
    float scale,
    cudaStream_t stream
) {
    int total_rows = batch_size * num_heads * seq_len_q;

    // Allocate temporary D buffer
    float* d_D = nullptr;
    cudaMallocAsync(&d_D, static_cast<size_t>(total_rows) * sizeof(float), stream);

    int block_dim = head_dim < 64 ? head_dim : 64;
    flash_attention_precompute_d_kernel<<<total_rows, block_dim, 0, stream>>>(
        d_D, grad_out, out, total_rows, head_dim
    );

    dim3 grid(seq_len_q, num_heads, batch_size);
    dim3 block(BLOCK_THREADS);

    flash_attention_backward_kernel<<<grid, block, 0, stream>>>(
        grad_q, grad_k, grad_v,
        d_D, grad_out, q, k, v, logsumexp,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        is_causal, scale
    );

    cudaFreeAsync(d_D, stream);
}

}  // namespace aic::cuda
#endif
