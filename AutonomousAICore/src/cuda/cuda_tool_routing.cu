// src/cuda/cuda_tool_routing.cu
#include "cuda/cuda_tool_routing.cuh"

#include <cmath>
#include <algorithm>

namespace aic::cuda {

#if AIC_WITH_CUDA

// Warp-reduction helper for dot product and norm summation
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: One block per tool (or 1 warp per tool)
__global__ void k_tool_embedding_similarity(
    const float* __restrict__ query_embed,
    const float* __restrict__ tool_embeds,
    float* __restrict__ scores,
    int num_tools,
    int embed_dim
) {
    int tool_idx = blockIdx.x;
    if (tool_idx >= num_tools) return;

    const float* t_ptr = tool_embeds + tool_idx * embed_dim;

    float thread_dot = 0.0f;
    float thread_q_norm2 = 0.0f;
    float thread_t_norm2 = 0.0f;

    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float q = query_embed[d];
        float t = t_ptr[d];
        thread_dot += q * t;
        thread_q_norm2 += q * q;
        thread_t_norm2 += t * t;
    }

    // Warp-level reduce
    thread_dot = warp_reduce_sum(thread_dot);
    thread_q_norm2 = warp_reduce_sum(thread_q_norm2);
    thread_t_norm2 = warp_reduce_sum(thread_t_norm2);

    __shared__ float s_dot[32];
    __shared__ float s_q[32];
    __shared__ float s_t[32];

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    if (lane == 0) {
        s_dot[wid] = thread_dot;
        s_q[wid] = thread_q_norm2;
        s_t[wid] = thread_t_norm2;
    }
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (wid == 0) {
        float block_dot = (lane < num_warps) ? s_dot[lane] : 0.0f;
        float block_q = (lane < num_warps) ? s_q[lane] : 0.0f;
        float block_t = (lane < num_warps) ? s_t[lane] : 0.0f;

        block_dot = warp_reduce_sum(block_dot);
        block_q = warp_reduce_sum(block_q);
        block_t = warp_reduce_sum(block_t);

        if (lane == 0) {
            float denom = sqrtf(block_q) * sqrtf(block_t) + 1e-8f;
            scores[tool_idx] = block_dot / denom;
        }
    }
}

void launch_tool_embedding_similarity(
    const float* query_embed,
    const float* tool_embeds,
    float* scores,
    int num_tools,
    int embed_dim,
    cudaStream_t stream
) {
    if (num_tools <= 0 || embed_dim <= 0) return;

    int threads = 128;
    if (embed_dim <= 64) threads = 64;
    else if (embed_dim >= 512) threads = 256;

    k_tool_embedding_similarity<<<num_tools, threads, 0, stream>>>(
        query_embed,
        tool_embeds,
        scores,
        num_tools,
        embed_dim
    );
}

#endif // AIC_WITH_CUDA

} // namespace aic::cuda
