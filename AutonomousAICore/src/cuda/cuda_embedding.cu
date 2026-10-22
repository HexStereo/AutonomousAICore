// src/cuda/cuda_embedding.cu
// High-performance CUDA kernels for Embedding lookup forward and backward scatter-add.
#include "cuda/cuda_embedding.cuh"
#include "core/error.hpp"

#include <cuda_runtime.h>

#define AIC_CUDA_CHECK_EMB(call)                                                       \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            AIC_THROW(CudaError, "CUDA error in embedding kernel: ",                   \
                      cudaGetErrorString(err__));                                      \
        }                                                                              \
    } while (false)

namespace aic::cuda {

namespace {

__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ tokens,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t N,
    int64_t D,
    int64_t V
) {
    int64_t total_elements = N * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int64_t token_idx = idx / D;
        int64_t dim_idx = idx % D;

        int64_t token_id = tokens[token_idx];
        if (token_id >= 0 && token_id < V) {
            out[idx] = weight[token_id * D + dim_idx];
        } else {
            out[idx] = 0.0f;
        }
    }
}

__global__ void embedding_backward_kernel(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ tokens,
    float* __restrict__ grad_weight,
    int64_t N,
    int64_t D,
    int64_t V
) {
    int64_t total_elements = N * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int64_t token_idx = idx / D;
        int64_t dim_idx = idx % D;

        int64_t token_id = tokens[token_idx];
        if (token_id >= 0 && token_id < V) {
            atomicAdd(&grad_weight[token_id * D + dim_idx], grad_output[idx]);
        }
    }
}

}  // namespace

void cuda_embedding_forward(
    const int64_t* tokens,
    const float* weight,
    float* out,
    int64_t N,
    int64_t D,
    int64_t V,
    cudaStream_t stream
) {
    int64_t total = N * D;
    if (total <= 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    embedding_forward_kernel<<<blocks, threads, 0, stream>>>(tokens, weight, out, N, D, V);
    AIC_CUDA_CHECK_EMB(cudaGetLastError());
}

void cuda_embedding_backward(
    const float* grad_output,
    const int64_t* tokens,
    float* grad_weight,
    int64_t N,
    int64_t D,
    int64_t V,
    cudaStream_t stream
) {
    int64_t total = N * D;
    if (total <= 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    embedding_backward_kernel<<<blocks, threads, 0, stream>>>(grad_output, tokens, grad_weight, N, D, V);
    AIC_CUDA_CHECK_EMB(cudaGetLastError());
}

}  // namespace aic::cuda
