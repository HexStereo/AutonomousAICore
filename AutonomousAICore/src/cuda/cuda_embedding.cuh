// src/cuda/cuda_embedding.cuh
// CUDA kernel declarations for Embedding lookup table.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aic::cuda {

void cuda_embedding_forward(
    const int64_t* tokens,
    const float* weight,
    float* out,
    int64_t N,
    int64_t D,
    int64_t V,
    cudaStream_t stream = nullptr
);

void cuda_embedding_backward(
    const float* grad_output,
    const int64_t* tokens,
    float* grad_weight,
    int64_t N,
    int64_t D,
    int64_t V,
    cudaStream_t stream = nullptr
);

}  // namespace aic::cuda
