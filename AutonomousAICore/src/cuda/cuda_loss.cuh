// src/cuda/cuda_loss.cuh
// CUDA kernel declarations for Cross-Entropy Loss forward and backward passes.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aic::cuda {

// Fused LogSumExp Cross Entropy forward
// logits: (N, C) float array in GPU memory
// targets_int64: (N) int64 array in GPU memory (if is_index_target == true)
// targets_float: (N, C) float array in GPU memory (if is_index_target == false)
// sample_losses: (N) float array output
// lse_out: (N) float array output
// valid_count_out: pointer to int64 count of non-ignored samples
void cuda_cross_entropy_forward(
    const float* logits,
    const int64_t* targets_int64,
    const float* targets_float,
    bool is_index_target,
    int64_t N,
    int64_t C,
    int64_t ignore_index,
    float* sample_losses,
    float* lse_out,
    int64_t* valid_count_out,
    cudaStream_t stream = nullptr
);

// Fused Cross Entropy analytical backward
// grad_output: incoming gradient (scalar or per-sample)
// logits: (N, C) float array
// targets_int64: (N) int64 array (if is_index_target == true)
// targets_float: (N, C) float array (if is_index_target == false)
// lse: (N) float array
// grad_logits: (N, C) float array output
void cuda_cross_entropy_backward(
    const float* grad_output,
    const float* logits,
    const int64_t* targets_int64,
    const float* targets_float,
    bool is_index_target,
    const float* lse,
    int64_t N,
    int64_t C,
    int64_t ignore_index,
    const char* reduction,
    int64_t valid_count,
    float* grad_logits,
    cudaStream_t stream = nullptr
);

}  // namespace aic::cuda
