// src/cuda/cuda_flash_attention.cuh
// CUDA FlashAttention-2 kernels header for NVIDIA Blackwell sm_120.
#pragma once

#include "tensor/tensor.hpp"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>

namespace aic::cuda {

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
    cudaStream_t stream = nullptr
);

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
    cudaStream_t stream = nullptr
);

}  // namespace aic::cuda
#endif
