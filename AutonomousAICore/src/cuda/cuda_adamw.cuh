// src/cuda/cuda_adamw.cuh
// CUDA kernel declaration for AdamW fused parameter update.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aic::cuda {

void cuda_adamw_step(
    float* p,
    const float* g,
    float* m,
    float* v,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    cudaStream_t stream = nullptr
);

}  // namespace aic::cuda
