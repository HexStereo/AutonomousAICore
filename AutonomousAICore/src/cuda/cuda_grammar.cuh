// src/cuda/cuda_grammar.cuh
// CUDA Logit Masking Kernel Declarations for Grammar-Constrained Decoding
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aic::cuda {

void launch_apply_logit_mask(
    float* d_logits,
    const uint8_t* d_valid_mask,
    int64_t B,
    int64_t V,
    float mask_value = -1e9f,
    cudaStream_t stream = nullptr
);

} // namespace aic::cuda
