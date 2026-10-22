// src/cuda/cuda_grammar.cu
// CUDA Kernel for Grammar-Constrained Logit Masking
#include "cuda_grammar.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace aic::cuda {

__global__ void apply_logit_mask_kernel(
    float* __restrict__ d_logits,
    const uint8_t* __restrict__ d_valid_mask,
    int64_t B,
    int64_t V,
    float mask_val
) {
    int64_t total = B * V;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t v = idx % V;
    if (d_valid_mask[v] == 0) {
        d_logits[idx] = mask_val;
    }
}

void launch_apply_logit_mask(
    float* d_logits,
    const uint8_t* d_valid_mask,
    int64_t B,
    int64_t V,
    float mask_value,
    cudaStream_t stream
) {
    int64_t total = B * V;
    if (total <= 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    apply_logit_mask_kernel<<<blocks, threads, 0, stream>>>(
        d_logits, d_valid_mask, B, V, mask_value
    );
}

} // namespace aic::cuda
