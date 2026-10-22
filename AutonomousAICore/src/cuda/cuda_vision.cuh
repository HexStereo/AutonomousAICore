// src/cuda/cuda_vision.cuh
// CUDA Vision Kernels for Blackwell sm_120: Bilinear Resizing and Patchification
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aic::cuda {

void launch_bilinear_resize(
    const float* d_in,
    float* d_out,
    int64_t B,
    int64_t C,
    int64_t in_H,
    int64_t in_W,
    int64_t out_H,
    int64_t out_W,
    cudaStream_t stream = nullptr
);

void launch_patchify(
    const float* d_images,
    float* d_patches,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t P,
    cudaStream_t stream = nullptr
);

} // namespace aic::cuda
