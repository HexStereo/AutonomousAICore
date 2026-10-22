// src/cuda/cuda_quantize.cuh
// CUDA Blackwell sm_120 Dequantization & Quantized Kernel Declarations
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aic::cuda {

void launch_dequantize_int4(
    const uint8_t* d_packed,
    const float* d_scales,
    const float* d_zero_points,
    float* d_output,
    int64_t numel,
    int32_t group_size,
    bool is_signed,
    cudaStream_t stream = nullptr
);

void launch_dequantize_int8(
    const uint8_t* d_packed,
    const float* d_scales,
    const float* d_zero_points,
    float* d_output,
    int64_t numel,
    int32_t group_size,
    bool is_signed,
    cudaStream_t stream = nullptr
);

void launch_dequantize_fp8(
    const uint8_t* d_packed,
    const float* d_scales,
    float* d_output,
    int64_t numel,
    int32_t group_size,
    bool is_e4m3,
    cudaStream_t stream = nullptr
);

} // namespace aic::cuda
