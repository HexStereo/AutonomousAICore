// src/cuda/cuda_quantize.cu
// CUDA Blackwell sm_120 Dequantization Kernels for INT4, INT8, and FP8
#include "cuda_quantize.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace aic::cuda {

__device__ inline float cuda_fp8_e4m3_to_float(uint8_t b) {
    uint32_t sign = (b >> 7) & 0x01;
    uint32_t exp  = (b >> 3) & 0x0F;
    uint32_t mant = b & 0x07;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
        float val = static_cast<float>(mant) / 8.0f * (1.0f / 64.0f);
        return sign ? -val : val;
    }
    // In E4M3, 0x7F (01111111) and 0xFF (11111111) represent NaN
    if (exp == 0x0F && mant == 0x07) {
        return __int_as_float(0x7fffffff);
    }
    // Normal: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
    int unbiased_exp = static_cast<int>(exp) - 7;
    float scale = std::ldexp(1.0f, unbiased_exp);
    float val = scale * (1.0f + static_cast<float>(mant) / 8.0f);
    return sign ? -val : val;
}

__device__ inline float cuda_fp8_e5m2_to_float(uint8_t b) {
    uint32_t sign = (b >> 7) & 0x01;
    uint32_t exp  = (b >> 2) & 0x1F;
    uint32_t mant = b & 0x03;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal: (-1)^sign * 2^(-14) * (mant / 4)
        float val = static_cast<float>(mant) / 4.0f * (1.0f / 16384.0f);
        return sign ? -val : val;
    }
    if (exp == 0x1F) {
        if (mant == 0) return sign ? -__int_as_float(0x7f800000) : __int_as_float(0x7f800000);
        return __int_as_float(0x7fffffff);
    }
    // Normal: (-1)^sign * 2^(exp - 15) * (1 + mant / 4)
    int unbiased_exp = static_cast<int>(exp) - 15;
    float scale = std::ldexp(1.0f, unbiased_exp);
    float val = scale * (1.0f + static_cast<float>(mant) / 4.0f);
    return sign ? -val : val;
}

__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ d_packed,
    const float* __restrict__ d_scales,
    const float* __restrict__ d_zero_points,
    float* __restrict__ d_output,
    int64_t numel,
    int32_t group_size,
    bool is_signed
) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = (numel + 1) / 2;
    if (pair_idx >= total_pairs) return;

    uint8_t byte = d_packed[pair_idx];
    uint8_t u_low = byte & 0x0F;
    uint8_t u_high = (byte >> 4) & 0x0F;

    float val0, val1;
    if (is_signed) {
        val0 = (u_low >= 8) ? static_cast<float>(static_cast<int>(u_low) - 16) : static_cast<float>(u_low);
        val1 = (u_high >= 8) ? static_cast<float>(static_cast<int>(u_high) - 16) : static_cast<float>(u_high);
    } else {
        val0 = static_cast<float>(u_low);
        val1 = static_cast<float>(u_high);
    }

    int64_t idx0 = pair_idx * 2;
    int64_t idx1 = idx0 + 1;

    int64_t g0 = (group_size > 0) ? (idx0 / group_size) : 0;
    float s0 = d_scales[g0];
    float zp0 = d_zero_points ? d_zero_points[g0] : 0.0f;
    d_output[idx0] = (val0 - zp0) * s0;

    if (idx1 < numel) {
        int64_t g1 = (group_size > 0) ? (idx1 / group_size) : 0;
        float s1 = d_scales[g1];
        float zp1 = d_zero_points ? d_zero_points[g1] : 0.0f;
        d_output[idx1] = (val1 - zp1) * s1;
    }
}

__global__ void dequantize_int8_kernel(
    const uint8_t* __restrict__ d_packed,
    const float* __restrict__ d_scales,
    const float* __restrict__ d_zero_points,
    float* __restrict__ d_output,
    int64_t numel,
    int32_t group_size,
    bool is_signed
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    uint8_t raw = d_packed[idx];
    float val;
    if (is_signed) {
        val = static_cast<float>(static_cast<int8_t>(raw));
    } else {
        val = static_cast<float>(raw);
    }

    int64_t g = (group_size > 0) ? (idx / group_size) : 0;
    float s = d_scales[g];
    float zp = d_zero_points ? d_zero_points[g] : 0.0f;
    d_output[idx] = (val - zp) * s;
}

__global__ void dequantize_fp8_kernel(
    const uint8_t* __restrict__ d_packed,
    const float* __restrict__ d_scales,
    float* __restrict__ d_output,
    int64_t numel,
    int32_t group_size,
    bool is_e4m3
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    uint8_t raw = d_packed[idx];
    float val = is_e4m3 ? cuda_fp8_e4m3_to_float(raw) : cuda_fp8_e5m2_to_float(raw);

    int64_t g = (group_size > 0) ? (idx / group_size) : 0;
    float s = d_scales ? d_scales[g] : 1.0f;
    d_output[idx] = val * s;
}

void launch_dequantize_int4(
    const uint8_t* d_packed,
    const float* d_scales,
    const float* d_zero_points,
    float* d_output,
    int64_t numel,
    int32_t group_size,
    bool is_signed,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    int64_t total_pairs = (numel + 1) / 2;
    int threads = 256;
    int blocks = static_cast<int>((total_pairs + threads - 1) / threads);

    dequantize_int4_kernel<<<blocks, threads, 0, stream>>>(
        d_packed, d_scales, d_zero_points, d_output, numel, group_size, is_signed
    );
}

void launch_dequantize_int8(
    const uint8_t* d_packed,
    const float* d_scales,
    const float* d_zero_points,
    float* d_output,
    int64_t numel,
    int32_t group_size,
    bool is_signed,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);

    dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(
        d_packed, d_scales, d_zero_points, d_output, numel, group_size, is_signed
    );
}

void launch_dequantize_fp8(
    const uint8_t* d_packed,
    const float* d_scales,
    float* d_output,
    int64_t numel,
    int32_t group_size,
    bool is_e4m3,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);

    dequantize_fp8_kernel<<<blocks, threads, 0, stream>>>(
        d_packed, d_scales, d_output, numel, group_size, is_e4m3
    );
}

} // namespace aic::cuda
