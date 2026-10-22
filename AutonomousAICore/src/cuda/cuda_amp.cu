// src/cuda/cuda_amp.cu
// High-performance CUDA kernels for Mixed Precision: Fused NaN/Inf detection & Gradient Unscaling.
#include "core/dtype.hpp"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace aic::cuda {

__global__ void has_nan_or_inf_kernel(const float* __restrict__ data, int64_t n, int* __restrict__ out_flag) {
    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int local_found = 0;
    for (int64_t i = idx; i < n; i += stride) {
        float v = data[i];
        if (isnan(v) || isinf(v)) {
            local_found = 1;
            break;
        }
    }

    // Warp-level vote
    int warp_found = __any_sync(0xFFFFFFFF, local_found);

    // Write to block output if any thread in warp found NaN/Inf
    if ((tid & 31) == 0 && warp_found) {
        atomicOr(out_flag, 1);
    }
}

__global__ void unscale_kernel_fp32(float* __restrict__ data, int64_t n, float inv_scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Vectorized 128-bit load/store where aligned
    int64_t n4 = n / 4;
    float4* data4 = reinterpret_cast<float4*>(data);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 v = data4[i];
        v.x *= inv_scale;
        v.y *= inv_scale;
        v.z *= inv_scale;
        v.w *= inv_scale;
        data4[i] = v;
    }

    // Remainder
    int64_t rem_start = n4 * 4;
    for (int64_t i = rem_start + idx; i < n; i += stride) {
        data[i] *= inv_scale;
    }
}

bool launch_has_nan_or_inf_gpu(const float* data, int64_t n, cudaStream_t stream = nullptr) {
    if (n <= 0) return false;

    int* d_flag = nullptr;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemsetAsync(d_flag, 0, sizeof(int), stream);

    int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 1024) num_blocks = 1024;

    has_nan_or_inf_kernel<<<num_blocks, block_size, 0, stream>>>(data, n, d_flag);

    int h_flag = 0;
    cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_flag);

    return h_flag != 0;
}

void launch_unscale_gpu(float* data, int64_t n, float inv_scale, cudaStream_t stream = nullptr) {
    if (n <= 0) return;

    int block_size = 256;
    int num_blocks = static_cast<int>((n / 4 + block_size - 1) / block_size);
    if (num_blocks <= 0) num_blocks = 1;
    if (num_blocks > 2048) num_blocks = 2048;

    unscale_kernel_fp32<<<num_blocks, block_size, 0, stream>>>(data, n, inv_scale);
}

}  // namespace aic::cuda

#endif
