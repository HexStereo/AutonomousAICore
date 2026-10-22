// src/cuda/cuda_loss.cu
// High-performance CUDA kernels for Cross-Entropy Loss with fused LogSumExp and analytical backward.
#include "cuda/cuda_loss.cuh"
#include "core/error.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>

#define AIC_CUDA_CHECK_LOSS(call)                                                      \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            AIC_THROW(CudaError, "CUDA error in loss kernel: ",                        \
                      cudaGetErrorString(err__));                                      \
        }                                                                              \
    } while (false)

namespace aic::cuda {

namespace {

__device__ __forceinline__ float warpReduceMaxLoss(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSumLoss(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceMaxLoss(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMaxLoss(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMaxLoss(val);
    return val;
}

__device__ __forceinline__ float blockReduceSumLoss(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSumLoss(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSumLoss(val);
    return val;
}

__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets_int64,
    const float* __restrict__ targets_float,
    bool is_index_target,
    int64_t N,
    int64_t C,
    int64_t ignore_index,
    float* __restrict__ sample_losses,
    float* __restrict__ lse_out,
    unsigned long long* __restrict__ valid_count_device
) {
    int64_t row = blockIdx.x;
    if (row >= N) return;

    const float* row_logits = logits + row * C;

    // Step 1: Thread-local max
    float local_max = -FLT_MAX;
    for (int64_t col = threadIdx.x; col < C; col += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[col]);
    }
    float row_max = blockReduceMaxLoss(local_max);
    __shared__ float s_row_max;
    if (threadIdx.x == 0) {
        s_row_max = row_max;
    }
    __syncthreads();
    row_max = s_row_max;

    // Step 2: Thread-local sum of exp(z - row_max)
    float local_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < C; col += blockDim.x) {
        local_sum += expf(row_logits[col] - row_max);
    }
    float row_sum = blockReduceSumLoss(local_sum);
    __shared__ float s_lse;
    if (threadIdx.x == 0) {
        s_lse = row_max + logf(fmaxf(row_sum, 1e-12f));
        lse_out[row] = s_lse;
    }
    __syncthreads();
    float lse = s_lse;

    // Step 3: Compute sample loss
    if (threadIdx.x == 0) {
        if (is_index_target) {
            int64_t target_idx = targets_int64[row];
            if (target_idx == ignore_index) {
                sample_losses[row] = 0.0f;
            } else {
                float target_logit = (target_idx >= 0 && target_idx < C) ? row_logits[target_idx] : 0.0f;
                sample_losses[row] = lse - target_logit;
                atomicAdd(valid_count_device, 1ULL);
            }
        } else {
            // Distribution target: sum_j(Y_j * z_j)
            // Handled below if parallel, or thread 0 for small C
            float dot = 0.0f;
            const float* row_target = targets_float + row * C;
            for (int64_t col = 0; col < C; ++col) {
                dot += row_target[col] * row_logits[col];
            }
            sample_losses[row] = lse - dot;
            atomicAdd(valid_count_device, 1ULL);
        }
    }
}

__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets_int64,
    const float* __restrict__ targets_float,
    bool is_index_target,
    const float* __restrict__ lse,
    int64_t N,
    int64_t C,
    int64_t ignore_index,
    int reduction_mode, // 0: mean, 1: sum, 2: none
    int64_t valid_count,
    float* __restrict__ grad_logits
) {
    int64_t row = blockIdx.x;
    if (row >= N) return;

    if (is_index_target && targets_int64[row] == ignore_index) {
        for (int64_t col = threadIdx.x; col < C; col += blockDim.x) {
            grad_logits[row * C + col] = 0.0f;
        }
        return;
    }

    float scale = 1.0f;
    if (reduction_mode == 0) { // mean
        float g = grad_output ? *grad_output : 1.0f;
        scale = valid_count > 0 ? (g / static_cast<float>(valid_count)) : 0.0f;
    } else if (reduction_mode == 1) { // sum
        scale = grad_output ? *grad_output : 1.0f;
    } else { // none
        scale = grad_output ? grad_output[row] : 1.0f;
    }

    float row_lse = lse[row];
    const float* row_logits = logits + row * C;
    float* row_grad = grad_logits + row * C;

    if (is_index_target) {
        int64_t target_idx = targets_int64[row];
        for (int64_t col = threadIdx.x; col < C; col += blockDim.x) {
            float prob = expf(row_logits[col] - row_lse);
            float target_ind = (col == target_idx) ? 1.0f : 0.0f;
            row_grad[col] = (prob - target_ind) * scale;
        }
    } else {
        const float* row_target = targets_float + row * C;
        for (int64_t col = threadIdx.x; col < C; col += blockDim.x) {
            float prob = expf(row_logits[col] - row_lse);
            float target_val = row_target[col];
            row_grad[col] = (prob - target_val) * scale;
        }
    }
}

}  // namespace

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
    cudaStream_t stream
) {
    if (N == 0) {
        if (valid_count_out) *valid_count_out = 0;
        return;
    }

    unsigned long long* d_valid_count = nullptr;
    AIC_CUDA_CHECK_LOSS(cudaMallocAsync(&d_valid_count, sizeof(unsigned long long), stream));
    AIC_CUDA_CHECK_LOSS(cudaMemsetAsync(d_valid_count, 0, sizeof(unsigned long long), stream));

    int threads = 256;
    if (C < 256) {
        threads = (static_cast<int>(C) + 31) / 32 * 32;
        if (threads < 32) threads = 32;
    }

    cross_entropy_forward_kernel<<<static_cast<unsigned int>(N), threads, 0, stream>>>(
        logits, targets_int64, targets_float, is_index_target,
        N, C, ignore_index, sample_losses, lse_out, d_valid_count
    );
    AIC_CUDA_CHECK_LOSS(cudaGetLastError());

    if (valid_count_out) {
        unsigned long long h_valid = 0;
        AIC_CUDA_CHECK_LOSS(cudaMemcpyAsync(&h_valid, d_valid_count, sizeof(unsigned long long),
                                            cudaMemcpyDeviceToHost, stream));
        AIC_CUDA_CHECK_LOSS(cudaStreamSynchronize(stream));
        *valid_count_out = static_cast<int64_t>(h_valid);
    }

    AIC_CUDA_CHECK_LOSS(cudaFreeAsync(d_valid_count, stream));
}

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
    cudaStream_t stream
) {
    if (N == 0) return;

    int red_mode = 0; // mean
    if (std::string(reduction) == "sum") red_mode = 1;
    else if (std::string(reduction) == "none") red_mode = 2;

    int threads = 256;
    if (C < 256) {
        threads = (static_cast<int>(C) + 31) / 32 * 32;
        if (threads < 32) threads = 32;
    }

    cross_entropy_backward_kernel<<<static_cast<unsigned int>(N), threads, 0, stream>>>(
        grad_output, logits, targets_int64, targets_float, is_index_target,
        lse, N, C, ignore_index, red_mode, valid_count, grad_logits
    );
    AIC_CUDA_CHECK_LOSS(cudaGetLastError());
}

}  // namespace aic::cuda
