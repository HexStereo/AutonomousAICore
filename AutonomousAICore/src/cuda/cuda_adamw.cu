// src/cuda/cuda_adamw.cu
// High-performance CUDA kernel for fused AdamW optimizer step.
#include "cuda/cuda_adamw.cuh"
#include "core/error.hpp"

#include <cuda_runtime.h>
#include <cmath>

#define AIC_CUDA_CHECK_ADAMW(call)                                                     \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            AIC_THROW(CudaError, "CUDA error in AdamW kernel: ",                       \
                      cudaGetErrorString(err__));                                      \
        }                                                                              \
    } while (false)

namespace aic::cuda {

namespace {

__global__ void adamw_kernel(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bc1,
    float bc2
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float grad = g[i];
        float param = p[i];

        // 1. Decoupled weight decay: theta = theta - lr * wd * theta
        param -= lr * wd * param;

        // 2. Exponential moving averages of gradient and square gradient
        float m_val = beta1 * m[i] + (1.0f - beta1) * grad;
        float v_val = beta2 * v[i] + (1.0f - beta2) * grad * grad;

        m[i] = m_val;
        v[i] = v_val;

        // 3. Bias-corrected moment estimates
        float m_hat = m_val / bc1;
        float v_hat = v_val / bc2;

        // 4. Update parameter
        param -= lr * (m_hat / (sqrtf(v_hat) + eps));
        p[i] = param;
    }
}

}  // namespace

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
    cudaStream_t stream
) {
    if (n <= 0) return;

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    adamw_kernel<<<blocks, threads, 0, stream>>>(
        p, g, m, v, n,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2
    );
    AIC_CUDA_CHECK_ADAMW(cudaGetLastError());
}

}  // namespace aic::cuda
