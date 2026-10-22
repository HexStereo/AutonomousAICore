// include/ops/matmul.hpp
// High-performance Matrix Multiplication (GEMM): 2D, batched, CPU SIMD, CUDA Tiled & Tensor Cores.
#pragma once

#include "tensor/tensor.hpp"

namespace aic {

// Public Matrix Multiplication Interface
// Supports:
// - 2D matrices: [M, K] x [K, N] -> [M, N]
// - Batched matrices: [..., M, K] x [..., K, N] -> [..., M, N] (with broadcasting)
Tensor matmul(const Tensor& a, const Tensor& b);

// GEMM implementation algorithms for benchmarking and control
enum class GemmAlgorithm {
    Auto,           // Dispatches optimal kernel based on device and hardware features
    CpuTiledSimd,   // Cache-tiled AVX2+FMA SIMD CPU kernel
    CudaTiled,      // 2D shared memory tiled CUDA kernel
    CudaRegTiled,   // Register-tiled 2D block CUDA kernel
    CudaTensorCore  // Hardware Tensor Core WMMA kernel (sm_120 Blackwell accelerated)
};

Tensor gemm(const Tensor& a, const Tensor& b,
            float alpha = 1.0f, float beta = 0.0f,
            const Tensor& bias = Tensor(),
            GemmAlgorithm algorithm = GemmAlgorithm::Auto);

}  // namespace aic
