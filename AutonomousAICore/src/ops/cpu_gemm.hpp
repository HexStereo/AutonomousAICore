// src/ops/cpu_gemm.hpp
// High-performance CPU GEMM using cache-blocking, AVX2+FMA SIMD, and multi-threading.
#pragma once

#include "tensor/tensor.hpp"

namespace aic::cpu {

// 2D CPU matrix multiplication: [M, K] x [K, N] -> [M, N]
Tensor gemm_2d(const Tensor& a, const Tensor& b);

// Batched CPU matrix multiplication: [..., M, K] x [..., K, N] -> [..., M, N]
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace aic::cpu
