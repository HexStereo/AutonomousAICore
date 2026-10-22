// src/cuda/cuda_gemm.cuh
// CUDA GEMM kernels header: Naive, Shared Memory Tiled, Register Tiled, and WMMA Tensor Cores.
#pragma once

#include "ops/matmul.hpp"
#include "tensor/tensor.hpp"

namespace aic::cuda {

// 2D CUDA GEMM with specific algorithm dispatch
Tensor gemm_2d(const Tensor& a, const Tensor& b, GemmAlgorithm algorithm = GemmAlgorithm::Auto);

// Batched CUDA Matrix Multiplication
Tensor matmul(const Tensor& a, const Tensor& b, GemmAlgorithm algorithm = GemmAlgorithm::Auto);

}  // namespace aic::cuda
