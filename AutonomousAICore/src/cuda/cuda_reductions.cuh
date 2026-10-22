// src/cuda/cuda_reductions.cuh
// CUDA reduction kernels header (sum, mean, max, min) for GPU tensors.
#pragma once

#include "tensor/tensor.hpp"

namespace aic::cuda {

// Full tensor reductions (returns 0-D scalar tensor)
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor max(const Tensor& a);
Tensor min(const Tensor& a);

// Reductions along a dimension
Tensor sum(const Tensor& a, int dim, bool keepdim);
Tensor mean(const Tensor& a, int dim, bool keepdim);
Tensor max(const Tensor& a, int dim, bool keepdim);
Tensor min(const Tensor& a, int dim, bool keepdim);

}  // namespace aic::cuda
