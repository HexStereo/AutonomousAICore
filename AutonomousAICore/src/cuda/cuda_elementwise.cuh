// src/cuda/cuda_elementwise.cuh
// CUDA elementwise kernels header for GPU tensor acceleration.
#pragma once

#include "tensor/tensor.hpp"

namespace aic::cuda {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor neg(const Tensor& a);
Tensor abs(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor silu(const Tensor& a);

}  // namespace aic::cuda
