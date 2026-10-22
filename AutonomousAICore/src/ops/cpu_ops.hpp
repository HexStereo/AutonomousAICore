// src/ops/cpu_ops.hpp
// CPU reference implementations for elementwise, unary, and reduction operations.
#pragma once

#include "tensor/tensor.hpp"

namespace aic::cpu {

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

Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor max(const Tensor& a);
Tensor min(const Tensor& a);

Tensor sum(const Tensor& a, int dim, bool keepdim);
Tensor mean(const Tensor& a, int dim, bool keepdim);
Tensor max(const Tensor& a, int dim, bool keepdim);
Tensor min(const Tensor& a, int dim, bool keepdim);

}  // namespace aic::cpu
