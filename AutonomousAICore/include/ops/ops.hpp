// include/ops/ops.hpp
// Tensor mathematical operations: elementwise arithmetic, unary activations, and reductions.
#pragma once

#include "tensor/tensor.hpp"

namespace aic {

// ============================================================================
// Elementwise Binary Operations (supporting automatic broadcasting)
// ============================================================================
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// In-place operations (target must be broadcast-compatible with source)
Tensor& add_(Tensor& a, const Tensor& b);
Tensor& sub_(Tensor& a, const Tensor& b);
Tensor& mul_(Tensor& a, const Tensor& b);
Tensor& div_(Tensor& a, const Tensor& b);

Tensor& add_(Tensor& a, float scalar);
Tensor& sub_(Tensor& a, float scalar);
Tensor& mul_(Tensor& a, float scalar);
Tensor& div_(Tensor& a, float scalar);

// Elementwise with Scalar
Tensor add(const Tensor& a, float scalar);
Tensor add(float scalar, const Tensor& a);
Tensor sub(const Tensor& a, float scalar);
Tensor sub(float scalar, const Tensor& a);
Tensor mul(const Tensor& a, float scalar);
Tensor mul(float scalar, const Tensor& a);
Tensor div(const Tensor& a, float scalar);
Tensor div(float scalar, const Tensor& a);

// ============================================================================
// Unary and Activation Functions
// ============================================================================
Tensor neg(const Tensor& a);
Tensor abs(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor silu(const Tensor& a);

// ============================================================================
// Reduction Operations
// ============================================================================
// Full-tensor reductions (returns 0-D scalar tensor)
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor max(const Tensor& a);
Tensor min(const Tensor& a);

// Reductions along a specific dimension
Tensor sum(const Tensor& a, int dim, bool keepdim = false);
Tensor mean(const Tensor& a, int dim, bool keepdim = false);
Tensor max(const Tensor& a, int dim, bool keepdim = false);
Tensor min(const Tensor& a, int dim, bool keepdim = false);

// ============================================================================
// Sequence and Tensor Manipulation Operations
// ============================================================================
Tensor softmax(const Tensor& a, int dim = -1);
Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);

// ============================================================================
// Operator Overloads
// ============================================================================
inline Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
inline Tensor operator+(const Tensor& a, float b) { return add(a, b); }
inline Tensor operator+(float a, const Tensor& b) { return add(a, b); }

inline Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
inline Tensor operator-(const Tensor& a, float b) { return sub(a, b); }
inline Tensor operator-(float a, const Tensor& b) { return sub(a, b); }

inline Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
inline Tensor operator*(const Tensor& a, float b) { return mul(a, b); }
inline Tensor operator*(float a, const Tensor& b) { return mul(a, b); }

inline Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
inline Tensor operator/(const Tensor& a, float b) { return div(a, b); }
inline Tensor operator/(float a, const Tensor& b) { return div(a, b); }

inline Tensor operator-(const Tensor& a) { return neg(a); }

inline Tensor& operator+=(Tensor& a, const Tensor& b) { return add_(a, b); }
inline Tensor& operator+=(Tensor& a, float b) { return add_(a, b); }

inline Tensor& operator-=(Tensor& a, const Tensor& b) { return sub_(a, b); }
inline Tensor& operator-=(Tensor& a, float b) { return sub_(a, b); }

inline Tensor& operator*=(Tensor& a, const Tensor& b) { return mul_(a, b); }
inline Tensor& operator*=(Tensor& a, float b) { return mul_(a, b); }

inline Tensor& operator/=(Tensor& a, const Tensor& b) { return div_(a, b); }
inline Tensor& operator/=(Tensor& a, float b) { return div_(a, b); }

}  // namespace aic
