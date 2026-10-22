// include/autograd/autograd.hpp
// Dynamic reverse-mode automatic differentiation engine and numerical gradcheck verification.
#pragma once

#include <functional>
#include <vector>

#include "autograd/node.hpp"
#include "tensor/tensor.hpp"

namespace aic::autograd {

class GradMode {
public:
    static bool is_enabled() noexcept;
    static void set_enabled(bool enabled) noexcept;
};

class NoGradGuard {
public:
    NoGradGuard() noexcept : prev_state_(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }
    ~NoGradGuard() noexcept {
        GradMode::set_enabled(prev_state_);
    }
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_state_;
};

// Top-level reverse-mode backward traversal
void backward(const Tensor& loss, const Tensor& gradient = Tensor(), bool retain_graph = false);

// Numerical gradient checking using centered finite differences: (f(x + eps) - f(x - eps)) / (2 * eps)
bool gradcheck(
    const std::function<Tensor(const std::vector<Tensor>&)>& func,
    const std::vector<Tensor>& inputs,
    float eps = 1e-4f,
    float atol = 1e-3f,
    float rtol = 1e-3f);

// Reduces broadcasted gradient tensor to match original unbroadcasted input shape
Tensor reduce_gradient_for_broadcast(const Tensor& grad, const Shape& original_shape);

}  // namespace aic::autograd
