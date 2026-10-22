// src/autograd/functions.cpp
// Implementation of differentiable backward functions and gradient broadcast reduction.
#include "autograd/functions.hpp"
#include "autograd/autograd.hpp"
#include "ops/ops.hpp"
#include "ops/matmul.hpp"

namespace aic::autograd {

Tensor reduce_gradient_for_broadcast(const Tensor& grad, const Shape& original_shape) {
    if (grad.shape() == original_shape) {
        return grad;
    }

    // If target shape is 0-D scalar, reduce everything to scalar
    if (original_shape.rank() == 0) {
        return sum(grad);
    }

    Tensor g = grad;
    int target_rank = original_shape.rank();
    int current_rank = g.rank();

    // 1. Reduce leading dimensions that were added due to broadcasting
    while (current_rank > target_rank) {
        g = sum(g, 0, false);
        current_rank = g.rank();
    }

    // 2. Reduce dimensions where original shape had size 1 but grad has size > 1
    for (int i = 0; i < target_rank; ++i) {
        if (original_shape[i] == 1 && g.dim(i) > 1) {
            g = sum(g, i, true);
        }
    }

    if (g.shape() != original_shape) {
        g = g.reshape(original_shape);
    }
    return g;
}

// ============================================================================
// Arithmetic Backward Functions
// ============================================================================
std::vector<Tensor> AddBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor grad_a = reduce_gradient_for_broadcast(g, shape_a_);
    Tensor grad_b = reduce_gradient_for_broadcast(g, shape_b_);
    return {grad_a, grad_b};
}

std::vector<Tensor> SubBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor grad_a = reduce_gradient_for_broadcast(g, shape_a_);
    Tensor grad_b = reduce_gradient_for_broadcast(-g, shape_b_);
    return {grad_a, grad_b};
}

std::vector<Tensor> MulBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor a = saved_a_.unpack("MulBackward");
    Tensor b = saved_b_.unpack("MulBackward");

    Tensor grad_a = reduce_gradient_for_broadcast(g * b, shape_a_);
    Tensor grad_b = reduce_gradient_for_broadcast(g * a, shape_b_);
    return {grad_a, grad_b};
}

std::vector<Tensor> DivBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor a = saved_a_.unpack("DivBackward");
    Tensor b = saved_b_.unpack("DivBackward");

    Tensor grad_a = reduce_gradient_for_broadcast(g / b, shape_a_);
    Tensor grad_b = reduce_gradient_for_broadcast(-g * a / (b * b), shape_b_);
    return {grad_a, grad_b};
}

std::vector<Tensor> PowBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor a = saved_a_.unpack("PowBackward");

    if (exponent_ == 1.0f) {
        return {g};
    } else if (exponent_ == 2.0f) {
        return {g * (2.0f * a)};
    } else {
        // g * p * a^(p - 1)
        Tensor p_scalar = Tensor::full(Shape({}), exponent_ - 1.0f, a.dtype(), a.device());
        // For general power on elements
        Tensor a_cpu = a.cpu();
        Tensor g_cpu = g.cpu();
        Tensor grad_cpu = Tensor::empty(a_cpu.shape(), a_cpu.dtype(), Device::cpu());
        float* dst = grad_cpu.data_ptr<float>();
        const float* src_a = a_cpu.data_ptr<float>();
        const float* src_g = g_cpu.data_ptr<float>();
        int64_t n = a_cpu.numel();
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = src_g[i] * exponent_ * std::pow(src_a[i], exponent_ - 1.0f);
        }
        return {grad_cpu.to(a.device())};
    }
}

std::vector<Tensor> NegBackward::backward(const std::vector<Tensor>& grad_outputs) {
    return {-grad_outputs[0]};
}

// ============================================================================
// Activation Backward Functions
// ============================================================================
std::vector<Tensor> ReluBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor x = saved_x_.unpack("ReluBackward");

    Tensor x_cpu = x.cpu();
    Tensor g_cpu = g.cpu();
    Tensor out_cpu = Tensor::empty(x_cpu.shape(), x_cpu.dtype(), Device::cpu());

    const float* px = x_cpu.data_ptr<float>();
    const float* pg = g_cpu.data_ptr<float>();
    float* pout = out_cpu.data_ptr<float>();
    int64_t n = x.numel();

    for (int64_t i = 0; i < n; ++i) {
        pout[i] = (px[i] > 0.0f) ? pg[i] : 0.0f;
    }
    return {out_cpu.to(x.device())};
}

std::vector<Tensor> TanhBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor y = saved_out_.unpack("TanhBackward");
    // d/dx tanh(x) = 1 - y^2
    Tensor dy = 1.0f - y * y;
    return {g * dy};
}

std::vector<Tensor> SigmoidBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor s = saved_out_.unpack("SigmoidBackward");
    // d/dx sigmoid(x) = s * (1 - s)
    Tensor ds = s * (1.0f - s);
    return {g * ds};
}

std::vector<Tensor> GeluBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor x = saved_x_.unpack("GeluBackward");

    // Standard approximation: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
    constexpr float c = 0.7978845608f;
    constexpr float k = 0.044715f;

    Tensor x2 = x * x;
    Tensor x3 = x2 * x;
    Tensor u = c * (x + k * x3);
    Tensor tanh_u = tanh(u);
    Tensor sech2_u = 1.0f - tanh_u * tanh_u;
    Tensor du_dx = c * (1.0f + 3.0f * k * x2);

    Tensor dgelu = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
    return {g * dgelu};
}

std::vector<Tensor> SiluBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor x = saved_x_.unpack("SiluBackward");

    Tensor s = sigmoid(x);
    // d/dx [x * s] = s * (1 + x * (1 - s))
    Tensor dsilu = s * (1.0f + x * (1.0f - s));
    return {g * dsilu};
}

std::vector<Tensor> ExpBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor y = saved_out_.unpack("ExpBackward");
    return {g * y};
}

std::vector<Tensor> LogBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor x = saved_x_.unpack("LogBackward");
    return {g / x};
}

std::vector<Tensor> SqrtBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor y = saved_out_.unpack("SqrtBackward");
    return {g / (2.0f * y)};
}

// ============================================================================
// Reduction Backward Functions
// ============================================================================
std::vector<Tensor> SumBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    if (!is_dim_reduce_) {
        // Full sum: scalar grad broadcasts to entire original input shape
        Tensor zeros = Tensor::zeros(input_shape_, g.dtype(), g.device());
        return {zeros + g};
    } else {
        Tensor grad_expanded = g;
        if (!keepdim_) {
            grad_expanded = grad_expanded.unsqueeze(dim_);
        }
        Tensor zeros = Tensor::zeros(input_shape_, g.dtype(), g.device());
        return {zeros + grad_expanded};
    }
}

std::vector<Tensor> MeanBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    if (!is_dim_reduce_) {
        float factor = 1.0f / static_cast<float>(input_shape_.numel());
        Tensor scaled_g = g * factor;
        Tensor zeros = Tensor::zeros(input_shape_, g.dtype(), g.device());
        return {zeros + scaled_g};
    } else {
        int r = input_shape_.rank();
        int norm_dim = dim_ < 0 ? dim_ + r : dim_;
        float factor = 1.0f / static_cast<float>(input_shape_[norm_dim]);
        Tensor grad_expanded = g * factor;
        if (!keepdim_) {
            grad_expanded = grad_expanded.unsqueeze(dim_);
        }
        Tensor zeros = Tensor::zeros(input_shape_, g.dtype(), g.device());
        return {zeros + grad_expanded};
    }
}

// ============================================================================
// Shape Transformation Backward Functions
// ============================================================================
std::vector<Tensor> ReshapeBackward::backward(const std::vector<Tensor>& grad_outputs) {
    return {grad_outputs[0].reshape(original_shape_)};
}

std::vector<Tensor> TransposeBackward::backward(const std::vector<Tensor>& grad_outputs) {
    return {grad_outputs[0].transpose(dim0_, dim1_)};
}

std::vector<Tensor> PermuteBackward::backward(const std::vector<Tensor>& grad_outputs) {
    int n = static_cast<int>(original_dims_.size());
    std::vector<int> inv_dims(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        inv_dims[static_cast<size_t>(original_dims_[static_cast<size_t>(i)])] = i;
    }
    return {grad_outputs[0].permute(inv_dims)};
}

std::vector<Tensor> MatMulBackward::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& g = grad_outputs[0];
    Tensor a = saved_a_.unpack("MatMulBackward");
    Tensor b = saved_b_.unpack("MatMulBackward");

    int ra = a.rank();
    int rb = b.rank();

    // dLoss/dA = g x B^T
    Tensor bt = b.transpose(rb - 2, rb - 1);
    Tensor grad_a = matmul(g, bt);
    grad_a = reduce_gradient_for_broadcast(grad_a, a.shape());

    // dLoss/dB = A^T x g
    Tensor at = a.transpose(ra - 2, ra - 1);
    Tensor grad_b = matmul(at, g);
    grad_b = reduce_gradient_for_broadcast(grad_b, b.shape());

    return {grad_a, grad_b};
}

}  // namespace aic::autograd
