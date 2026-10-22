// src/loss/mse.cpp
// Mean Squared Error (MSE) loss with dedicated autograd backward node.
#include "loss/mse.hpp"
#include "autograd/autograd.hpp"
#include "autograd/node.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

namespace aic {

namespace {

inline autograd::Edge make_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return autograd::Edge{nullptr, 0};
    }
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

class MSEBackward : public autograd::Node {
public:
    MSEBackward(Tensor diff, std::string reduction, int64_t numel)
        : saved_diff_(std::move(diff)), reduction_(std::move(reduction)), numel_(numel) {}

    std::string name() const override { return "MSEBackward"; }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& g = grad_outputs[0];
        Tensor diff = saved_diff_.unpack("MSEBackward");

        float scale = 2.0f;
        if (reduction_ == "mean") {
            scale = numel_ > 0 ? (2.0f / static_cast<float>(numel_)) : 0.0f;
        }

        Tensor grad_input;
        if (reduction_ == "none") {
            grad_input = diff * (2.0f * g);
        } else {
            float g_scalar = g.defined() ? g.cpu().item<float>() : 1.0f;
            grad_input = diff * (scale * g_scalar);
        }

        return {grad_input, -grad_input};
    }

private:
    autograd::SavedVariable saved_diff_;
    std::string reduction_;
    int64_t numel_;
};

}  // namespace

Tensor mse_loss(const Tensor& input, const Tensor& target, const std::string& reduction) {
    AIC_CHECK(input.defined() && target.defined(), InvalidArgument,
              "mse_loss inputs must be defined tensors");
    AIC_CHECK(input.device() == target.device(), DeviceMismatch,
              "input and target must be on the same device");
    AIC_CHECK(input.shape() == target.shape(), ShapeMismatch,
              "input shape ", input.shape().str(), " must match target shape ", target.shape().str());
    AIC_CHECK(reduction == "mean" || reduction == "sum" || reduction == "none", InvalidArgument,
              "reduction must be 'mean', 'sum', or 'none'");

    Tensor diff = input - target;
    Tensor sq = diff * diff;

    Tensor result;
    if (reduction == "none") {
        result = sq;
    } else if (reduction == "sum") {
        result = sum(sq);
    } else { // "mean"
        result = mean(sq);
    }

    if (autograd::GradMode::is_enabled() && (input.requires_grad() || target.requires_grad())) {
        result.set_requires_grad(true);
        result.set_leaf(false);
        auto node = std::make_shared<MSEBackward>(diff, reduction, input.numel());
        node->add_next_edge(make_edge(input));
        node->add_next_edge(make_edge(target));
        result.set_grad_fn(std::move(node));
    }

    return result;
}

}  // namespace aic
