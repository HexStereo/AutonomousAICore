// include/nn/swiglu.hpp
// Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.
#pragma once

#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "ops/ops.hpp"

namespace aic::nn {

class SwiGLU : public Module {
public:
    SwiGLU(int64_t d_model, int64_t d_hidden, Device device = Device::cpu())
        : d_model_(d_model), d_hidden_(d_hidden) {
        
        gate_proj_ = register_module("gate_proj", std::make_shared<Linear>(d_model_, d_hidden_, /*has_bias=*/false, device));
        up_proj_   = register_module("up_proj",   std::make_shared<Linear>(d_model_, d_hidden_, /*has_bias=*/false, device));
        down_proj_ = register_module("down_proj", std::make_shared<Linear>(d_hidden_, d_model_, /*has_bias=*/false, device));
    }

    Tensor forward(const Tensor& x) override {
        // 1. Gate projection + SiLU activation
        Tensor gate = silu(gate_proj_->forward(x));

        // 2. Up projection
        Tensor up = up_proj_->forward(x);

        // 3. Elementwise gating product: silu(gate) * up
        Tensor hidden = mul(gate, up);

        // 4. Down projection back to d_model
        return down_proj_->forward(hidden);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    std::shared_ptr<Linear> gate_proj() const { return gate_proj_; }
    std::shared_ptr<Linear> up_proj() const { return up_proj_; }
    std::shared_ptr<Linear> down_proj() const { return down_proj_; }

    void set_gate_proj(std::shared_ptr<Linear> mod) { gate_proj_ = register_module("gate_proj", mod); }
    void set_up_proj(std::shared_ptr<Linear> mod) { up_proj_ = register_module("up_proj", mod); }
    void set_down_proj(std::shared_ptr<Linear> mod) { down_proj_ = register_module("down_proj", mod); }

    int64_t d_model() const noexcept { return d_model_; }
    int64_t d_hidden() const noexcept { return d_hidden_; }

private:
    int64_t d_model_;
    int64_t d_hidden_;
    std::shared_ptr<Linear> gate_proj_;
    std::shared_ptr<Linear> up_proj_;
    std::shared_ptr<Linear> down_proj_;
};

}  // namespace aic::nn
