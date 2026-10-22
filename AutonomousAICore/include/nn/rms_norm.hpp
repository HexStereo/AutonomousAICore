// include/nn/rms_norm.hpp
// Root Mean Square Layer Normalization (RMSNorm) with learnable scale parameter.
#pragma once

#include "nn/module.hpp"
#include "ops/ops.hpp"

namespace aic::nn {

class RMSNorm : public Module {
public:
    explicit RMSNorm(int64_t dim, float eps = 1e-6f, Device device = Device::cpu())
        : dim_(dim), eps_(eps) {
        Tensor w = Tensor::ones(Shape({dim_}), DType::Float32, device);
        weight_ = register_parameter("weight", std::move(w));
    }

    Tensor forward(const Tensor& x) override {
        // x: [..., dim]
        // 1. Calculate x^2
        Tensor x2 = mul(x, x);

        // 2. Mean along last dimension with keepdim=true
        Tensor mean_x2 = mean(x2, -1, /*keepdim=*/true);

        // 3. sqrt(mean(x^2) + eps)
        Tensor rms = sqrt(add(mean_x2, eps_));

        // 4. Normalized x: x / rms
        Tensor x_norm = div(x, rms);

        // 5. Affine scaling: x_norm * weight
        return mul(x_norm, *weight_);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    std::shared_ptr<Tensor> weight() const { return weight_; }
    int64_t dim() const noexcept { return dim_; }
    float eps() const noexcept { return eps_; }

private:
    int64_t dim_;
    float eps_;
    std::shared_ptr<Tensor> weight_;
};

}  // namespace aic::nn
