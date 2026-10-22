// include/nn/linear.hpp
// Fully connected Linear layer with Kaiming uniform weight initialization.
#pragma once

#include "nn/module.hpp"
#include "ops/matmul.hpp"
#include "ops/ops.hpp"
#include <cmath>
#include <iostream>

namespace aic::nn {

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool has_bias = true, Device device = Device::cpu())
        : in_features_(in_features), out_features_(out_features), has_bias_(has_bias) {
        
        // Kaiming uniform initialization: bound = 1 / sqrt(in_features)
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        Tensor w = Tensor::uniform(Shape({out_features_, in_features_}), -bound, bound, DType::Float32, device);
        weight_ = register_parameter("weight", std::move(w));

        if (has_bias_) {
            Tensor b = Tensor::zeros(Shape({out_features_}), DType::Float32, device);
            bias_ = register_parameter("bias", std::move(b));
        }
    }

    virtual ~Linear() = default;

    virtual Tensor forward(const Tensor& x) override {
        // x: [..., in_features]
        // weight_: [out_features, in_features]
        // weight_^T: [in_features, out_features]
        Tensor w_t = weight_->transpose(0, 1);
        Tensor y = matmul(x, w_t);
        if (has_bias_ && bias_) {
            y = y + *bias_;
        }
        return y;
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    std::shared_ptr<Tensor> weight() const { return weight_; }
    std::shared_ptr<Tensor> bias() const { return bias_; }
    int64_t in_features() const noexcept { return in_features_; }
    int64_t out_features() const noexcept { return out_features_; }
    bool has_bias() const noexcept { return has_bias_; }

protected:
    Linear() : in_features_(0), out_features_(0), has_bias_(false) {}

    int64_t in_features_;
    int64_t out_features_;
    bool has_bias_;
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
};

}  // namespace aic::nn
