// include/loss/mse.hpp
// Mean Squared Error (MSE) Loss function with analytical backward pass.
#pragma once

#include <string>
#include "tensor/tensor.hpp"

namespace aic {

// Functional MSE loss: L = (input - target)^2
// reduction: "none" | "mean" | "sum"
Tensor mse_loss(const Tensor& input, const Tensor& target,
                const std::string& reduction = "mean");

// Object-oriented MSELoss wrapper
class MSELoss {
public:
    explicit MSELoss(std::string reduction = "mean")
        : reduction_(std::move(reduction)) {}

    Tensor operator()(const Tensor& input, const Tensor& target) const {
        return mse_loss(input, target, reduction_);
    }

    const std::string& reduction() const noexcept { return reduction_; }

private:
    std::string reduction_ = "mean";
};

}  // namespace aic
