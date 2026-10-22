// include/loss/cross_entropy.hpp
// Cross-Entropy Loss with fused LogSumExp numerical stabilization and analytical backward gradient.
#pragma once

#include <cstdint>
#include <string>
#include "tensor/tensor.hpp"

namespace aic {

// Functional cross entropy loss
// input: logits of shape (..., C) where C is the number of classes.
// target: either class indices of shape (...) (Int64/Int32), or probability distributions of shape (..., C) (Float32).
// ignore_index: specifies a target value that is ignored and does not contribute to the input gradient.
// reduction: "none" | "mean" | "sum"
Tensor cross_entropy(const Tensor& input, const Tensor& target,
                     int64_t ignore_index = -100,
                     const std::string& reduction = "mean");

// Object-oriented CrossEntropyLoss wrapper
class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(int64_t ignore_index = -100, std::string reduction = "mean")
        : ignore_index_(ignore_index), reduction_(std::move(reduction)) {}

    Tensor operator()(const Tensor& input, const Tensor& target) const {
        return cross_entropy(input, target, ignore_index_, reduction_);
    }

    int64_t ignore_index() const noexcept { return ignore_index_; }
    const std::string& reduction() const noexcept { return reduction_; }

private:
    int64_t ignore_index_ = -100;
    std::string reduction_ = "mean";
};

}  // namespace aic
