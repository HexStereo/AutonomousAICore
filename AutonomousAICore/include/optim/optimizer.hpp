// include/optim/optimizer.hpp
// Base class for numerical optimization algorithms.
#pragma once

#include <memory>
#include <vector>
#include "tensor/tensor.hpp"

namespace aic::optim {

class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<Tensor>> params)
        : params_(std::move(params)) {}

    virtual ~Optimizer() = default;

    // Perform a single optimization step (parameter update)
    virtual void step() = 0;

    // Reset gradients of all managed parameters to zero (or nullptr)
    virtual void zero_grad() {
        for (auto& p : params_) {
            if (p && p->defined()) {
                p->zero_grad();
            }
        }
    }

    const std::vector<std::shared_ptr<Tensor>>& params() const noexcept { return params_; }

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
};

}  // namespace aic::optim
