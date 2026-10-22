// include/optim/adamw.hpp
// AdamW optimizer with decoupled weight decay and fused moment updates.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "optim/optimizer.hpp"

namespace aic::optim {

struct AdamWOptions {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
};

struct AdamWParamState {
    int64_t step = 0;
    Tensor exp_avg;     // First moment vector m
    Tensor exp_avg_sq;  // Second moment vector v
};

class AdamW : public Optimizer {
public:
    explicit AdamW(std::vector<std::shared_ptr<Tensor>> params, AdamWOptions options = {})
        : Optimizer(std::move(params)), options_(options) {}

    void step() override;

    const AdamWOptions& options() const noexcept { return options_; }
    AdamWOptions& options() noexcept { return options_; }

    const AdamWParamState* state(const Tensor* p) const {
        auto it = state_.find(p);
        return it != state_.end() ? &it->second : nullptr;
    }

private:
    AdamWOptions options_;
    std::unordered_map<const Tensor*, AdamWParamState> state_;
};

}  // namespace aic::optim
