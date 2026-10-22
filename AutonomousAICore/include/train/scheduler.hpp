// include/train/scheduler.hpp
// Learning rate scheduler with linear warmup and cosine annealing.
#pragma once

#include <cstdint>
#include <cmath>
#include <numbers>

namespace aic::train {

struct CosineWarmupSchedulerOptions {
    float initial_lr = 0.0f;
    float max_lr = 1e-3f;
    float min_lr = 1e-5f;
    int64_t warmup_steps = 100;
    int64_t total_steps = 1000;
};

class CosineWarmupScheduler {
public:
    explicit CosineWarmupScheduler(CosineWarmupSchedulerOptions options = {})
        : options_(options) {}

    float get_lr(int64_t current_step) const {
        if (current_step < options_.warmup_steps) {
            if (options_.warmup_steps <= 0) return options_.max_lr;
            float ratio = static_cast<float>(current_step) / static_cast<float>(options_.warmup_steps);
            return options_.initial_lr + ratio * (options_.max_lr - options_.initial_lr);
        }

        if (current_step >= options_.total_steps) {
            return options_.min_lr;
        }

        int64_t decay_steps = options_.total_steps - options_.warmup_steps;
        if (decay_steps <= 0) return options_.min_lr;

        float ratio = static_cast<float>(current_step - options_.warmup_steps) / static_cast<float>(decay_steps);
        float coeff = 0.5f * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * ratio));
        return options_.min_lr + coeff * (options_.max_lr - options_.min_lr);
    }

    const CosineWarmupSchedulerOptions& options() const noexcept { return options_; }

private:
    CosineWarmupSchedulerOptions options_;
};

}  // namespace aic::train
