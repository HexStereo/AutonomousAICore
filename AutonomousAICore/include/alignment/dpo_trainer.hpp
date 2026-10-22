// include/alignment/dpo_trainer.hpp
// Direct Preference Optimization (DPO) Trainer for Model Alignment.
#pragma once

#include <functional>
#include <memory>
#include "alignment/dpo_loss.hpp"
#include "core/dtype.hpp"
#include "nn/module.hpp"
#include "optim/adamw.hpp"
#include "optim/clip_grad.hpp"
#include "train/amp.hpp"

namespace aic::alignment {

struct DPOConfig {
    float beta = 0.1f;
    float max_grad_norm = 1.0f;
    float label_smoothing = 0.0f;
    bool enable_amp = false;
    DType amp_dtype = DType::Float16;
};

class DPOTrainer {
public:
    DPOTrainer(
        std::shared_ptr<nn::Module> policy_model,
        std::shared_ptr<nn::Module> ref_model,
        std::shared_ptr<optim::AdamW> optimizer,
        DPOConfig config = {}
    );

    // Executes a single DPO optimization step
    DPOMetrics step(
        const Tensor& chosen_input_ids,
        const Tensor& chosen_labels,
        const Tensor& rejected_input_ids,
        const Tensor& rejected_labels
    );

    const DPOConfig& config() const noexcept { return config_; }
    DPOConfig& config() noexcept { return config_; }

    std::shared_ptr<nn::Module> policy_model() const noexcept { return policy_model_; }
    std::shared_ptr<nn::Module> ref_model() const noexcept { return ref_model_; }
    std::shared_ptr<optim::AdamW> optimizer() const noexcept { return optimizer_; }

    train::DynamicLossScaler& loss_scaler() noexcept { return loss_scaler_; }
    const train::DynamicLossScaler& loss_scaler() const noexcept { return loss_scaler_; }

    int64_t step_count() const noexcept { return step_count_; }

    std::function<void(const DPOMetrics&)> on_step_end;

private:
    std::shared_ptr<nn::Module> policy_model_;
    std::shared_ptr<nn::Module> ref_model_;
    std::shared_ptr<optim::AdamW> optimizer_;
    DPOConfig config_;
    train::DynamicLossScaler loss_scaler_;
    int64_t step_count_ = 0;
};

}  // namespace aic::alignment
