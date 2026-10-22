// src/alignment/dpo_trainer.cpp
// Direct Preference Optimization (DPO) Trainer implementation.
#include "alignment/dpo_trainer.hpp"
#include "autograd/autograd.hpp"
#include "core/error.hpp"

namespace aic::alignment {

DPOTrainer::DPOTrainer(
    std::shared_ptr<nn::Module> policy_model,
    std::shared_ptr<nn::Module> ref_model,
    std::shared_ptr<optim::AdamW> optimizer,
    DPOConfig config
) : policy_model_(std::move(policy_model)),
    ref_model_(std::move(ref_model)),
    optimizer_(std::move(optimizer)),
    config_(config) {
    AIC_CHECK(policy_model_ != nullptr, InvalidArgument, "policy_model cannot be null");
    AIC_CHECK(ref_model_ != nullptr, InvalidArgument, "ref_model cannot be null");
    AIC_CHECK(optimizer_ != nullptr, InvalidArgument, "optimizer cannot be null");
    AIC_CHECK(config_.beta > 0.0f, InvalidArgument, "beta must be positive");

    ref_model_->eval();
}

DPOMetrics DPOTrainer::step(
    const Tensor& chosen_input_ids,
    const Tensor& chosen_labels,
    const Tensor& rejected_input_ids,
    const Tensor& rejected_labels
) {
    policy_model_->train();
    ref_model_->eval();

    Tensor pi_chosen_logps;
    Tensor pi_rejected_logps;
    Tensor ref_chosen_logps;
    Tensor ref_rejected_logps;

    // 1. Forward passes for policy network (with autograd & optional autocast)
    {
        train::AutocastGuard autocast(config_.enable_amp, config_.amp_dtype);
        pi_chosen_logps = compute_sequence_log_probs(*policy_model_, chosen_input_ids, chosen_labels);
        pi_rejected_logps = compute_sequence_log_probs(*policy_model_, rejected_input_ids, rejected_labels);
    }

    // 2. Forward passes for frozen reference network (NoGrad)
    {
        autograd::NoGradGuard no_grad;
        train::AutocastGuard autocast(config_.enable_amp, config_.amp_dtype);
        ref_chosen_logps = compute_sequence_log_probs(*ref_model_, chosen_input_ids, chosen_labels);
        ref_rejected_logps = compute_sequence_log_probs(*ref_model_, rejected_input_ids, rejected_labels);
    }

    // 3. Compute DPO Loss
    auto [loss, metrics] = compute_dpo_loss(
        pi_chosen_logps,
        pi_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        config_.beta,
        config_.label_smoothing
    );

    // 4. Backward Pass & Gradient Updates
    if (config_.enable_amp) {
        Tensor scaled_loss = loss_scaler_.scale(loss);
        scaled_loss.backward();

        bool ok = loss_scaler_.unscale_and_update(policy_model_->parameters());
        if (!ok) {
            optimizer_->zero_grad();
            if (on_step_end) on_step_end(metrics);
            return metrics;
        }
    } else {
        loss.backward();
    }

    // 5. Clip Gradient Norm & Optimizer Step
    optim::clip_grad_norm_(policy_model_->parameters(), config_.max_grad_norm);
    optimizer_->step();
    optimizer_->zero_grad();

    step_count_++;

    if (on_step_end) {
        on_step_end(metrics);
    }

    return metrics;
}

}  // namespace aic::alignment
