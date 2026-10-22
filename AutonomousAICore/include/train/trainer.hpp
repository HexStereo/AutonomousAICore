// include/train/trainer.hpp
// Training Engine with gradient accumulation, evaluation, and telemetry hooks.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "data/dataloader.hpp"
#include "loss/loss.hpp"
#include "nn/module.hpp"
#include "optim/adamw.hpp"
#include "optim/clip_grad.hpp"
#include "train/amp.hpp"
#include "train/checkpoint.hpp"
#include "train/scheduler.hpp"

namespace aic::train {

struct TrainingConfig {
    int64_t max_steps = 100;
    int64_t grad_accum_steps = 1;
    float max_grad_norm = 1.0f;
    int64_t eval_interval = 25;
    int64_t save_interval = 50;
    std::string checkpoint_dir = "checkpoints";
    bool enable_amp = false;
    DType amp_dtype = DType::Float16;
};

struct StepMetrics {
    int64_t step = 0;
    float loss = 0.0f;
    float grad_norm = 0.0f;
    float lr = 0.0f;
    float tokens_per_sec = 0.0f;
    float loss_scale = 1.0f;
    bool step_skipped = false;
};

class Trainer {
public:
    Trainer(
        std::shared_ptr<nn::Module> model,
        std::shared_ptr<optim::AdamW> optimizer,
        CosineWarmupScheduler scheduler,
        std::shared_ptr<data::DataLoader> train_loader,
        std::shared_ptr<data::DataLoader> val_loader = nullptr,
        TrainingConfig config = {}
    );

    // Runs training for up to config_.max_steps
    void train();

    // Runs a single optimization step (with micro-steps for gradient accumulation)
    StepMetrics step();

    // Evaluates validation loss across val_loader
    float evaluate();

    const TrainingConfig& config() const noexcept { return config_; }
    TrainingConfig& config() noexcept { return config_; }

    int64_t current_step() const noexcept { return current_step_; }
    float best_val_loss() const noexcept { return best_val_loss_; }

    std::shared_ptr<nn::Module> model() const noexcept { return model_; }
    std::shared_ptr<optim::AdamW> optimizer() const noexcept { return optimizer_; }
    CosineWarmupScheduler& scheduler() noexcept { return scheduler_; }
    const CosineWarmupScheduler& scheduler() const noexcept { return scheduler_; }
    std::shared_ptr<data::DataLoader> train_loader() const noexcept { return train_loader_; }
    std::shared_ptr<data::DataLoader> val_loader() const noexcept { return val_loader_; }
    void set_current_step(int64_t step) noexcept { current_step_ = step; }

    DynamicLossScaler& loss_scaler() noexcept { return loss_scaler_; }
    const DynamicLossScaler& loss_scaler() const noexcept { return loss_scaler_; }

    // Telemetry hook: called after each optimizer step with latest metrics
    std::function<void(const StepMetrics&)> on_step_end;

private:
    std::shared_ptr<nn::Module> model_;
    std::shared_ptr<optim::AdamW> optimizer_;
    CosineWarmupScheduler scheduler_;
    std::shared_ptr<data::DataLoader> train_loader_;
    std::shared_ptr<data::DataLoader> val_loader_;
    TrainingConfig config_;
    DynamicLossScaler loss_scaler_;

    int64_t current_step_ = 0;
    float best_val_loss_ = 1e9f;
};

}  // namespace aic::train
