// src/train/trainer.cpp
// Training Engine implementation with gradient accumulation and telemetry callbacks.
#include "train/trainer.hpp"
#include "autograd/autograd.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>

namespace aic::train {

Trainer::Trainer(
    std::shared_ptr<nn::Module> model,
    std::shared_ptr<optim::AdamW> optimizer,
    CosineWarmupScheduler scheduler,
    std::shared_ptr<data::DataLoader> train_loader,
    std::shared_ptr<data::DataLoader> val_loader,
    TrainingConfig config
) : model_(std::move(model)),
    optimizer_(std::move(optimizer)),
    scheduler_(scheduler),
    train_loader_(std::move(train_loader)),
    val_loader_(std::move(val_loader)),
    config_(config) {
    AIC_CHECK(model_ != nullptr, InvalidArgument, "Trainer requires a non-null model");
    AIC_CHECK(optimizer_ != nullptr, InvalidArgument, "Trainer requires a non-null optimizer");
    AIC_CHECK(train_loader_ != nullptr, InvalidArgument, "Trainer requires a non-null train_loader");
    AIC_CHECK(config_.grad_accum_steps > 0, InvalidArgument, "grad_accum_steps must be positive");

    if (!config_.checkpoint_dir.empty()) {
        std::filesystem::create_directories(config_.checkpoint_dir);
    }
}

StepMetrics Trainer::step() {
    model_->train();
    auto start_time = std::chrono::high_resolution_clock::now();

    float total_loss = 0.0f;
    int64_t total_tokens = 0;

    for (int64_t micro = 0; micro < config_.grad_accum_steps; ++micro) {
        if (!train_loader_->has_next()) {
            train_loader_->reset();
        }
        auto [batch_x, batch_y] = train_loader_->next();
        total_tokens += batch_x.numel();

        Tensor logits;
        Tensor loss;
        {
            AutocastGuard autocast(config_.enable_amp, config_.amp_dtype);
            logits = model_->forward(batch_x);
            loss = cross_entropy(logits, batch_y, -100, "mean");
        }

        total_loss += loss.cpu().item<float>();

        // Scale loss if AMP is enabled, and divide by grad_accum_steps if > 1
        Tensor backward_loss = loss;
        if (config_.enable_amp) {
            backward_loss = loss_scaler_.scale(backward_loss);
        }
        if (config_.grad_accum_steps > 1) {
            backward_loss = backward_loss / static_cast<float>(config_.grad_accum_steps);
        }
        backward_loss.backward();
    }

    bool step_skipped = false;
    float grad_norm = 0.0f;

    if (config_.enable_amp) {
        // Unscale gradients and check for inf/nan
        bool ok = loss_scaler_.unscale_and_update(model_->parameters());
        if (!ok) {
            step_skipped = true;
            // Clear corrupted gradients so next step starts fresh
            optimizer_->zero_grad();
        }
    }

    if (!step_skipped) {
        // Clip gradient norm
        grad_norm = optim::clip_grad_norm_(model_->parameters(), config_.max_grad_norm);

        // Learning rate update
        float lr = scheduler_.get_lr(current_step_);
        optimizer_->options().lr = lr;

        // Optimizer step
        optimizer_->step();
        optimizer_->zero_grad();

        current_step_++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    float tokens_per_sec = elapsed_sec > 0.0 ? static_cast<float>(static_cast<double>(total_tokens) / elapsed_sec) : 0.0f;

    StepMetrics metrics;
    metrics.step = current_step_;
    metrics.loss = total_loss / static_cast<float>(config_.grad_accum_steps);
    metrics.grad_norm = grad_norm;
    metrics.lr = scheduler_.get_lr(current_step_);
    metrics.tokens_per_sec = tokens_per_sec;
    metrics.loss_scale = config_.enable_amp ? loss_scaler_.current_scale() : 1.0f;
    metrics.step_skipped = step_skipped;

    if (on_step_end) {
        on_step_end(metrics);
    }

    return metrics;
}

float Trainer::evaluate() {
    if (!val_loader_) return 0.0f;

    model_->eval();
    autograd::NoGradGuard no_grad;

    val_loader_->reset();
    float total_val_loss = 0.0f;
    int64_t batch_count = 0;

    while (val_loader_->has_next()) {
        auto [val_x, val_y] = val_loader_->next();
        Tensor logits;
        Tensor loss;
        {
            AutocastGuard autocast(config_.enable_amp, config_.amp_dtype);
            logits = model_->forward(val_x);
            loss = cross_entropy(logits, val_y, -100, "mean");
        }
        total_val_loss += loss.cpu().item<float>();
        batch_count++;
    }

    model_->train();
    return batch_count > 0 ? (total_val_loss / static_cast<float>(batch_count)) : 0.0f;
}

void Trainer::train() {
    while (current_step_ < config_.max_steps) {
        StepMetrics m = step();

        if (val_loader_ && (current_step_ % config_.eval_interval == 0)) {
            float val_loss = evaluate();
            if (val_loss < best_val_loss_) {
                best_val_loss_ = val_loss;
                CheckpointMetadata meta;
                meta.step = current_step_;
                meta.best_val_loss = best_val_loss_;
                meta.description = "best_model";
                std::string best_path = config_.checkpoint_dir + "/best_model.aic";
                save_checkpoint(best_path, *model_, optimizer_.get(), meta);
            }
        }

        if (current_step_ % config_.save_interval == 0) {
            CheckpointMetadata meta;
            meta.step = current_step_;
            meta.best_val_loss = best_val_loss_;
            meta.description = "periodic_checkpoint";
            std::string ckpt_path = config_.checkpoint_dir + "/model_step_" + std::to_string(current_step_) + ".aic";
            save_checkpoint(ckpt_path, *model_, optimizer_.get(), meta);
        }
    }
}

}  // namespace aic::train
