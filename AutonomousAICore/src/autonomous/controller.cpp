// src/autonomous/controller.cpp
// Autonomous Optimization Controller implementation.
#include "autonomous/controller.hpp"
#include "core/error.hpp"
#include "train/checkpoint.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace aic::autonomous {

const char* action_name(ControllerAction action) noexcept {
    switch (action) {
        case ControllerAction::None: return "None";
        case ControllerAction::RollbackCheckpoint: return "RollbackCheckpoint";
        case ControllerAction::ScaleLearningRate: return "ScaleLearningRate";
        case ControllerAction::TightenGradientClipping: return "TightenGradientClipping";
        case ControllerAction::TuneGradientAccumulation: return "TuneGradientAccumulation";
        case ControllerAction::Halt: return "Halt";
        default: return "Unknown";
    }
}

const char* health_state_name(HealthState state) noexcept {
    switch (state) {
        case HealthState::Optimal: return "Optimal";
        case HealthState::Warning: return "Warning";
        case HealthState::Diverging: return "Diverging";
        case HealthState::Recovering: return "Recovering";
        case HealthState::Halted: return "Halted";
        default: return "Unknown";
    }
}

AutonomousController::AutonomousController(AutonomousConfig config)
    : config_(std::move(config)) {
    if (!config_.checkpoint_dir.empty()) {
        std::filesystem::create_directories(config_.checkpoint_dir);
    }
}

void AutonomousController::reset() {
    telemetry_.reset();
    health_state_ = HealthState::Optimal;
    last_action_ = ControllerAction::None;
    last_reason_.clear();
    last_stable_checkpoint_.clear();
    last_checkpoint_step_ = 0;
    rollback_count_ = 0;
    consecutive_rollbacks_ = 0;
}

void AutonomousController::save_stable_checkpoint(train::Trainer& trainer, int64_t step) {
    if (config_.checkpoint_dir.empty() || !trainer.model()) return;

    std::string path = config_.checkpoint_dir + "/stable_step_" + std::to_string(step) + ".aic";
    train::CheckpointMetadata meta;
    meta.step = step;
    meta.best_val_loss = trainer.best_val_loss();
    meta.description = "autonomous_stable_checkpoint";

    train::save_checkpoint(path, *trainer.model(), trainer.optimizer().get(), meta);
    last_stable_checkpoint_ = path;
    last_checkpoint_step_ = step;
}

ControllerAction AutonomousController::evaluate(
    train::Trainer& trainer,
    const train::StepMetrics& metrics,
    int64_t model_params,
    int cuda_device_index
) {
    auto snapshot = telemetry_.record_step(
        metrics.step,
        metrics.loss,
        metrics.grad_norm,
        metrics.lr,
        metrics.tokens_per_sec,
        /*step_duration_sec=*/metrics.tokens_per_sec > 0.0f ? 1.0 / static_cast<double>(metrics.tokens_per_sec) : 0.0,
        /*total_tokens=*/0,
        model_params,
        cuda_device_index
    );

    last_action_ = ControllerAction::None;
    last_reason_.clear();

    // 1. Critical Anomaly: NaN / Inf loss or gradients
    if (snapshot.detected_anomaly == AnomalyType::NanOrInfLoss ||
        snapshot.detected_anomaly == AnomalyType::NanOrInfGradient) {
        last_reason_ = "NaN/Inf detected in loss or gradient";

        if (config_.enable_rollback && !last_stable_checkpoint_.empty()) {
            if (consecutive_rollbacks_ >= config_.max_rollback_retries) {
                health_state_ = HealthState::Halted;
                last_action_ = ControllerAction::Halt;
                last_reason_ += " (max rollbacks exceeded: halting)";
            } else {
                health_state_ = HealthState::Recovering;
                last_action_ = ControllerAction::RollbackCheckpoint;
            }
        } else {
            health_state_ = HealthState::Diverging;
            last_action_ = ControllerAction::ScaleLearningRate;
        }
        return last_action_;
    }

    // 2. Loss Divergence: sustained loss increase above long-term EMA
    if (snapshot.detected_anomaly == AnomalyType::LossDivergence) {
        last_reason_ = "Loss divergence detected: current loss significantly exceeds long-term EMA";

        if (config_.enable_rollback && !last_stable_checkpoint_.empty() &&
            consecutive_rollbacks_ < config_.max_rollback_retries) {
            health_state_ = HealthState::Diverging;
            last_action_ = ControllerAction::RollbackCheckpoint;
        } else {
            health_state_ = HealthState::Warning;
            last_action_ = ControllerAction::ScaleLearningRate;
        }
        return last_action_;
    }

    // 3. Gradient Explosion: sudden spike in L2 gradient norm
    if (snapshot.detected_anomaly == AnomalyType::GradientExplosion) {
        last_reason_ = "Gradient norm explosion spike detected";
        health_state_ = HealthState::Warning;

        if (config_.enable_grad_clip_tuning) {
            last_action_ = ControllerAction::TightenGradientClipping;
        } else {
            last_action_ = ControllerAction::ScaleLearningRate;
        }
        return last_action_;
    }

    // 4. Memory Pressure: GPU VRAM near exhaustion
    if (snapshot.detected_anomaly == AnomalyType::MemoryCritical) {
        last_reason_ = "Critical GPU memory pressure detected (<8% free VRAM)";
        health_state_ = HealthState::Warning;

        if (config_.enable_auto_grad_accum &&
            trainer.config().grad_accum_steps > config_.min_grad_accum_steps) {
            last_action_ = ControllerAction::TuneGradientAccumulation;
        }
        return last_action_;
    }

    // 5. Healthy State
    health_state_ = HealthState::Optimal;
    if (consecutive_rollbacks_ > 0) {
        consecutive_rollbacks_ = 0;
    }

    // Auto-save stable checkpoint on schedule or initial baseline at step 1
    if (last_checkpoint_step_ == 0 || (metrics.step - last_checkpoint_step_ >= config_.checkpoint_interval)) {
        save_stable_checkpoint(trainer, metrics.step);
    }

    return last_action_;
}

bool AutonomousController::execute_action(train::Trainer& trainer, ControllerAction action) {
    switch (action) {
        case ControllerAction::RollbackCheckpoint: {
            if (last_stable_checkpoint_.empty()) {
                return false;
            }

            // 1. Restore model parameters from last stable checkpoint
            train::load_checkpoint(last_stable_checkpoint_, *trainer.model(), trainer.optimizer().get());

            // 2. Decay learning rate
            if (trainer.optimizer()) {
                float old_lr = trainer.optimizer()->options().lr;
                float new_lr = old_lr * config_.lr_decay_on_rollback;
                trainer.optimizer()->options().lr = new_lr;
            }

            // 3. Advance/skip batch in train_loader if possible to bypass divergent training sample
            if (trainer.train_loader() && trainer.train_loader()->has_next()) {
                (void)trainer.train_loader()->next();
            }

            rollback_count_++;
            consecutive_rollbacks_++;
            health_state_ = HealthState::Recovering;
            return true;
        }

        case ControllerAction::TightenGradientClipping: {
            if (config_.enable_grad_clip_tuning) {
                float current_clip = trainer.config().max_grad_norm;
                float new_clip = std::max(config_.min_grad_clip_norm, current_clip * config_.tighten_clip_factor);
                trainer.config().max_grad_norm = new_clip;
                return true;
            }
            return false;
        }

        case ControllerAction::ScaleLearningRate: {
            if (trainer.optimizer()) {
                float old_lr = trainer.optimizer()->options().lr;
                float new_lr = old_lr * 0.8f;
                trainer.optimizer()->options().lr = new_lr;
                return true;
            }
            return false;
        }

        case ControllerAction::TuneGradientAccumulation: {
            if (config_.enable_auto_grad_accum) {
                // If memory is critical, decrease accumulation steps
                int64_t current = trainer.config().grad_accum_steps;
                if (telemetry_.is_memory_critical()) {
                    trainer.config().grad_accum_steps = std::max(config_.min_grad_accum_steps, current - 1);
                } else if (telemetry_.latest().memory.gpu_used_ratio() < 0.5f && current < config_.max_grad_accum_steps) {
                    trainer.config().grad_accum_steps = std::min(config_.max_grad_accum_steps, current + 1);
                }
                return true;
            }
            return false;
        }

        case ControllerAction::Halt:
            health_state_ = HealthState::Halted;
            return false;

        case ControllerAction::None:
        default:
            return true;
    }
}

ControllerAction AutonomousController::step(
    train::Trainer& trainer,
    const train::StepMetrics& metrics,
    int64_t model_params,
    int cuda_device_index
) {
    ControllerAction act = evaluate(trainer, metrics, model_params, cuda_device_index);
    if (act != ControllerAction::None) {
        execute_action(trainer, act);
    }
    return act;
}

}  // namespace aic::autonomous
