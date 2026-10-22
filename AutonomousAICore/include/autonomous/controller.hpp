// include/autonomous/controller.hpp
// Autonomous Optimization Controller: Self-healing, adaptive hyperparameter tuning,
// checkpoint rollback, and anomaly mitigation.
#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "autonomous/telemetry.hpp"
#include "train/trainer.hpp"

namespace aic::autonomous {

enum class ControllerAction {
    None = 0,
    RollbackCheckpoint,
    ScaleLearningRate,
    TightenGradientClipping,
    TuneGradientAccumulation,
    Halt
};

const char* action_name(ControllerAction action) noexcept;

inline std::ostream& operator<<(std::ostream& os, ControllerAction action) {
    return os << action_name(action);
}

enum class HealthState {
    Optimal = 0,
    Warning,
    Diverging,
    Recovering,
    Halted
};

const char* health_state_name(HealthState state) noexcept;

inline std::ostream& operator<<(std::ostream& os, HealthState state) {
    return os << health_state_name(state);
}

struct AutonomousConfig {
    bool enable_rollback = true;
    int max_rollback_retries = 3;
    float lr_decay_on_rollback = 0.5f;

    bool enable_grad_clip_tuning = true;
    float tighten_clip_factor = 0.7f;
    float min_grad_clip_norm = 0.1f;

    bool enable_auto_grad_accum = true;
    int64_t max_grad_accum_steps = 8;
    int64_t min_grad_accum_steps = 1;

    int64_t checkpoint_interval = 5;
    std::string checkpoint_dir = "autonomous_ckpts";
};

class AutonomousController {
public:
    explicit AutonomousController(AutonomousConfig config = {});

    // Evaluates the latest step metrics, queries hardware telemetry, updates health state,
    // and determines what autonomous intervention (if any) is required.
    ControllerAction evaluate(
        train::Trainer& trainer,
        const train::StepMetrics& metrics,
        int64_t model_params = 0,
        int cuda_device_index = 0
    );

    // Applies the determined action to the trainer (e.g., executing rollback, adjusting LR, clipping, etc.)
    bool execute_action(train::Trainer& trainer, ControllerAction action);

    // High-level integration helper: evaluates and applies action in a single call.
    ControllerAction step(
        train::Trainer& trainer,
        const train::StepMetrics& metrics,
        int64_t model_params = 0,
        int cuda_device_index = 0
    );

    const TelemetryTracker& telemetry() const noexcept { return telemetry_; }
    TelemetryTracker& telemetry() noexcept { return telemetry_; }

    HealthState health_state() const noexcept { return health_state_; }
    ControllerAction last_action() const noexcept { return last_action_; }
    const std::string& last_reason() const noexcept { return last_reason_; }

    int rollback_count() const noexcept { return rollback_count_; }
    int consecutive_rollbacks() const noexcept { return consecutive_rollbacks_; }
    const std::string& last_stable_checkpoint() const noexcept { return last_stable_checkpoint_; }

    const AutonomousConfig& config() const noexcept { return config_; }
    AutonomousConfig& config() noexcept { return config_; }

    void reset();

private:
    void save_stable_checkpoint(train::Trainer& trainer, int64_t step);

    AutonomousConfig config_;
    TelemetryTracker telemetry_;
    HealthState health_state_ = HealthState::Optimal;
    ControllerAction last_action_ = ControllerAction::None;
    std::string last_reason_;

    std::string last_stable_checkpoint_;
    int64_t last_checkpoint_step_ = 0;
    int rollback_count_ = 0;
    int consecutive_rollbacks_ = 0;
};

}  // namespace aic::autonomous
