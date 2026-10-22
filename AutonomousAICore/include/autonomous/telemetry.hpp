// include/autonomous/telemetry.hpp
// Real-time telemetry tracking for deep learning training and hardware utilization.
#pragma once

#include <cmath>
#include <cstdint>
#include <deque>
#include <ostream>
#include <string>
#include <vector>

namespace aic::autonomous {

enum class AnomalyType {
    None = 0,
    NanOrInfLoss,
    NanOrInfGradient,
    LossDivergence,
    GradientExplosion,
    GradientVanishing,
    MemoryCritical
};

const char* anomaly_name(AnomalyType type) noexcept;

inline std::ostream& operator<<(std::ostream& os, AnomalyType type) {
    return os << anomaly_name(type);
}

struct HardwareMemoryStats {
    size_t host_total_bytes = 0;
    size_t host_free_bytes = 0;
    size_t gpu_total_bytes = 0;
    size_t gpu_free_bytes = 0;

    float host_used_ratio() const noexcept {
        return host_total_bytes > 0
            ? static_cast<float>(host_total_bytes - host_free_bytes) / static_cast<float>(host_total_bytes)
            : 0.0f;
    }

    float gpu_used_ratio() const noexcept {
        return gpu_total_bytes > 0
            ? static_cast<float>(gpu_total_bytes - gpu_free_bytes) / static_cast<float>(gpu_total_bytes)
            : 0.0f;
    }
};

struct TelemetrySnapshot {
    int64_t step = 0;
    float loss = 0.0f;
    float loss_ema_short = 0.0f;
    float loss_ema_long = 0.0f;
    float loss_variance = 0.0f;
    float grad_norm = 0.0f;
    float grad_norm_ema = 0.0f;
    float lr = 0.0f;
    float tokens_per_sec = 0.0f;
    float estimated_tflops = 0.0f;
    HardwareMemoryStats memory;
    AnomalyType detected_anomaly = AnomalyType::None;

    std::string summary() const;
    std::string to_json() const;
};

class TelemetryTracker {
public:
    explicit TelemetryTracker(size_t window_size = 50, float alpha_short = 0.8f, float alpha_long = 0.98f);

    // Records a new step and updates moving averages, variance, throughput, and hardware memory
    TelemetrySnapshot record_step(
        int64_t step,
        float loss,
        float grad_norm,
        float lr,
        float tokens_per_sec,
        double step_duration_sec,
        int64_t total_tokens,
        int64_t model_params = 0,
        int cuda_device_index = 0
    );

    // Anomaly predicates
    bool is_nan_or_inf() const noexcept;
    bool is_diverging(float threshold_ratio = 1.4f, int min_consecutive = 3) const noexcept;
    bool is_gradient_exploding(float spike_factor = 4.0f) const noexcept;
    bool is_gradient_vanishing(float min_norm = 1e-7f) const noexcept;
    bool is_memory_critical(float min_free_ratio = 0.08f) const noexcept;

    // Latest metrics
    const TelemetrySnapshot& latest() const noexcept { return latest_snapshot_; }
    const std::deque<TelemetrySnapshot>& history() const noexcept { return history_; }
    size_t count() const noexcept { return history_.size(); }

    void reset();

    static HardwareMemoryStats query_hardware_memory(int cuda_device_index = 0);

private:
    size_t window_size_;
    float alpha_short_;
    float alpha_long_;

    float loss_ema_short_ = 0.0f;
    float loss_ema_long_ = 0.0f;
    float grad_norm_ema_ = 0.0f;
    int consecutive_divergence_ = 0;

    std::deque<TelemetrySnapshot> history_;
    TelemetrySnapshot latest_snapshot_;
};

}  // namespace aic::autonomous
