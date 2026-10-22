// src/autonomous/telemetry.cpp
// Real-time telemetry tracking and hardware performance counters.
#include "autonomous/telemetry.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace aic::autonomous {

const char* anomaly_name(AnomalyType type) noexcept {
    switch (type) {
        case AnomalyType::None: return "None";
        case AnomalyType::NanOrInfLoss: return "NanOrInfLoss";
        case AnomalyType::NanOrInfGradient: return "NanOrInfGradient";
        case AnomalyType::LossDivergence: return "LossDivergence";
        case AnomalyType::GradientExplosion: return "GradientExplosion";
        case AnomalyType::GradientVanishing: return "GradientVanishing";
        case AnomalyType::MemoryCritical: return "MemoryCritical";
        default: return "Unknown";
    }
}

std::string TelemetrySnapshot::summary() const {
    std::ostringstream os;
    os << "[Step " << step << "] "
       << "Loss: " << std::fixed << std::setprecision(4) << loss
       << " (EMA-S: " << loss_ema_short << ", EMA-L: " << loss_ema_long << ") "
       << "| GradNorm: " << std::setprecision(3) << grad_norm
       << " | LR: " << std::scientific << std::setprecision(2) << lr << std::fixed
       << " | " << std::setprecision(1) << tokens_per_sec << " tok/s";

    if (estimated_tflops > 0.0f) {
        os << " (" << std::setprecision(2) << estimated_tflops << " TFLOP/s)";
    }

    if (memory.gpu_total_bytes > 0) {
        double used_gb = static_cast<double>(memory.gpu_total_bytes - memory.gpu_free_bytes) / 1e9;
        double tot_gb = static_cast<double>(memory.gpu_total_bytes) / 1e9;
        os << " | VRAM: " << std::setprecision(1) << used_gb << "/" << tot_gb << " GB ("
           << std::setprecision(0) << (memory.gpu_used_ratio() * 100.0f) << "%)";
    }

    if (detected_anomaly != AnomalyType::None) {
        os << " [ANOMALY: " << anomaly_name(detected_anomaly) << "]";
    }
    return os.str();
}

std::string TelemetrySnapshot::to_json() const {
    std::ostringstream os;
    os << "{"
       << "\"step\":" << step << ","
       << "\"loss\":" << loss << ","
       << "\"loss_ema_short\":" << loss_ema_short << ","
       << "\"loss_ema_long\":" << loss_ema_long << ","
       << "\"loss_variance\":" << loss_variance << ","
       << "\"grad_norm\":" << grad_norm << ","
       << "\"grad_norm_ema\":" << grad_norm_ema << ","
       << "\"lr\":" << lr << ","
       << "\"tokens_per_sec\":" << tokens_per_sec << ","
       << "\"estimated_tflops\":" << estimated_tflops << ","
       << "\"gpu_used_ratio\":" << memory.gpu_used_ratio() << ","
       << "\"host_used_ratio\":" << memory.host_used_ratio() << ","
       << "\"anomaly\":\"" << anomaly_name(detected_anomaly) << "\""
       << "}";
    return os.str();
}

TelemetryTracker::TelemetryTracker(size_t window_size, float alpha_short, float alpha_long)
    : window_size_(window_size),
      alpha_short_(alpha_short),
      alpha_long_(alpha_long) {}

void TelemetryTracker::reset() {
    loss_ema_short_ = 0.0f;
    loss_ema_long_ = 0.0f;
    grad_norm_ema_ = 0.0f;
    consecutive_divergence_ = 0;
    history_.clear();
    latest_snapshot_ = TelemetrySnapshot{};
}

HardwareMemoryStats TelemetryTracker::query_hardware_memory(int cuda_device_index) {
    HardwareMemoryStats stats;

#if defined(_WIN32)
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        stats.host_total_bytes = static_cast<size_t>(mem_info.ullTotalPhys);
        stats.host_free_bytes = static_cast<size_t>(mem_info.ullAvailPhys);
    }
#endif

#if AIC_WITH_CUDA
    size_t free_b = 0, total_b = 0;
    if (cudaSetDevice(cuda_device_index) == cudaSuccess &&
        cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        stats.gpu_total_bytes = total_b;
        stats.gpu_free_bytes = free_b;
    }
#else
    (void)cuda_device_index;
#endif

    return stats;
}

TelemetrySnapshot TelemetryTracker::record_step(
    int64_t step,
    float loss,
    float grad_norm,
    float lr,
    float tokens_per_sec,
    double step_duration_sec,
    int64_t total_tokens,
    int64_t model_params,
    int cuda_device_index
) {
    TelemetrySnapshot s;
    s.step = step;
    s.loss = loss;
    s.grad_norm = grad_norm;
    s.lr = lr;
    s.tokens_per_sec = tokens_per_sec;
    s.memory = query_hardware_memory(cuda_device_index);

    // Compute estimated TFLOP/s: (6 * P * tokens) / (time * 1e12)
    if (model_params > 0) {
        if (tokens_per_sec > 0.0f) {
            double flops = 6.0 * static_cast<double>(model_params) * static_cast<double>(tokens_per_sec);
            s.estimated_tflops = static_cast<float>(flops / 1e12);
        } else if (step_duration_sec > 0.0 && total_tokens > 0) {
            double flops = 6.0 * static_cast<double>(model_params) * static_cast<double>(total_tokens);
            s.estimated_tflops = static_cast<float>((flops / step_duration_sec) / 1e12);
        }
    }

    // Check for NaN or Inf immediately
    if (std::isnan(loss) || std::isinf(loss)) {
        s.detected_anomaly = AnomalyType::NanOrInfLoss;
    } else if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
        s.detected_anomaly = AnomalyType::NanOrInfGradient;
    }

    if (history_.empty()) {
        loss_ema_short_ = loss;
        loss_ema_long_ = loss;
        grad_norm_ema_ = grad_norm;
    } else {
        if (!std::isnan(loss) && !std::isinf(loss)) {
            loss_ema_short_ = alpha_short_ * loss_ema_short_ + (1.0f - alpha_short_) * loss;
            loss_ema_long_ = alpha_long_ * loss_ema_long_ + (1.0f - alpha_long_) * loss;
        }
        if (!std::isnan(grad_norm) && !std::isinf(grad_norm)) {
            grad_norm_ema_ = alpha_short_ * grad_norm_ema_ + (1.0f - alpha_short_) * grad_norm;
        }
    }

    s.loss_ema_short = loss_ema_short_;
    s.loss_ema_long = loss_ema_long_;
    s.grad_norm_ema = grad_norm_ema_;

    // Calculate rolling loss variance across sliding window
    history_.push_back(s);
    if (history_.size() > window_size_) {
        history_.pop_front();
    }

    if (history_.size() >= 2) {
        double sum = 0.0;
        double sum_sq = 0.0;
        size_t n = 0;
        for (const auto& item : history_) {
            if (!std::isnan(item.loss) && !std::isinf(item.loss)) {
                sum += item.loss;
                sum_sq += item.loss * item.loss;
                n++;
            }
        }
        if (n >= 2) {
            double mean = sum / static_cast<double>(n);
            double var = (sum_sq / static_cast<double>(n)) - (mean * mean);
            s.loss_variance = var > 0.0 ? static_cast<float>(var) : 0.0f;
        }
    }

    // Update consecutive divergence tracking
    if (!std::isnan(loss) && !std::isinf(loss) && loss_ema_long_ > 1e-6f && loss > 1.4f * loss_ema_long_) {
        consecutive_divergence_++;
    } else {
        consecutive_divergence_ = 0;
    }

    latest_snapshot_ = s;

    // Anomaly detection rules
    if (s.detected_anomaly == AnomalyType::None) {
        if (is_diverging()) {
            s.detected_anomaly = AnomalyType::LossDivergence;
        } else if (is_gradient_exploding()) {
            s.detected_anomaly = AnomalyType::GradientExplosion;
        } else if (is_gradient_vanishing()) {
            s.detected_anomaly = AnomalyType::GradientVanishing;
        } else if (is_memory_critical()) {
            s.detected_anomaly = AnomalyType::MemoryCritical;
        }
    }

    latest_snapshot_ = s;
    // Replace the back with the fully populated snapshot
    history_.back() = s;
    return s;
}

bool TelemetryTracker::is_nan_or_inf() const noexcept {
    return std::isnan(latest_snapshot_.loss) || std::isinf(latest_snapshot_.loss) ||
           std::isnan(latest_snapshot_.grad_norm) || std::isinf(latest_snapshot_.grad_norm);
}

bool TelemetryTracker::is_diverging(float threshold_ratio, int min_consecutive) const noexcept {
    if (history_.size() < 3) return false;
    if (std::abs(threshold_ratio - 1.4f) < 1e-4f) {
        return consecutive_divergence_ >= min_consecutive;
    }
    return loss_ema_long_ > 1e-6f &&
           (latest_snapshot_.loss > threshold_ratio * loss_ema_long_) &&
           (consecutive_divergence_ >= min_consecutive);
}

bool TelemetryTracker::is_gradient_exploding(float spike_factor) const noexcept {
    if (history_.size() < 5) return false;
    if (grad_norm_ema_ > 1e-6f) {
        return latest_snapshot_.grad_norm > spike_factor * grad_norm_ema_;
    }
    return latest_snapshot_.grad_norm > 100.0f;
}

bool TelemetryTracker::is_gradient_vanishing(float min_norm) const noexcept {
    if (history_.size() < 5) return false;
    return latest_snapshot_.grad_norm < min_norm;
}

bool TelemetryTracker::is_memory_critical(float min_free_ratio) const noexcept {
    if (latest_snapshot_.memory.gpu_total_bytes > 0) {
        float free_ratio = static_cast<float>(latest_snapshot_.memory.gpu_free_bytes) /
                           static_cast<float>(latest_snapshot_.memory.gpu_total_bytes);
        return free_ratio < min_free_ratio;
    }
    return false;
}

}  // namespace aic::autonomous
