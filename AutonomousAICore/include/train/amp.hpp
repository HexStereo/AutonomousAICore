// include/train/amp.hpp
// Automatic Mixed Precision (AMP) and Dynamic Loss Scaling.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/dtype.hpp"
#include "tensor/tensor.hpp"

namespace aic::train {

// Thread-local Autocast state and RAII scope guard
bool is_autocast_enabled() noexcept;
DType get_autocast_dtype() noexcept;
void set_autocast_enabled(bool enabled, DType dtype = DType::Float16) noexcept;

class AutocastGuard {
public:
    explicit AutocastGuard(bool enabled = true, DType dtype = DType::Float16);
    ~AutocastGuard();

    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;
    AutocastGuard(AutocastGuard&&) noexcept = default;
    AutocastGuard& operator=(AutocastGuard&&) noexcept = default;

private:
    bool prev_enabled_;
    DType prev_dtype_;
};

struct DynamicLossScalerOptions {
    float initial_scale = 65536.0f; // 2^16
    float growth_factor = 2.0f;
    float backoff_factor = 0.5f;
    int growth_interval = 1000;
    float min_scale = 1.0f;
    float max_scale = 16777216.0f;  // 2^24
};

class DynamicLossScaler {
public:
    explicit DynamicLossScaler(DynamicLossScalerOptions options = {});

    // Scales loss tensor: loss * current_scale
    Tensor scale(const Tensor& loss) const;

    // Checks for NaN or Inf across all gradients.
    // If NaN/Inf found: skips step, reduces scale by backoff_factor, returns false.
    // If valid: unscales all gradients (grad / scale), tracks growth, returns true.
    bool unscale_and_update(const std::vector<std::shared_ptr<Tensor>>& parameters);

    float current_scale() const noexcept { return current_scale_; }
    void set_scale(float s) noexcept { current_scale_ = s; }

    int consecutive_good_steps() const noexcept { return consecutive_good_steps_; }
    int skipped_steps() const noexcept { return skipped_steps_; }

    const DynamicLossScalerOptions& options() const noexcept { return options_; }
    DynamicLossScalerOptions& options() noexcept { return options_; }

    void reset();

    // Utility: checks if tensor has NaN or Inf on CPU or GPU
    static bool has_nan_or_inf(const Tensor& tensor);

    // Utility: unscales tensor in-place on CPU or GPU
    static void unscale_tensor_(Tensor& tensor, float inv_scale);

private:
    DynamicLossScalerOptions options_;
    float current_scale_ = 65536.0f;
    int consecutive_good_steps_ = 0;
    int skipped_steps_ = 0;
};

}  // namespace aic::train
