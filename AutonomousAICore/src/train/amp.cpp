// src/train/amp.cpp
// Automatic Mixed Precision (AMP) and Dynamic Loss Scaler Implementation.
#include "train/amp.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
namespace aic::cuda {
bool launch_has_nan_or_inf_gpu(const float* data, int64_t n, cudaStream_t stream = nullptr);
void launch_unscale_gpu(float* data, int64_t n, float inv_scale, cudaStream_t stream = nullptr);
}
#endif

namespace aic::train {

namespace {

thread_local bool t_autocast_enabled = false;
thread_local DType t_autocast_dtype = DType::Float16;

}  // namespace

bool is_autocast_enabled() noexcept {
    return t_autocast_enabled;
}

DType get_autocast_dtype() noexcept {
    return t_autocast_dtype;
}

void set_autocast_enabled(bool enabled, DType dtype) noexcept {
    t_autocast_enabled = enabled;
    t_autocast_dtype = dtype;
}

AutocastGuard::AutocastGuard(bool enabled, DType dtype)
    : prev_enabled_(is_autocast_enabled()),
      prev_dtype_(get_autocast_dtype()) {
    set_autocast_enabled(enabled, dtype);
}

AutocastGuard::~AutocastGuard() {
    set_autocast_enabled(prev_enabled_, prev_dtype_);
}

DynamicLossScaler::DynamicLossScaler(DynamicLossScalerOptions options)
    : options_(options),
      current_scale_(options.initial_scale) {}

void DynamicLossScaler::reset() {
    current_scale_ = options_.initial_scale;
    consecutive_good_steps_ = 0;
    skipped_steps_ = 0;
}

Tensor DynamicLossScaler::scale(const Tensor& loss) const {
    AIC_CHECK(loss.defined(), InvalidArgument, "Cannot scale undefined loss tensor");
    return mul(loss, current_scale_);
}

bool DynamicLossScaler::has_nan_or_inf(const Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) return false;

    if (tensor.is_cpu()) {
        const float* ptr = tensor.contiguous().data_ptr<float>();
        int64_t n = tensor.numel();
        for (int64_t i = 0; i < n; ++i) {
            float v = ptr[i];
            if (std::isnan(v) || std::isinf(v)) {
                return true;
            }
        }
        return false;
    }

#if AIC_WITH_CUDA
    if (tensor.is_cuda()) {
        return cuda::launch_has_nan_or_inf_gpu(
            tensor.contiguous().data_ptr<float>(),
            tensor.numel()
        );
    }
#endif

    return false;
}

void DynamicLossScaler::unscale_tensor_(Tensor& tensor, float inv_scale) {
    if (!tensor.defined() || tensor.numel() == 0) return;

    if (tensor.is_cpu()) {
        float* ptr = tensor.data_ptr<float>();
        int64_t n = tensor.numel();
        for (int64_t i = 0; i < n; ++i) {
            ptr[i] *= inv_scale;
        }
        return;
    }

#if AIC_WITH_CUDA
    if (tensor.is_cuda()) {
        cuda::launch_unscale_gpu(
            tensor.data_ptr<float>(),
            tensor.numel(),
            inv_scale
        );
        return;
    }
#endif

    mul_(tensor, inv_scale);
}

bool DynamicLossScaler::unscale_and_update(const std::vector<std::shared_ptr<Tensor>>& parameters) {
    // 1. Scan all gradients for NaN or Inf
    for (const auto& p : parameters) {
        if (!p || !p->defined()) continue;
        Tensor g = p->grad();
        if (!g.defined()) continue;

        if (has_nan_or_inf(g)) {
            // Gradient overflow detected: backoff scale and skip optimizer step
            current_scale_ = std::max(options_.min_scale, current_scale_ * options_.backoff_factor);
            consecutive_good_steps_ = 0;
            skipped_steps_++;
            return false;
        }
    }

    // 2. Unscale all gradients in-place
    float inv_scale = 1.0f / current_scale_;
    for (const auto& p : parameters) {
        if (!p || !p->defined()) continue;
        Tensor g = p->grad();
        if (!g.defined()) continue;

        unscale_tensor_(g, inv_scale);
    }

    // 3. Track stable steps and apply scale growth
    consecutive_good_steps_++;
    if (consecutive_good_steps_ >= options_.growth_interval) {
        current_scale_ = std::min(options_.max_scale, current_scale_ * options_.growth_factor);
        consecutive_good_steps_ = 0;
    }

    return true;
}

}  // namespace aic::train
