// tests/unit/test_amp.cpp
// Unit and Parity tests for Automatic Mixed Precision (AMP) and Dynamic Loss Scaler.
#include "train/amp.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "models/transformer_lm.hpp"
#include "ops/ops.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"
#include "train/trainer.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace aic;
using namespace aic::train;

namespace {

Tensor make_tensor(const std::vector<float>& values, Shape shape, Device device = Device::cpu()) {
    Tensor t = Tensor::empty(shape, DType::Float32, Device::cpu());
    std::memcpy(t.data_ptr<float>(), values.data(), values.size() * sizeof(float));
    if (device.is_cuda()) {
        return t.to(device);
    }
    return t;
}

}  // namespace

AIC_TEST(AutocastTest, ScopeGuardThreadLocal) {
    ASSERT_FALSE(is_autocast_enabled());
    ASSERT_EQ(static_cast<int>(get_autocast_dtype()), static_cast<int>(DType::Float16));

    {
        AutocastGuard guard(true, DType::BFloat16);
        ASSERT_TRUE(is_autocast_enabled());
        ASSERT_EQ(static_cast<int>(get_autocast_dtype()), static_cast<int>(DType::BFloat16));

        {
            AutocastGuard inner_guard(false, DType::Float16);
            ASSERT_FALSE(is_autocast_enabled());
            ASSERT_EQ(static_cast<int>(get_autocast_dtype()), static_cast<int>(DType::Float16));
        }

        ASSERT_TRUE(is_autocast_enabled());
        ASSERT_EQ(static_cast<int>(get_autocast_dtype()), static_cast<int>(DType::BFloat16));
    }

    ASSERT_FALSE(is_autocast_enabled());
}

AIC_TEST(LossScalerTest, ScaleAndUnscaleCPU) {
    DynamicLossScalerOptions opts;
    opts.initial_scale = 1024.0f;
    DynamicLossScaler scaler(opts);

    ASSERT_EQ(scaler.current_scale(), 1024.0f);

    Tensor loss = make_tensor({2.5f}, Shape({1}), Device::cpu());
    Tensor scaled_loss = scaler.scale(loss);
    ASSERT_NEAR(scaled_loss.item<float>(), 2560.0f, 1e-4f);

    auto param = std::make_shared<Tensor>(Tensor::zeros(Shape({4}), DType::Float32, Device::cpu()));
    param->set_requires_grad(true);

    // Provide a gradient scaled by 1024.0f
    Tensor g = make_tensor({1024.0f, 2048.0f, -512.0f, 0.0f}, Shape({4}), Device::cpu());
    param->set_grad(g);

    bool ok = scaler.unscale_and_update({param});
    ASSERT_TRUE(ok);
    ASSERT_EQ(scaler.consecutive_good_steps(), 1);
    ASSERT_EQ(scaler.skipped_steps(), 0);

    const float* g_data = param->grad().data_ptr<float>();
    ASSERT_NEAR(g_data[0], 1.0f, 1e-5f);
    ASSERT_NEAR(g_data[1], 2.0f, 1e-5f);
    ASSERT_NEAR(g_data[2], -0.5f, 1e-5f);
    ASSERT_NEAR(g_data[3], 0.0f, 1e-5f);
}

AIC_TEST(LossScalerTest, NanInfDetectionAndBackoffCPU) {
    DynamicLossScalerOptions opts;
    opts.initial_scale = 1024.0f;
    opts.backoff_factor = 0.5f;
    opts.min_scale = 1.0f;
    DynamicLossScaler scaler(opts);

    auto param = std::make_shared<Tensor>(Tensor::zeros(Shape({4}), DType::Float32, Device::cpu()));
    param->set_requires_grad(true);

    // Test with NaN
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    Tensor g_nan = make_tensor({1.0f, nan_val, 2.0f, 3.0f}, Shape({4}), Device::cpu());
    param->set_grad(g_nan);

    bool ok = scaler.unscale_and_update({param});
    ASSERT_FALSE(ok);
    ASSERT_EQ(scaler.current_scale(), 512.0f);
    ASSERT_EQ(scaler.consecutive_good_steps(), 0);
    ASSERT_EQ(scaler.skipped_steps(), 1);

    // Test with Inf
    float inf_val = std::numeric_limits<float>::infinity();
    Tensor g_inf = make_tensor({1.0f, 2.0f, inf_val, 3.0f}, Shape({4}), Device::cpu());
    param->set_grad(g_inf);

    ok = scaler.unscale_and_update({param});
    ASSERT_FALSE(ok);
    ASSERT_EQ(scaler.current_scale(), 256.0f);
    ASSERT_EQ(scaler.skipped_steps(), 2);
}

AIC_TEST(LossScalerTest, GrowthSchedule) {
    DynamicLossScalerOptions opts;
    opts.initial_scale = 128.0f;
    opts.growth_factor = 2.0f;
    opts.growth_interval = 3;
    opts.max_scale = 1024.0f;
    DynamicLossScaler scaler(opts);

    auto param = std::make_shared<Tensor>(Tensor::zeros(Shape({2}), DType::Float32, Device::cpu()));
    param->set_requires_grad(true);

    for (int step = 1; step <= 6; ++step) {
        Tensor g = make_tensor({1.0f, 2.0f}, Shape({2}), Device::cpu());
        param->set_grad(g);
        bool ok = scaler.unscale_and_update({param});
        ASSERT_TRUE(ok);
        if (step < 3) {
            ASSERT_EQ(scaler.current_scale(), 128.0f);
        } else if (step >= 3 && step < 6) {
            ASSERT_EQ(scaler.current_scale(), 256.0f);
        } else if (step == 6) {
            ASSERT_EQ(scaler.current_scale(), 512.0f);
        }
    }
}

#if AIC_WITH_CUDA
AIC_TEST(LossScalerTest, CudaNanInfAndUnscaleGPU) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available for GPU AMP tests\n";
        return;
    }

    Device gpu = Device::cuda(0);
    DynamicLossScalerOptions opts;
    opts.initial_scale = 2048.0f;
    DynamicLossScaler scaler(opts);

    // 1. Verify unscale on GPU with vectorized float4
    std::vector<float> h_grads(1024);
    for (size_t i = 0; i < h_grads.size(); ++i) {
        h_grads[i] = static_cast<float>(i + 1) * 2048.0f;
    }
    Tensor g_gpu = make_tensor(h_grads, Shape({1024}), gpu);
    auto param_gpu = std::make_shared<Tensor>(Tensor::zeros(Shape({1024}), DType::Float32, gpu));
    param_gpu->set_requires_grad(true);
    param_gpu->set_grad(g_gpu);

    bool ok = scaler.unscale_and_update({param_gpu});
    ASSERT_TRUE(ok);
    ASSERT_EQ(scaler.current_scale(), 2048.0f);

    Tensor result_cpu = param_gpu->grad().cpu();
    const float* r_data = result_cpu.data_ptr<float>();
    for (size_t i = 0; i < 16; ++i) {
        ASSERT_NEAR(r_data[i], static_cast<float>(i + 1), 1e-4f);
    }

    // 2. Verify GPU NaN detection with warp vote reduction
    h_grads[513] = std::numeric_limits<float>::quiet_NaN();
    Tensor g_nan_gpu = make_tensor(h_grads, Shape({1024}), gpu);
    param_gpu->set_grad(g_nan_gpu);

    ok = scaler.unscale_and_update({param_gpu});
    ASSERT_FALSE(ok);
    ASSERT_EQ(scaler.current_scale(), 1024.0f);
    ASSERT_EQ(scaler.skipped_steps(), 1);
}
#endif
