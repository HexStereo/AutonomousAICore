// tests/unit/test_flash_attention.cpp
// Comprehensive tests for FlashAttention-2: Parity, Causal Masking, Autograd, Scaling, and CUDA.
#include "ops/flash_attention.hpp"
#include "autograd/autograd.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "nn/attention.hpp"
#include "ops/matmul.hpp"
#include "ops/ops.hpp"
#include "optim/adamw.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace aic;
using namespace aic::ops;
using namespace aic::nn;
using namespace aic::optim;

namespace {

Tensor create_random_tensor(const Shape& shape, Device device, float mean = 0.0f, float stddev = 1.0f, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    int64_t n = shape.numel();
    std::vector<float> data(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = dist(gen);
    }
    return Tensor::from_blob(data.data(), shape, DType::Float32, Device::cpu()).clone().to(device);
}

// Reference standard attention implementation
Tensor standard_attention(const Tensor& q, const Tensor& k, const Tensor& v, bool is_causal, float scale) {
    // q: [B, H, Sq, D], k: [B, H, Sk, D], v: [B, H, Sk, D]
    int64_t Sq = q.dim(2);
    int64_t Sk = k.dim(2);

    Tensor k_t = k.transpose(2, 3);
    Tensor scores = matmul(q, k_t);
    scores = mul(scores, scale);

    if (is_causal && Sq > 1) {
        std::vector<float> mask_vals(static_cast<size_t>(Sq * Sk), 0.0f);
        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t j = 0; j < Sk; ++j) {
                if (j > i) {
                    mask_vals[static_cast<size_t>(i * Sk + j)] = -1e9f;
                }
            }
        }
        Tensor mask = Tensor::from_blob(mask_vals.data(), Shape({1, 1, Sq, Sk}), DType::Float32, Device::cpu()).clone().to(q.device());
        scores = add(scores, mask);
    }

    Tensor weights = softmax(scores, -1);
    return matmul(weights, v);
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    Tensor diff = sub(a.cpu().contiguous(), b.cpu().contiguous());
    const float* ptr = diff.data_ptr<float>();
    float max_val = 0.0f;
    for (int64_t i = 0; i < diff.numel(); ++i) {
        float v = std::abs(ptr[i]);
        if (v > max_val) max_val = v;
    }
    return max_val;
}

}  // namespace

AIC_TEST(FlashAttentionTest, NumericalParityForwardCPU) {
    int64_t B = 2;
    int64_t H = 2;
    int64_t S = 8;
    int64_t D = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Tensor q = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 101);
    Tensor k = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 102);
    Tensor v = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 103);

    Tensor ref_out = standard_attention(q, k, v, /*is_causal=*/false, scale);
    Tensor flash_out = flash_attention(q, k, v, /*is_causal=*/false, scale);

    float err = max_abs_diff(ref_out, flash_out);
    ASSERT_NEAR(err, 0.0f, 1e-4f);
}

AIC_TEST(FlashAttentionTest, CausalMaskingParityCPU) {
    int64_t B = 2;
    int64_t H = 2;
    int64_t S = 12;
    int64_t D = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Tensor q = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 201);
    Tensor k = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 202);
    Tensor v = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 203);

    Tensor ref_out = standard_attention(q, k, v, /*is_causal=*/true, scale);
    Tensor flash_out = flash_attention(q, k, v, /*is_causal=*/true, scale);

    float err = max_abs_diff(ref_out, flash_out);
    ASSERT_NEAR(err, 0.0f, 1e-4f);
}

AIC_TEST(FlashAttentionTest, AutogradGradientParityCPU) {
    int64_t B = 1;
    int64_t H = 2;
    int64_t S = 6;
    int64_t D = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Reference standard attention autograd
    Tensor q1 = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 301);
    Tensor k1 = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 302);
    Tensor v1 = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 303);
    q1.set_requires_grad(true);
    k1.set_requires_grad(true);
    v1.set_requires_grad(true);

    Tensor out1 = standard_attention(q1, k1, v1, /*is_causal=*/true, scale);
    Tensor loss1 = sum(out1);
    loss1.backward();

    // FlashAttention autograd
    Tensor q2 = q1.detach().clone();
    Tensor k2 = k1.detach().clone();
    Tensor v2 = v1.detach().clone();
    q2.set_requires_grad(true);
    k2.set_requires_grad(true);
    v2.set_requires_grad(true);

    Tensor out2 = flash_attention(q2, k2, v2, /*is_causal=*/true, scale);
    Tensor loss2 = sum(out2);
    loss2.backward();

    float err_q = max_abs_diff(q1.grad(), q2.grad());
    float err_k = max_abs_diff(k1.grad(), k2.grad());
    float err_v = max_abs_diff(v1.grad(), v2.grad());

    ASSERT_NEAR(err_q, 0.0f, 5e-3f);
    ASSERT_NEAR(err_k, 0.0f, 5e-3f);
    ASSERT_NEAR(err_v, 0.0f, 5e-3f);
}

AIC_TEST(FlashAttentionTest, LongSequenceScalingMultiTile) {
    int64_t B = 1;
    int64_t H = 2;
    int64_t S = 64;  // Crosses multiple Bc (32) boundaries
    int64_t D = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Tensor q = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.2f, 401);
    Tensor k = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.2f, 402);
    Tensor v = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.2f, 403);

    Tensor ref_out = standard_attention(q, k, v, /*is_causal=*/true, scale);
    Tensor flash_out = flash_attention(q, k, v, /*is_causal=*/true, scale);

    float err = max_abs_diff(ref_out, flash_out);
    ASSERT_NEAR(err, 0.0f, 1e-4f);
}

AIC_TEST(FlashAttentionTest, CausalSelfAttentionModuleIntegration) {
    int64_t d_model = 32;
    int64_t n_heads = 4;
    int64_t B = 2;
    int64_t S = 8;

    CausalSelfAttention attn(d_model, n_heads, 64, Device::cpu());

    Tensor x = create_random_tensor(Shape({B, S, d_model}), Device::cpu(), 0.0f, 0.5f, 501);

    // Standard forward
    attn.set_use_flash_attention(false);
    Tensor out_std = attn.forward(x);

    // FlashAttention forward
    attn.set_use_flash_attention(true);
    Tensor out_flash = attn.forward(x);

    float err = max_abs_diff(out_std, out_flash);
    ASSERT_NEAR(err, 0.0f, 2e-4f);

    // Autograd backward verification through module
    x.set_requires_grad(true);
    Tensor out = attn.forward(x);
    Tensor loss = sum(out);
    loss.backward();

    ASSERT_TRUE(x.grad().defined());
    ASSERT_EQ(x.grad().numel(), x.numel());
    ASSERT_TRUE(attn.q_proj()->weight()->grad().defined());
    ASSERT_TRUE(attn.out_proj()->weight()->grad().defined());
}

AIC_TEST(FlashAttentionTest, CudaExecutionAndParity) {
    if (!is_cuda_available()) {
        std::cout << "[SKIP] CUDA not available on this device" << std::endl;
        return;
    }

    Device gpu_dev = Device::cuda(0);
    int64_t B = 2;
    int64_t H = 2;
    int64_t S = 16;
    int64_t D = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Tensor q_cpu = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 601);
    Tensor k_cpu = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 602);
    Tensor v_cpu = create_random_tensor(Shape({B, H, S, D}), Device::cpu(), 0.0f, 0.5f, 603);

    Tensor q_gpu = q_cpu.clone().to(gpu_dev);
    Tensor k_gpu = k_cpu.clone().to(gpu_dev);
    Tensor v_gpu = v_cpu.clone().to(gpu_dev);

    // Forward comparison
    Tensor out_cpu = flash_attention(q_cpu, k_cpu, v_cpu, /*is_causal=*/true, scale);
    Tensor out_gpu = flash_attention(q_gpu, k_gpu, v_gpu, /*is_causal=*/true, scale);

    float fwd_err = max_abs_diff(out_cpu, out_gpu.cpu());
    ASSERT_NEAR(fwd_err, 0.0f, 1e-3f);

    // Autograd backward comparison on GPU
    q_gpu.set_requires_grad(true);
    k_gpu.set_requires_grad(true);
    v_gpu.set_requires_grad(true);

    Tensor gpu_loss = sum(flash_attention(q_gpu, k_gpu, v_gpu, /*is_causal=*/true, scale));
    gpu_loss.backward();

    q_cpu.set_requires_grad(true);
    k_cpu.set_requires_grad(true);
    v_cpu.set_requires_grad(true);
    Tensor cpu_loss = sum(flash_attention(q_cpu, k_cpu, v_cpu, /*is_causal=*/true, scale));
    cpu_loss.backward();

    float bwd_q_err = max_abs_diff(q_cpu.grad(), q_gpu.grad().cpu());
    float bwd_k_err = max_abs_diff(k_cpu.grad(), k_gpu.grad().cpu());
    float bwd_v_err = max_abs_diff(v_cpu.grad(), v_gpu.grad().cpu());

    ASSERT_NEAR(bwd_q_err, 0.0f, 5e-3f);
    ASSERT_NEAR(bwd_k_err, 0.0f, 5e-3f);
    ASSERT_NEAR(bwd_v_err, 0.0f, 5e-3f);
}

AIC_TEST(FlashAttentionTest, EndToEndTrainingConvergence) {
    int64_t d_model = 32;
    int64_t n_heads = 4;
    int64_t B = 2;
    int64_t S = 8;

    CausalSelfAttention attn(d_model, n_heads, 64, Device::cpu());
    attn.set_use_flash_attention(true);

    Tensor x = create_random_tensor(Shape({B, S, d_model}), Device::cpu(), 0.0f, 0.5f, 701);
    Tensor target = create_random_tensor(Shape({B, S, d_model}), Device::cpu(), 0.0f, 0.5f, 702);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    AdamWOptions opts;
    opts.lr = 0.005f;
    opts.weight_decay = 0.0f;
    AdamW opt(attn.parameters(), opts);

    for (int step = 0; step < 15; ++step) {
        opt.zero_grad();
        Tensor pred = attn.forward(x);
        Tensor diff = sub(pred, target);
        Tensor loss = mul(sum(mul(diff, diff)), 1.0f / static_cast<float>(diff.numel()));

        float l_val = loss.item<float>();
        if (step == 0) initial_loss = l_val;
        final_loss = l_val;

        loss.backward();
        opt.step();
    }

    ASSERT_TRUE(final_loss < initial_loss);
}
