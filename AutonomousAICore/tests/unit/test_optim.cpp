// tests/unit/test_optim.cpp
// Verification of Loss functions, Numerical stability, Gradient Clipping, and AdamW optimizer.
#include "loss/loss.hpp"
#include "optim/optim.hpp"
#include "nn/linear.hpp"
#include "ops/ops.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <iostream>

using namespace aic;
using namespace aic::optim;

AIC_TEST(LossTest, CrossEntropyExactMath) {
    // 2 samples, 3 classes
    Tensor logits = Tensor::zeros(Shape({2, 3}), DType::Float32, Device::cpu());
    float* z = logits.data_ptr<float>();
    z[0] = 1.0f; z[1] = 2.0f; z[2] = 3.0f;
    z[3] = 1.0f; z[4] = -1.0f; z[5] = 0.0f;

    Tensor targets = Tensor::zeros(Shape({2}), DType::Int64, Device::cpu());
    int64_t* y = targets.data_ptr<int64_t>();
    y[0] = 2; // class 2 for sample 0
    y[1] = 0; // class 0 for sample 1

    Tensor loss = cross_entropy(logits, targets, -100, "mean");
    float loss_val = loss.item<float>();
    // Precomputed exact mathematical value: ~0.407606
    float expected = 0.407606f;
    ASSERT_NEAR(loss_val, expected, 1e-4f);

    Tensor sum_loss = cross_entropy(logits, targets, -100, "sum");
    ASSERT_NEAR(sum_loss.item<float>(), expected * 2.0f, 1e-4f);
}

AIC_TEST(LossTest, CrossEntropyNumericalStability) {
    // Large values that would cause exp() overflow without LogSumExp stabilization
    Tensor logits = Tensor::zeros(Shape({2, 4}), DType::Float32, Device::cpu());
    float* z = logits.data_ptr<float>();
    z[0] = 1000.0f; z[1] = 1001.0f; z[2] = 999.0f; z[3] = 998.0f;
    z[4] = -1000.0f; z[5] = -999.0f; z[6] = -1001.0f; z[7] = -1002.0f;

    Tensor targets = Tensor::zeros(Shape({2}), DType::Int64, Device::cpu());
    int64_t* y = targets.data_ptr<int64_t>();
    y[0] = 1;
    y[1] = 5 - 4; // index 1

    logits.set_requires_grad(true);
    Tensor loss = cross_entropy(logits, targets, -100, "mean");
    float val = loss.item<float>();

    ASSERT_FALSE(std::isnan(val));
    ASSERT_FALSE(std::isinf(val));
    ASSERT_TRUE(val > 0.0f);

    loss.backward();
    ASSERT_TRUE(logits.grad().defined());
    const float* gz = logits.grad().data_ptr<float>();
    for (int i = 0; i < 8; ++i) {
        ASSERT_FALSE(std::isnan(gz[i]));
        ASSERT_FALSE(std::isinf(gz[i]));
    }
}

AIC_TEST(LossTest, CrossEntropyAnalyticalGradcheck) {
    Tensor logits = Tensor::randn(Shape({3, 5}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    logits.set_requires_grad(true);

    Tensor targets = Tensor::zeros(Shape({3}), DType::Int64, Device::cpu());
    int64_t* y = targets.data_ptr<int64_t>();
    y[0] = 1; y[1] = 4; y[2] = 0;

    Tensor loss = cross_entropy(logits, targets, -100, "mean");
    loss.backward();

    Tensor analytical_grad = logits.grad().clone();

    // Numerical finite differences
    const float h = 1e-3f;
    float* z_ptr = logits.data_ptr<float>();
    const float* a_ptr = analytical_grad.data_ptr<float>();

    for (int64_t i = 0; i < logits.numel(); ++i) {
        float orig = z_ptr[i];

        z_ptr[i] = orig + h;
        float loss_pos = cross_entropy(logits, targets, -100, "mean").item<float>();

        z_ptr[i] = orig - h;
        float loss_neg = cross_entropy(logits, targets, -100, "mean").item<float>();

        z_ptr[i] = orig;

        float num_grad = (loss_pos - loss_neg) / (2.0f * h);
        ASSERT_NEAR(a_ptr[i], num_grad, 1e-3f);
    }
}

AIC_TEST(LossTest, CrossEntropy3DAndIgnoreIndex) {
    // Typical LLM output shape: [Batch=2, Seq=3, Vocab=5]
    Tensor logits = Tensor::randn(Shape({2, 3, 5}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    logits.set_requires_grad(true);

    Tensor targets = Tensor::zeros(Shape({2, 3}), DType::Int64, Device::cpu());
    int64_t* y = targets.data_ptr<int64_t>();
    y[0] = 0;    y[1] = 2;    y[2] = -100; // token 2 ignored
    y[3] = -100; y[4] = 4;    y[5] = 1;    // token 3 ignored

    Tensor loss = cross_entropy(logits, targets, -100, "mean");
    loss.backward();

    ASSERT_TRUE(logits.grad().defined());
    const float* gz = logits.grad().data_ptr<float>();

    // Ignored positions (index 2 and 3) must have exactly 0 gradient across all classes
    for (int c = 0; c < 5; ++c) {
        ASSERT_EQ(gz[2 * 5 + c], 0.0f);
        ASSERT_EQ(gz[3 * 5 + c], 0.0f);
    }

    // Active positions must have non-zero gradients
    bool has_nonzero = false;
    for (int c = 0; c < 5; ++c) {
        if (std::abs(gz[0 * 5 + c]) > 1e-6f) has_nonzero = true;
    }
    ASSERT_TRUE(has_nonzero);
}

AIC_TEST(LossTest, CrossEntropyGPUParity) {
    if (!is_cuda_available()) return;

    Tensor logits_cpu = Tensor::randn(Shape({4, 16}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    logits_cpu.set_requires_grad(true);

    Tensor targets_cpu = Tensor::zeros(Shape({4}), DType::Int64, Device::cpu());
    int64_t* y = targets_cpu.data_ptr<int64_t>();
    y[0] = 3; y[1] = 15; y[2] = 0; y[3] = 7;

    Tensor loss_cpu = cross_entropy(logits_cpu, targets_cpu, -100, "mean");
    loss_cpu.backward();

    Tensor logits_gpu = logits_cpu.cuda();
    logits_gpu.set_requires_grad(true);
    Tensor targets_gpu = targets_cpu.cuda();

    Tensor loss_gpu = cross_entropy(logits_gpu, targets_gpu, -100, "mean");
    loss_gpu.backward();

    ASSERT_NEAR(loss_cpu.item<float>(), loss_gpu.cpu().item<float>(), 1e-4f);

    Tensor g_cpu = logits_cpu.grad();
    Tensor g_gpu = logits_gpu.grad().cpu();
    const float* p_cpu = g_cpu.data_ptr<float>();
    const float* p_gpu = g_gpu.data_ptr<float>();
    for (int64_t i = 0; i < logits_cpu.numel(); ++i) {
        ASSERT_NEAR(p_cpu[i], p_gpu[i], 1e-4f);
    }
}

AIC_TEST(LossTest, MSELossCPUAndGPU) {
    Tensor a = Tensor::randn(Shape({4, 8}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    Tensor b = Tensor::randn(Shape({4, 8}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    a.set_requires_grad(true);

    Tensor loss_cpu = mse_loss(a, b, "mean");
    loss_cpu.backward();

    ASSERT_TRUE(a.grad().defined());

    if (is_cuda_available()) {
        Tensor a_gpu = a.cuda();
        Tensor b_gpu = b.cuda();
        a_gpu.set_requires_grad(true);

        Tensor loss_gpu = mse_loss(a_gpu, b_gpu, "mean");
        loss_gpu.backward();

        ASSERT_NEAR(loss_cpu.item<float>(), loss_gpu.cpu().item<float>(), 1e-4f);

        Tensor g_gpu_cpu = a_gpu.grad().cpu();
        const float* p_c = a.grad().data_ptr<float>();
        const float* p_g = g_gpu_cpu.data_ptr<float>();
        for (int64_t i = 0; i < a.numel(); ++i) {
            ASSERT_NEAR(p_c[i], p_g[i], 1e-4f);
        }
    }
}

AIC_TEST(OptimTest, ClipGradNorm) {
    auto p1 = std::make_shared<Tensor>(Tensor::zeros(Shape({3}), DType::Float32, Device::cpu()));
    auto p2 = std::make_shared<Tensor>(Tensor::zeros(Shape({4}), DType::Float32, Device::cpu()));

    Tensor g1 = Tensor::full(Shape({3}), 3.0f, DType::Float32, Device::cpu()); // 3 * 3^2 = 27
    Tensor g2 = Tensor::full(Shape({4}), 4.0f, DType::Float32, Device::cpu()); // 4 * 4^2 = 64
    // total norm sq = 27 + 64 = 91. total norm = sqrt(91) ~ 9.53939
    p1->set_grad(g1);
    p2->set_grad(g2);

    float norm = clip_grad_norm_({p1, p2}, 5.0f);
    ASSERT_NEAR(norm, std::sqrt(91.0f), 1e-3f);

    // After clipping with max_norm=5.0, new total norm must be ~5.0
    float new_norm = clip_grad_norm_({p1, p2}, 100.0f); // with large threshold, just measures norm
    ASSERT_NEAR(new_norm, 5.0f, 1e-3f);
}

AIC_TEST(OptimTest, AdamWStepMath) {
    auto p = std::make_shared<Tensor>(Tensor::ones(Shape({2}), DType::Float32, Device::cpu()));
    Tensor g = Tensor::full(Shape({2}), 0.1f, DType::Float32, Device::cpu());
    p->set_grad(g);

    AdamWOptions opts;
    opts.lr = 0.01f;
    opts.weight_decay = 0.1f;
    opts.beta1 = 0.9f;
    opts.beta2 = 0.999f;
    opts.eps = 1e-8f;

    AdamW opt({p}, opts);
    opt.step();

    const auto* state = opt.state(p.get());
    ASSERT_TRUE(state != nullptr);
    ASSERT_EQ(state->step, 1);

    // Initial param = 1.0. After decoupled weight decay (1 - 0.01 * 0.1) = 0.999.
    // m1 = (1 - 0.9) * 0.1 = 0.01. m_hat = 0.01 / (1 - 0.9) = 0.1.
    // v1 = (1 - 0.999) * 0.01 = 0.00001. v_hat = 0.00001 / (1 - 0.999) = 0.01.
    // sqrt(v_hat) = 0.1.
    // update = 0.01 * (0.1 / 0.1) = 0.01.
    // final param = 0.999 - 0.01 = 0.989.
    const float* ptr = p->data_ptr<float>();
    ASSERT_NEAR(ptr[0], 0.989f, 1e-4f);
    ASSERT_NEAR(ptr[1], 0.989f, 1e-4f);
}

AIC_TEST(OptimTest, ConvergenceLinearModel) {
    // Train a simple linear regression model y = X W + b on CPU and GPU
    auto run_convergence = [](Device dev) {
        int64_t N = 32;
        int64_t in_features = 4;
        int64_t out_features = 1;

        // Ground truth W_true = [1.5, -2.0, 3.0, -0.5]^T, b_true = 0.5
        Tensor X = Tensor::randn(Shape({N, in_features}), 0.0f, 1.0f, DType::Float32, dev);
        Tensor W_true = Tensor::empty(Shape({in_features, out_features}), DType::Float32, dev);
        {
            Tensor W_cpu = Tensor::empty(Shape({in_features, out_features}), DType::Float32, Device::cpu());
            float* w = W_cpu.data_ptr<float>();
            w[0] = 1.5f; w[1] = -2.0f; w[2] = 3.0f; w[3] = -0.5f;
            W_true = W_cpu.to(dev);
        }
        Tensor Y_true = matmul(X, W_true) + 0.5f;

        // Learnable linear layer
        auto linear = std::make_shared<nn::Linear>(in_features, out_features, true);
        linear->to(dev);

        AdamWOptions opts;
        opts.lr = 0.1f;
        opts.weight_decay = 0.0f;
        AdamW opt(linear->parameters(), opts);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int step = 0; step < 80; ++step) {
            opt.zero_grad();
            Tensor pred = linear->forward(X);
            Tensor loss = mse_loss(pred, Y_true, "mean");

            float l = loss.cpu().item<float>();
            if (step == 0) initial_loss = l;
            final_loss = l;

            loss.backward();
            opt.step();
        }

        std::cout << "\n[" << dev.str() << " AdamW Linear Convergence] Initial loss: "
                  << initial_loss << " -> Final loss: " << final_loss << std::endl;
        ASSERT_TRUE(initial_loss > 0.5f);
        ASSERT_TRUE(final_loss < 0.05f);
    };

    run_convergence(Device::cpu());
    if (is_cuda_available()) {
        run_convergence(Device::cuda(0));
    }
}
