// tests/unit/test_autograd.cpp
// Comprehensive tests for reverse-mode autograd, DAG traversal, version tracking, and gradcheck.
#include "autograd/autograd.hpp"
#include "autograd/functions.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>

using namespace aic;
using namespace aic::autograd;

AIC_TEST(Autograd, BasicPolynomial) {
    // y = x^2 + 3*x - 5
    // dy/dx = 2*x + 3
    Tensor x = Tensor::full(Shape({1}), 4.0f);
    x.set_requires_grad(true);

    Tensor y = x * x + 3.0f * x - 5.0f;
    ASSERT_NEAR(y.item<float>(), 23.0f, 1e-5f);

    y.backward();

    ASSERT_TRUE(x.grad().defined());
    ASSERT_NEAR(x.grad().item<float>(), 11.0f, 1e-5f); // 2*4 + 3 = 11
}

AIC_TEST(Autograd, DiamondGraphGradientAccumulation) {
    // c = a * b + a
    // dc/da = b + 1
    // dc/db = a
    Tensor a = Tensor::full(Shape({2, 2}), 3.0f);
    Tensor b = Tensor::full(Shape({2, 2}), 5.0f);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor c = a * b + a;
    Tensor loss = sum(c);
    loss.backward();

    ASSERT_TRUE(a.grad().defined());
    ASSERT_TRUE(b.grad().defined());

    const float* ga = a.grad().data_ptr<float>();
    const float* gb = b.grad().data_ptr<float>();

    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(ga[i], 6.0f, 1e-5f); // 5 + 1 = 6
        ASSERT_NEAR(gb[i], 3.0f, 1e-5f); // 3
    }
}

AIC_TEST(Autograd, BroadcastingReduction) {
    // a: [2, 3], b: [3] -> c: [2, 3] = a + b
    // loss = sum(c)
    // grad_a = ones([2, 3])
    // grad_b = sum over dim 0 = [2, 2, 2]
    Tensor a = Tensor::full(Shape({2, 3}), 1.0f);
    Tensor b = Tensor::full(Shape({3}), 2.0f);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor c = a + b;
    Tensor loss = sum(c);
    loss.backward();

    ASSERT_EQ(a.grad().shape(), Shape({2, 3}));
    ASSERT_EQ(b.grad().shape(), Shape({3}));

    for (int i = 0; i < 6; ++i) {
        ASSERT_NEAR(a.grad().data_ptr<float>()[i], 1.0f, 1e-5f);
    }
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(b.grad().data_ptr<float>()[i], 2.0f, 1e-5f);
    }
}

AIC_TEST(Autograd, ActivationsBackward) {
    // Test tanh and sigmoid backward
    Tensor x = Tensor::full(Shape({1}), 0.0f);
    x.set_requires_grad(true);

    Tensor y = tanh(x);
    y.backward();
    // d/dx tanh(0) = 1 - 0^2 = 1.0
    ASSERT_NEAR(x.grad().item<float>(), 1.0f, 1e-5f);

    x.zero_grad();
    Tensor s = sigmoid(x);
    s.backward();
    // d/dx sigmoid(0) = 0.5 * (1 - 0.5) = 0.25
    ASSERT_NEAR(x.grad().item<float>(), 0.25f, 1e-5f);
}

AIC_TEST(Autograd, ShapeTransformationsBackward) {
    // x: [2, 3] -> transpose(0, 1) -> [3, 2]
    Tensor x = Tensor::arange(1.0f, 7.0f).reshape(Shape({2, 3}));
    x.set_requires_grad(true);

    Tensor xt = x.transpose(0, 1);
    ASSERT_EQ(xt.shape(), Shape({3, 2}));

    Tensor loss = sum(xt * 2.0f);
    loss.backward();

    ASSERT_EQ(x.grad().shape(), Shape({2, 3}));
    for (int i = 0; i < 6; ++i) {
        ASSERT_NEAR(x.grad().data_ptr<float>()[i], 2.0f, 1e-5f);
    }
}

AIC_TEST(Autograd, InPlaceVersionCheck) {
    Tensor a = Tensor::full(Shape({2, 2}), 2.0f);
    Tensor b = Tensor::full(Shape({2, 2}), 3.0f);
    b.set_requires_grad(true);

    // 1. Modifying leaf that requires grad in-place throws immediately
    Tensor delta = Tensor::full(Shape({2, 2}), 1.0f);
    ASSERT_THROW(b += delta, aic::Error);

    // 2. Modifying saved variable version throws during backward
    Tensor x = Tensor::full(Shape({2, 2}), 2.0f);
    Tensor y = Tensor::full(Shape({2, 2}), 3.0f);
    y.set_requires_grad(true);

    Tensor z = x * y;
    x.bump_version(); // simulate in-place modification on saved variable x
    ASSERT_THROW(z.backward(), aic::Error);
}

AIC_TEST(Autograd, NumericalGradcheck) {
    // 1. Polynomial check
    auto poly_fn = [](const std::vector<Tensor>& inps) -> Tensor {
        return sum(inps[0] * inps[0] + 3.0f * inps[0]);
    };
    Tensor x1 = Tensor::arange(1.0f, 5.0f).reshape(Shape({2, 2}));
    x1.set_requires_grad(true);
    ASSERT_TRUE(gradcheck(poly_fn, {x1}, 1e-4f, 1e-3f, 1e-3f));

    // 2. Multi-input product check with broadcasting
    auto prod_fn = [](const std::vector<Tensor>& inps) -> Tensor {
        return sum(inps[0] * inps[1]);
    };
    Tensor a = 0.2f * Tensor::arange(1.0f, 7.0f).reshape(Shape({2, 3}));
    Tensor b = 0.2f * Tensor::arange(1.0f, 4.0f).reshape(Shape({1, 3}));
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    ASSERT_TRUE(gradcheck(prod_fn, {a, b}, 1e-3f, 1e-3f, 1e-3f));

    // 3. Nonlinear activations (GELU + SiLU)
    auto act_fn = [](const std::vector<Tensor>& inps) -> Tensor {
        return sum(gelu(inps[0]) + silu(inps[0]));
    };
    Tensor x2 = Tensor::arange(-2.0f, 2.0f, 0.8f);
    x2.set_requires_grad(true);
    ASSERT_TRUE(gradcheck(act_fn, {x2}, 1e-4f, 1e-3f, 1e-3f));

    // 4. Mean dimensional reduction
    auto mean_fn = [](const std::vector<Tensor>& inps) -> Tensor {
        return sum(mean(inps[0], 1, true));
    };
    Tensor x3 = 0.2f * Tensor::arange(1.0f, 9.0f).reshape(Shape({2, 4}));
    x3.set_requires_grad(true);
    ASSERT_TRUE(gradcheck(mean_fn, {x3}, 1e-3f, 2e-3f, 2e-3f));
}

AIC_TEST(Autograd, GPUExecution) {
    if (get_cuda_device_count() == 0) {
        return;
    }

    Device gpu = Device::cuda(0);
    Tensor x = Tensor::full(Shape({3, 3}), 2.0f, DType::Float32, gpu);
    x.set_requires_grad(true);

    Tensor y = x * x + 5.0f * x;
    Tensor loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().defined());
    ASSERT_EQ(x.grad().device(), gpu);

    Tensor grad_cpu = x.grad().cpu();
    const float* p = grad_cpu.data_ptr<float>();
    for (int i = 0; i < 9; ++i) {
        // dy/dx = 2*2 + 5 = 9
        ASSERT_NEAR(p[i], 9.0f, 1e-4f);
    }
}
