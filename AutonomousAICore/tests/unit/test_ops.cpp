// tests/unit/test_ops.cpp
// Unit and numerical parity tests for tensor mathematical operations and CUDA acceleration.
#include "ops/ops.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <vector>

using namespace aic;

AIC_TEST(OpsCPU, BasicArithmetic) {
    Tensor a = Tensor::full(Shape({2, 3}), 4.0f);
    Tensor b = Tensor::full(Shape({2, 3}), 2.0f);

    Tensor c_add = a + b;
    Tensor c_sub = a - b;
    Tensor c_mul = a * b;
    Tensor c_div = a / b;

    ASSERT_EQ(c_add.shape(), Shape({2, 3}));
    const float* p_add = c_add.data_ptr<float>();
    const float* p_sub = c_sub.data_ptr<float>();
    const float* p_mul = c_mul.data_ptr<float>();
    const float* p_div = c_div.data_ptr<float>();

    for (int i = 0; i < 6; ++i) {
        ASSERT_NEAR(p_add[i], 6.0f, 1e-6f);
        ASSERT_NEAR(p_sub[i], 2.0f, 1e-6f);
        ASSERT_NEAR(p_mul[i], 8.0f, 1e-6f);
        ASSERT_NEAR(p_div[i], 2.0f, 1e-6f);
    }
}

AIC_TEST(OpsCPU, Broadcasting) {
    // a: [2, 3], b: [1, 3] -> out: [2, 3]
    Tensor a = Tensor::arange(0.0f, 6.0f).reshape(Shape({2, 3}));
    Tensor b = Tensor::full(Shape({1, 3}), 10.0f);

    Tensor c = a + b;
    ASSERT_EQ(c.shape(), Shape({2, 3}));
    const float* p = c.data_ptr<float>();
    ASSERT_NEAR(p[0], 10.0f, 1e-6f); // 0 + 10
    ASSERT_NEAR(p[1], 11.0f, 1e-6f); // 1 + 10
    ASSERT_NEAR(p[2], 12.0f, 1e-6f); // 2 + 10
    ASSERT_NEAR(p[3], 13.0f, 1e-6f); // 3 + 10
    ASSERT_NEAR(p[4], 14.0f, 1e-6f); // 4 + 10
    ASSERT_NEAR(p[5], 15.0f, 1e-6f); // 5 + 10
}

AIC_TEST(OpsCPU, ScalarArithmetic) {
    Tensor a = Tensor::full(Shape({2, 2}), 5.0f);
    Tensor add_s = a + 3.0f;
    Tensor sub_s = a - 2.0f;
    Tensor mul_s = 2.0f * a;
    Tensor div_s = a / 2.0f;

    const float* p_add = add_s.data_ptr<float>();
    const float* p_sub = sub_s.data_ptr<float>();
    const float* p_mul = mul_s.data_ptr<float>();
    const float* p_div = div_s.data_ptr<float>();

    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(p_add[i], 8.0f, 1e-6f);
        ASSERT_NEAR(p_sub[i], 3.0f, 1e-6f);
        ASSERT_NEAR(p_mul[i], 10.0f, 1e-6f);
        ASSERT_NEAR(p_div[i], 2.5f, 1e-6f);
    }
}

AIC_TEST(OpsCPU, UnaryActivations) {
    Tensor x = Tensor::arange(-2.0f, 3.0f, 1.0f); // [-2, -1, 0, 1, 2]
    Tensor s = sigmoid(x);
    Tensor t = tanh(x);
    Tensor g = gelu(x);
    Tensor si = silu(x);

    const float* ps = s.data_ptr<float>();
    const float* pt = t.data_ptr<float>();
    const float* pg = g.data_ptr<float>();
    const float* psi = si.data_ptr<float>();

    // Test zero points
    ASSERT_NEAR(ps[2], 0.5f, 1e-5f);
    ASSERT_NEAR(pt[2], 0.0f, 1e-5f);
    ASSERT_NEAR(pg[2], 0.0f, 1e-5f);
    ASSERT_NEAR(psi[2], 0.0f, 1e-5f);

    // Test monotonicity / bounds
    ASSERT_TRUE(ps[0] < ps[1] && ps[1] < ps[2] && ps[2] < ps[3] && ps[3] < ps[4]);
    ASSERT_NEAR(exp(Tensor::zeros(Shape({1}))).item<float>(), 1.0f, 1e-6f);
}

AIC_TEST(OpsCPU, FullReductions) {
    Tensor a = Tensor::arange(1.0f, 6.0f); // [1, 2, 3, 4, 5] -> sum=15, mean=3, max=5, min=1

    Tensor s = sum(a);
    Tensor m = mean(a);
    Tensor mx = max(a);
    Tensor mn = min(a);

    ASSERT_NEAR(s.item<float>(), 15.0f, 1e-6f);
    ASSERT_NEAR(m.item<float>(), 3.0f, 1e-6f);
    ASSERT_NEAR(mx.item<float>(), 5.0f, 1e-6f);
    ASSERT_NEAR(mn.item<float>(), 1.0f, 1e-6f);
}

AIC_TEST(OpsCPU, DimensionalReductions) {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Tensor a = Tensor::arange(1.0f, 7.0f).reshape(Shape({2, 3}));

    // Sum along dim 0 -> [5, 7, 9], shape [3] (keepdim=false)
    Tensor sum_d0 = sum(a, 0, false);
    ASSERT_EQ(sum_d0.shape(), Shape({3}));
    ASSERT_NEAR(sum_d0.data_ptr<float>()[0], 5.0f, 1e-6f);
    ASSERT_NEAR(sum_d0.data_ptr<float>()[1], 7.0f, 1e-6f);
    ASSERT_NEAR(sum_d0.data_ptr<float>()[2], 9.0f, 1e-6f);

    // Sum along dim 1 -> [6, 15], shape [2, 1] (keepdim=true)
    Tensor sum_d1 = sum(a, 1, true);
    ASSERT_EQ(sum_d1.shape(), Shape({2, 1}));
    ASSERT_NEAR(sum_d1.data_ptr<float>()[0], 6.0f, 1e-6f);
    ASSERT_NEAR(sum_d1.data_ptr<float>()[1], 15.0f, 1e-6f);

    // Max along dim 1
    Tensor max_d1 = max(a, 1, false);
    ASSERT_EQ(max_d1.shape(), Shape({2}));
    ASSERT_NEAR(max_d1.data_ptr<float>()[0], 3.0f, 1e-6f);
    ASSERT_NEAR(max_d1.data_ptr<float>()[1], 6.0f, 1e-6f);
}

AIC_TEST(OpsCPU, InPlaceOperations) {
    Tensor a = Tensor::full(Shape({2, 2}), 10.0f);
    Tensor b = Tensor::full(Shape({2, 2}), 3.0f);

    a += b;
    ASSERT_NEAR(a.data_ptr<float>()[0], 13.0f, 1e-6f);

    a -= b;
    ASSERT_NEAR(a.data_ptr<float>()[0], 10.0f, 1e-6f);

    a *= b;
    ASSERT_NEAR(a.data_ptr<float>()[0], 30.0f, 1e-6f);

    a /= b;
    ASSERT_NEAR(a.data_ptr<float>()[0], 10.0f, 1e-6f);
}

AIC_TEST(OpsGPU, ParityWithCPU) {
    if (get_cuda_device_count() == 0) {
        return; // Skip on CPU-only machines
    }

    Device gpu = Device::cuda(0);

    // 1. Elementwise arithmetic parity
    Tensor a_cpu = Tensor::arange(1.0f, 101.0f).reshape(Shape({10, 10}));
    Tensor b_cpu = Tensor::arange(101.0f, 201.0f).reshape(Shape({10, 10}));

    Tensor a_gpu = a_cpu.to(gpu);
    Tensor b_gpu = b_cpu.to(gpu);

    Tensor c_cpu_add = a_cpu + b_cpu;
    Tensor c_gpu_add = (a_gpu + b_gpu).cpu();
    for (int i = 0; i < 100; ++i) {
        ASSERT_NEAR(c_cpu_add.data_ptr<float>()[i], c_gpu_add.data_ptr<float>()[i], 1e-5f);
    }

    Tensor c_cpu_mul = a_cpu * b_cpu;
    Tensor c_gpu_mul = (a_gpu * b_gpu).cpu();
    for (int i = 0; i < 100; ++i) {
        ASSERT_NEAR(c_cpu_mul.data_ptr<float>()[i], c_gpu_mul.data_ptr<float>()[i], 1e-3f);
    }

    // 2. Activations parity (GELU, SiLU, Tanh, Sigmoid)
    Tensor x_cpu = Tensor::arange(-5.0f, 5.0f, 0.1f);
    Tensor x_gpu = x_cpu.to(gpu);

    Tensor gelu_cpu = gelu(x_cpu);
    Tensor gelu_gpu = gelu(x_gpu).cpu();
    for (int64_t i = 0; i < x_cpu.numel(); ++i) {
        ASSERT_NEAR(gelu_cpu.data_ptr<float>()[i], gelu_gpu.data_ptr<float>()[i], 1e-4f);
    }

    Tensor silu_cpu = silu(x_cpu);
    Tensor silu_gpu = silu(x_gpu).cpu();
    for (int64_t i = 0; i < x_cpu.numel(); ++i) {
        ASSERT_NEAR(silu_cpu.data_ptr<float>()[i], silu_gpu.data_ptr<float>()[i], 1e-4f);
    }

    // 3. Full Reductions parity
    Tensor sum_cpu = sum(x_cpu);
    Tensor sum_gpu = sum(x_gpu).cpu();
    ASSERT_NEAR(sum_cpu.item<float>(), sum_gpu.item<float>(), 1e-3f);

    Tensor mean_cpu = mean(x_cpu);
    Tensor mean_gpu = mean(x_gpu).cpu();
    ASSERT_NEAR(mean_cpu.item<float>(), mean_gpu.item<float>(), 1e-4f);

    Tensor max_cpu = max(x_cpu);
    Tensor max_gpu = max(x_gpu).cpu();
    ASSERT_NEAR(max_cpu.item<float>(), max_gpu.item<float>(), 1e-4f);

    Tensor min_cpu = min(x_cpu);
    Tensor min_gpu = min(x_gpu).cpu();
    ASSERT_NEAR(min_cpu.item<float>(), min_gpu.item<float>(), 1e-4f);

    // 4. Dimensional Reductions parity
    Tensor mat_cpu = Tensor::arange(1.0f, 25.0f).reshape(Shape({4, 6}));
    Tensor mat_gpu = mat_cpu.to(gpu);

    Tensor red_d0_cpu = sum(mat_cpu, 0, false);
    Tensor red_d0_gpu = sum(mat_gpu, 0, false).cpu();
    for (int i = 0; i < 6; ++i) {
        ASSERT_NEAR(red_d0_cpu.data_ptr<float>()[i], red_d0_gpu.data_ptr<float>()[i], 1e-4f);
    }

    Tensor red_d1_cpu = mean(mat_cpu, 1, true);
    Tensor red_d1_gpu = mean(mat_gpu, 1, true).cpu();
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(red_d1_cpu.data_ptr<float>()[i], red_d1_gpu.data_ptr<float>()[i], 1e-4f);
    }

    // 5. In-place on GPU
    Tensor g_val = Tensor::full(Shape({4}), 10.0f, DType::Float32, gpu);
    Tensor g_add = Tensor::full(Shape({4}), 5.0f, DType::Float32, gpu);
    g_val += g_add;
    Tensor g_cpu = g_val.cpu();
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(g_cpu.data_ptr<float>()[i], 15.0f, 1e-5f);
    }
}
