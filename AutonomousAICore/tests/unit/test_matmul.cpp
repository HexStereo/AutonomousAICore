// tests/unit/test_matmul.cpp
// Unit and numerical parity tests for CPU SIMD GEMM, CUDA Tiled/TensorCore GEMM, and MatMul Autograd.
#include "ops/matmul.hpp"
#include "ops/ops.hpp"
#include "autograd/autograd.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <chrono>
#include <iostream>
#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace aic;

AIC_TEST(MatMulCPU, Basic2D) {
    // A: [2, 3], B: [3, 2]
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 1], [2, 3]]
    // C[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    Tensor a = Tensor::arange(1.0f, 7.0f).reshape(Shape({2, 3}));
    std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
    Tensor b = Tensor::from_blob(b_data.data(), Shape({3, 2})).clone();

    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), Shape({2, 2}));

    const float* p = c.data_ptr<float>();
    ASSERT_NEAR(p[0], 31.0f, 1e-5f);
    ASSERT_NEAR(p[1], 19.0f, 1e-5f);
    ASSERT_NEAR(p[2], 85.0f, 1e-5f);
    ASSERT_NEAR(p[3], 55.0f, 1e-5f);
}

AIC_TEST(MatMulCPU, ArbitrarySizesSimd) {
    // Test size that exercises SIMD loop unrolling and edge remainders: 67 x 79 x 83
    int64_t M = 67, K = 79, N = 83;
    Tensor a = Tensor::full(Shape({M, K}), 1.0f);
    Tensor b = Tensor::full(Shape({K, N}), 2.0f);

    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), Shape({M, N}));

    const float* p = c.data_ptr<float>();
    float expected = static_cast<float>(K) * 2.0f; // 79 * 2 = 158
    for (int64_t i = 0; i < M * N; ++i) {
        ASSERT_NEAR(p[i], expected, 1e-4f);
    }
}

AIC_TEST(MatMulCPU, BatchedMatMul) {
    // A: [2, 3, 4], B: [2, 4, 5] -> C: [2, 3, 5]
    Tensor a = Tensor::full(Shape({2, 3, 4}), 1.5f);
    Tensor b = Tensor::full(Shape({2, 4, 5}), 2.0f);

    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), Shape({2, 3, 5}));

    const float* p = c.data_ptr<float>();
    float expected = 4.0f * (1.5f * 2.0f); // 12.0
    for (int64_t i = 0; i < c.numel(); ++i) {
        ASSERT_NEAR(p[i], expected, 1e-4f);
    }
}

AIC_TEST(MatMulGPU, ParityAndTensorCores) {
    if (get_cuda_device_count() == 0) {
        return;
    }

    Device gpu = Device::cuda(0);
    int64_t M = 128, K = 128, N = 128;

    Tensor a_cpu = Tensor::arange(0.01f, 0.01f + static_cast<float>(M * K) * 0.01f, 0.01f).reshape(Shape({M, K}));
    Tensor b_cpu = Tensor::arange(0.02f, 0.02f + static_cast<float>(K * N) * 0.02f, 0.02f).reshape(Shape({K, N}));

    Tensor c_cpu = matmul(a_cpu, b_cpu);

    Tensor a_gpu = a_cpu.to(gpu);
    Tensor b_gpu = b_cpu.to(gpu);

    // 1. Shared Memory Tiled Kernel
    Tensor c_gpu_tiled = gemm(a_gpu, b_gpu, 1.0f, 0.0f, Tensor(), GemmAlgorithm::CudaTiled).cpu();
    for (int64_t i = 0; i < M * N; ++i) {
        ASSERT_NEAR(c_cpu.data_ptr<float>()[i], c_gpu_tiled.data_ptr<float>()[i], 1e-2f);
    }

    // 2. 2D Block + Register Tiled Kernel
    Tensor c_gpu_reg = gemm(a_gpu, b_gpu, 1.0f, 0.0f, Tensor(), GemmAlgorithm::CudaRegTiled).cpu();
    for (int64_t i = 0; i < M * N; ++i) {
        ASSERT_NEAR(c_cpu.data_ptr<float>()[i], c_gpu_reg.data_ptr<float>()[i], 1e-2f);
    }

    // 3. Hardware Tensor Cores (WMMA)
    Tensor c_gpu_wmma = gemm(a_gpu, b_gpu, 1.0f, 0.0f, Tensor(), GemmAlgorithm::CudaTensorCore).cpu();
    for (int64_t i = 0; i < M * N; ++i) {
        float expected = c_cpu.data_ptr<float>()[i];
        float actual = c_gpu_wmma.data_ptr<float>()[i];
        float rel_err = std::abs(expected - actual) / (std::abs(expected) + 1.0f);
        ASSERT_TRUE(rel_err < 1e-2f);
    }

    // 4. Batched CUDA MatMul
    Tensor ba_cpu = Tensor::full(Shape({4, 16, 32}), 1.0f);
    Tensor bb_cpu = Tensor::full(Shape({4, 32, 24}), 0.5f);
    Tensor bc_cpu = matmul(ba_cpu, bb_cpu);

    Tensor bc_gpu = matmul(ba_cpu.to(gpu), bb_cpu.to(gpu)).cpu();
    for (int64_t i = 0; i < bc_cpu.numel(); ++i) {
        ASSERT_NEAR(bc_cpu.data_ptr<float>()[i], bc_gpu.data_ptr<float>()[i], 1e-3f);
    }
}

AIC_TEST(MatMulAutograd, GradientVerification) {
    // C = A x B
    // loss = sum(C)
    // dL/dA = 1 x B^T
    // dL/dB = A^T x 1
    Tensor a = Tensor::arange(1.0f, 7.0f).reshape(Shape({2, 3}));
    Tensor b = Tensor::arange(1.0f, 13.0f).reshape(Shape({3, 4}));
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), Shape({2, 4}));

    Tensor loss = sum(c);
    loss.backward();

    ASSERT_TRUE(a.grad().defined());
    ASSERT_TRUE(b.grad().defined());
    ASSERT_EQ(a.grad().shape(), Shape({2, 3}));
    ASSERT_EQ(b.grad().shape(), Shape({3, 4}));

    // Analytical gradient check using finite differences gradcheck
    auto mm_fn = [](const std::vector<Tensor>& inps) -> Tensor {
        return sum(matmul(inps[0], inps[1]));
    };

    Tensor x = (0.1f * Tensor::arange(1.0f, 7.0f)).reshape(Shape({2, 3}));
    Tensor y = (0.1f * Tensor::arange(1.0f, 13.0f)).reshape(Shape({3, 4}));
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    ASSERT_TRUE(autograd::gradcheck(mm_fn, {x, y}, 1e-3f, 2e-3f, 2e-3f));
}

AIC_TEST(MatMulBenchmark, FlopsCalculation) {
    int64_t M = 1024, K = 1024, N = 1024;
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    Tensor a = Tensor::full(Shape({M, K}), 1.0f);
    Tensor b = Tensor::full(Shape({K, N}), 1.0f);

    // Warmup CPU
    Tensor c_cpu = matmul(a, b);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 3; ++iter) {
        c_cpu = matmul(a, b);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_sec = std::chrono::duration<double>(t1 - t0).count() / 3.0;
    double cpu_gflops = (flops / cpu_sec) / 1e9;

    std::cout << "\n[CPU GEMM Benchmark " << M << "x" << N << "x" << K << "]: "
              << (cpu_sec * 1000.0) << " ms (" << cpu_gflops << " GFLOP/s)\n";

#if AIC_WITH_CUDA
    if (get_cuda_device_count() > 0) {
        Device gpu = Device::cuda(0);
        Tensor a_gpu = a.to(gpu);
        Tensor b_gpu = b.to(gpu);

        // Warmup GPU
        Tensor c_gpu = matmul(a_gpu, b_gpu);
        cudaDeviceSynchronize();

        auto gt0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 10; ++iter) {
            c_gpu = gemm(a_gpu, b_gpu, 1.0f, 0.0f, Tensor(), GemmAlgorithm::CudaRegTiled);
        }
        cudaDeviceSynchronize();
        auto gt1 = std::chrono::high_resolution_clock::now();
        double gpu_sec = std::chrono::duration<double>(gt1 - gt0).count() / 10.0;
        double gpu_tflops = (flops / gpu_sec) / 1e12;

        std::cout << "[GPU GEMM Benchmark (RTX 5090 Blackwell " << M << "x" << N << "x" << K << ")]: "
                  << (gpu_sec * 1000.0) << " ms (" << gpu_tflops << " TFLOP/s)\n";
    }
#endif
}
