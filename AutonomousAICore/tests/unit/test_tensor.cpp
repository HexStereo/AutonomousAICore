// tests/unit/test_tensor.cpp
#include "tensor/tensor.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

using namespace aic;

AIC_TEST(TensorTest, FactoryFunctions) {
    Tensor z = Tensor::zeros(Shape({2, 3}), DType::Float32);
    ASSERT_EQ(z.numel(), 6);
    ASSERT_TRUE(z.is_contiguous());
    const float* z_data = z.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ(z_data[i], 0.0f);
    }

    Tensor o = Tensor::ones(Shape({4}), DType::Float32);
    ASSERT_EQ(o.numel(), 4);
    const float* o_data = o.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(o_data[i], 1.0f);
    }

    Tensor a = Tensor::arange(0.0f, 5.0f, 1.0f, DType::Float32);
    ASSERT_EQ(a.numel(), 5);
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(a.data_ptr<float>()[i], static_cast<float>(i));
    }
}

AIC_TEST(TensorTest, ReshapeAndView) {
    Tensor t = Tensor::arange(0.0f, 12.0f, 1.0f);
    Tensor v = t.view(Shape({3, 4}));
    ASSERT_EQ(v.rank(), 2);
    ASSERT_EQ(v.dim(0), 3);
    ASSERT_EQ(v.dim(1), 4);
    // Shares storage
    ASSERT_EQ(t.raw_data(), v.raw_data());

    Tensor r = t.reshape(Shape({2, 6}));
    ASSERT_EQ(r.dim(0), 2);
    ASSERT_EQ(r.dim(1), 6);
}

AIC_TEST(TensorTest, TransposeAndPermute) {
    Tensor t = Tensor::arange(0.0f, 6.0f, 1.0f).view(Shape({2, 3}));
    // Matrix:
    // [0, 1, 2]
    // [3, 4, 5]
    Tensor tr = t.transpose(0, 1);
    ASSERT_EQ(tr.dim(0), 3);
    ASSERT_EQ(tr.dim(1), 2);
    ASSERT_FALSE(tr.is_contiguous());

    // When making contiguous, data must be reordered
    Tensor contig = tr.contiguous();
    ASSERT_TRUE(contig.is_contiguous());
    const float* p = contig.data_ptr<float>();
    // Transposed matrix:
    // [0, 3]
    // [1, 4]
    // [2, 5]
    ASSERT_EQ(p[0], 0.0f);
    ASSERT_EQ(p[1], 3.0f);
    ASSERT_EQ(p[2], 1.0f);
    ASSERT_EQ(p[3], 4.0f);
    ASSERT_EQ(p[4], 2.0f);
    ASSERT_EQ(p[5], 5.0f);
}

AIC_TEST(TensorTest, SqueezeAndUnsqueeze) {
    Tensor t = Tensor::zeros(Shape({1, 3, 1, 4}));
    Tensor s = t.squeeze();
    ASSERT_EQ(s.shape(), Shape({3, 4}));

    Tensor u = s.unsqueeze(0);
    ASSERT_EQ(u.shape(), Shape({1, 3, 4}));
}

AIC_TEST(TensorTest, ItemScalar) {
    Tensor s = Tensor::full(Shape({1}), 42.5f);
    ASSERT_NEAR(s.item<float>(), 42.5f, 1e-5);

    Tensor multi = Tensor::zeros(Shape({2, 2}));
    ASSERT_THROW(multi.item<float>(), Error);
}

AIC_TEST(TensorTest, TypeCasting) {
    Tensor f32 = Tensor::arange(0.0f, 4.0f, 1.0f, DType::Float32);
    Tensor f16 = f32.to(DType::Float16);
    ASSERT_EQ(f16.dtype(), DType::Float16);
    ASSERT_EQ(f16.nbytes(), 4 * sizeof(uint16_t));

    Tensor back = f16.to(DType::Float32);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(back.data_ptr<float>()[i], static_cast<float>(i), 1e-3);
    }
}

AIC_TEST(TensorTest, CudaAllocationAndTransfer) {
#if AIC_WITH_CUDA
    if (!is_cuda_available()) return;

    Device dev = Device::cuda(0);
    Tensor t_gpu = Tensor::zeros(Shape({8, 8}), DType::Float32, dev);
    ASSERT_TRUE(t_gpu.is_cuda());
    ASSERT_EQ(t_gpu.device().index, 0);
    ASSERT_EQ(t_gpu.numel(), 64);

    Tensor t_cpu = t_gpu.cpu();
    ASSERT_TRUE(t_cpu.is_cpu());
    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(t_cpu.data_ptr<float>()[i], 0.0f);
    }

    // Host -> Device -> Host roundtrip with known numbers
    Tensor src = Tensor::arange(1.0f, 17.0f, 1.0f, DType::Float32, Device::cpu());
    Tensor dst_gpu = src.to(dev);
    ASSERT_TRUE(dst_gpu.is_cuda());

    Tensor roundtrip = dst_gpu.to(Device::cpu());
    ASSERT_TRUE(roundtrip.is_cpu());
    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(roundtrip.data_ptr<float>()[i], static_cast<float>(i + 1));
    }
#endif
}

