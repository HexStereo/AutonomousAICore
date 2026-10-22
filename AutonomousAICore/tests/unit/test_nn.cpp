// tests/unit/test_nn.cpp
// Verification and numerical tests for Neural Network Layers & Transformer Architecture.
#include "nn/nn.hpp"
#include "ops/ops.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <iostream>

using namespace aic;
using namespace aic::nn;

AIC_TEST(NNModule, ParameterRegistrationAndHierarchy) {
    auto linear = std::make_shared<Linear>(16, 32, /*has_bias=*/true);
    ASSERT_EQ(linear->parameters().size(), 2); // weight and bias

    auto named = linear->named_parameters();
    ASSERT_EQ(named.size(), 2);
    bool has_weight = (named[0].first == "weight" || named[1].first == "weight");
    bool has_bias = (named[0].first == "bias" || named[1].first == "bias");
    ASSERT_TRUE(has_weight && has_bias);

    // Test zero_grad
    for (auto& p : linear->parameters()) {
        p->set_grad(Tensor::ones(p->shape()));
        ASSERT_TRUE(p->grad().defined());
    }
    linear->zero_grad();
    for (auto& p : linear->parameters()) {
        ASSERT_FALSE(p->grad().defined());
    }
}

AIC_TEST(NNLinear, ForwardAndBackward) {
    int64_t in_features = 4;
    int64_t out_features = 3;
    Linear linear(in_features, out_features, /*has_bias=*/true);

    Tensor x = Tensor::arange(1.0f, 9.0f).reshape(Shape({2, 4})); // B=2, in=4
    x.set_requires_grad(true);

    Tensor y = linear.forward(x);
    ASSERT_EQ(y.shape(), Shape({2, 3}));

    Tensor loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().defined());
    ASSERT_TRUE(linear.weight()->grad().defined());
    ASSERT_TRUE(linear.bias()->grad().defined());

    ASSERT_EQ(x.grad().shape(), x.shape());
    ASSERT_EQ(linear.weight()->grad().shape(), linear.weight()->shape());
    ASSERT_EQ(linear.bias()->grad().shape(), linear.bias()->shape());
}

AIC_TEST(NNRMSNorm, ForwardAndUnitVariance) {
    int64_t dim = 64;
    RMSNorm norm(dim, 1e-6f);

    Tensor x = Tensor::randn(Shape({4, 16, dim}), 0.0f, 2.5f);
    x.set_requires_grad(true);

    Tensor y = norm.forward(x);
    ASSERT_EQ(y.shape(), Shape({4, 16, dim}));

    // Verify RMS is approximately 1.0 along normalized dimension
    Tensor y2 = mul(y, y);
    Tensor mean_y2 = mean(y2, -1);
    const float* p = mean_y2.data_ptr<float>();
    for (int64_t i = 0; i < mean_y2.numel(); ++i) {
        ASSERT_NEAR(p[i], 1.0f, 1e-3f);
    }

    // Verify gradient flow through RMSNorm
    Tensor loss = sum(y);
    loss.backward();
    ASSERT_TRUE(x.grad().defined());
    ASSERT_TRUE(norm.weight()->grad().defined());
}

AIC_TEST(NNRoPE, RotationalSymmetryAndOffset) {
    int64_t head_dim = 16;
    int64_t seq_len = 8;
    RotaryPositionEmbedding rope(head_dim, 128);

    Tensor x = Tensor::randn(Shape({1, 1, seq_len, head_dim}), 0.0f, 1.0f);
    Tensor x_rot = rope.forward(x);

    // RoPE preserves vector Euclidean norm for each position
    Tensor norm_orig = sum(mul(x, x), -1);
    Tensor norm_rot = sum(mul(x_rot, x_rot), -1);

    const float* p_orig = norm_orig.data_ptr<float>();
    const float* p_rot = norm_rot.data_ptr<float>();
    for (int64_t i = 0; i < norm_orig.numel(); ++i) {
        ASSERT_NEAR(p_orig[i], p_rot[i], 1e-4f);
    }

    // Verify offset start_pos: slice from full sequence matches single step with start_pos
    int64_t step_pos = 3;
    Tensor single_x = x.slice(2, step_pos, step_pos + 1);
    Tensor single_rot = rope.forward(single_x, step_pos);

    Tensor slice_full_rot = x_rot.slice(2, step_pos, step_pos + 1);
    const float* p1 = single_rot.data_ptr<float>();
    const float* p2 = slice_full_rot.data_ptr<float>();
    for (int64_t i = 0; i < head_dim; ++i) {
        ASSERT_NEAR(p1[i], p2[i], 1e-5f);
    }
}

AIC_TEST(NNSwiGLU, ForwardAndBackward) {
    int64_t d_model = 32;
    int64_t d_hidden = 64;
    SwiGLU ffn(d_model, d_hidden);

    Tensor x = Tensor::randn(Shape({2, 8, d_model}), 0.0f, 1.0f);
    x.set_requires_grad(true);

    Tensor y = ffn.forward(x);
    ASSERT_EQ(y.shape(), Shape({2, 8, d_model}));

    Tensor loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().defined());
    ASSERT_TRUE(ffn.gate_proj()->weight()->grad().defined());
    ASSERT_TRUE(ffn.up_proj()->weight()->grad().defined());
    ASSERT_TRUE(ffn.down_proj()->weight()->grad().defined());
}

AIC_TEST(NNAttention, CausalMaskAndKVCacheParity) {
    int64_t d_model = 32;
    int64_t n_heads = 4;
    CausalSelfAttention attn(d_model, n_heads);

    Tensor x = Tensor::randn(Shape({1, 4, d_model}), 0.0f, 1.0f);

    // 1. Process full sequence at once without cache
    Tensor out_full = attn.forward(x);
    ASSERT_EQ(out_full.shape(), Shape({1, 4, d_model}));

    // 2. Process token-by-token using KV cache (autoregressive mode)
    KVCache cache;
    std::vector<Tensor> step_outputs;

    for (int64_t pos = 0; pos < 4; ++pos) {
        Tensor token_x = x.slice(1, pos, pos + 1);
        Tensor token_out = attn.forward_with_cache(token_x, &cache, pos);
        step_outputs.push_back(token_out);
    }

    Tensor out_cached = cat(step_outputs, 1);
    ASSERT_EQ(out_cached.shape(), out_full.shape());

    // 3. Verify exact mathematical parity between parallel and cached generation
    const float* p_full = out_full.data_ptr<float>();
    const float* p_cached = out_cached.data_ptr<float>();
    for (int64_t i = 0; i < out_full.numel(); ++i) {
        ASSERT_NEAR(p_full[i], p_cached[i], 1e-4f);
    }
}

AIC_TEST(NNTransformerBlock, EndToEndCPUAndGPU) {
    int64_t d_model = 32;
    int64_t n_heads = 4;
    int64_t d_hidden = 64;

    auto block = std::make_shared<TransformerBlock>(d_model, n_heads, d_hidden);

    // CPU Forward and Backward
    Tensor x_cpu = Tensor::randn(Shape({2, 6, d_model}), 0.0f, 1.0f);
    x_cpu.set_requires_grad(true);

    Tensor y_cpu = block->forward(x_cpu);
    ASSERT_EQ(y_cpu.shape(), Shape({2, 6, d_model}));

    Tensor loss_cpu = sum(y_cpu);
    loss_cpu.backward();
    ASSERT_TRUE(x_cpu.grad().defined());

    // GPU Execution on Blackwell RTX 5090 if available
    if (get_cuda_device_count() > 0) {
        Device gpu = Device::cuda(0);
        block->zero_grad();
        block->to(gpu);

        Tensor x_gpu = Tensor::randn(Shape({2, 6, d_model}), 0.0f, 1.0f, DType::Float32, gpu);
        x_gpu.set_requires_grad(true);

        Tensor y_gpu = block->forward(x_gpu);
        ASSERT_EQ(y_gpu.shape(), Shape({2, 6, d_model}));
        ASSERT_TRUE(y_gpu.is_cuda());

        Tensor loss_gpu = sum(y_gpu);
        loss_gpu.backward();

        ASSERT_TRUE(x_gpu.grad().defined());
        ASSERT_TRUE(x_gpu.grad().is_cuda());
    }
}
