// tests/unit/test_moe.cpp
// Unit tests for Mixture-of-Experts (MoE) Sparse Routing & Dynamic Load-Balancing Engine.
#include "core/device.hpp"
#include "loss/cross_entropy.hpp"
#include "models/moe_transformer_lm.hpp"
#include "nn/moe.hpp"
#include "ops/ops.hpp"
#include "optim/adamw.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <numeric>
#include <vector>

using namespace aic;
using namespace aic::nn;
using namespace aic::models;
using namespace aic::optim;

AIC_TEST(MoETest, RouterTopKMath) {
    int64_t d_model = 64;
    int64_t num_experts = 8;
    int64_t top_k = 2;
    float aux_loss_coef = 0.01f;

    MoERouter router(d_model, num_experts, top_k, aux_loss_coef);

    ASSERT_EQ(router.d_model(), d_model);
    ASSERT_EQ(router.num_experts(), num_experts);
    ASSERT_EQ(router.top_k(), top_k);

    int64_t B = 2;
    int64_t S = 10;
    int64_t total_tokens = B * S;
    Tensor x = Tensor::randn(Shape({B, S, d_model}));

    RouterOutput r_out = router.route(x);

    ASSERT_TRUE(r_out.expert_indices.shape() == Shape({total_tokens, top_k}));
    ASSERT_TRUE(r_out.routing_weights.shape() == Shape({total_tokens, top_k}));
    ASSERT_EQ(static_cast<int64_t>(r_out.expert_counts.size()), num_experts);

    const int64_t* idx_ptr = r_out.expert_indices.data_ptr<int64_t>();
    const float* w_ptr = r_out.routing_weights.data_ptr<float>();

    int64_t total_dispatched = 0;
    for (int64_t e : r_out.expert_counts) {
        total_dispatched += e;
    }
    ASSERT_EQ(total_dispatched, total_tokens * top_k);

    for (int64_t i = 0; i < total_tokens; ++i) {
        int64_t e0 = idx_ptr[i * top_k + 0];
        int64_t e1 = idx_ptr[i * top_k + 1];

        // Experts must be in valid range and distinct
        ASSERT_TRUE(e0 >= 0 && e0 < num_experts);
        ASSERT_TRUE(e1 >= 0 && e1 < num_experts);
        ASSERT_TRUE(e0 != e1);

        // Routing weights must sum to 1.0f
        float w0 = w_ptr[i * top_k + 0];
        float w1 = w_ptr[i * top_k + 1];
        ASSERT_TRUE(w0 > 0.0f && w0 <= 1.0f);
        ASSERT_TRUE(w1 > 0.0f && w1 <= 1.0f);
        ASSERT_NEAR(w0 + w1, 1.0f, 1e-4f);
    }
}

AIC_TEST(MoETest, RouterAuxiliaryLoss) {
    int64_t d_model = 32;
    int64_t num_experts = 4;
    int64_t top_k = 2;
    float aux_loss_coef = 0.05f;

    MoERouter router(d_model, num_experts, top_k, aux_loss_coef);
    Tensor x = Tensor::randn(Shape({1, 16, d_model}));

    RouterOutput r_out = router.route(x);
    Tensor aux_loss = r_out.aux_loss;

    ASSERT_EQ(aux_loss.numel(), 1);
    float val = aux_loss.item<float>();
    ASSERT_TRUE(val > 0.0f);

    // Verify autograd graph connectivity: backprop into gate linear weights
    aux_loss.backward();
    ASSERT_TRUE(router.gate()->weight()->grad().defined());

    const float* g_ptr = router.gate()->weight()->grad().data_ptr<float>();
    int64_t num_g = router.gate()->weight()->grad().numel();
    float max_abs_g = 0.0f;
    for (int64_t i = 0; i < num_g; ++i) {
        float ag = std::abs(g_ptr[i]);
        if (ag > max_abs_g) max_abs_g = ag;
    }
    ASSERT_TRUE(max_abs_g > 1e-8f);

    // Verify that aux_loss_coef = 0 returns zero loss
    MoERouter router_zero(d_model, num_experts, top_k, 0.0f);
    RouterOutput r_out_zero = router_zero.route(x);
    ASSERT_NEAR(r_out_zero.aux_loss.item<float>(), 0.0f, 1e-7f);
}

AIC_TEST(MoETest, MoELayerSparseDispatch) {
    int64_t d_model = 32;
    int64_t d_hidden = 64;
    int64_t num_experts = 4;
    int64_t top_k = 1;

    MoELayer moe(d_model, d_hidden, num_experts, top_k, 0.01f);
    ASSERT_EQ(static_cast<int64_t>(moe.experts().size()), num_experts);

    Tensor x = Tensor::randn(Shape({2, 4, d_model}));
    Tensor y = moe.forward(x);

    ASSERT_TRUE(y.shape() == Shape({2, 4, d_model}));
    ASSERT_EQ(static_cast<int64_t>(moe.last_expert_counts().size()), num_experts);

    int64_t total_dispatched = 0;
    for (int64_t c : moe.last_expert_counts()) {
        total_dispatched += c;
    }
    ASSERT_EQ(total_dispatched, 8); // 8 tokens total, top-1

    double imbalance = moe.load_imbalance_ratio();
    ASSERT_TRUE(imbalance >= 1.0);
}

AIC_TEST(MoETest, MoELayerAutogradEndToEnd) {
    int64_t d_model = 16;
    int64_t d_hidden = 32;
    int64_t num_experts = 4;
    int64_t top_k = 2;

    MoELayer moe(d_model, d_hidden, num_experts, top_k, 0.02f);

    Tensor x = Tensor::randn(Shape({1, 4, d_model}));
    x.set_requires_grad(true);

    Tensor y = moe.forward(x);
    Tensor loss = add(sum(mul(y, y)), moe.last_aux_loss());
    loss.backward();

    // Verify input gradient
    ASSERT_TRUE(x.grad().defined());
    ASSERT_TRUE(x.grad().numel() == x.numel());

    // Verify router gate gradient
    ASSERT_TRUE(moe.router()->gate()->weight()->grad().defined());

    // Verify active experts have defined gradients
    for (int64_t e = 0; e < num_experts; ++e) {
        if (moe.last_expert_counts()[static_cast<size_t>(e)] > 0) {
            auto exp = moe.experts()[static_cast<size_t>(e)];
            ASSERT_TRUE(exp->gate_proj()->weight()->grad().defined());
            ASSERT_TRUE(exp->up_proj()->weight()->grad().defined());
            ASSERT_TRUE(exp->down_proj()->weight()->grad().defined());
        }
    }
}

AIC_TEST(MoETest, MoETransformerBlockExecution) {
    int64_t d_model = 32;
    int64_t n_heads = 2;
    int64_t d_hidden = 64;
    int64_t num_experts = 4;
    int64_t top_k = 2;

    MoETransformerBlock block(d_model, n_heads, d_hidden, num_experts, top_k);

    Tensor x = Tensor::randn(Shape({2, 6, d_model}));
    Tensor y = block.forward(x);

    ASSERT_TRUE(y.shape() == Shape({2, 6, d_model}));
    ASSERT_TRUE(block.last_aux_loss().numel() == 1);

    // Test with KVCache forward_with_cache
    KVCache cache;
    Tensor single_step = Tensor::randn(Shape({1, 1, d_model}));
    Tensor step_out = block.forward_with_cache(single_step, &cache, 0);
    ASSERT_TRUE(step_out.shape() == Shape({1, 1, d_model}));
}

AIC_TEST(MoETest, MoETransformerLMTrainingConvergence) {
    MoETransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.aux_loss_coef = 0.01f;
    cfg.max_seq_len = 64;

    MoETransformerLM model(cfg);
    AdamWOptions opt_opts;
    opt_opts.lr = 0.02f;
    AdamW opt(model.parameters(), opt_opts);

    // Repetitive synthetic sequence
    int64_t S = 8;
    Tensor input_ids = Tensor::empty(Shape({1, S}), DType::Int64);
    Tensor targets = Tensor::empty(Shape({1, S}), DType::Int64);

    int64_t* in_ptr = input_ids.data_ptr<int64_t>();
    int64_t* tgt_ptr = targets.data_ptr<int64_t>();

    for (int64_t s = 0; s < S; ++s) {
        in_ptr[s] = (s * 3) % cfg.vocab_size;
        tgt_ptr[s] = ((s + 1) * 3) % cfg.vocab_size;
    }

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int step = 0; step < 15; ++step) {
        opt.zero_grad();
        Tensor logits = model.forward(input_ids);
        Tensor main_loss = cross_entropy(logits, targets);
        Tensor aux_loss = model.total_aux_loss();
        Tensor total_loss = add(main_loss, aux_loss);

        float l_val = total_loss.item<float>();
        if (step == 0) initial_loss = l_val;
        final_loss = l_val;

        total_loss.backward();
        opt.step();
    }

    // Verify loss convergence
    ASSERT_TRUE(final_loss < initial_loss);
}

#if AIC_WITH_CUDA
AIC_TEST(MoETest, CudaExecutionAndParity) {
    if (!is_cuda_available()) return;

    Device dev = Device::cuda(0);
    int64_t d_model = 32;
    int64_t d_hidden = 64;
    int64_t num_experts = 4;
    int64_t top_k = 2;

    MoELayer moe(d_model, d_hidden, num_experts, top_k, 0.01f, dev);
    ASSERT_TRUE(moe.router()->gate()->weight()->is_cuda());

    Tensor x = Tensor::randn(Shape({2, 8, d_model}), 0.0f, 1.0f, DType::Float32, dev);
    Tensor y = moe.forward(x);

    ASSERT_TRUE(y.is_cuda());
    ASSERT_TRUE(y.shape() == Shape({2, 8, d_model}));

    // Backprop on CUDA
    Tensor loss = add(mean(mul(y, y)), moe.last_aux_loss());
    loss.backward();

    ASSERT_TRUE(moe.router()->gate()->weight()->grad().defined());
    ASSERT_TRUE(moe.router()->gate()->weight()->grad().is_cuda());

    // Test MoETransformerLM on CUDA
    MoETransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.max_seq_len = 64;

    MoETransformerLM cuda_model(cfg, dev);
    ASSERT_TRUE(cuda_model.device().is_cuda());

    Tensor input_ids = Tensor::empty(Shape({1, 4}), DType::Int64, dev);
    int64_t host_ids[4] = {1, 5, 12, 20};
    cudaMemcpy(input_ids.data_ptr<int64_t>(), host_ids, sizeof(host_ids), cudaMemcpyHostToDevice);

    Tensor logits = cuda_model.forward(input_ids);
    ASSERT_TRUE(logits.is_cuda());
    ASSERT_TRUE(logits.shape() == Shape({1, 4, 32}));

    // Test autoregressive generation on CUDA
    std::vector<int64_t> prompt = {1, 5};
    std::vector<int64_t> gen = cuda_model.generate(prompt, 3, 0.8f, 10, 0.9f, -1, 123);
    ASSERT_EQ(gen.size(), static_cast<size_t>(5));
    ASSERT_EQ(gen[0], 1);
    ASSERT_EQ(gen[1], 5);
}
#endif
