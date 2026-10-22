// tests/unit/test_dpo.cpp
// Unit and Integration tests for Direct Preference Optimization (DPO) and Self-Play Alignment.
#include "alignment/alignment.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "models/transformer_lm.hpp"
#include "ops/ops.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using namespace aic;
using namespace aic::alignment;
using namespace aic::models;

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

AIC_TEST(DPOLossTest, ExactAnalyticalMath) {
    // Test Case 1: Identical policies -> u = 0 -> loss = ln(2) ~ 0.693147
    {
        Tensor pi_w = make_tensor({-2.0f}, Shape({1}));
        Tensor pi_l = make_tensor({-5.0f}, Shape({1}));
        Tensor ref_w = make_tensor({-2.0f}, Shape({1}));
        Tensor ref_l = make_tensor({-5.0f}, Shape({1}));

        auto [loss, metrics] = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, /*beta=*/0.1f);
        float expected_loss = std::log(2.0f);

        ASSERT_NEAR(loss.item<float>(), expected_loss, 1e-5f);
        ASSERT_NEAR(metrics.loss, expected_loss, 1e-5f);
        ASSERT_NEAR(metrics.chosen_reward, 0.0f, 1e-5f);
        ASSERT_NEAR(metrics.rejected_reward, 0.0f, 1e-5f);
        ASSERT_NEAR(metrics.reward_margin, 0.0f, 1e-5f);
        ASSERT_EQ(metrics.accuracy, 0.0f);
    }

    // Test Case 2: Positive margin:
    // pi_w = -1.0, ref_w = -2.0 -> r_w = 0.1 * 1.0 = 0.1
    // pi_l = -4.0, ref_l = -3.0 -> r_l = 0.1 * -1.0 = -0.1
    // u = 0.1 - (-0.1) = 0.2
    // loss = ln(1 + e^(-0.2)) ~ 0.5981388
    {
        Tensor pi_w = make_tensor({-1.0f}, Shape({1}));
        Tensor pi_l = make_tensor({-4.0f}, Shape({1}));
        Tensor ref_w = make_tensor({-2.0f}, Shape({1}));
        Tensor ref_l = make_tensor({-3.0f}, Shape({1}));

        auto [loss, metrics] = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, /*beta=*/0.1f);
        float expected_loss = std::log1p(std::exp(-0.2f));

        ASSERT_NEAR(loss.item<float>(), expected_loss, 1e-5f);
        ASSERT_NEAR(metrics.chosen_reward, 0.1f, 1e-5f);
        ASSERT_NEAR(metrics.rejected_reward, -0.1f, 1e-5f);
        ASSERT_NEAR(metrics.reward_margin, 0.2f, 1e-5f);
        ASSERT_EQ(metrics.accuracy, 1.0f);
    }
}

AIC_TEST(DPOLossTest, AutogradGradientDirections) {
    Tensor pi_w = make_tensor({-2.0f}, Shape({1}));
    Tensor pi_l = make_tensor({-2.0f}, Shape({1}));
    Tensor ref_w = make_tensor({-2.0f}, Shape({1}));
    Tensor ref_l = make_tensor({-2.0f}, Shape({1}));

    pi_w.set_requires_grad(true);
    pi_l.set_requires_grad(true);

    auto [loss, metrics] = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, /*beta=*/0.1f);
    loss.backward();

    // With u = 0, sigma(-u) = 0.5
    // dL/d(pi_w) = -beta * 0.5 = -0.05
    // dL/d(pi_l) = +beta * 0.5 = +0.05
    float g_w = pi_w.grad().item<float>();
    float g_l = pi_l.grad().item<float>();

    ASSERT_NEAR(g_w, -0.05f, 1e-5f);
    ASSERT_NEAR(g_l, 0.05f, 1e-5f);
}

AIC_TEST(DPOTrainerTest, PolicyOptimizationConvergence) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 32;
    cfg.max_seq_len = 16;

    auto policy = std::make_shared<TransformerLM>(cfg, Device::cpu());
    auto ref = std::make_shared<TransformerLM>(cfg, Device::cpu());

    optim::AdamWOptions opt_opts;
    opt_opts.lr = 0.05f;
    auto optimizer = std::make_shared<optim::AdamW>(policy->parameters(), opt_opts);

    DPOConfig dpo_cfg;
    dpo_cfg.beta = 0.5f;
    DPOTrainer trainer(policy, ref, optimizer, dpo_cfg);

    // Chosen and Rejected sequences of length 4
    std::vector<int64_t> c_ids = {1, 2, 3, 4};
    std::vector<int64_t> c_lbl = {-100, 2, 3, 4};
    std::vector<int64_t> r_ids = {1, 2, 5, 6};
    std::vector<int64_t> r_lbl = {-100, 2, 5, 6};

    Tensor chosen_in = Tensor::empty(Shape({1, 4}), DType::Int64, Device::cpu());
    Tensor chosen_lb = Tensor::empty(Shape({1, 4}), DType::Int64, Device::cpu());
    Tensor rejected_in = Tensor::empty(Shape({1, 4}), DType::Int64, Device::cpu());
    Tensor rejected_lb = Tensor::empty(Shape({1, 4}), DType::Int64, Device::cpu());

    std::memcpy(chosen_in.data_ptr<int64_t>(), c_ids.data(), 4 * sizeof(int64_t));
    std::memcpy(chosen_lb.data_ptr<int64_t>(), c_lbl.data(), 4 * sizeof(int64_t));
    std::memcpy(rejected_in.data_ptr<int64_t>(), r_ids.data(), 4 * sizeof(int64_t));
    std::memcpy(rejected_lb.data_ptr<int64_t>(), r_lbl.data(), 4 * sizeof(int64_t));

    DPOMetrics initial_metrics = trainer.step(chosen_in, chosen_lb, rejected_in, rejected_lb);
    DPOMetrics final_metrics;

    for (int step = 0; step < 8; ++step) {
        final_metrics = trainer.step(chosen_in, chosen_lb, rejected_in, rejected_lb);
    }

    // After 8 optimization steps, reward margin should increase and loss decrease
    ASSERT_TRUE(final_metrics.reward_margin > initial_metrics.reward_margin);
    ASSERT_TRUE(final_metrics.loss < initial_metrics.loss);
}

AIC_TEST(SelfPlayTest, BatchTensorPreparationAndMasking) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 32;

    auto model = std::make_shared<TransformerLM>(cfg, Device::cpu());
    data::Tokenizer tokenizer = data::Tokenizer::create_default();

    SelfPlayEngine engine(model, tokenizer);

    SelfPlayPreferencePair pair;
    pair.prompt = "Solve 2+2:";
    pair.chosen_response = " Answer is 4.";
    pair.rejected_response = " Answer is 5.";
    pair.chosen_score = 0.95f;
    pair.rejected_score = 0.10f;

    BatchTensors bt = engine.prepare_batch_tensors(pair, Device::cpu());

    ASSERT_EQ(bt.chosen_input_ids.rank(), 2);
    ASSERT_EQ(bt.chosen_labels.rank(), 2);
    ASSERT_EQ(bt.rejected_input_ids.rank(), 2);
    ASSERT_EQ(bt.rejected_labels.rank(), 2);

    // Prompt tokens must be masked with -100
    auto prompt_tokens = tokenizer.encode(pair.prompt);
    const int64_t* c_lbl_ptr = bt.chosen_labels.data_ptr<int64_t>();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        ASSERT_EQ(c_lbl_ptr[i], -100);
    }
}
