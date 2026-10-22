// tests/unit/test_speculative.cpp
// Unit and Integration tests for Speculative Decoding & Verification.
#include "inference/speculative_decoder.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "models/transformer_lm.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace aic;
using namespace aic::inference;
using namespace aic::models;

AIC_TEST(SpeculativeTest, VerifyAndSampleAcceptanceMath) {
    std::mt19937 rng(42);
    int64_t vocab_size = 4;

    // Target has higher probability for candidate token 0:
    // P_target = [0.6, 0.2, 0.1, 0.1]
    // P_draft  = [0.3, 0.3, 0.2, 0.2]
    // P_target(0) / P_draft(0) = 0.6 / 0.3 = 2.0 -> min(1, 2.0) = 1.0 -> ALWAYS ACCEPTED
    std::vector<float> p_target = {0.6f, 0.2f, 0.1f, 0.1f};
    std::vector<float> p_draft  = {0.3f, 0.3f, 0.2f, 0.2f};

    int accept_count = 0;
    int trials = 100;
    for (int i = 0; i < trials; ++i) {
        auto [accepted, tok] = SpeculativeDecoder::verify_and_sample(
            p_target.data(), p_draft.data(), /*candidate=*/0, vocab_size, rng
        );
        if (accepted && tok == 0) accept_count++;
    }

    ASSERT_EQ(accept_count, trials);
}

AIC_TEST(SpeculativeTest, VerifyAndSampleRejectionAndRecovery) {
    std::mt19937 rng(42);
    int64_t vocab_size = 4;

    // Candidate token 0 has 0 probability in Target:
    // P_target = [0.0, 0.5, 0.5, 0.0]
    // P_draft  = [0.8, 0.1, 0.1, 0.0]
    // P_target(0) / P_draft(0) = 0.0 -> ALWAYS REJECTED
    // Residual distribution will only have mass on token 1 and token 2.
    std::vector<float> p_target = {0.0f, 0.5f, 0.5f, 0.0f};
    std::vector<float> p_draft  = {0.8f, 0.1f, 0.1f, 0.0f};

    int reject_count = 0;
    for (int i = 0; i < 50; ++i) {
        auto [accepted, tok] = SpeculativeDecoder::verify_and_sample(
            p_target.data(), p_draft.data(), /*candidate=*/0, vocab_size, rng
        );
        if (!accepted) {
            reject_count++;
            // Recovery token must be either 1 or 2 (never 0 or 3)
            ASSERT_TRUE(tok == 1 || tok == 2);
        }
    }

    ASSERT_EQ(reject_count, 50);
}

AIC_TEST(SpeculativeTest, EndToEndIdenticalModelsFullAcceptance) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 32;
    cfg.max_seq_len = 32;

    auto target_model = std::make_shared<TransformerLM>(cfg, Device::cpu());
    // Draft model shares identical weights
    auto draft_model = target_model;

    data::Tokenizer tokenizer = data::Tokenizer::create_default();
    SpeculativeConfig s_cfg;
    s_cfg.draft_lookahead = 3;
    s_cfg.temperature = 0.01f; // Greediest sampling

    SpeculativeDecoder decoder(target_model, draft_model, tokenizer, s_cfg);

    std::vector<int64_t> prompt = {1, 2, 3};
    auto output = decoder.generate(prompt, 6);

    // Should have generated 6 new tokens -> total 9 tokens
    ASSERT_EQ(output.size(), 9);

    // Target steps should be significantly fewer than generated tokens (due to parallel verification)
    ASSERT_TRUE(decoder.metrics().target_forward_steps < decoder.metrics().total_tokens_generated);
}

AIC_TEST(SpeculativeTest, HeterogeneousDraftAndTargetDecoding) {
    TransformerLMConfig target_cfg;
    target_cfg.vocab_size = 64;
    target_cfg.hidden_dim = 32;
    target_cfg.num_layers = 2;
    target_cfg.num_heads = 4;
    target_cfg.intermediate_dim = 64;

    TransformerLMConfig draft_cfg = target_cfg;
    draft_cfg.num_layers = 1; // Smaller / faster draft model
    draft_cfg.hidden_dim = 32;

    Device dev = is_cuda_available() ? Device::cuda(0) : Device::cpu();

    auto target = std::make_shared<TransformerLM>(target_cfg, dev);
    auto draft = std::make_shared<TransformerLM>(draft_cfg, dev);

    data::Tokenizer tokenizer = data::Tokenizer::create_default();
    SpeculativeConfig s_cfg;
    s_cfg.draft_lookahead = 2;

    SpeculativeDecoder decoder(target, draft, tokenizer, s_cfg);

    std::vector<int64_t> prompt = {5, 10};
    auto out = decoder.generate(prompt, 8);

    ASSERT_EQ(out.size(), 10);
    ASSERT_EQ(decoder.metrics().total_tokens_generated, 8);
    for (int64_t tok : out) {
        ASSERT_TRUE(tok >= 0 && tok < target_cfg.vocab_size);
    }
}
