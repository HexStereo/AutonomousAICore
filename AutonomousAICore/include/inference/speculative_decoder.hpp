// include/inference/speculative_decoder.hpp
// Speculative Decoding & Verification Engine for Accelerated LLM Inference.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "tensor/tensor.hpp"

namespace aic::inference {

struct SpeculativeConfig {
    int draft_lookahead = 4;        // K: Number of candidate tokens proposed by draft model per step
    float temperature = 0.8f;       // Sampling temperature
    int top_k = 20;                 // Top-K filtering
    float top_p = 0.9f;             // Top-P nucleus filtering
    int64_t eos_token_id = 0;       // End of sequence token
};

struct SpeculativeMetrics {
    int64_t total_tokens_generated = 0;
    int64_t draft_tokens_proposed = 0;
    int64_t draft_tokens_accepted = 0;
    int64_t target_forward_steps = 0;
    double total_time_ms = 0.0;

    float acceptance_rate() const noexcept {
        return draft_tokens_proposed > 0
            ? static_cast<float>(draft_tokens_accepted) / static_cast<float>(draft_tokens_proposed)
            : 0.0f;
    }

    float tokens_per_target_step() const noexcept {
        return target_forward_steps > 0
            ? static_cast<float>(total_tokens_generated) / static_cast<float>(target_forward_steps)
            : 0.0f;
    }
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(
        std::shared_ptr<models::TransformerLM> target_model,
        std::shared_ptr<models::TransformerLM> draft_model,
        data::Tokenizer tokenizer,
        SpeculativeConfig config = {}
    );

    // Executes speculative autoregressive decoding
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt_tokens,
        int max_new_tokens
    );

    // High-level text generation helper
    std::string generate_text(
        const std::string& prompt_text,
        int max_new_tokens
    );

    const SpeculativeConfig& config() const noexcept { return config_; }
    SpeculativeConfig& config() noexcept { return config_; }

    const SpeculativeMetrics& metrics() const noexcept { return metrics_; }
    void reset_metrics() noexcept { metrics_ = SpeculativeMetrics{}; }

    // Utility: Samples token from distribution with temperature and top-k/top-p
    static int64_t sample_token(
        const float* logits,
        int64_t vocab_size,
        float temperature,
        int top_k,
        float top_p,
        std::mt19937& rng
    );

    // Utility: Rejection sampling verification between draft and target probabilities:
    // Returns pair: <is_accepted, recovery_token>
    static std::pair<bool, int64_t> verify_and_sample(
        const float* target_probs,
        const float* draft_probs,
        int64_t candidate_token,
        int64_t vocab_size,
        std::mt19937& rng
    );

private:
    std::shared_ptr<models::TransformerLM> target_model_;
    std::shared_ptr<models::TransformerLM> draft_model_;
    data::Tokenizer tokenizer_;
    SpeculativeConfig config_;
    SpeculativeMetrics metrics_;
    std::mt19937 rng_;
};

}  // namespace aic::inference
