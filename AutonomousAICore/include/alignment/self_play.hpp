// include/alignment/self_play.hpp
// Autonomous Reasoning Self-Play Engine: Generates DPO preference pairs via MCTS & SelfVerifier.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "alignment/dpo_loss.hpp"
#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "reasoning/mcts_engine.hpp"
#include "reasoning/verifier.hpp"
#include "tensor/tensor.hpp"

namespace aic::alignment {

struct SelfPlayPreferencePair {
    std::string prompt;
    std::string chosen_response;
    std::string rejected_response;
    float chosen_score = 0.0f;
    float rejected_score = 0.0f;
    bool has_contradiction_rejected = false;
};

struct BatchTensors {
    Tensor chosen_input_ids;
    Tensor chosen_labels;
    Tensor rejected_input_ids;
    Tensor rejected_labels;
};

struct SelfPlayConfig {
    int num_rollouts_per_prompt = 3;
    float min_score_margin = 0.1f;
    int max_seq_len = 32;
};

class SelfPlayEngine {
public:
    SelfPlayEngine(
        std::shared_ptr<models::TransformerLM> model,
        data::Tokenizer tokenizer,
        reasoning::ReasoningConfig reasoning_cfg = {},
        reasoning::SelfVerifier verifier = {},
        SelfPlayConfig config = {}
    );

    // Generates a preference pair from a reasoning prompt by evaluating candidate rollouts
    bool generate_pair(
        const std::string& prompt,
        SelfPlayPreferencePair& out_pair
    );

    // Converts a preference pair into model input_ids and labels (masking prompt tokens with -100)
    BatchTensors prepare_batch_tensors(
        const SelfPlayPreferencePair& pair,
        Device device = Device::cpu()
    );

    const SelfPlayConfig& config() const noexcept { return config_; }
    SelfPlayConfig& config() noexcept { return config_; }

    reasoning::SelfVerifier& verifier() noexcept { return verifier_; }
    const reasoning::SelfVerifier& verifier() const noexcept { return verifier_; }

private:
    std::shared_ptr<models::TransformerLM> model_;
    data::Tokenizer tokenizer_;
    reasoning::ReasoningConfig reasoning_cfg_;
    reasoning::SelfVerifier verifier_;
    SelfPlayConfig config_;
};

}  // namespace aic::alignment
