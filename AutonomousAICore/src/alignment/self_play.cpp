// src/alignment/self_play.cpp
// Autonomous Reasoning Self-Play Engine implementation.
#include "alignment/self_play.hpp"
#include "core/error.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace aic::alignment {

SelfPlayEngine::SelfPlayEngine(
    std::shared_ptr<models::TransformerLM> model,
    data::Tokenizer tokenizer,
    reasoning::ReasoningConfig reasoning_cfg,
    reasoning::SelfVerifier verifier,
    SelfPlayConfig config
) : model_(std::move(model)),
    tokenizer_(std::move(tokenizer)),
    reasoning_cfg_(reasoning_cfg),
    verifier_(std::move(verifier)),
    config_(config) {
    AIC_CHECK(model_ != nullptr, InvalidArgument, "model cannot be null");
    AIC_CHECK(config_.num_rollouts_per_prompt >= 2, InvalidArgument, "num_rollouts_per_prompt must be >= 2");
}

bool SelfPlayEngine::generate_pair(
    const std::string& prompt,
    SelfPlayPreferencePair& out_pair
) {
    struct Candidate {
        std::string text;
        reasoning::VerificationResult verification;
    };

    std::vector<Candidate> candidates;

    // 1. Generate primary trace using Tree-of-Thoughts Reasoning Engine
    reasoning::ReasoningSearchEngine engine(model_, tokenizer_, reasoning_cfg_, verifier_);
    reasoning::ReasoningTrace best_trace = engine.reason(prompt);

    std::string best_text = best_trace.to_string();
    auto v_best = verifier_.verify_text(best_text);
    candidates.push_back({best_text, v_best});

    // 2. Generate alternative rollouts with varying temperature and sampling
    for (int r = 1; r < config_.num_rollouts_per_prompt; ++r) {
        reasoning::ReasoningConfig alt_cfg = reasoning_cfg_;
        alt_cfg.temperature = reasoning_cfg_.temperature + 0.2f * static_cast<float>(r);
        alt_cfg.max_search_iterations = std::max(2, reasoning_cfg_.max_search_iterations / 2);

        reasoning::ReasoningSearchEngine alt_engine(model_, tokenizer_, alt_cfg, verifier_);
        reasoning::ReasoningTrace alt_trace = alt_engine.reason(prompt);
        std::string alt_text = alt_trace.to_string();
        auto v_alt = verifier_.verify_text(alt_text);
        candidates.push_back({alt_text, v_alt});
    }

    // Sort descending by verification score
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.verification.score > b.verification.score;
    });

    const auto& chosen = candidates.front();
    const auto& rejected = candidates.back();

    float margin = chosen.verification.score - rejected.verification.score;
    if (margin < config_.min_score_margin && !rejected.verification.is_contradiction) {
        // Not enough distinction to form an unambiguous preference pair
        return false;
    }

    out_pair.prompt = prompt;
    out_pair.chosen_response = chosen.text;
    out_pair.rejected_response = rejected.text;
    out_pair.chosen_score = chosen.verification.score;
    out_pair.rejected_score = rejected.verification.score;
    out_pair.has_contradiction_rejected = rejected.verification.is_contradiction;

    return true;
}

BatchTensors SelfPlayEngine::prepare_batch_tensors(
    const SelfPlayPreferencePair& pair,
    Device device
) {
    auto prompt_tokens = tokenizer_.encode(pair.prompt);
    auto chosen_tokens = tokenizer_.encode(pair.chosen_response);
    auto rejected_tokens = tokenizer_.encode(pair.rejected_response);

    // Build chosen sequence
    std::vector<int64_t> full_chosen = prompt_tokens;
    full_chosen.insert(full_chosen.end(), chosen_tokens.begin(), chosen_tokens.end());
    if (static_cast<int>(full_chosen.size()) > config_.max_seq_len) {
        full_chosen.resize(static_cast<size_t>(config_.max_seq_len));
    }

    // Build rejected sequence
    std::vector<int64_t> full_rejected = prompt_tokens;
    full_rejected.insert(full_rejected.end(), rejected_tokens.begin(), rejected_tokens.end());
    if (static_cast<int>(full_rejected.size()) > config_.max_seq_len) {
        full_rejected.resize(static_cast<size_t>(config_.max_seq_len));
    }

    size_t S_c = full_chosen.size();
    size_t S_r = full_rejected.size();

    // Prepare CPU vectors for chosen
    std::vector<int64_t> c_inputs = full_chosen;
    std::vector<int64_t> c_labels(S_c, -100);
    for (size_t i = 0; i < S_c; ++i) {
        if (i >= prompt_tokens.size()) {
            c_labels[i] = full_chosen[i];
        }
    }

    // Prepare CPU vectors for rejected
    std::vector<int64_t> r_inputs = full_rejected;
    std::vector<int64_t> r_labels(S_r, -100);
    for (size_t i = 0; i < S_r; ++i) {
        if (i >= prompt_tokens.size()) {
            r_labels[i] = full_rejected[i];
        }
    }

    BatchTensors bt;
    bt.chosen_input_ids = Tensor::empty(Shape({1, static_cast<int64_t>(S_c)}), DType::Int64, Device::cpu());
    bt.chosen_labels = Tensor::empty(Shape({1, static_cast<int64_t>(S_c)}), DType::Int64, Device::cpu());
    bt.rejected_input_ids = Tensor::empty(Shape({1, static_cast<int64_t>(S_r)}), DType::Int64, Device::cpu());
    bt.rejected_labels = Tensor::empty(Shape({1, static_cast<int64_t>(S_r)}), DType::Int64, Device::cpu());

    std::memcpy(bt.chosen_input_ids.data_ptr<int64_t>(), c_inputs.data(), S_c * sizeof(int64_t));
    std::memcpy(bt.chosen_labels.data_ptr<int64_t>(), c_labels.data(), S_c * sizeof(int64_t));
    std::memcpy(bt.rejected_input_ids.data_ptr<int64_t>(), r_inputs.data(), S_r * sizeof(int64_t));
    std::memcpy(bt.rejected_labels.data_ptr<int64_t>(), r_labels.data(), S_r * sizeof(int64_t));

    if (device.is_cuda()) {
        bt.chosen_input_ids = bt.chosen_input_ids.to(device);
        bt.chosen_labels = bt.chosen_labels.to(device);
        bt.rejected_input_ids = bt.rejected_input_ids.to(device);
        bt.rejected_labels = bt.rejected_labels.to(device);
    }

    return bt;
}

}  // namespace aic::alignment
