// include/reasoning/mcts_engine.hpp
// Monte Carlo Tree Search (MCTS) & Tree-of-Thoughts Autonomous Reasoning Engine.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "reasoning/thought_node.hpp"
#include "reasoning/verifier.hpp"

namespace aic::reasoning {

struct ReasoningConfig {
    int max_search_iterations = 15;
    int beam_width = 3;           // Number of alternative thoughts expanded per node
    int max_depth = 5;            // Maximum sequential reasoning steps
    int max_tokens_per_step = 12; // Maximum tokens per thought step
    float c_puct = 1.414f;        // Exploration constant in PUCT formula
    float temperature = 0.8f;     // Sampling temperature for thought generation
    int top_k = 20;               // Top-k filtering for thought sampling
    float top_p = 0.9f;           // Nucleus sampling threshold
};

struct ReasoningTrace {
    std::string prompt;
    std::vector<ThoughtStep> chain;
    std::string final_conclusion;
    float cumulative_confidence = 0.0f;
    int nodes_explored = 0;
    int pruned_branches = 0;
    double search_time_ms = 0.0;

    std::string to_string() const;
};

class ReasoningSearchEngine {
public:
    ReasoningSearchEngine(
        std::shared_ptr<models::TransformerLM> model,
        data::Tokenizer tokenizer,
        ReasoningConfig config = {},
        SelfVerifier verifier = {}
    );

    // Executes Tree-of-Thoughts MCTS search to discover the optimal reasoning trajectory
    ReasoningTrace reason(const std::string& prompt_text);

    const ReasoningConfig& config() const noexcept { return config_; }
    ReasoningConfig& config() noexcept { return config_; }

    const SelfVerifier& verifier() const noexcept { return verifier_; }
    SelfVerifier& verifier() noexcept { return verifier_; }

private:
    std::shared_ptr<models::TransformerLM> model_;
    data::Tokenizer tokenizer_;
    ReasoningConfig config_;
    SelfVerifier verifier_;
};

}  // namespace aic::reasoning
