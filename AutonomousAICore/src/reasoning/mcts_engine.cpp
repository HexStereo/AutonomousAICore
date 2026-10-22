// src/reasoning/mcts_engine.cpp
// Tree-of-Thoughts & MCTS Search Engine implementation.
#include "reasoning/mcts_engine.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace aic::reasoning {

std::string ReasoningTrace::to_string() const {
    std::ostringstream os;
    os << "=======================================================\n"
       << "   AUTONOMOUS REASONING TRACE (Tree-of-Thoughts / MCTS)\n"
       << "=======================================================\n"
       << "Prompt: \"" << prompt << "\"\n"
       << "Search Stats:\n"
       << "  Nodes Explored:   " << nodes_explored << "\n"
       << "  Pruned Branches:  " << pruned_branches << "\n"
       << "  Search Latency:   " << std::fixed << std::setprecision(2) << search_time_ms << " ms\n"
       << "  Confidence Score: " << std::setprecision(3) << cumulative_confidence << "\n"
       << "-------------------------------------------------------\n"
       << "Reasoning Trajectory (" << chain.size() << " steps):\n";

    for (size_t i = 0; i < chain.size(); ++i) {
        os << "  [Step " << (i + 1) << " | Confidence: "
           << std::fixed << std::setprecision(2) << chain[i].score << "]: "
           << chain[i].text << "\n";
    }

    if (!final_conclusion.empty()) {
        os << "-------------------------------------------------------\n"
           << "Final Conclusion: " << final_conclusion << "\n";
    }
    os << "=======================================================\n";
    return os.str();
}

ReasoningSearchEngine::ReasoningSearchEngine(
    std::shared_ptr<models::TransformerLM> model,
    data::Tokenizer tokenizer,
    ReasoningConfig config,
    SelfVerifier verifier
) : model_(std::move(model)),
    tokenizer_(std::move(tokenizer)),
    config_(config),
    verifier_(std::move(verifier)) {}

ReasoningTrace ReasoningSearchEngine::reason(const std::string& prompt_text) {
    auto start_time = std::chrono::high_resolution_clock::now();

    ReasoningTrace trace;
    trace.prompt = prompt_text;

    std::vector<int64_t> prompt_tokens = tokenizer_.encode(prompt_text);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_token_id() >= 0 ? tokenizer_.bos_token_id() : 1);
    }

    // 1. Initialize root node with prompt
    ThoughtStep root_step;
    root_step.tokens = prompt_tokens;
    root_step.text = prompt_text;
    root_step.score = 1.0f;
    root_step.is_valid = true;

    auto root = std::make_shared<ThoughtNode>(std::weak_ptr<ThoughtNode>{}, root_step, /*depth=*/0);
    trace.nodes_explored = 1;
    trace.pruned_branches = 0;

    // 2. MCTS Search Iterations
    for (int iter = 0; iter < config_.max_search_iterations; ++iter) {
        // Selection: traverse down tree using PUCT
        auto curr = root;
        while (!curr->is_leaf() && curr->depth() < config_.max_depth && !curr->step().is_terminal) {
            auto next_child = curr->select_best_child(config_.c_puct);
            if (!next_child) break;
            curr = next_child;
        }

        // Terminal condition check
        if (curr->step().is_terminal || curr->depth() >= config_.max_depth) {
            curr->backpropagate(curr->step().score);
            continue;
        }

        // Expansion: generate beam_width distinct thoughts
        std::vector<int64_t> context = curr->accumulated_tokens();

        for (int b = 0; b < config_.beam_width; ++b) {
            uint64_t seed = static_cast<uint64_t>(iter * 1000 + b + 1);
            auto gen_tokens = model_->generate(
                context,
                config_.max_tokens_per_step,
                config_.temperature,
                config_.top_k,
                config_.top_p,
                /*eos_token_id=*/-1,
                seed
            );

            if (gen_tokens.size() > context.size()) {
                std::vector<int64_t> new_tokens(gen_tokens.begin() + context.size(), gen_tokens.end());
                std::string new_text = tokenizer_.decode(new_tokens);

                ThoughtStep cand_step;
                cand_step.tokens = std::move(new_tokens);
                cand_step.text = std::move(new_text);

                // Self-verification
                auto ver_res = verifier_.verify_step(prompt_tokens, context, cand_step);
                cand_step.score = ver_res.score;
                cand_step.is_valid = ver_res.is_valid;
                cand_step.is_terminal = ver_res.is_conclusive;

                if (!cand_step.is_valid) {
                    trace.pruned_branches++;
                }

                float prior = std::max(0.01f, ver_res.score);
                auto child = std::make_shared<ThoughtNode>(curr, std::move(cand_step), curr->depth() + 1, prior);
                curr->add_child(child);
                trace.nodes_explored++;
            }
        }

        // Evaluation & Backpropagation: propagate value of best child
        auto best_new = curr->select_best_child(config_.c_puct);
        if (best_new) {
            best_new->backpropagate(best_new->step().score);
        } else {
            curr->backpropagate(curr->step().score);
        }
    }

    // 3. Extract Best Reasoning Trajectory (Max Visits / Max Q)
    auto curr = root;
    float sum_conf = 0.0f;

    while (!curr->is_leaf()) {
        std::shared_ptr<ThoughtNode> best_next = nullptr;
        int max_visits = -1;
        float max_q = -1.0f;

        for (const auto& child : curr->children()) {
            if (!child || !child->step().is_valid) continue;
            // Primary criteria: visits; Secondary: Q-value
            if (child->visits() > max_visits || (child->visits() == max_visits && child->q_value() > max_q)) {
                max_visits = child->visits();
                max_q = child->q_value();
                best_next = child;
            }
        }

        if (!best_next) break;
        trace.chain.push_back(best_next->step());
        sum_conf += best_next->step().score;

        if (best_next->step().is_terminal) {
            break;
        }
        curr = best_next;
    }

    if (!trace.chain.empty()) {
        trace.cumulative_confidence = sum_conf / static_cast<float>(trace.chain.size());
        trace.final_conclusion = trace.chain.back().text;
    } else {
        trace.cumulative_confidence = 0.0f;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    trace.search_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return trace;
}

}  // namespace aic::reasoning
