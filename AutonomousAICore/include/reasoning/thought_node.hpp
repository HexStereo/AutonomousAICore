// include/reasoning/thought_node.hpp
// Tree-of-Thoughts & Monte Carlo Tree Search (MCTS) Node Representation.
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aic::reasoning {

struct ThoughtStep {
    std::vector<int64_t> tokens;
    std::string text;
    float log_prob = 0.0f;
    float score = 0.5f;       // Verifier score in [0.0, 1.0]
    bool is_terminal = false; // Indicates conclusion of reasoning
    bool is_valid = true;     // False if rejected/pruned by verifier
};

class ThoughtNode : public std::enable_shared_from_this<ThoughtNode> {
public:
    ThoughtNode(
        std::weak_ptr<ThoughtNode> parent = {},
        ThoughtStep step = {},
        int depth = 0,
        float prior_prob = 1.0f
    );

    // Tree navigation
    std::shared_ptr<ThoughtNode> parent() const noexcept { return parent_.lock(); }
    const std::vector<std::shared_ptr<ThoughtNode>>& children() const noexcept { return children_; }
    void add_child(std::shared_ptr<ThoughtNode> child);

    bool is_leaf() const noexcept { return children_.empty(); }
    bool is_root() const noexcept { return parent_.expired(); }
    int depth() const noexcept { return depth_; }

    const ThoughtStep& step() const noexcept { return step_; }
    ThoughtStep& step() noexcept { return step_; }

    // MCTS Statistics
    int visits() const noexcept { return visits_; }
    float total_value() const noexcept { return total_value_; }
    float prior_prob() const noexcept { return prior_prob_; }

    // Q-value (mean action value)
    float q_value() const noexcept {
        return visits_ > 0 ? (total_value_ / static_cast<float>(visits_)) : step_.score;
    }

    // PUCT (Predictor Upper Confidence bounds for Trees) formula:
    // PUCT = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    float compute_puct(float c_puct, int parent_visits) const;

    // Selects the child node that maximizes PUCT among valid children
    std::shared_ptr<ThoughtNode> select_best_child(float c_puct = 1.414f) const;

    // Backpropagates value update up to the root
    void backpropagate(float value);

    // Traverses ancestry from root to this node, collecting all tokens
    std::vector<int64_t> accumulated_tokens() const;

    // Traverses ancestry from root to this node, collecting the textual reasoning chain
    std::string full_reasoning_trace() const;

private:
    std::weak_ptr<ThoughtNode> parent_;
    std::vector<std::shared_ptr<ThoughtNode>> children_;
    ThoughtStep step_;
    int depth_ = 0;
    float prior_prob_ = 1.0f;

    int visits_ = 0;
    float total_value_ = 0.0f;
};

}  // namespace aic::reasoning
