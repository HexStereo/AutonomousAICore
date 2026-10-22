// src/reasoning/thought_node.cpp
// Tree-of-Thoughts & MCTS Node Implementation.
#include "reasoning/thought_node.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace aic::reasoning {

ThoughtNode::ThoughtNode(
    std::weak_ptr<ThoughtNode> parent,
    ThoughtStep step,
    int depth,
    float prior_prob
) : parent_(std::move(parent)),
    step_(std::move(step)),
    depth_(depth),
    prior_prob_(prior_prob) {}

void ThoughtNode::add_child(std::shared_ptr<ThoughtNode> child) {
    if (child) {
        children_.push_back(std::move(child));
    }
}

float ThoughtNode::compute_puct(float c_puct, int parent_visits) const {
    if (!step_.is_valid) {
        return -1e9f; // Pruned/invalid child
    }

    float q = q_value();
    float exploration = c_puct * prior_prob_ *
        (std::sqrt(static_cast<float>(std::max(1, parent_visits))) / (1.0f + static_cast<float>(visits_)));
    return q + exploration;
}

std::shared_ptr<ThoughtNode> ThoughtNode::select_best_child(float c_puct) const {
    if (children_.empty()) return nullptr;

    std::shared_ptr<ThoughtNode> best_child = nullptr;
    float max_puct = -std::numeric_limits<float>::infinity();

    for (const auto& child : children_) {
        if (!child || !child->step().is_valid) continue;
        float puct = child->compute_puct(c_puct, visits_);
        if (puct > max_puct) {
            max_puct = puct;
            best_child = child;
        }
    }

    // Fallback: if all valid children are somehow tied or only 1 exists
    if (!best_child && !children_.empty()) {
        best_child = children_.front();
    }
    return best_child;
}

void ThoughtNode::backpropagate(float value) {
    visits_++;
    total_value_ += value;

    if (auto p = parent_.lock()) {
        p->backpropagate(value);
    }
}

std::vector<int64_t> ThoughtNode::accumulated_tokens() const {
    std::vector<int64_t> result;
    const ThoughtNode* curr = this;
    std::vector<const ThoughtNode*> path;

    while (curr) {
        path.push_back(curr);
        auto p = curr->parent_.lock();
        curr = p.get();
    }

    std::reverse(path.begin(), path.end());
    for (const auto* node : path) {
        result.insert(result.end(), node->step_.tokens.begin(), node->step_.tokens.end());
    }
    return result;
}

std::string ThoughtNode::full_reasoning_trace() const {
    std::ostringstream os;
    const ThoughtNode* curr = this;
    std::vector<const ThoughtNode*> path;

    while (curr) {
        if (!curr->step_.text.empty()) {
            path.push_back(curr);
        }
        auto p = curr->parent_.lock();
        curr = p.get();
    }

    std::reverse(path.begin(), path.end());
    for (size_t i = 0; i < path.size(); ++i) {
        os << "[Step " << (i + 1) << " (Score: " << std::fixed << path[i]->step_.score << ")]: "
           << path[i]->step_.text << "\n";
    }
    return os.str();
}

}  // namespace aic::reasoning
