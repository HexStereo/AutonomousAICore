// include/reasoning/verifier.hpp
// Self-Verification, Invariant Checking, and Confidence Scoring for Reasoning Steps.
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "reasoning/thought_node.hpp"

namespace aic::reasoning {

struct VerificationResult {
    float score = 0.5f;          // Verification confidence in [0.0, 1.0]
    bool is_valid = true;        // False if step fails invariant or is contradictory
    bool is_contradiction = false;
    bool is_conclusive = false;  // True if step reaches final answer / conclusion
    std::string reason;
};

struct VerifierConfig {
    float rejection_threshold = 0.25f;
    float repetition_penalty = 0.3f;
    float deduction_bonus = 0.15f;
    size_t max_consecutive_repeat = 3;
};

class SelfVerifier {
public:
    SelfVerifier() = default;
    explicit SelfVerifier(VerifierConfig config);

    // Evaluates a proposed reasoning step against the accumulated reasoning history
    VerificationResult verify_step(
        const std::vector<int64_t>& prompt_tokens,
        const std::vector<int64_t>& history_tokens,
        const ThoughtStep& candidate_step
    ) const;

    // Evaluates a complete reasoning text or trace string
    VerificationResult verify_text(std::string_view text) const;

    const VerifierConfig& config() const noexcept { return config_; }
    VerifierConfig& config() noexcept { return config_; }

private:
    float score_repetition(const std::vector<int64_t>& tokens) const;
    bool detect_contradiction(std::string_view text) const;
    bool detect_conclusion(std::string_view text) const;
    float score_reasoning_flow(std::string_view text) const;

    VerifierConfig config_;
};

}  // namespace aic::reasoning
