// src/reasoning/verifier.cpp
// Self-Verifier and Thought Consistency Checker.
#include "reasoning/verifier.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace aic::reasoning {

namespace {

std::string to_lower(std::string_view s) {
    std::string out(s.begin(), s.end());
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

}  // namespace

SelfVerifier::SelfVerifier(VerifierConfig config)
    : config_(std::move(config)) {}

float SelfVerifier::score_repetition(const std::vector<int64_t>& tokens) const {
    if (tokens.size() < 3) return 0.0f;

    size_t consecutive_repeats = 1;
    size_t max_consecutive = 1;

    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i - 1]) {
            consecutive_repeats++;
            max_consecutive = std::max(max_consecutive, consecutive_repeats);
        } else {
            consecutive_repeats = 1;
        }
    }

    if (max_consecutive >= config_.max_consecutive_repeat) {
        return config_.repetition_penalty;
    }
    return 0.0f;
}

bool SelfVerifier::detect_contradiction(std::string_view text) const {
    std::string lower = to_lower(text);
    const std::string_view contradictions[] = {
        "contradiction",
        "is impossible",
        "cannot be true and false",
        "not equal to itself",
        "1 = 0",
        "false is true"
    };

    for (const auto& pat : contradictions) {
        if (lower.find(pat) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool SelfVerifier::detect_conclusion(std::string_view text) const {
    std::string lower = to_lower(text);
    const std::string_view conclusions[] = {
        "answer is",
        "in conclusion",
        "therefore the answer",
        "result:",
        "q.e.d",
        "final answer",
        "thus we conclude"
    };

    for (const auto& pat : conclusions) {
        if (lower.find(pat) != std::string::npos) {
            return true;
        }
    }
    return false;
}

float SelfVerifier::score_reasoning_flow(std::string_view text) const {
    std::string lower = to_lower(text);
    const std::string_view deduction_markers[] = {
        "step",
        "first",
        "then",
        "next",
        "because",
        "since",
        "implies",
        "therefore",
        "hence",
        "thus"
    };

    float bonus = 0.0f;
    for (const auto& marker : deduction_markers) {
        if (lower.find(marker) != std::string::npos) {
            bonus += 0.05f;
            if (bonus >= config_.deduction_bonus) break;
        }
    }
    return std::min(bonus, config_.deduction_bonus);
}

VerificationResult SelfVerifier::verify_step(
    const std::vector<int64_t>& /*prompt_tokens*/,
    const std::vector<int64_t>& /*history_tokens*/,
    const ThoughtStep& candidate_step
) const {
    VerificationResult res;
    res.score = candidate_step.score > 0.0f ? candidate_step.score : 0.5f;

    // 1. Contradiction Check
    if (detect_contradiction(candidate_step.text)) {
        res.is_contradiction = true;
        res.is_valid = false;
        res.score = 0.05f;
        res.reason = "Logical contradiction detected";
        return res;
    }

    // 2. Token Repetition Penalty
    float rep_penalty = score_repetition(candidate_step.tokens);
    if (rep_penalty > 0.0f) {
        res.score -= rep_penalty;
    }

    // 3. Logical Reasoning Flow Bonus
    float flow_bonus = score_reasoning_flow(candidate_step.text);
    res.score += flow_bonus;

    // 4. Log Probability Confidence Integration (if available)
    if (!candidate_step.tokens.empty() && candidate_step.log_prob != 0.0f) {
        float avg_log_prob = candidate_step.log_prob / static_cast<float>(candidate_step.tokens.size());
        // Map typical avg_log_prob [-4.0, 0.0] to [0.1, 0.9] via sigmoid
        float prob_conf = 1.0f / (1.0f + std::exp(-(avg_log_prob + 1.5f)));
        res.score = 0.6f * res.score + 0.4f * prob_conf;
    }

    // Clamp score to [0.01, 0.99]
    res.score = std::clamp(res.score, 0.01f, 0.99f);

    // 5. Conclusion Check
    res.is_conclusive = detect_conclusion(candidate_step.text);

    // 6. Validity / Rejection Threshold
    if (res.score < config_.rejection_threshold) {
        res.is_valid = false;
        res.reason = "Verification score below rejection threshold";
    } else {
        res.is_valid = true;
        res.reason = res.is_conclusive ? "Conclusive reasoning step" : "Valid deduction step";
    }

    return res;
}

VerificationResult SelfVerifier::verify_text(std::string_view text) const {
    ThoughtStep step;
    step.text = std::string(text);
    step.score = 0.5f;
    return verify_step({}, {}, step);
}

}  // namespace aic::reasoning
