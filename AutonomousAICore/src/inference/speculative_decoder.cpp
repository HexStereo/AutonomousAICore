// src/inference/speculative_decoder.cpp
// Speculative Decoding & Verification Engine implementation.
#include "inference/speculative_decoder.hpp"
#include "autograd/autograd.hpp"
#include "core/error.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>

namespace aic::inference {

namespace {

std::vector<float> compute_softmax_distribution(
    const float* logits,
    int64_t vocab_size,
    float temperature
) {
    std::vector<float> probs(static_cast<size_t>(vocab_size));
    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    float max_logit = logits[0] * inv_temp;
    for (int64_t i = 1; i < vocab_size; ++i) {
        float l = logits[i] * inv_temp;
        if (l > max_logit) max_logit = l;
    }

    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab_size; ++i) {
        float e = std::exp(logits[i] * inv_temp - max_logit);
        probs[static_cast<size_t>(i)] = e;
        sum_exp += e;
    }

    float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-12));
    for (int64_t i = 0; i < vocab_size; ++i) {
        probs[static_cast<size_t>(i)] *= inv_sum;
    }

    return probs;
}

}  // namespace

SpeculativeDecoder::SpeculativeDecoder(
    std::shared_ptr<models::TransformerLM> target_model,
    std::shared_ptr<models::TransformerLM> draft_model,
    data::Tokenizer tokenizer,
    SpeculativeConfig config
) : target_model_(std::move(target_model)),
    draft_model_(std::move(draft_model)),
    tokenizer_(std::move(tokenizer)),
    config_(config),
    rng_(1337) {
    AIC_CHECK(target_model_ != nullptr, InvalidArgument, "target_model cannot be null");
    AIC_CHECK(draft_model_ != nullptr, InvalidArgument, "draft_model cannot be null");
    AIC_CHECK(config_.draft_lookahead > 0, InvalidArgument, "draft_lookahead must be positive");

    target_model_->eval();
    draft_model_->eval();
}

int64_t SpeculativeDecoder::sample_token(
    const float* logits,
    int64_t vocab_size,
    float temperature,
    int top_k,
    float top_p,
    std::mt19937& rng
) {
    auto probs = compute_softmax_distribution(logits, vocab_size, temperature);

    struct TokenProb {
        int64_t id;
        float prob;
    };

    std::vector<TokenProb> candidates(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
        candidates[static_cast<size_t>(i)] = {i, probs[static_cast<size_t>(i)]};
    }

    std::sort(candidates.begin(), candidates.end(), [](const TokenProb& a, const TokenProb& b) {
        return a.prob > b.prob;
    });

    // Top-K
    if (top_k > 0 && top_k < static_cast<int>(candidates.size())) {
        candidates.resize(static_cast<size_t>(top_k));
    }

    // Top-P Nucleus
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        size_t cutoff = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumsum += candidates[i].prob;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
    }

    // Renormalize
    float sum_p = 0.0f;
    for (const auto& c : candidates) sum_p += c.prob;
    if (sum_p <= 0.0f) return candidates.front().id;

    std::uniform_real_distribution<float> dist(0.0f, sum_p);
    float r = dist(rng);
    float acc = 0.0f;
    for (const auto& c : candidates) {
        acc += c.prob;
        if (r <= acc) {
            return c.id;
        }
    }

    return candidates.back().id;
}

std::pair<bool, int64_t> SpeculativeDecoder::verify_and_sample(
    const float* target_probs,
    const float* draft_probs,
    int64_t candidate_token,
    int64_t vocab_size,
    std::mt19937& rng
) {
    float p_target = target_probs[candidate_token];
    float p_draft = std::max(draft_probs[candidate_token], 1e-12f);

    float acceptance_ratio = std::min(1.0f, p_target / p_draft);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);

    if (u < acceptance_ratio) {
        // Token accepted!
        return {true, candidate_token};
    }

    // Token rejected: sample recovery token from residual distribution max(0, P_target - P_draft)
    std::vector<float> residual(static_cast<size_t>(vocab_size));
    float sum_residual = 0.0f;

    for (int64_t i = 0; i < vocab_size; ++i) {
        float diff = std::max(0.0f, target_probs[i] - draft_probs[i]);
        residual[static_cast<size_t>(i)] = diff;
        sum_residual += diff;
    }

    if (sum_residual <= 1e-12f) {
        // Fallback: sample directly from target distribution
        float r = dist(rng);
        float acc = 0.0f;
        for (int64_t i = 0; i < vocab_size; ++i) {
            acc += target_probs[i];
            if (r <= acc) return {false, i};
        }
        return {false, 0};
    }

    float r = dist(rng) * sum_residual;
    float acc = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        acc += residual[static_cast<size_t>(i)];
        if (r <= acc) {
            return {false, i};
        }
    }

    return {false, vocab_size - 1};
}

std::vector<int64_t> SpeculativeDecoder::generate(
    const std::vector<int64_t>& prompt_tokens,
    int max_new_tokens
) {
    AIC_CHECK(!prompt_tokens.empty(), InvalidArgument, "prompt_tokens cannot be empty");
    AIC_CHECK(max_new_tokens > 0, InvalidArgument, "max_new_tokens must be positive");

    autograd::NoGradGuard no_grad;
    auto start_time = std::chrono::high_resolution_clock::now();

    Device dev = target_model_->device();
    int64_t vocab_size = target_model_->config().vocab_size;

    std::vector<int64_t> output_seq = prompt_tokens;
    int generated_count = 0;

    while (generated_count < max_new_tokens) {
        int lookahead = std::min(config_.draft_lookahead, max_new_tokens - generated_count);

        // 1. Propose 'lookahead' draft tokens sequentially with the Draft Model
        std::vector<int64_t> draft_proposed;
        std::vector<std::vector<float>> draft_probs_history;
        draft_proposed.reserve(static_cast<size_t>(lookahead));

        std::vector<int64_t> draft_input = output_seq;

        for (int k = 0; k < lookahead; ++k) {
            int64_t seq_len = static_cast<int64_t>(draft_input.size());
            Tensor in_t = Tensor::from_blob(draft_input.data(), Shape({1, seq_len}), DType::Int64, Device::cpu())
                            .clone()
                            .to(dev);

            Tensor logits = draft_model_->forward(in_t);
            Tensor last_logit = logits.slice(1, seq_len - 1, seq_len).squeeze(1).cpu();
            const float* logit_ptr = last_logit.data_ptr<float>();

            auto probs = compute_softmax_distribution(logit_ptr, vocab_size, config_.temperature);
            draft_probs_history.push_back(probs);

            int64_t token = sample_token(
                logit_ptr, vocab_size, config_.temperature,
                config_.top_k, config_.top_p, rng_
            );

            draft_proposed.push_back(token);
            draft_input.push_back(token);
            metrics_.draft_tokens_proposed++;
        }

        // 2. Parallel Target Model Forward Pass on (output_seq + draft_proposed)
        int64_t target_in_len = static_cast<int64_t>(draft_input.size());
        Tensor target_in_t = Tensor::from_blob(draft_input.data(), Shape({1, target_in_len}), DType::Int64, Device::cpu())
                                .clone()
                                .to(dev);

        Tensor target_logits = target_model_->forward(target_in_t);
        metrics_.target_forward_steps++;

        // 3. Sequential Verification & Rejection Sampling
        int64_t orig_len = static_cast<int64_t>(output_seq.size());
        bool all_accepted = true;

        for (int k = 0; k < lookahead; ++k) {
            int64_t eval_pos = orig_len - 1 + k;
            Tensor target_pos_logits = target_logits.slice(1, eval_pos, eval_pos + 1).squeeze(1).cpu();
            auto target_probs = compute_softmax_distribution(target_pos_logits.data_ptr<float>(), vocab_size, config_.temperature);

            auto [accepted, recovery_token] = verify_and_sample(
                target_probs.data(),
                draft_probs_history[static_cast<size_t>(k)].data(),
                draft_proposed[static_cast<size_t>(k)],
                vocab_size,
                rng_
            );

            if (accepted) {
                output_seq.push_back(draft_proposed[static_cast<size_t>(k)]);
                generated_count++;
                metrics_.draft_tokens_accepted++;
                metrics_.total_tokens_generated++;

                if (draft_proposed[static_cast<size_t>(k)] == config_.eos_token_id || generated_count >= max_new_tokens) {
                    all_accepted = false;
                    break;
                }
            } else {
                // Rejected: accept recovery token and terminate draft window
                output_seq.push_back(recovery_token);
                generated_count++;
                metrics_.total_tokens_generated++;
                all_accepted = false;
                break;
            }
        }

        // 4. Bonus token if all draft tokens were accepted
        if (all_accepted && generated_count < max_new_tokens) {
            int64_t last_pos = target_in_len - 1;
            Tensor bonus_logits = target_logits.slice(1, last_pos, last_pos + 1).squeeze(1).cpu();
            int64_t bonus_token = sample_token(
                bonus_logits.data_ptr<float>(), vocab_size, config_.temperature,
                config_.top_k, config_.top_p, rng_
            );
            output_seq.push_back(bonus_token);
            generated_count++;
            metrics_.total_tokens_generated++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.total_time_ms += std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return output_seq;
}

std::string SpeculativeDecoder::generate_text(
    const std::string& prompt_text,
    int max_new_tokens
) {
    auto prompt_tokens = tokenizer_.encode(prompt_text);
    auto result_tokens = generate(prompt_tokens, max_new_tokens);
    return tokenizer_.decode(result_tokens);
}

}  // namespace aic::inference
