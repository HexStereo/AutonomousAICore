// src/models/transformer_lm.cpp
// Decoder-Only Transformer Language Model implementation with KV-caching.
#include "models/transformer_lm.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace aic::models {

TransformerLM::TransformerLM(TransformerLMConfig config, Device device)
    : config_(config) {
    AIC_CHECK(config_.vocab_size > 0, InvalidArgument, "vocab_size must be positive");
    AIC_CHECK(config_.hidden_dim > 0, InvalidArgument, "hidden_dim must be positive");
    AIC_CHECK(config_.num_layers > 0, InvalidArgument, "num_layers must be positive");
    AIC_CHECK(config_.num_heads > 0, InvalidArgument, "num_heads must be positive");

    tok_embeddings_ = register_module(
        "tok_embeddings",
        std::make_shared<nn::Embedding>(config_.vocab_size, config_.hidden_dim, device)
    );

    layers_.reserve(static_cast<size_t>(config_.num_layers));
    for (int64_t i = 0; i < config_.num_layers; ++i) {
        auto block = std::make_shared<nn::TransformerBlock>(
            config_.hidden_dim,
            config_.num_heads,
            config_.intermediate_dim,
            config_.max_seq_len,
            config_.rms_norm_eps,
            device
        );
        layers_.push_back(register_module("layers." + std::to_string(i), block));
    }

    norm_ = register_module(
        "norm",
        std::make_shared<nn::RMSNorm>(config_.hidden_dim, config_.rms_norm_eps, device)
    );

    lm_head_ = register_module(
        "lm_head",
        std::make_shared<nn::Linear>(config_.hidden_dim, config_.vocab_size, /*has_bias=*/false, device)
    );
}

Tensor TransformerLM::forward(const Tensor& input_ids) {
    Tensor h = tok_embeddings_->forward(input_ids);
    return forward_embeddings(h);
}

Tensor TransformerLM::forward_embeddings(const Tensor& embeddings) {
    Tensor h = embeddings;
    for (auto& layer : layers_) {
        h = layer->forward(h);
    }
    h = norm_->forward(h);
    return lm_head_->forward(h);
}

Tensor TransformerLM::forward_step(
    const Tensor& input_ids,
    int64_t start_pos,
    std::vector<nn::KVCache>& kv_caches
) {
    Tensor h = tok_embeddings_->forward(input_ids);
    return forward_embeddings_step(h, start_pos, kv_caches);
}

Tensor TransformerLM::forward_embeddings_step(
    const Tensor& embeddings,
    int64_t start_pos,
    std::vector<nn::KVCache>& kv_caches
) {
    AIC_CHECK(kv_caches.size() == layers_.size(), InvalidArgument,
              "kv_caches size (", kv_caches.size(), ") must match num_layers (", layers_.size(), ")");

    Tensor h = embeddings;
    for (size_t l = 0; l < layers_.size(); ++l) {
        h = layers_[l]->forward_with_cache(h, &kv_caches[l], start_pos);
    }
    h = norm_->forward(h);
    return lm_head_->forward(h);
}

std::vector<int64_t> TransformerLM::generate(
    const std::vector<int64_t>& prompt,
    int64_t max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    int64_t eos_token_id,
    uint64_t seed
) {
    AIC_CHECK(max_new_tokens >= 0, InvalidArgument, "max_new_tokens must be non-negative");

    std::vector<int64_t> current_prompt = prompt;
    if (current_prompt.empty()) {
        current_prompt.push_back(0);
    }

    std::vector<int64_t> output_tokens = current_prompt;
    output_tokens.reserve(current_prompt.size() + static_cast<size_t>(max_new_tokens));

    std::vector<nn::KVCache> kv_caches(layers_.size());
    std::mt19937_64 rng(seed);
    Device dev = tok_embeddings_->weight()->device();

    // 1. Prefill prompt tokens into KV-caches step by step
    Tensor last_logits;
    for (size_t i = 0; i < current_prompt.size(); ++i) {
        Tensor single_cpu = Tensor::empty(Shape({1, 1}), DType::Int64, Device::cpu());
        single_cpu.data_ptr<int64_t>()[0] = current_prompt[i];
        Tensor single_token = dev.is_cuda() ? single_cpu.to(dev) : single_cpu;

        last_logits = forward_step(single_token, static_cast<int64_t>(i), kv_caches);
    }

    // 2. Autoregressive token-by-token generation loop
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        // Extract 1D logits from last_logits [1, 1, V]
        Tensor cpu_logits = last_logits.cpu();
        const float* l_ptr = cpu_logits.data_ptr<float>();
        int64_t V = config_.vocab_size;

        int64_t next_token = 0;

        if (temperature <= 0.001f || top_k == 1) {
            // Greedy argmax sampling
            next_token = static_cast<int64_t>(
                std::max_element(l_ptr, l_ptr + V) - l_ptr
            );
        } else {
            // Temperature scaling and softmax
            std::vector<float> scaled_logits(V);
            float max_l = l_ptr[0] / temperature;
            for (int64_t v = 0; v < V; ++v) {
                float val = l_ptr[v] / temperature;
                scaled_logits[static_cast<size_t>(v)] = val;
                if (val > max_l) max_l = val;
            }

            std::vector<float> probs(V);
            double sum_p = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                float p = std::exp(scaled_logits[static_cast<size_t>(v)] - max_l);
                probs[static_cast<size_t>(v)] = p;
                sum_p += p;
            }
            for (auto& p : probs) {
                p /= static_cast<float>(sum_p);
            }

            // Top-K filtering
            if (top_k > 0 && top_k < V) {
                std::vector<int64_t> indices(static_cast<size_t>(V));
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                    [&probs](int64_t a, int64_t b) { return probs[a] > probs[b]; }
                );

                std::vector<float> filtered(V, 0.0f);
                double fsum = 0.0;
                for (int i = 0; i < top_k; ++i) {
                    filtered[indices[i]] = probs[indices[i]];
                    fsum += probs[indices[i]];
                }
                for (int i = 0; i < top_k; ++i) {
                    filtered[indices[i]] /= static_cast<float>(fsum);
                }
                probs = std::move(filtered);
            }

            // Top-P (nucleus) filtering
            if (top_p < 1.0f && top_p > 0.0f) {
                std::vector<std::pair<float, int64_t>> p_idx;
                p_idx.reserve(static_cast<size_t>(V));
                for (int64_t v = 0; v < V; ++v) {
                    if (probs[v] > 0.0f) {
                        p_idx.push_back({probs[v], v});
                    }
                }
                std::sort(p_idx.begin(), p_idx.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; }
                );

                double cumsum = 0.0;
                size_t cutoff = p_idx.size();
                for (size_t i = 0; i < p_idx.size(); ++i) {
                    cumsum += p_idx[i].first;
                    if (cumsum >= top_p) {
                        cutoff = i + 1;
                        break;
                    }
                }

                std::vector<float> nucleus_probs(V, 0.0f);
                double nsum = 0.0;
                for (size_t i = 0; i < cutoff; ++i) {
                    nucleus_probs[p_idx[i].second] = p_idx[i].first;
                    nsum += p_idx[i].first;
                }
                for (size_t i = 0; i < cutoff; ++i) {
                    nucleus_probs[p_idx[i].second] /= static_cast<float>(nsum);
                }
                probs = std::move(nucleus_probs);
            }

            // Categorical random sampling
            std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
            next_token = dist(rng);
        }

        output_tokens.push_back(next_token);

        if (eos_token_id >= 0 && next_token == eos_token_id) {
            break; // Finished generation
        }

        // Run forward step for newly generated token
        Tensor next_cpu = Tensor::empty(Shape({1, 1}), DType::Int64, Device::cpu());
        next_cpu.data_ptr<int64_t>()[0] = next_token;
        Tensor next_tensor = dev.is_cuda() ? next_cpu.to(dev) : next_cpu;

        int64_t next_pos = static_cast<int64_t>(current_prompt.size()) + step;
        last_logits = forward_step(next_tensor, next_pos, kv_caches);
    }

    return output_tokens;
}

}  // namespace aic::models
