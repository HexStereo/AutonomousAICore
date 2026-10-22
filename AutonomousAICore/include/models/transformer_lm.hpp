// include/models/transformer_lm.hpp
// Complete Decoder-Only Autoregressive Transformer Language Model with KV-caching.
#pragma once

#include "nn/module.hpp"
#include "nn/embedding.hpp"
#include "nn/transformer_block.hpp"
#include "nn/rms_norm.hpp"
#include "nn/linear.hpp"
#include "nn/attention.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace aic::models {

struct TransformerLMConfig {
    int64_t vocab_size = 320;
    int64_t hidden_dim = 128;
    int64_t num_layers = 4;
    int64_t num_heads = 4;
    int64_t intermediate_dim = 384;
    int64_t max_seq_len = 512;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;
};

class TransformerLM : public nn::Module {
public:
    explicit TransformerLM(TransformerLMConfig config, Device device = Device::cpu());

    // Standard sequence forward pass: input_ids [B, S] -> logits [B, S, V]
    Tensor forward(const Tensor& input_ids) override;

    Tensor operator()(const Tensor& input_ids) {
        return forward(input_ids);
    }

    // Forward directly from dense embedding sequences [B, S, hidden_dim]
    Tensor forward_embeddings(const Tensor& embeddings);

    // Step-by-step autoregressive forward pass with KV caching for fast generation
    Tensor forward_step(const Tensor& input_ids, int64_t start_pos, std::vector<nn::KVCache>& kv_caches);

    // Forward step directly from dense embedding sequence [B, S, hidden_dim]
    Tensor forward_embeddings_step(const Tensor& embeddings, int64_t start_pos, std::vector<nn::KVCache>& kv_caches);

    // High-level generation loop with temperature, top-k, and nucleus (top-p) sampling
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt,
        int64_t max_new_tokens,
        float temperature = 1.0f,
        int top_k = 50,
        float top_p = 0.9f,
        int64_t eos_token_id = -1,
        uint64_t seed = 42
    );

    const TransformerLMConfig& config() const noexcept { return config_; }
    Device device() const noexcept { return tok_embeddings_ ? tok_embeddings_->weight()->device() : Device::cpu(); }
    std::shared_ptr<nn::Embedding> tok_embeddings() const noexcept { return tok_embeddings_; }
    const std::vector<std::shared_ptr<nn::TransformerBlock>>& layers() const noexcept { return layers_; }
    std::shared_ptr<nn::RMSNorm> norm() const noexcept { return norm_; }
    std::shared_ptr<nn::Linear> lm_head() const noexcept { return lm_head_; }
    void set_lm_head(std::shared_ptr<nn::Linear> head) { lm_head_ = register_module("lm_head", head); }

private:
    TransformerLMConfig config_;
    std::shared_ptr<nn::Embedding> tok_embeddings_;
    std::vector<std::shared_ptr<nn::TransformerBlock>> layers_;
    std::shared_ptr<nn::RMSNorm> norm_;
    std::shared_ptr<nn::Linear> lm_head_;
};

}  // namespace aic::models
