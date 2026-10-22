// include/models/moe_transformer_lm.hpp
// Decoder-Only Mixture-of-Experts (MoE) Language Model with Dynamic Routing & KV-caching.
#pragma once

#include "nn/attention.hpp"
#include "nn/embedding.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/moe.hpp"
#include "nn/rms_norm.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace aic::models {

struct MoETransformerLMConfig {
    int64_t vocab_size = 320;
    int64_t hidden_dim = 128;
    int64_t num_layers = 4;
    int64_t num_heads = 4;
    int64_t intermediate_dim = 256;
    int64_t num_experts = 4;
    int64_t top_k = 2;
    float aux_loss_coef = 0.01f;
    int64_t max_seq_len = 512;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

class MoETransformerLM : public nn::Module {
public:
    explicit MoETransformerLM(MoETransformerLMConfig config, Device device = Device::cpu());

    // Standard sequence forward pass: input_ids [B, S] -> logits [B, S, V]
    Tensor forward(const Tensor& input_ids) override;

    Tensor operator()(const Tensor& input_ids) {
        return forward(input_ids);
    }

    // Step-by-step autoregressive forward pass with KV caching for fast generation
    Tensor forward_step(const Tensor& input_ids, int64_t start_pos, std::vector<nn::KVCache>& kv_caches);

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

    // Differentiable total auxiliary load-balancing loss summed over all MoE layers
    Tensor total_aux_loss() const;

    const MoETransformerLMConfig& config() const noexcept { return config_; }
    Device device() const noexcept { return tok_embeddings_ ? tok_embeddings_->weight()->device() : Device::cpu(); }
    std::shared_ptr<nn::Embedding> tok_embeddings() const noexcept { return tok_embeddings_; }
    const std::vector<std::shared_ptr<nn::MoETransformerBlock>>& layers() const noexcept { return layers_; }
    std::shared_ptr<nn::RMSNorm> norm() const noexcept { return norm_; }
    std::shared_ptr<nn::Linear> lm_head() const noexcept { return lm_head_; }

private:
    MoETransformerLMConfig config_;
    std::shared_ptr<nn::Embedding> tok_embeddings_;
    std::vector<std::shared_ptr<nn::MoETransformerBlock>> layers_;
    std::shared_ptr<nn::RMSNorm> norm_;
    std::shared_ptr<nn::Linear> lm_head_;
};

}  // namespace aic::models
