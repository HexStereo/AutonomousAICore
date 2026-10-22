// include/nn/moe.hpp
// Mixture-of-Experts (MoE) Sparse Routing & Load Balancing Engine.
// Provides Top-K expert gating, auxiliary load-balancing regularization loss,
// sparse token dispatch across parallel SwiGLU experts, and MoE Transformer Blocks.
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "nn/attention.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/rms_norm.hpp"
#include "nn/swiglu.hpp"
#include "ops/ops.hpp"

namespace aic::nn {

// Configuration for Mixture-of-Experts
struct MoEConfig {
    int64_t num_experts{8};
    int64_t top_k{2};
    float aux_loss_coef{0.01f};
};

// Output of the Gating Router
struct RouterOutput {
    Tensor routing_weights;             // Normalized softmax weights over top-k [N, top_k]
    Tensor expert_indices;              // Indices of selected experts [N, top_k] (Int64)
    Tensor aux_loss;                    // Differentiable auxiliary load-balancing loss (scalar)
    std::vector<int64_t> expert_counts; // Number of tokens dispatched to each expert
};

// Gating Router Network
// Maps token representations to expert logits, selects top-k experts per token,
// computes normalized softmax gating weights, and evaluates the auxiliary load-balancing loss.
class MoERouter : public Module {
public:
    MoERouter(int64_t d_model, int64_t num_experts = 8, int64_t top_k = 2,
              float aux_loss_coef = 0.01f, Device device = Device::cpu());

    RouterOutput route(const Tensor& x);

    Tensor forward(const Tensor& x) override {
        return route(x).routing_weights;
    }

    int64_t d_model() const noexcept { return d_model_; }
    int64_t num_experts() const noexcept { return num_experts_; }
    int64_t top_k() const noexcept { return top_k_; }
    float aux_loss_coef() const noexcept { return aux_loss_coef_; }
    std::shared_ptr<Linear> gate() const noexcept { return gate_; }

private:
    int64_t d_model_;
    int64_t num_experts_;
    int64_t top_k_;
    float aux_loss_coef_;
    std::shared_ptr<Linear> gate_;
};

// Sparse Mixture-of-Experts Feed-Forward Layer
// Replaces standard dense FFN with E parallel SwiGLU experts. For each token,
// only top-k experts are dynamically activated, scaling parameter capacity with constant compute.
class MoELayer : public Module {
public:
    MoELayer(int64_t d_model, int64_t d_hidden, int64_t num_experts = 8, int64_t top_k = 2,
             float aux_loss_coef = 0.01f, Device device = Device::cpu());

    Tensor forward(const Tensor& x) override;

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    Tensor last_aux_loss() const noexcept { return last_aux_loss_; }
    const std::vector<int64_t>& last_expert_counts() const noexcept { return last_expert_counts_; }
    double load_imbalance_ratio() const;

    int64_t d_model() const noexcept { return d_model_; }
    int64_t d_hidden() const noexcept { return d_hidden_; }
    int64_t num_experts() const noexcept { return num_experts_; }
    int64_t top_k() const noexcept { return top_k_; }

    std::shared_ptr<MoERouter> router() const noexcept { return router_; }
    const std::vector<std::shared_ptr<SwiGLU>>& experts() const noexcept { return experts_; }

private:
    int64_t d_model_;
    int64_t d_hidden_;
    int64_t num_experts_;
    int64_t top_k_;

    std::shared_ptr<MoERouter> router_;
    std::vector<std::shared_ptr<SwiGLU>> experts_;

    Tensor last_aux_loss_;
    std::vector<int64_t> last_expert_counts_;
};

// Pre-norm Transformer Block combining RMSNorm, CausalSelfAttention, and MoE Feed-Forward Layer.
class MoETransformerBlock : public Module {
public:
    MoETransformerBlock(int64_t d_model, int64_t n_heads, int64_t d_hidden,
                        int64_t num_experts = 8, int64_t top_k = 2,
                        int64_t max_seq_len = 4096, float norm_eps = 1e-6f,
                        float aux_loss_coef = 0.01f, Device device = Device::cpu());

    Tensor forward_with_cache(const Tensor& x, KVCache* cache = nullptr, int64_t start_pos = 0);
    Tensor forward(const Tensor& x) override;

    Tensor operator()(const Tensor& x, KVCache* cache = nullptr, int64_t start_pos = 0) {
        return forward_with_cache(x, cache, start_pos);
    }

    Tensor last_aux_loss() const noexcept { return moe_ ? moe_->last_aux_loss() : Tensor(); }

    std::shared_ptr<RMSNorm> norm1() const noexcept { return norm1_; }
    std::shared_ptr<CausalSelfAttention> attn() const noexcept { return attn_; }
    std::shared_ptr<RMSNorm> norm2() const noexcept { return norm2_; }
    std::shared_ptr<MoELayer> moe() const noexcept { return moe_; }
    int64_t d_model() const noexcept { return d_model_; }

private:
    int64_t d_model_;
    std::shared_ptr<RMSNorm> norm1_;
    std::shared_ptr<CausalSelfAttention> attn_;
    std::shared_ptr<RMSNorm> norm2_;
    std::shared_ptr<MoELayer> moe_;
};

}  // namespace aic::nn
