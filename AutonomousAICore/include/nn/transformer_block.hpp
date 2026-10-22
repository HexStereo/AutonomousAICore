// include/nn/transformer_block.hpp
// Pre-norm Transformer Block combining RMSNorm, CausalSelfAttention, and SwiGLU FFN.
#pragma once

#include "nn/attention.hpp"
#include "nn/module.hpp"
#include "nn/rms_norm.hpp"
#include "nn/swiglu.hpp"
#include "ops/ops.hpp"

namespace aic::nn {

class TransformerBlock : public Module {
public:
    TransformerBlock(int64_t d_model, int64_t n_heads, int64_t d_hidden,
                     int64_t max_seq_len = 4096, float norm_eps = 1e-6f,
                     Device device = Device::cpu())
        : d_model_(d_model) {
        
        norm1_ = register_module("norm1", std::make_shared<RMSNorm>(d_model_, norm_eps, device));
        attn_  = register_module("attn",  std::make_shared<CausalSelfAttention>(d_model_, n_heads, max_seq_len, device));
        norm2_ = register_module("norm2", std::make_shared<RMSNorm>(d_model_, norm_eps, device));
        ffn_   = register_module("ffn",   std::make_shared<SwiGLU>(d_model_, d_hidden, device));
    }

    Tensor forward_with_cache(const Tensor& x, KVCache* cache = nullptr, int64_t start_pos = 0) {
        // Pre-norm Attention with residual connection: x = x + Attention(Norm1(x))
        Tensor norm1_out = norm1_->forward(x);
        Tensor attn_out = attn_->forward_with_cache(norm1_out, cache, start_pos);
        Tensor x1 = add(x, attn_out);

        // Pre-norm Feed-Forward with residual connection: x = x + SwiGLU(Norm2(x))
        Tensor norm2_out = norm2_->forward(x1);
        Tensor ffn_out = ffn_->forward(norm2_out);
        return add(x1, ffn_out);
    }

    Tensor forward(const Tensor& x) override {
        return forward_with_cache(x, nullptr, 0);
    }

    Tensor operator()(const Tensor& x, KVCache* cache = nullptr, int64_t start_pos = 0) {
        return forward_with_cache(x, cache, start_pos);
    }

    std::shared_ptr<RMSNorm> norm1() const { return norm1_; }
    std::shared_ptr<CausalSelfAttention> attn() const { return attn_; }
    std::shared_ptr<RMSNorm> norm2() const { return norm2_; }
    std::shared_ptr<SwiGLU> ffn() const { return ffn_; }
    int64_t d_model() const noexcept { return d_model_; }

private:
    int64_t d_model_;
    std::shared_ptr<RMSNorm> norm1_;
    std::shared_ptr<CausalSelfAttention> attn_;
    std::shared_ptr<RMSNorm> norm2_;
    std::shared_ptr<SwiGLU> ffn_;
};

}  // namespace aic::nn
