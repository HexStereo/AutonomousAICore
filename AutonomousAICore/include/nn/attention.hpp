// include/nn/attention.hpp
// Multi-Head Causal Self-Attention with RoPE and Key-Value (KV) Cache support.
#pragma once

#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/rope.hpp"
#include "ops/flash_attention.hpp"
#include "ops/matmul.hpp"
#include "ops/ops.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace aic::nn {

struct KVCache {
    Tensor key_cache;
    Tensor value_cache;

    void reset() {
        key_cache = Tensor();
        value_cache = Tensor();
    }

    int64_t current_seq_len() const {
        if (!key_cache.defined()) return 0;
        return key_cache.dim(2);
    }
};

class CausalSelfAttention : public Module {
public:
    CausalSelfAttention(int64_t d_model, int64_t n_heads, int64_t max_seq_len = 4096, Device device = Device::cpu())
        : d_model_(d_model), n_heads_(n_heads), head_dim_(d_model / n_heads), use_flash_attention_(false) {
        
        AIC_CHECK(d_model_ % n_heads_ == 0, InvalidArgument,
                  "d_model (", d_model_, ") must be divisible by n_heads (", n_heads_, ")");

        q_proj_   = register_module("q_proj",   std::make_shared<Linear>(d_model_, d_model_, /*has_bias=*/false, device));
        k_proj_   = register_module("k_proj",   std::make_shared<Linear>(d_model_, d_model_, /*has_bias=*/false, device));
        v_proj_   = register_module("v_proj",   std::make_shared<Linear>(d_model_, d_model_, /*has_bias=*/false, device));
        out_proj_ = register_module("out_proj", std::make_shared<Linear>(d_model_, d_model_, /*has_bias=*/false, device));

        rope_ = register_module("rope", std::make_shared<RotaryPositionEmbedding>(head_dim_, max_seq_len, 10000.0f, device));
    }

    Tensor forward_with_cache(const Tensor& x, KVCache* cache = nullptr, int64_t start_pos = 0) {
        AIC_CHECK(x.rank() == 3, InvalidArgument, "CausalSelfAttention expects 3D tensor [B, S, D], got rank ", x.rank());
        int64_t B = x.dim(0);
        int64_t S = x.dim(1);
        AIC_CHECK(x.dim(2) == d_model_, ShapeMismatch, "Input feature dimension mismatch");

        // 1. Compute Q, K, V projections
        Tensor q = q_proj_->forward(x);
        Tensor k = k_proj_->forward(x);
        Tensor v = v_proj_->forward(x);

        // 2. Reshape to multi-head: [B, S, n_heads, head_dim] -> transpose -> [B, n_heads, S, head_dim]
        q = q.reshape(Shape({B, S, n_heads_, head_dim_})).transpose(1, 2);
        k = k.reshape(Shape({B, S, n_heads_, head_dim_})).transpose(1, 2);
        v = v.reshape(Shape({B, S, n_heads_, head_dim_})).transpose(1, 2);

        // 3. Apply Rotary Position Embedding to Q and K
        q = rope_->forward(q, start_pos);
        k = rope_->forward(k, start_pos);

        // 4. Update Key-Value Cache if present
        if (cache != nullptr) {
            if (cache->key_cache.defined() && cache->key_cache.numel() > 0) {
                k = cat({cache->key_cache, k}, 2);
                v = cat({cache->value_cache, v}, 2);
            }
            cache->key_cache = k;
            cache->value_cache = v;
        }

        Tensor context;
        if (use_flash_attention_ && cache == nullptr && start_pos == 0) {
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            context = ops::flash_attention(q, k, v, /*is_causal=*/(S > 1), scale);
        } else {
            int64_t S_k = k.dim(2);

            // 5. Scaled dot-product attention scores: [B, n_heads, S, S_k]
            Tensor k_t = k.transpose(2, 3);
            Tensor scores = matmul(q, k_t);

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            scores = mul(scores, scale);

            // 6. Causal masking: prevent query at pos_q from attending to future tokens
            if (S > 1) {
                std::vector<float> mask_vals(static_cast<size_t>(S * S_k), 0.0f);
                for (int64_t i = 0; i < S; ++i) {
                    int64_t pos_q = start_pos + i;
                    for (int64_t j = 0; j < S_k; ++j) {
                        if (j > pos_q) {
                            mask_vals[static_cast<size_t>(i * S_k + j)] = -1e9f;
                        }
                    }
                }
                Tensor mask = Tensor::from_blob(mask_vals.data(), Shape({1, 1, S, S_k})).clone().to(x.device());
                scores = add(scores, mask);
            }

            // 7. Softmax over last dimension
            Tensor attn_weights = softmax(scores, -1);

            // 8. Weighted context: [B, n_heads, S, head_dim]
            context = matmul(attn_weights, v);
        }

        // 9. Transpose back: [B, S, n_heads, head_dim] -> flatten -> [B, S, d_model]
        context = context.transpose(1, 2).reshape(Shape({B, S, d_model_}));

        // 10. Output linear projection
        return out_proj_->forward(context);
    }

    Tensor forward(const Tensor& x) override {
        return forward_with_cache(x, nullptr, 0);
    }

    Tensor operator()(const Tensor& x, KVCache* cache = nullptr, int64_t start_pos = 0) {
        return forward_with_cache(x, cache, start_pos);
    }

    int64_t d_model() const noexcept { return d_model_; }
    int64_t n_heads() const noexcept { return n_heads_; }
    int64_t head_dim() const noexcept { return head_dim_; }
    std::shared_ptr<RotaryPositionEmbedding> rope() const { return rope_; }

    std::shared_ptr<Linear> q_proj() const { return q_proj_; }
    std::shared_ptr<Linear> k_proj() const { return k_proj_; }
    std::shared_ptr<Linear> v_proj() const { return v_proj_; }
    std::shared_ptr<Linear> out_proj() const { return out_proj_; }

    void set_q_proj(std::shared_ptr<Linear> mod) { q_proj_ = register_module("q_proj", mod); }
    void set_k_proj(std::shared_ptr<Linear> mod) { k_proj_ = register_module("k_proj", mod); }
    void set_v_proj(std::shared_ptr<Linear> mod) { v_proj_ = register_module("v_proj", mod); }
    void set_out_proj(std::shared_ptr<Linear> mod) { out_proj_ = register_module("out_proj", mod); }

    bool use_flash_attention() const noexcept { return use_flash_attention_; }
    void set_use_flash_attention(bool val) noexcept { use_flash_attention_ = val; }

private:
    int64_t d_model_;
    int64_t n_heads_;
    int64_t head_dim_;
    bool use_flash_attention_;

    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> out_proj_;
    std::shared_ptr<RotaryPositionEmbedding> rope_;
};

}  // namespace aic::nn
