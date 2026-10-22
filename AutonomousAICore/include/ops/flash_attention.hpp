// include/ops/flash_attention.hpp
// FlashAttention-2: Fused Scaled Dot-Product Attention with Online Softmax Tiling & Autograd.
#pragma once

#include <memory>
#include <vector>

#include "autograd/node.hpp"
#include "core/error.hpp"
#include "tensor/tensor.hpp"

namespace aic::ops {

struct FlashAttentionForwardResult {
    Tensor output;     // [B, H, Sq, D]
    Tensor logsumexp;   // [B, H, Sq] (stored for backward pass, L = m + log(l))
};

struct FlashAttentionBackwardResult {
    Tensor grad_q;     // [B, H, Sq, D]
    Tensor grad_k;     // [B, H, Sk, D]
    Tensor grad_v;     // [B, H, Sk, D]
};

// Forward pass: online softmax tiling (O(N) memory IO, no N x N materialization)
FlashAttentionForwardResult flash_attention_forward(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    bool is_causal = true,
    float scale = 0.0f
);

// Backward pass: exact gradients dQ, dK, dV recomputing attention blocks from L
FlashAttentionBackwardResult flash_attention_backward(
    const Tensor& grad_out,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& logsumexp,
    bool is_causal = true,
    float scale = 0.0f
);

// Autograd-integrated top-level API: returns output tensor connected to computational DAG
Tensor flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    bool is_causal = true,
    float scale = 0.0f
);

}  // namespace aic::ops

namespace aic::autograd {

class FlashAttentionBackwardNode : public Node {
public:
    FlashAttentionBackwardNode(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        const Tensor& out,
        const Tensor& logsumexp,
        bool is_causal,
        float scale
    );

    std::string name() const override { return "FlashAttentionBackwardNode"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_q_;
    SavedVariable saved_k_;
    SavedVariable saved_v_;
    SavedVariable saved_out_;
    SavedVariable saved_lse_;
    bool is_causal_;
    float scale_;
};

}  // namespace aic::autograd
