// src/nn/moe.cpp
// Mixture-of-Experts (MoE) Sparse Routing & Load Balancing Engine.
#include "nn/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace aic::nn {

// ============================================================================
// MoERouter Implementation
// ============================================================================

MoERouter::MoERouter(int64_t d_model, int64_t num_experts, int64_t top_k,
                     float aux_loss_coef, Device device)
    : d_model_(d_model),
      num_experts_(num_experts),
      top_k_(std::min(top_k, num_experts)),
      aux_loss_coef_(aux_loss_coef) {
    AIC_CHECK(d_model_ > 0, InvalidArgument, "d_model must be positive, got ", d_model_);
    AIC_CHECK(num_experts_ > 0, InvalidArgument, "num_experts must be positive, got ", num_experts_);
    AIC_CHECK(top_k_ > 0, InvalidArgument, "top_k must be positive, got ", top_k_);

    gate_ = register_module("gate", std::make_shared<Linear>(d_model_, num_experts_, /*has_bias=*/false, device));
}

RouterOutput MoERouter::route(const Tensor& x) {
    AIC_CHECK(x.dtype() == DType::Float32, DTypeMismatch, "MoERouter expects Float32 tensor, got ", dtype_name(x.dtype()));
    AIC_CHECK(x.dim(-1) == d_model_, ShapeMismatch,
              "MoERouter last dimension must match d_model (", d_model_, "), got ", x.dim(-1));

    // 1. Router linear projection: [..., d_model] -> [..., num_experts]
    Tensor logits = gate_->forward(x);
    int64_t total_tokens = logits.numel() / num_experts_;
    AIC_CHECK(total_tokens > 0, InvalidArgument, "MoERouter requires at least 1 token in forward pass");

    // 2. Softmax probabilities over all experts: [..., num_experts]
    Tensor probs = softmax(logits, -1);

    // 3. Average probability allocated to each expert across all tokens: \bar{P}_e
    Tensor probs_2d = probs.reshape({total_tokens, num_experts_});
    Tensor mean_probs = mean(probs_2d, 0); // shape: [num_experts]

    // 4. Extract top-k experts per token and token dispatch counts
    Tensor logits_cpu = logits.detach().cpu();
    const float* lptr = logits_cpu.data_ptr<float>();

    std::vector<int64_t> expert_counts(static_cast<size_t>(num_experts_), 0);
    std::vector<int64_t> topk_indices(static_cast<size_t>(total_tokens * top_k_));
    std::vector<float> topk_weights(static_cast<size_t>(total_tokens * top_k_));

    for (int64_t i = 0; i < total_tokens; ++i) {
        const float* row = lptr + i * num_experts_;
        std::vector<std::pair<float, int64_t>> candidates(static_cast<size_t>(num_experts_));
        for (int64_t e = 0; e < num_experts_; ++e) {
            candidates[static_cast<size_t>(e)] = {row[e], e};
        }
        std::partial_sort(candidates.begin(), candidates.begin() + top_k_, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float max_val = candidates[0].first;
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < top_k_; ++j) {
            sum_exp += std::exp(candidates[static_cast<size_t>(j)].first - max_val);
        }
        for (int64_t j = 0; j < top_k_; ++j) {
            int64_t exp_idx = candidates[static_cast<size_t>(j)].second;
            topk_indices[static_cast<size_t>(i * top_k_ + j)] = exp_idx;
            topk_weights[static_cast<size_t>(i * top_k_ + j)] =
                std::exp(candidates[static_cast<size_t>(j)].first - max_val) / (sum_exp + 1e-9f);
            expert_counts[static_cast<size_t>(exp_idx)]++;
        }
    }

    // 5. Differentiable auxiliary load-balancing loss:
    //    L_aux = alpha * E * \sum_{e=0}^{E-1} f_e * \bar{P}_e
    Tensor aux_loss;
    if (aux_loss_coef_ > 0.0f && num_experts_ > 1) {
        Tensor f_tensor = Tensor::empty({num_experts_}, DType::Float32, x.device());
        std::vector<float> f_data(static_cast<size_t>(num_experts_));
        float scale = (static_cast<float>(num_experts_) * aux_loss_coef_) /
                      (static_cast<float>(total_tokens) * static_cast<float>(top_k_));
        for (int64_t e = 0; e < num_experts_; ++e) {
            f_data[static_cast<size_t>(e)] = static_cast<float>(expert_counts[static_cast<size_t>(e)]) * scale;
        }

        if (x.is_cpu()) {
            std::memcpy(f_tensor.data_ptr<float>(), f_data.data(), sizeof(float) * static_cast<size_t>(num_experts_));
        } else {
#if AIC_WITH_CUDA
            cudaMemcpy(f_tensor.data_ptr<float>(), f_data.data(),
                       sizeof(float) * static_cast<size_t>(num_experts_), cudaMemcpyHostToDevice);
#endif
        }
        aux_loss = sum(mul(mean_probs, f_tensor));
    } else {
        aux_loss = Tensor::zeros({1}, DType::Float32, x.device());
    }

    // 6. Construct outputs
    RouterOutput out;
    out.aux_loss = aux_loss;
    out.expert_counts = std::move(expert_counts);

    out.expert_indices = Tensor::empty({total_tokens, top_k_}, DType::Int64, x.device());
    out.routing_weights = Tensor::empty({total_tokens, top_k_}, DType::Float32, x.device());

    if (x.is_cpu()) {
        std::memcpy(out.expert_indices.data_ptr<int64_t>(), topk_indices.data(),
                    sizeof(int64_t) * topk_indices.size());
        std::memcpy(out.routing_weights.data_ptr<float>(), topk_weights.data(),
                    sizeof(float) * topk_weights.size());
    } else {
#if AIC_WITH_CUDA
        cudaMemcpy(out.expert_indices.data_ptr<int64_t>(), topk_indices.data(),
                   sizeof(int64_t) * topk_indices.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(out.routing_weights.data_ptr<float>(), topk_weights.data(),
                   sizeof(float) * topk_weights.size(), cudaMemcpyHostToDevice);
#endif
    }

    return out;
}

// ============================================================================
// MoELayer Implementation
// ============================================================================

MoELayer::MoELayer(int64_t d_model, int64_t d_hidden, int64_t num_experts, int64_t top_k,
                   float aux_loss_coef, Device device)
    : d_model_(d_model),
      d_hidden_(d_hidden),
      num_experts_(num_experts),
      top_k_(std::min(top_k, num_experts)) {
    AIC_CHECK(d_model_ > 0, InvalidArgument, "d_model must be positive");
    AIC_CHECK(d_hidden_ > 0, InvalidArgument, "d_hidden must be positive");
    AIC_CHECK(num_experts_ > 0, InvalidArgument, "num_experts must be positive");
    AIC_CHECK(top_k_ > 0, InvalidArgument, "top_k must be positive");

    router_ = register_module("router", std::make_shared<MoERouter>(d_model_, num_experts_, top_k_, aux_loss_coef, device));

    experts_.reserve(static_cast<size_t>(num_experts_));
    for (int64_t e = 0; e < num_experts_; ++e) {
        auto exp = std::make_shared<SwiGLU>(d_model_, d_hidden_, device);
        experts_.push_back(register_module("experts." + std::to_string(e), exp));
    }
}

Tensor MoELayer::forward(const Tensor& x) {
    AIC_CHECK(x.dtype() == DType::Float32, DTypeMismatch, "MoELayer expects Float32 tensor");
    AIC_CHECK(x.dim(-1) == d_model_, ShapeMismatch,
              "MoELayer input last dim must match d_model (", d_model_, "), got ", x.dim(-1));

    // 1. Router forward pass
    RouterOutput router_out = router_->route(x);
    last_aux_loss_ = router_out.aux_loss;
    last_expert_counts_ = router_out.expert_counts;

    int r = x.rank();
    int64_t total_tokens = x.numel() / d_model_;
    if (total_tokens == 0) {
        return Tensor::zeros(x.shape(), x.dtype(), x.device());
    }

    // 2. Differentiable routing weight computation:
    //    logits = router.gate(x)
    //    probs = softmax(logits, -1)
    //    M = binary top-k mask [..., num_experts]
    //    masked_probs = probs * M
    //    normalized_gates = masked_probs / (sum(masked_probs, -1, keepdim=True) + 1e-9)
    Tensor logits = router_->gate()->forward(x);
    Tensor probs = softmax(logits, -1);

    // Build binary mask on host and transfer to target device
    Tensor mask = Tensor::empty(probs.shape(), DType::Float32, x.device());
    std::vector<float> mask_host(static_cast<size_t>(total_tokens * num_experts_), 0.0f);

    Tensor topk_cpu = router_out.expert_indices.cpu();
    const int64_t* topk_ptr = topk_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < total_tokens; ++i) {
        for (int64_t k = 0; k < top_k_; ++k) {
            int64_t exp_idx = topk_ptr[i * top_k_ + k];
            mask_host[static_cast<size_t>(i * num_experts_ + exp_idx)] = 1.0f;
        }
    }

    if (x.is_cpu()) {
        std::memcpy(mask.data_ptr<float>(), mask_host.data(), sizeof(float) * mask_host.size());
    } else {
#if AIC_WITH_CUDA
        cudaMemcpy(mask.data_ptr<float>(), mask_host.data(),
                   sizeof(float) * mask_host.size(), cudaMemcpyHostToDevice);
#endif
    }

    Tensor masked_probs = mul(probs, mask);
    Tensor norm_denom = add(sum(masked_probs, -1, /*keepdim=*/true), 1e-9f);
    Tensor normalized_gates = div(masked_probs, norm_denom); // shape: [..., num_experts]

    // 3. Sparse Dispatch: Only activate experts with non-zero token assignments
    std::vector<int64_t> u_dims(static_cast<size_t>(r), 1);
    u_dims.back() = num_experts_;
    Shape u_shape(std::move(u_dims));

    Tensor accum;
    bool accum_init = false;

    for (int64_t e = 0; e < num_experts_; ++e) {
        if (last_expert_counts_[static_cast<size_t>(e)] == 0) {
            // True conditional sparsity: inactive expert is completely skipped!
            continue;
        }

        // Construct 1-hot unit vector along last dimension
        Tensor u_e = Tensor::zeros(u_shape, DType::Float32, x.device());
        float one_val = 1.0f;
        if (x.is_cpu()) {
            u_e.data_ptr<float>()[e] = one_val;
        } else {
#if AIC_WITH_CUDA
            cudaMemcpy(u_e.data_ptr<float>() + e, &one_val, sizeof(float), cudaMemcpyHostToDevice);
#endif
        }

        // Extract column e of normalized_gates: [..., 1]
        Tensor w_e = sum(mul(normalized_gates, u_e), -1, /*keepdim=*/true);

        // Forward through active expert SwiGLU: [..., d_model]
        Tensor expert_out = experts_[static_cast<size_t>(e)]->forward(x);

        // Scale by routing weight: [..., d_model] * [..., 1] -> [..., d_model]
        Tensor weighted = mul(expert_out, w_e);

        if (!accum_init) {
            accum = weighted;
            accum_init = true;
        } else {
            accum = add(accum, weighted);
        }
    }

    if (!accum_init) {
        return Tensor::zeros(x.shape(), x.dtype(), x.device());
    }

    return accum;
}

double MoELayer::load_imbalance_ratio() const {
    if (last_expert_counts_.empty()) return 1.0;
    int64_t max_c = *std::max_element(last_expert_counts_.begin(), last_expert_counts_.end());
    double total = static_cast<double>(
        std::accumulate(last_expert_counts_.begin(), last_expert_counts_.end(), int64_t{0}));
    double avg = total / static_cast<double>(last_expert_counts_.size());
    if (avg <= 1e-9) return 1.0;
    return static_cast<double>(max_c) / avg;
}

// ============================================================================
// MoETransformerBlock Implementation
// ============================================================================

MoETransformerBlock::MoETransformerBlock(int64_t d_model, int64_t n_heads, int64_t d_hidden,
                                         int64_t num_experts, int64_t top_k,
                                         int64_t max_seq_len, float norm_eps,
                                         float aux_loss_coef, Device device)
    : d_model_(d_model) {
    AIC_CHECK(d_model_ > 0, InvalidArgument, "d_model must be positive");

    norm1_ = register_module("norm1", std::make_shared<RMSNorm>(d_model_, norm_eps, device));
    attn_  = register_module("attn",  std::make_shared<CausalSelfAttention>(d_model_, n_heads, max_seq_len, device));
    norm2_ = register_module("norm2", std::make_shared<RMSNorm>(d_model_, norm_eps, device));
    moe_   = register_module("moe",   std::make_shared<MoELayer>(d_model_, d_hidden, num_experts, top_k, aux_loss_coef, device));
}

Tensor MoETransformerBlock::forward_with_cache(const Tensor& x, KVCache* cache, int64_t start_pos) {
    // 1. Pre-norm Attention with residual connection: x1 = x + Attention(Norm1(x))
    Tensor norm1_out = norm1_->forward(x);
    Tensor attn_out = attn_->forward_with_cache(norm1_out, cache, start_pos);
    Tensor x1 = add(x, attn_out);

    // 2. Pre-norm MoE Feed-Forward with residual connection: x2 = x1 + MoE(Norm2(x1))
    Tensor norm2_out = norm2_->forward(x1);
    Tensor moe_out = moe_->forward(norm2_out);
    return add(x1, moe_out);
}

Tensor MoETransformerBlock::forward(const Tensor& x) {
    return forward_with_cache(x, nullptr, 0);
}

}  // namespace aic::nn
