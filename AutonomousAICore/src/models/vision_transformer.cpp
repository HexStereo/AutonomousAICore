// src/models/vision_transformer.cpp
// Implementation of Vision Transformer (ViT) Encoder with Bidirectional Attention (Aşama 23)
#include "models/vision_transformer.hpp"
#include "core/error.hpp"

namespace aic::models {

// ----------------------------------------------------------------------------
// PatchEmbedder Implementation
// ----------------------------------------------------------------------------
PatchEmbedder::PatchEmbedder(const ViTConfig& config, Device device)
    : patch_size_(config.patch_size), num_patches_(config.num_patches()) {
    AIC_CHECK(patch_size_ > 0, InvalidArgument, "patch_size must be positive");
    AIC_CHECK(num_patches_ > 0, InvalidArgument, "num_patches must be positive");

    proj_ = register_module(
        "proj",
        std::make_shared<nn::Linear>(config.patch_dim(), config.hidden_dim, /*has_bias=*/true, device)
    );

    Tensor pos = Tensor::uniform(
        Shape({1, num_patches_, config.hidden_dim}),
        -0.02f,
        0.02f,
        DType::Float32,
        device
    );
    pos_embed_ = register_parameter("pos_embed", std::move(pos));
}

Tensor PatchEmbedder::forward(const Tensor& images) {
    Tensor patches;
    if (images.rank() == 4) {
        patches = data::Image::batch_patchify(images, static_cast<int>(patch_size_));
    } else if (images.rank() == 3) {
        patches = images;
    } else {
        AIC_THROW(ShapeMismatch, "PatchEmbedder expects 4D images [B, C, H, W] or 3D patches [B, N_p, D_p]");
    }

    Tensor x = proj_->forward(patches);
    return add(x, *pos_embed_);
}

// ----------------------------------------------------------------------------
// VisionAttention Implementation
// ----------------------------------------------------------------------------
VisionAttention::VisionAttention(int64_t d_model, int64_t n_heads, Device device)
    : d_model_(d_model), n_heads_(n_heads), head_dim_(d_model / n_heads) {
    AIC_CHECK(d_model % n_heads == 0, InvalidArgument, "d_model must be divisible by n_heads");

    q_proj_   = register_module("q_proj",   std::make_shared<nn::Linear>(d_model, d_model, /*has_bias=*/false, device));
    k_proj_   = register_module("k_proj",   std::make_shared<nn::Linear>(d_model, d_model, /*has_bias=*/false, device));
    v_proj_   = register_module("v_proj",   std::make_shared<nn::Linear>(d_model, d_model, /*has_bias=*/false, device));
    out_proj_ = register_module("out_proj", std::make_shared<nn::Linear>(d_model, d_model, /*has_bias=*/false, device));
}

Tensor VisionAttention::forward(const Tensor& x) {
    AIC_CHECK(x.rank() == 3, InvalidArgument, "VisionAttention expects 3D tensor [B, S, D], got rank ", x.rank());
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);

    Tensor q = q_proj_->forward(x).reshape(Shape({B, S, n_heads_, head_dim_})).transpose(1, 2);
    Tensor k = k_proj_->forward(x).reshape(Shape({B, S, n_heads_, head_dim_})).transpose(1, 2);
    Tensor v = v_proj_->forward(x).reshape(Shape({B, S, n_heads_, head_dim_})).transpose(1, 2);

    Tensor k_t = k.transpose(2, 3);
    Tensor scores = matmul(q, k_t);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    scores = mul(scores, scale);

    // Bidirectional spatial attention: full softmax over all patches without causal masking
    Tensor attn_weights = softmax(scores, -1);
    Tensor context = matmul(attn_weights, v).transpose(1, 2).reshape(Shape({B, S, d_model_}));

    return out_proj_->forward(context);
}

// ----------------------------------------------------------------------------
// VisionTransformerBlock Implementation
// ----------------------------------------------------------------------------
VisionTransformerBlock::VisionTransformerBlock(
    int64_t d_model,
    int64_t n_heads,
    int64_t d_intermediate,
    float norm_eps,
    Device device
) {
    norm1_ = register_module("norm1", std::make_shared<nn::RMSNorm>(d_model, norm_eps, device));
    attn_  = register_module("attn",  std::make_shared<VisionAttention>(d_model, n_heads, device));
    norm2_ = register_module("norm2", std::make_shared<nn::RMSNorm>(d_model, norm_eps, device));
    ffn_   = register_module("ffn",   std::make_shared<nn::SwiGLU>(d_model, d_intermediate, device));
}

Tensor VisionTransformerBlock::forward(const Tensor& x) {
    Tensor norm1_out = norm1_->forward(x);
    Tensor attn_out = attn_->forward(norm1_out);
    Tensor x1 = add(x, attn_out);

    Tensor norm2_out = norm2_->forward(x1);
    Tensor ffn_out = ffn_->forward(norm2_out);
    return add(x1, ffn_out);
}

// ----------------------------------------------------------------------------
// VisionTransformer Implementation
// ----------------------------------------------------------------------------
VisionTransformer::VisionTransformer(ViTConfig config, Device device)
    : config_(config) {
    patch_embed_ = register_module("patch_embed", std::make_shared<PatchEmbedder>(config_, device));

    for (int64_t i = 0; i < config_.num_layers; ++i) {
        auto block = std::make_shared<VisionTransformerBlock>(
            config_.hidden_dim,
            config_.num_heads,
            config_.intermediate_dim,
            config_.norm_eps,
            device
        );
        blocks_.push_back(register_module("blocks." + std::to_string(i), block));
    }

    norm_ = register_module(
        "norm",
        std::make_shared<nn::RMSNorm>(config_.hidden_dim, config_.norm_eps, device)
    );
}

Tensor VisionTransformer::forward(const Tensor& images) {
    Tensor x = patch_embed_->forward(images);
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    return norm_->forward(x);
}

} // namespace aic::models
