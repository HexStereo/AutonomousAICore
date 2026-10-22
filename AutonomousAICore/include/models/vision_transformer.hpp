// include/models/vision_transformer.hpp
// Vision Transformer (ViT) Encoder with Bidirectional Attention and Patch Embedding (Aşama 23)
#pragma once

#include "data/image.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/rms_norm.hpp"
#include "nn/swiglu.hpp"
#include "ops/matmul.hpp"
#include "ops/ops.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace aic::models {

struct ViTConfig {
    int64_t image_size = 224;
    int64_t patch_size = 16;
    int64_t in_channels = 3;
    int64_t hidden_dim = 256;
    int64_t num_layers = 4;
    int64_t num_heads = 4;
    int64_t intermediate_dim = 512;
    float norm_eps = 1e-6f;

    int64_t num_patches() const noexcept {
        return (image_size / patch_size) * (image_size / patch_size);
    }

    int64_t patch_dim() const noexcept {
        return in_channels * patch_size * patch_size;
    }
};

class PatchEmbedder : public nn::Module {
public:
    PatchEmbedder(const ViTConfig& config, Device device = Device::cpu());

    Tensor forward(const Tensor& images) override;

    int64_t patch_size() const noexcept { return patch_size_; }
    int64_t num_patches() const noexcept { return num_patches_; }

private:
    int64_t patch_size_;
    int64_t num_patches_;
    std::shared_ptr<nn::Linear> proj_;
    std::shared_ptr<Tensor> pos_embed_;
};

class VisionAttention : public nn::Module {
public:
    VisionAttention(int64_t d_model, int64_t n_heads, Device device = Device::cpu());

    Tensor forward(const Tensor& x) override;

private:
    int64_t d_model_;
    int64_t n_heads_;
    int64_t head_dim_;
    std::shared_ptr<nn::Linear> q_proj_;
    std::shared_ptr<nn::Linear> k_proj_;
    std::shared_ptr<nn::Linear> v_proj_;
    std::shared_ptr<nn::Linear> out_proj_;
};

class VisionTransformerBlock : public nn::Module {
public:
    VisionTransformerBlock(
        int64_t d_model,
        int64_t n_heads,
        int64_t d_intermediate,
        float norm_eps = 1e-6f,
        Device device = Device::cpu()
    );

    Tensor forward(const Tensor& x) override;

private:
    std::shared_ptr<nn::RMSNorm> norm1_;
    std::shared_ptr<VisionAttention> attn_;
    std::shared_ptr<nn::RMSNorm> norm2_;
    std::shared_ptr<nn::SwiGLU> ffn_;
};

class VisionTransformer : public nn::Module {
public:
    explicit VisionTransformer(ViTConfig config, Device device = Device::cpu());

    // Input: images tensor [B, C, H, W] or pre-patchified [B, N_p, patch_dim]
    // Output: contextualized visual patch tokens [B, N_p, hidden_dim]
    Tensor forward(const Tensor& images) override;

    const ViTConfig& config() const noexcept { return config_; }
    int64_t num_patches() const noexcept { return config_.num_patches(); }
    int64_t hidden_dim() const noexcept { return config_.hidden_dim; }

private:
    ViTConfig config_;
    std::shared_ptr<PatchEmbedder> patch_embed_;
    std::vector<std::shared_ptr<VisionTransformerBlock>> blocks_;
    std::shared_ptr<nn::RMSNorm> norm_;
};

} // namespace aic::models
