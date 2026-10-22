// include/models/vision_language_lm.hpp
// Vision-Language Multimodal Foundation Model (VLM) Architecture (Aşama 23)
// Bridges VisionTransformer (ViT) and Decoder-Only TransformerLM via Multimodal Projector.
#pragma once

#include "models/transformer_lm.hpp"
#include "models/vision_transformer.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "ops/ops.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace aic::models {

class MultimodalProjector : public nn::Module {
public:
    MultimodalProjector(int64_t in_features, int64_t out_features, Device device = Device::cpu());

    Tensor forward(const Tensor& x) override;

    int64_t in_features() const noexcept { return in_features_; }
    int64_t out_features() const noexcept { return out_features_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    std::shared_ptr<nn::Linear> linear1_;
    std::shared_ptr<nn::Linear> linear2_;
};

struct VisionLanguageLMConfig {
    ViTConfig vit_config;
    TransformerLMConfig lm_config;
};

class VisionLanguageLM : public nn::Module {
public:
    explicit VisionLanguageLM(VisionLanguageLMConfig config, Device device = Device::cpu());

    // Encode images directly into LLM token embedding space [B, N_p, lm_hidden_dim]
    Tensor encode_images(const Tensor& images);

    // Multimodal forward pass:
    // images: [B, C, H, W] or [B, N_p, patch_dim]
    // text_input_ids: [B, S_text]
    // Output: logits [B, N_p + S_text, vocab_size]
    Tensor forward(const Tensor& images, const Tensor& text_input_ids);

    Tensor forward(const Tensor& text_input_ids) override {
        return language_model_->forward(text_input_ids);
    }

    // Multimodal autoregressive text generation given an image and prompt
    std::vector<int64_t> generate(
        const Tensor& image,
        const std::vector<int64_t>& prompt_tokens,
        int64_t max_new_tokens = 32,
        float temperature = 0.8f,
        int top_k = 20,
        float top_p = 0.9f,
        int64_t eos_token_id = -1
    );

    const VisionLanguageLMConfig& config() const noexcept { return config_; }
    std::shared_ptr<VisionTransformer> vision_encoder() const noexcept { return vision_encoder_; }
    std::shared_ptr<MultimodalProjector> mm_projector() const noexcept { return mm_projector_; }
    std::shared_ptr<TransformerLM> language_model() const noexcept { return language_model_; }
    Device device() const noexcept { return language_model_->device(); }

private:
    VisionLanguageLMConfig config_;
    std::shared_ptr<VisionTransformer> vision_encoder_;
    std::shared_ptr<MultimodalProjector> mm_projector_;
    std::shared_ptr<TransformerLM> language_model_;
};

} // namespace aic::models
