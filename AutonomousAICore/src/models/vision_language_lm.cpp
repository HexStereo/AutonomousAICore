// src/models/vision_language_lm.cpp
// Implementation of Vision-Language Multimodal Model (VLM) Architecture (Aşama 23)
#include "models/vision_language_lm.hpp"
#include "core/error.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace aic::models {

// ----------------------------------------------------------------------------
// MultimodalProjector Implementation
// ----------------------------------------------------------------------------
MultimodalProjector::MultimodalProjector(int64_t in_features, int64_t out_features, Device device)
    : in_features_(in_features), out_features_(out_features) {
    AIC_CHECK(in_features > 0 && out_features > 0, InvalidArgument, "Projector features must be positive");

    linear1_ = register_module("linear1", std::make_shared<nn::Linear>(in_features, out_features, true, device));
    linear2_ = register_module("linear2", std::make_shared<nn::Linear>(out_features, out_features, true, device));
}

Tensor MultimodalProjector::forward(const Tensor& x) {
    Tensor h = linear1_->forward(x);
    h = gelu(h);
    return linear2_->forward(h);
}

// ----------------------------------------------------------------------------
// VisionLanguageLM Implementation
// ----------------------------------------------------------------------------
VisionLanguageLM::VisionLanguageLM(VisionLanguageLMConfig config, Device device)
    : config_(config) {
    vision_encoder_ = register_module(
        "vision_encoder",
        std::make_shared<VisionTransformer>(config_.vit_config, device)
    );

    mm_projector_ = register_module(
        "mm_projector",
        std::make_shared<MultimodalProjector>(
            config_.vit_config.hidden_dim,
            config_.lm_config.hidden_dim,
            device
        )
    );

    language_model_ = register_module(
        "language_model",
        std::make_shared<TransformerLM>(config_.lm_config, device)
    );
}

Tensor VisionLanguageLM::encode_images(const Tensor& images) {
    // 1. Run ViT vision encoder to obtain spatial patch tokens [B, N_p, vit_hidden_dim]
    Tensor vit_tokens = vision_encoder_->forward(images);

    // 2. Project visual tokens into LLM embedding space [B, N_p, lm_hidden_dim]
    return mm_projector_->forward(vit_tokens);
}

Tensor VisionLanguageLM::forward(const Tensor& images, const Tensor& text_input_ids) {
    AIC_CHECK(text_input_ids.rank() == 2, InvalidArgument,
              "text_input_ids must be 2D [B, S_text], got rank ", text_input_ids.rank());

    int64_t B = text_input_ids.dim(0);

    // 1. Project images to visual embeddings: [B, N_p, lm_hidden_dim]
    Tensor visual_embeds = encode_images(images);
    AIC_CHECK(visual_embeds.dim(0) == B, ShapeMismatch,
              "Batch size mismatch between images (", visual_embeds.dim(0), ") and text (", B, ")");

    // 2. Lookup text token embeddings: [B, S_text, lm_hidden_dim]
    Tensor text_embeds = language_model_->tok_embeddings()->forward(text_input_ids);

    // 3. Multimodal prefix concatenation along sequence dimension (dim=1):
    // [B, N_p + S_text, lm_hidden_dim]
    Tensor fused_embeds = cat({visual_embeds, text_embeds}, 1);

    // 4. Forward dense multimodal embeddings through the autoregressive Transformer blocks
    return language_model_->forward_embeddings(fused_embeds);
}

std::vector<int64_t> VisionLanguageLM::generate(
    const Tensor& image,
    const std::vector<int64_t>& prompt_tokens,
    int64_t max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    int64_t eos_token_id
) {
    Device dev = device();
    Tensor img = (image.rank() == 3) ? image.reshape(Shape({1, image.dim(0), image.dim(1), image.dim(2)})) : image;
    img = img.to(dev);

    // 1. Encode visual features into LLM space: [1, N_p, lm_hidden_dim]
    Tensor visual_embeds = encode_images(img);

    Tensor fused_embeds;
    if (!prompt_tokens.empty()) {
        Tensor prompt_t = Tensor::from_blob(
            const_cast<int64_t*>(prompt_tokens.data()),
            Shape({1, static_cast<int64_t>(prompt_tokens.size())}),
            DType::Int64,
            Device::cpu()
        ).to(dev);

        Tensor text_embeds = language_model_->tok_embeddings()->forward(prompt_t);
        fused_embeds = cat({visual_embeds, text_embeds}, 1);
    } else {
        fused_embeds = visual_embeds;
    }

    int64_t total_prefill = fused_embeds.dim(1);
    std::vector<nn::KVCache> kv_caches(language_model_->layers().size());

    // 2. Prefill step: process visual and text prefix in parallel
    Tensor logits = language_model_->forward_embeddings_step(fused_embeds, 0, kv_caches);

    int64_t V = logits.dim(-1);
    Tensor last_logits = logits.slice(1, total_prefill - 1, total_prefill).squeeze(1).squeeze(0);

    std::vector<int64_t> output_tokens;
    output_tokens.reserve(static_cast<size_t>(max_new_tokens));

    std::mt19937_64 rng(42);

    auto sample_token = [&](const Tensor& logit_vec) -> int64_t {
        Tensor cpu_l = logit_vec.cpu().contiguous();
        const float* p_logits = cpu_l.data_ptr<float>();

        if (temperature <= 0.001f || top_k == 1) {
            return static_cast<int64_t>(std::max_element(p_logits, p_logits + V) - p_logits);
        }

        std::vector<float> scaled_logits(static_cast<size_t>(V));
        float max_l = p_logits[0] / temperature;
        for (int64_t v = 0; v < V; ++v) {
            float val = p_logits[v] / temperature;
            scaled_logits[static_cast<size_t>(v)] = val;
            if (val > max_l) max_l = val;
        }

        std::vector<float> probs(static_cast<size_t>(V));
        double sum_p = 0.0;
        for (int64_t v = 0; v < V; ++v) {
            float p = std::exp(scaled_logits[static_cast<size_t>(v)] - max_l);
            probs[static_cast<size_t>(v)] = p;
            sum_p += p;
        }
        for (auto& p : probs) {
            p /= static_cast<float>(sum_p);
        }

        if (top_k > 0 && top_k < V) {
            std::vector<int64_t> indices(static_cast<size_t>(V));
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                [&probs](int64_t a, int64_t b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; }
            );

            std::vector<float> filtered(static_cast<size_t>(V), 0.0f);
            double fsum = 0.0;
            for (int i = 0; i < top_k; ++i) {
                filtered[static_cast<size_t>(indices[static_cast<size_t>(i)])] = probs[static_cast<size_t>(indices[static_cast<size_t>(i)])];
                fsum += probs[static_cast<size_t>(indices[static_cast<size_t>(i)])];
            }
            for (int i = 0; i < top_k; ++i) {
                filtered[static_cast<size_t>(indices[static_cast<size_t>(i)])] /= static_cast<float>(fsum);
            }
            probs = std::move(filtered);
        }

        if (top_p > 0.0f && top_p < 1.0f) {
            std::vector<std::pair<float, int64_t>> p_idx;
            p_idx.reserve(static_cast<size_t>(V));
            for (int64_t v = 0; v < V; ++v) {
                if (probs[static_cast<size_t>(v)] > 0.0f) {
                    p_idx.push_back({probs[static_cast<size_t>(v)], v});
                }
            }
            std::sort(p_idx.begin(), p_idx.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );

            double cumsum = 0.0;
            size_t cutoff = p_idx.size();
            for (size_t i = 0; i < p_idx.size(); ++i) {
                cumsum += p_idx[i].first;
                if (cumsum >= top_p) {
                    cutoff = i + 1;
                    break;
                }
            }

            std::vector<float> nucleus_probs(static_cast<size_t>(V), 0.0f);
            double nsum = 0.0;
            for (size_t i = 0; i < cutoff; ++i) {
                nucleus_probs[static_cast<size_t>(p_idx[i].second)] = p_idx[i].first;
                nsum += p_idx[i].first;
            }
            if (nsum > 0.0) {
                for (size_t i = 0; i < cutoff; ++i) {
                    nucleus_probs[static_cast<size_t>(p_idx[i].second)] /= static_cast<float>(nsum);
                }
                probs = std::move(nucleus_probs);
            }
        }

        std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
        return dist(rng);
    };

    int64_t next_token = sample_token(last_logits);
    output_tokens.push_back(next_token);

    if (eos_token_id >= 0 && next_token == eos_token_id) {
        return output_tokens;
    }

    // 3. Autoregressive decode loop
    for (int64_t step = 1; step < max_new_tokens; ++step) {
        int64_t pos = total_prefill + step - 1;
        Tensor in_tok = Tensor::from_blob(&next_token, Shape({1, 1}), DType::Int64, Device::cpu()).to(dev);
        Tensor step_logits = language_model_->forward_step(in_tok, pos, kv_caches);
        Tensor cur_logits = step_logits.squeeze(1).squeeze(0);

        next_token = sample_token(cur_logits);
        output_tokens.push_back(next_token);

        if (eos_token_id >= 0 && next_token == eos_token_id) {
            break;
        }
    }

    return output_tokens;
}

} // namespace aic::models
