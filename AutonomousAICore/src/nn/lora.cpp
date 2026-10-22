// src/nn/lora.cpp
// Low-Rank Adaptation (LoRA) implementation.
#include "nn/lora.hpp"
#include "models/transformer_lm.hpp"
#include "core/error.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace aic::nn {

namespace {

uint32_t compute_adler32(const uint8_t* data, size_t len) {
    const uint32_t MOD_ADLER = 65521;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    return (b << 16) | a;
}

constexpr uint32_t AIC_LORA_MAGIC = 0x4C4F5241; // 'LORA'
constexpr uint32_t AIC_LORA_VERSION = 1;

}  // namespace

void LoRALinear::init_adapter_weights(Device device) {
    AIC_CHECK(rank_ > 0, InvalidArgument, "LoRA rank must be positive, got ", rank_);
    scaling_ = alpha_ / static_cast<float>(rank_);

    // Kaiming uniform bound for A: 1 / sqrt(in_features)
    float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
    Tensor a = Tensor::uniform(Shape({rank_, in_features_}), -bound, bound, DType::Float32, device);
    lora_A_ = register_parameter("lora_A", std::move(a));

    // B initialized to strictly zero so delta_W = 0 at start
    Tensor b = Tensor::zeros(Shape({out_features_, rank_}), DType::Float32, device);
    lora_B_ = register_parameter("lora_B", std::move(b));
}

LoRALinear::LoRALinear(int64_t in_features, int64_t out_features, int64_t rank, float alpha,
                       bool has_bias, float dropout, Device device)
    : Linear(in_features, out_features, has_bias, device),
      rank_(rank), alpha_(alpha), dropout_(dropout), merged_(false) {
    // In standalone LoRA, freeze base weights so only adapter matrices receive gradients
    weight_->set_requires_grad(false);
    if (has_bias_ && bias_) {
        bias_->set_requires_grad(false);
    }
    init_adapter_weights(device);
}

LoRALinear::LoRALinear(std::shared_ptr<Linear> base, int64_t rank, float alpha, float dropout)
    : Linear(base->in_features(), base->out_features(), base->has_bias(), base->weight()->device()),
      rank_(rank), alpha_(alpha), dropout_(dropout), merged_(false), base_linear_(base) {
    // Share base's existing pre-trained weight and bias
    weight_ = base->weight();
    bias_ = base->bias();
    parameters_["weight"] = weight_;
    if (bias_) {
        parameters_["bias"] = bias_;
    }
    // Freeze base parameters
    weight_->set_requires_grad(false);
    if (bias_) {
        bias_->set_requires_grad(false);
    }
    init_adapter_weights(weight_->device());
}

Tensor LoRALinear::forward(const Tensor& x) {
    if (merged_) {
        // Zero-latency merged inference
        return Linear::forward(x);
    }

    // Base branch: h_base = x @ W_0^T + b
    Tensor base_out = Linear::forward(x);

    // LoRA low-rank branch: h_lora = (alpha / r) * ((x @ A^T) @ B^T)
    Tensor lora_a_t = lora_A_->transpose(0, 1);
    Tensor x_a = matmul(x, lora_a_t);

    Tensor lora_b_t = lora_B_->transpose(0, 1);
    Tensor lora_out = matmul(x_a, lora_b_t);

    Tensor scaled_lora = lora_out * scaling_;
    return base_out + scaled_lora;
}

void LoRALinear::merge() {
    if (merged_) return;
    // delta_W = scaling * (B @ A) -> [out_features, in_features]
    Tensor delta_w = matmul(*lora_B_, *lora_A_) * scaling_;
    *weight_ = *weight_ + delta_w;
    merged_ = true;
}

void LoRALinear::unmerge() {
    if (!merged_) return;
    Tensor delta_w = matmul(*lora_B_, *lora_A_) * scaling_;
    *weight_ = *weight_ - delta_w;
    merged_ = false;
}

LoRAAdapter::LoRAAdapter(std::shared_ptr<models::TransformerLM> model, LoRAConfig config)
    : model_(std::move(model)), config_(std::move(config)) {
    AIC_CHECK(model_ != nullptr, InvalidArgument, "Model cannot be null");
    apply();
}

void LoRAAdapter::apply() {
    AIC_CHECK(model_ != nullptr, InvalidArgument, "Model cannot be null");
    lora_layers_.clear();
    layer_names_.clear();

    // 1. Freeze ALL base parameters
    for (auto& p : model_->parameters()) {
        if (p) {
            p->set_requires_grad(false);
        }
    }

    // 2. Iterate over all transformer layers
    const auto& layers = model_->layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        auto block = layers[i];
        auto attn = block->attn();
        auto ffn = block->ffn();
        std::string layer_prefix = "layers." + std::to_string(i);

        for (const auto& target : config_.target_modules) {
            if (target == "q_proj") {
                auto lora = std::make_shared<LoRALinear>(attn->q_proj(), config_.rank, config_.alpha, config_.dropout);
                attn->set_q_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".attn.q_proj");
            } else if (target == "k_proj") {
                auto lora = std::make_shared<LoRALinear>(attn->k_proj(), config_.rank, config_.alpha, config_.dropout);
                attn->set_k_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".attn.k_proj");
            } else if (target == "v_proj") {
                auto lora = std::make_shared<LoRALinear>(attn->v_proj(), config_.rank, config_.alpha, config_.dropout);
                attn->set_v_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".attn.v_proj");
            } else if (target == "out_proj") {
                auto lora = std::make_shared<LoRALinear>(attn->out_proj(), config_.rank, config_.alpha, config_.dropout);
                attn->set_out_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".attn.out_proj");
            } else if (target == "gate_proj") {
                auto lora = std::make_shared<LoRALinear>(ffn->gate_proj(), config_.rank, config_.alpha, config_.dropout);
                ffn->set_gate_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".ffn.gate_proj");
            } else if (target == "up_proj") {
                auto lora = std::make_shared<LoRALinear>(ffn->up_proj(), config_.rank, config_.alpha, config_.dropout);
                ffn->set_up_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".ffn.up_proj");
            } else if (target == "down_proj") {
                auto lora = std::make_shared<LoRALinear>(ffn->down_proj(), config_.rank, config_.alpha, config_.dropout);
                ffn->set_down_proj(lora);
                lora_layers_.push_back(lora);
                layer_names_.push_back(layer_prefix + ".ffn.down_proj");
            }
        }
    }
}

int64_t LoRAAdapter::num_trainable_parameters() const {
    int64_t total = 0;
    for (const auto& layer : lora_layers_) {
        if (layer->lora_A()) total += layer->lora_A()->numel();
        if (layer->lora_B()) total += layer->lora_B()->numel();
    }
    return total;
}

int64_t LoRAAdapter::num_total_parameters() const {
    int64_t total = 0;
    for (const auto& p : model_->parameters()) {
        if (p) total += p->numel();
    }
    return total;
}

float LoRAAdapter::trainable_percentage() const {
    int64_t total = num_total_parameters();
    if (total == 0) return 0.0f;
    return (static_cast<float>(num_trainable_parameters()) / static_cast<float>(total)) * 100.0f;
}

std::vector<std::shared_ptr<Tensor>> LoRAAdapter::trainable_parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& layer : lora_layers_) {
        params.push_back(layer->lora_A());
        params.push_back(layer->lora_B());
    }
    return params;
}

void LoRAAdapter::merge_all() {
    for (auto& layer : lora_layers_) {
        layer->merge();
    }
}

void LoRAAdapter::unmerge_all() {
    for (auto& layer : lora_layers_) {
        layer->unmerge();
    }
}

bool LoRAAdapter::is_merged() const {
    if (lora_layers_.empty()) return false;
    return lora_layers_[0]->is_merged();
}

void LoRAAdapter::save_adapter(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    AIC_CHECK(out.is_open(), IOError, "Failed to open file for saving LoRA adapter: ", filepath);

    uint32_t magic = AIC_LORA_MAGIC;
    uint32_t version = AIC_LORA_VERSION;
    int64_t rank = config_.rank;
    float alpha = config_.alpha;
    uint32_t num_layers = static_cast<uint32_t>(lora_layers_.size());

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    out.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (size_t i = 0; i < lora_layers_.size(); ++i) {
        const std::string& name = layer_names_[i];
        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(name.data(), name_len);

        const auto& layer = lora_layers_[i];
        Tensor a = layer->lora_A()->cpu().contiguous();
        Tensor b = layer->lora_B()->cpu().contiguous();

        int64_t a_numel = a.numel();
        int64_t b_numel = b.numel();
        out.write(reinterpret_cast<const char*>(&a_numel), sizeof(a_numel));
        out.write(reinterpret_cast<const char*>(a.data_ptr<float>()), a_numel * sizeof(float));

        out.write(reinterpret_cast<const char*>(&b_numel), sizeof(b_numel));
        out.write(reinterpret_cast<const char*>(b.data_ptr<float>()), b_numel * sizeof(float));

        uint32_t chk_a = compute_adler32(reinterpret_cast<const uint8_t*>(a.data_ptr<float>()), a_numel * sizeof(float));
        uint32_t chk_b = compute_adler32(reinterpret_cast<const uint8_t*>(b.data_ptr<float>()), b_numel * sizeof(float));
        out.write(reinterpret_cast<const char*>(&chk_a), sizeof(chk_a));
        out.write(reinterpret_cast<const char*>(&chk_b), sizeof(chk_b));
    }
}

void LoRAAdapter::load_adapter(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    AIC_CHECK(in.is_open(), IOError, "Failed to open file for loading LoRA adapter: ", filepath);

    uint32_t magic = 0, version = 0;
    int64_t rank = 0;
    float alpha = 0.0f;
    uint32_t num_layers = 0;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    in.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    AIC_CHECK(magic == AIC_LORA_MAGIC, FormatError, "Invalid LoRA magic header");
    AIC_CHECK(version == AIC_LORA_VERSION, FormatError, "Unsupported LoRA version: ", version);
    AIC_CHECK(rank == config_.rank, InvalidArgument, "Rank mismatch in loaded adapter: ", rank, " vs configured ", config_.rank);
    AIC_CHECK(num_layers == lora_layers_.size(), InvalidArgument, "Layer count mismatch: ", num_layers, " vs ", lora_layers_.size());

    for (size_t i = 0; i < num_layers; ++i) {
        uint32_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        in.read(name.data(), name_len);

        int64_t a_numel = 0;
        in.read(reinterpret_cast<char*>(&a_numel), sizeof(a_numel));

        auto layer = lora_layers_[i];
        AIC_CHECK(a_numel == layer->lora_A()->numel(), ShapeMismatch, "lora_A shape mismatch in layer ", name);

        Tensor a_tensor = Tensor::empty(layer->lora_A()->shape(), DType::Float32, Device::cpu());
        in.read(reinterpret_cast<char*>(a_tensor.data_ptr<float>()), a_numel * sizeof(float));

        int64_t b_numel = 0;
        in.read(reinterpret_cast<char*>(&b_numel), sizeof(b_numel));
        AIC_CHECK(b_numel == layer->lora_B()->numel(), ShapeMismatch, "lora_B shape mismatch in layer ", name);

        Tensor b_tensor = Tensor::empty(layer->lora_B()->shape(), DType::Float32, Device::cpu());
        in.read(reinterpret_cast<char*>(b_tensor.data_ptr<float>()), b_numel * sizeof(float));

        uint32_t expected_chk_a = 0, expected_chk_b = 0;
        in.read(reinterpret_cast<char*>(&expected_chk_a), sizeof(expected_chk_a));
        in.read(reinterpret_cast<char*>(&expected_chk_b), sizeof(expected_chk_b));

        uint32_t actual_chk_a = compute_adler32(reinterpret_cast<const uint8_t*>(a_tensor.data_ptr<float>()), a_numel * sizeof(float));
        uint32_t actual_chk_b = compute_adler32(reinterpret_cast<const uint8_t*>(b_tensor.data_ptr<float>()), b_numel * sizeof(float));

        AIC_CHECK(actual_chk_a == expected_chk_a, FormatError, "Adler-32 checksum mismatch for lora_A in layer ", name);
        AIC_CHECK(actual_chk_b == expected_chk_b, FormatError, "Adler-32 checksum mismatch for lora_B in layer ", name);

        *layer->lora_A() = a_tensor.to(layer->lora_A()->device());
        layer->lora_A()->set_requires_grad(true);

        *layer->lora_B() = b_tensor.to(layer->lora_B()->device());
        layer->lora_B()->set_requires_grad(true);
    }
}


}  // namespace aic::nn
