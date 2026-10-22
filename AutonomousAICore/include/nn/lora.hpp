// include/nn/lora.hpp
// Low-Rank Adaptation (LoRA) Parameter-Efficient Fine-Tuning (PEFT) Engine.
// Provides low-rank decomposed linear layers (W_0 + alpha/r * B * A),
// selective parameter freezing, zero-latency weight merging, and adapter serialization.
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "ops/matmul.hpp"
#include "ops/ops.hpp"

namespace aic::models {
class TransformerLM;
}

namespace aic::nn {

// Configuration for Low-Rank Adaptation
struct LoRAConfig {
    int64_t rank{8};
    float alpha{16.0f};
    float dropout{0.0f};
    std::vector<std::string> target_modules{"q_proj", "v_proj"};
};

// Low-Rank Adapted Linear Layer
// Decomposes weight updates as delta_W = (alpha / r) * (B @ A)
// where A in R^{r x d_in} is initialized with Kaiming uniform distribution
// and B in R^{d_out x r} is initialized to zero, ensuring delta_W = 0 at step 0.
class LoRALinear : public Linear {
public:
    // Standalone constructor
    LoRALinear(int64_t in_features, int64_t out_features, int64_t rank = 8, float alpha = 16.0f,
               bool has_bias = false, float dropout = 0.0f, Device device = Device::cpu());

    // Wrap an existing Linear layer, taking over its weight/bias, freezing them, and attaching adapter
    LoRALinear(std::shared_ptr<Linear> base, int64_t rank = 8, float alpha = 16.0f, float dropout = 0.0f);

    ~LoRALinear() override = default;

    // Forward pass: computes h = W_0(x) + (alpha / r) * ((x @ A^T) @ B^T)
    // If merged_ == true, executes pure W_eff(x) with zero additional latency.
    Tensor forward(const Tensor& x) override;

    // Fuses delta_W into base weight matrix: W_eff = W_0 + (alpha / r) * (B @ A)
    void merge();

    // Subtracts delta_W from base weight matrix, restoring original decoupled weights
    void unmerge();

    bool is_merged() const noexcept { return merged_; }
    int64_t rank() const noexcept { return rank_; }
    float alpha() const noexcept { return alpha_; }
    float scaling() const noexcept { return scaling_; }
    float dropout() const noexcept { return dropout_; }

    std::shared_ptr<Tensor> lora_A() const noexcept { return lora_A_; }
    std::shared_ptr<Tensor> lora_B() const noexcept { return lora_B_; }
    std::shared_ptr<Linear> base_linear() const noexcept { return base_linear_; }

private:
    void init_adapter_weights(Device device);

    int64_t rank_{8};
    float alpha_{16.0f};
    float scaling_{2.0f};
    float dropout_{0.0f};
    bool merged_{false};

    std::shared_ptr<Linear> base_linear_;
    std::shared_ptr<Tensor> lora_A_;
    std::shared_ptr<Tensor> lora_B_;
};

// Model-Wide LoRA Adapter Manager
// Handles selective freezing, automatic layer replacement in TransformerLM,
// parameter inspection, unified merging/unmerging, and compact adapter serialization.
class LoRAAdapter {
public:
    LoRAAdapter(std::shared_ptr<models::TransformerLM> model, LoRAConfig config);

    // Freezes all base model parameters and injects LoRALinear into specified target modules
    void apply();

    // Parameter inspection
    int64_t num_trainable_parameters() const;
    int64_t num_total_parameters() const;
    float trainable_percentage() const;

    // Retrieves only the trainable adapter parameters (A and B matrices) for the optimizer
    std::vector<std::shared_ptr<Tensor>> trainable_parameters() const;

    // Fuses all adapters into base weights across the entire model for zero-overhead inference
    void merge_all();

    // Unmerges all adapters across the entire model to resume fine-tuning
    void unmerge_all();

    bool is_merged() const;

    // Saves only the low-rank adapter weights (.aic_lora format) with Adler-32 checksums
    void save_adapter(const std::string& filepath) const;

    // Loads low-rank adapter weights (.aic_lora format) and verifies data integrity
    void load_adapter(const std::string& filepath);

    const std::vector<std::shared_ptr<LoRALinear>>& lora_layers() const noexcept { return lora_layers_; }
    const std::vector<std::string>& layer_names() const noexcept { return layer_names_; }
    const LoRAConfig& config() const noexcept { return config_; }
    std::shared_ptr<models::TransformerLM> model() const noexcept { return model_; }

private:
    std::shared_ptr<models::TransformerLM> model_;
    LoRAConfig config_;
    std::vector<std::shared_ptr<LoRALinear>> lora_layers_;
    std::vector<std::string> layer_names_;
};

}  // namespace aic::nn
