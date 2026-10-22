// tests/unit/test_lora.cpp
// Unit tests for Low-Rank Adaptation (LoRA) and PEFT Engine.
#include "core/device.hpp"
#include "loss/cross_entropy.hpp"
#include "loss/mse.hpp"
#include "models/transformer_lm.hpp"
#include "nn/linear.hpp"
#include "nn/lora.hpp"
#include "ops/ops.hpp"
#include "optim/adamw.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

using namespace aic;
using namespace aic::nn;
using namespace aic::models;
using namespace aic::optim;

AIC_TEST(LoRATest, InitializationAndZeroDelta) {
    int64_t in_features = 64;
    int64_t out_features = 128;
    int64_t rank = 8;
    float alpha = 16.0f;

    auto base = std::make_shared<Linear>(in_features, out_features, /*has_bias=*/true);
    LoRALinear lora(base, rank, alpha);

    ASSERT_EQ(lora.in_features(), in_features);
    ASSERT_EQ(lora.out_features(), out_features);
    ASSERT_EQ(lora.rank(), rank);
    ASSERT_NEAR(lora.alpha(), alpha, 1e-5f);
    ASSERT_NEAR(lora.scaling(), alpha / static_cast<float>(rank), 1e-5f);
    ASSERT_FALSE(lora.is_merged());

    // A shape: [rank, in_features]
    ASSERT_TRUE(lora.lora_A()->shape() == Shape({rank, in_features}));
    ASSERT_TRUE(lora.lora_A()->requires_grad());

    // B shape: [out_features, rank], strictly all zeros
    ASSERT_TRUE(lora.lora_B()->shape() == Shape({out_features, rank}));
    ASSERT_TRUE(lora.lora_B()->requires_grad());

    const float* b_data = lora.lora_B()->data_ptr<float>();
    int64_t b_numel = lora.lora_B()->numel();
    for (int64_t i = 0; i < b_numel; ++i) {
        ASSERT_NEAR(b_data[i], 0.0f, 1e-7f);
    }

    // Base weight and bias must be frozen
    ASSERT_FALSE(lora.weight()->requires_grad());
    ASSERT_FALSE(lora.bias()->requires_grad());

    // Forward pass must match base model bit-for-bit at step 0
    Tensor x = Tensor::randn(Shape({2, 16, in_features}));
    Tensor y_base = base->forward(x);
    Tensor y_lora = lora.forward(x);

    ASSERT_TRUE(y_base.shape() == y_lora.shape());
    const float* base_ptr = y_base.data_ptr<float>();
    const float* lora_ptr = y_lora.data_ptr<float>();
    int64_t numel = y_base.numel();
    for (int64_t i = 0; i < numel; ++i) {
        ASSERT_NEAR(base_ptr[i], lora_ptr[i], 1e-5f);
    }
}

AIC_TEST(LoRATest, AutogradGradientsAndParameterFreezing) {
    int64_t in_features = 32;
    int64_t out_features = 32;
    int64_t rank = 4;
    float alpha = 8.0f;

    auto base = std::make_shared<Linear>(in_features, out_features, /*has_bias=*/false);
    LoRALinear lora(base, rank, alpha);

    Tensor x = Tensor::randn(Shape({4, in_features}));
    Tensor y = lora.forward(x);

    // Compute simple loss: mean(y^2)
    Tensor loss = mean(mul(y, y));
    loss.backward();

    // Base weight must have NO gradient accumulated (strictly frozen)
    ASSERT_FALSE(lora.weight()->grad().defined());

    // LoRA A and B must receive non-zero gradients
    ASSERT_TRUE(lora.lora_A()->grad().defined());
    ASSERT_TRUE(lora.lora_B()->grad().defined());
    ASSERT_TRUE(lora.lora_A()->grad().numel() > 0);
    ASSERT_TRUE(lora.lora_B()->grad().numel() > 0);

    // Gradients should be finite and non-NaN
    const float* g_a = lora.lora_A()->grad().data_ptr<float>();
    for (int64_t i = 0; i < lora.lora_A()->numel(); ++i) {
        ASSERT_FALSE(std::isnan(g_a[i]));
        ASSERT_FALSE(std::isinf(g_a[i]));
    }
}

AIC_TEST(LoRATest, WeightMergingAndZeroLatencyParity) {
    int64_t in_features = 32;
    int64_t out_features = 48;
    int64_t rank = 4;
    float alpha = 8.0f;

    auto base = std::make_shared<Linear>(in_features, out_features, /*has_bias=*/true);
    LoRALinear lora(base, rank, alpha);

    // Give non-zero values to B to simulate training
    Tensor b_new = Tensor::uniform(Shape({out_features, rank}), -0.5f, 0.5f);
    *lora.lora_B() = b_new;

    Tensor x = Tensor::randn(Shape({2, 8, in_features}));

    // 1. Unmerged forward pass
    Tensor y_unmerged = lora.forward(x);

    // 2. Merge weights
    lora.merge();
    ASSERT_TRUE(lora.is_merged());

    // 3. Merged forward pass (executes standard Linear GEMM W_eff)
    Tensor y_merged = lora.forward(x);

    // Outputs must match within numerical floating point precision
    ASSERT_TRUE(y_unmerged.shape() == y_merged.shape());
    const float* unmerged_ptr = y_unmerged.data_ptr<float>();
    const float* merged_ptr = y_merged.data_ptr<float>();
    for (int64_t i = 0; i < y_unmerged.numel(); ++i) {
        ASSERT_NEAR(unmerged_ptr[i], merged_ptr[i], 1e-4f);
    }

    // 4. Unmerge weights and verify restoration
    lora.unmerge();
    ASSERT_FALSE(lora.is_merged());
    Tensor y_restored = lora.forward(x);
    const float* restored_ptr = y_restored.data_ptr<float>();
    for (int64_t i = 0; i < y_unmerged.numel(); ++i) {
        ASSERT_NEAR(unmerged_ptr[i], restored_ptr[i], 1e-4f);
    }
}

AIC_TEST(LoRATest, TransformerLMAdaptationAndParameterReduction) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.intermediate_dim = 128;
    cfg.max_seq_len = 64;

    auto model = std::make_shared<TransformerLM>(cfg);

    int64_t total_before = 0;
    for (auto& p : model->parameters()) {
        total_before += p->numel();
    }

    // Adapt Attention Q and V projections with rank=4
    LoRAConfig lora_cfg;
    lora_cfg.rank = 4;
    lora_cfg.alpha = 8.0f;
    lora_cfg.target_modules = {"q_proj", "v_proj"};

    LoRAAdapter adapter(model, lora_cfg);

    int64_t trainable = adapter.num_trainable_parameters();
    int64_t total = adapter.num_total_parameters();
    float pct = adapter.trainable_percentage();

    // 2 layers * 2 projections (q, v) = 4 LoRALinear layers
    ASSERT_EQ(adapter.lora_layers().size(), 4);
    ASSERT_EQ(adapter.layer_names().size(), 4);

    // For each layer: A is [4, 64] = 256, B is [64, 4] = 256 -> 512 params per projection
    // 4 projections * 512 = 2048 trainable parameters
    ASSERT_EQ(trainable, 4 * (4 * 64 + 64 * 4));
    ASSERT_TRUE(total > 0);
    ASSERT_TRUE(pct < 5.0f); // Trainable parameters are < 5% of total model!

    // Verify all base parameters are frozen
    for (auto& p : model->parameters()) {
        if (p->numel() != 4 * 64 && p->numel() != 64 * 4) {
            // Not an adapter parameter
            ASSERT_FALSE(p->requires_grad());
        }
    }

    // Verify inference generation still works seamlessly
    std::vector<int64_t> prompt = {1, 5, 12};
    auto tokens = model->generate(prompt, 5, 0.8f);
    ASSERT_EQ(tokens.size(), 8);

    // Verify merge_all and unmerge_all
    adapter.merge_all();
    ASSERT_TRUE(adapter.is_merged());
    adapter.unmerge_all();
    ASSERT_FALSE(adapter.is_merged());
}

AIC_TEST(LoRATest, FineTuningConvergenceOnlyAdaptersChange) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.max_seq_len = 32;

    auto model = std::make_shared<TransformerLM>(cfg);

    LoRAConfig lora_cfg;
    lora_cfg.rank = 4;
    lora_cfg.alpha = 8.0f;
    lora_cfg.target_modules = {"q_proj", "v_proj"};
    LoRAAdapter adapter(model, lora_cfg);

    // Snapshot base model weights before fine-tuning
    std::vector<float> base_w_snapshot;
    auto base_w = model->layers()[0]->attn()->q_proj()->weight();
    const float* base_w_ptr = base_w->data_ptr<float>();
    int64_t base_w_numel = base_w->numel();
    base_w_snapshot.assign(base_w_ptr, base_w_ptr + base_w_numel);

    // Setup dummy target sequence
    std::vector<int32_t> input_tokens = {1, 10, 20, 30, 40, 50};
    Tensor input_ids = Tensor::zeros(Shape({1, 6}), DType::Int32);
    std::memcpy(input_ids.data_ptr<int32_t>(), input_tokens.data(), 6 * sizeof(int32_t));

    std::vector<int32_t> target_tokens = {10, 20, 30, 40, 50, 60};
    Tensor targets = Tensor::zeros(Shape({1, 6}), DType::Int32);
    std::memcpy(targets.data_ptr<int32_t>(), target_tokens.data(), 6 * sizeof(int32_t));

    CrossEntropyLoss ce_loss;
    AdamWOptions opt_opts;
    opt_opts.lr = 0.05f;
    AdamW optimizer(adapter.trainable_parameters(), opt_opts);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    // Train only LoRA adapters for 15 steps
    for (int step = 0; step < 15; ++step) {
        optimizer.zero_grad();
        Tensor logits = model->forward(input_ids);
        Tensor loss = ce_loss(logits, targets);
        float loss_val = loss.item<float>();
        if (step == 0) initial_loss = loss_val;
        final_loss = loss_val;

        loss.backward();
        optimizer.step();
    }

    // Verify loss decreased
    ASSERT_TRUE(final_loss < initial_loss);

    // Verify base weights are 100% BIT-EXACT UNCHANGED
    const float* base_w_after = base_w->data_ptr<float>();
    for (int64_t i = 0; i < base_w_numel; ++i) {
        ASSERT_NEAR(base_w_snapshot[i], base_w_after[i], 1e-7f);
    }

    // Verify adapter weights HAVE changed from zero
    auto lora_b = adapter.lora_layers()[0]->lora_B();
    const float* b_data = lora_b->data_ptr<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < lora_b->numel(); ++i) {
        if (std::abs(b_data[i]) > 1e-7f) {
            has_nonzero = true;
            break;
        }
    }
    ASSERT_TRUE(has_nonzero);
}

AIC_TEST(LoRATest, SerializationAndIntegrityCheck) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;

    auto model1 = std::make_shared<TransformerLM>(cfg);
    LoRAConfig lora_cfg;
    lora_cfg.rank = 4;
    lora_cfg.alpha = 8.0f;
    lora_cfg.target_modules = {"q_proj", "v_proj"};

    LoRAAdapter adapter1(model1, lora_cfg);

    // Put non-zero values into adapters
    for (auto& layer : adapter1.lora_layers()) {
        *layer->lora_B() = Tensor::uniform(layer->lora_B()->shape(), -0.2f, 0.2f);
    }

    std::string test_path = "test_adapter.aic_lora";
    adapter1.save_adapter(test_path);
    ASSERT_TRUE(std::filesystem::exists(test_path));

    // Create fresh model and adapter
    auto model2 = std::make_shared<TransformerLM>(cfg);
    LoRAAdapter adapter2(model2, lora_cfg);

    // Load saved adapter weights into adapter2
    adapter2.load_adapter(test_path);

    // Verify loaded weights match adapter1 bit-for-bit
    for (size_t l = 0; l < adapter1.lora_layers().size(); ++l) {
        auto b1 = adapter1.lora_layers()[l]->lora_B();
        auto b2 = adapter2.lora_layers()[l]->lora_B();
        const float* p1 = b1->data_ptr<float>();
        const float* p2 = b2->data_ptr<float>();
        for (int64_t i = 0; i < b1->numel(); ++i) {
            ASSERT_NEAR(p1[i], p2[i], 1e-6f);
        }
    }

    // Clean up
    std::filesystem::remove(test_path);
}

#if AIC_WITH_CUDA
AIC_TEST(LoRATest, CudaExecutionAndParity) {
    if (!is_cuda_available()) return;

    Device dev = Device::cuda(0);
    int64_t in_features = 64;
    int64_t out_features = 64;
    int64_t rank = 8;
    float alpha = 16.0f;

    auto base = std::make_shared<Linear>(in_features, out_features, /*has_bias=*/true, dev);
    LoRALinear lora(base, rank, alpha);

    ASSERT_TRUE(lora.lora_A()->is_cuda());
    ASSERT_TRUE(lora.lora_B()->is_cuda());
    ASSERT_TRUE(lora.weight()->is_cuda());

    Tensor x = Tensor::randn(Shape({2, 16, in_features}), 0.0f, 1.0f, DType::Float32, dev);
    Tensor y = lora.forward(x);
    ASSERT_TRUE(y.is_cuda());
    ASSERT_TRUE(y.shape() == Shape({2, 16, out_features}));

    // Backprop on CUDA
    Tensor loss = mean(mul(y, y));
    loss.backward();

    ASSERT_FALSE(lora.weight()->grad().defined());
    ASSERT_TRUE(lora.lora_A()->grad().defined());
    ASSERT_TRUE(lora.lora_B()->grad().defined());

    // Merge on CUDA
    lora.merge();
    ASSERT_TRUE(lora.is_merged());
    Tensor y_merged = lora.forward(x);
    ASSERT_TRUE(y_merged.is_cuda());

    lora.unmerge();
    ASSERT_FALSE(lora.is_merged());
}
#endif
