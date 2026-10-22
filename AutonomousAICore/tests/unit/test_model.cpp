// tests/unit/test_model.cpp
// Verification of Embedding, TransformerLM architecture, Autoregressive generation, and KV caching.
#include "models/models.hpp"
#include "loss/loss.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <iostream>

using namespace aic;
using namespace aic::nn;
using namespace aic::models;

AIC_TEST(EmbeddingTest, ForwardBackwardCPUAndGPU) {
    int64_t V = 16;
    int64_t D = 8;
    Embedding emb(V, D, Device::cpu());

    Tensor input_ids = Tensor::empty(Shape({2, 3}), DType::Int64, Device::cpu());
    int64_t* ids = input_ids.data_ptr<int64_t>();
    ids[0] = 0; ids[1] = 5; ids[2] = 12;
    ids[3] = 3; ids[4] = 5; ids[5] = 1;

    Tensor out = emb.forward(input_ids);
    ASSERT_EQ(out.shape(), Shape({2, 3, D}));

    // Verify row values match weight
    const float* w = emb.weight()->data_ptr<float>();
    const float* o = out.data_ptr<float>();
    for (int64_t d = 0; d < D; ++d) {
        ASSERT_EQ(o[0 * D + d], w[0 * D + d]);
        ASSERT_EQ(o[1 * D + d], w[5 * D + d]);
        ASSERT_EQ(o[4 * D + d], w[5 * D + d]); // repeated token 5
    }

    // Backward pass
    Tensor loss = sum(out);
    loss.backward();

    ASSERT_TRUE(emb.weight()->grad().defined());
    const float* gw = emb.weight()->grad().data_ptr<float>();
    // Token 5 appeared twice, so its gradient should be 2.0 for all dimensions
    for (int64_t d = 0; d < D; ++d) {
        ASSERT_NEAR(gw[5 * D + d], 2.0f, 1e-4f);
        ASSERT_NEAR(gw[0 * D + d], 1.0f, 1e-4f);
        ASSERT_NEAR(gw[7 * D + d], 0.0f, 1e-4f); // not in input
    }

    if (is_cuda_available()) {
        Embedding emb_gpu(V, D, Device::cuda(0));
        emb_gpu.weight()->clone(); // ensure allocation
        Tensor ids_gpu = input_ids.cuda();
        Tensor out_gpu = emb_gpu.forward(ids_gpu);
        ASSERT_TRUE(out_gpu.is_cuda());
        ASSERT_EQ(out_gpu.shape(), Shape({2, 3, D}));

        Tensor loss_gpu = sum(out_gpu);
        loss_gpu.backward();
        ASSERT_TRUE(emb_gpu.weight()->grad().is_cuda());
    }
}

AIC_TEST(TransformerLMTest, ParameterCountAndHierarchy) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 96;

    TransformerLM model(cfg, Device::cpu());
    auto params = model.parameters();
    ASSERT_FALSE(params.empty());

    auto named_params = model.named_parameters();
    bool has_tok_emb = false;
    bool has_norm = false;
    bool has_head = false;
    bool has_attn = false;

    for (const auto& [name, p] : named_params) {
        if (name.find("tok_embeddings") != std::string::npos) has_tok_emb = true;
        if (name.find("norm") != std::string::npos) has_norm = true;
        if (name.find("lm_head") != std::string::npos) has_head = true;
        if (name.find("attn") != std::string::npos) has_attn = true;
    }

    ASSERT_TRUE(has_tok_emb);
    ASSERT_TRUE(has_norm);
    ASSERT_TRUE(has_head);
    ASSERT_TRUE(has_attn);
}

AIC_TEST(TransformerLMTest, FullSequenceForwardBackward) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;

    TransformerLM model(cfg, Device::cpu());

    int64_t B = 2;
    int64_t S = 4;
    Tensor input_ids = Tensor::zeros(Shape({B, S}), DType::Int64, Device::cpu());
    Tensor targets = Tensor::zeros(Shape({B, S}), DType::Int64, Device::cpu());
    for (int64_t i = 0; i < B * S; ++i) {
        input_ids.data_ptr<int64_t>()[i] = i % cfg.vocab_size;
        targets.data_ptr<int64_t>()[i] = (i + 1) % cfg.vocab_size;
    }

    Tensor logits = model.forward(input_ids);
    ASSERT_EQ(logits.shape(), Shape({B, S, cfg.vocab_size}));

    Tensor loss = cross_entropy(logits, targets, -100, "mean");
    ASSERT_TRUE(loss.item<float>() > 0.0f);

    loss.backward();

    for (const auto& [name, p] : model.named_parameters()) {
        if (!p->grad().defined()) {
            std::cout << "[MISSING GRAD] " << name << std::endl;
        }
        ASSERT_TRUE(p->grad().defined());
    }
}

AIC_TEST(TransformerLMTest, KVCacheAutoregressiveParity) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;

    TransformerLM model(cfg, Device::cpu());

    // Prompt tokens [3, 7, 15]
    std::vector<int64_t> prompt = {3, 7, 15};
    Tensor input_seq = Tensor::empty(Shape({1, 3}), DType::Int64, Device::cpu());
    input_seq.data_ptr<int64_t>()[0] = 3;
    input_seq.data_ptr<int64_t>()[1] = 7;
    input_seq.data_ptr<int64_t>()[2] = 15;

    // Parallel forward pass
    Tensor parallel_logits = model.forward(input_seq);

    // Cached step-by-step forward pass
    std::vector<KVCache> kv_caches(cfg.num_layers);
    Tensor cached_last_logits;

    for (size_t pos = 0; pos < prompt.size(); ++pos) {
        Tensor single = Tensor::empty(Shape({1, 1}), DType::Int64, Device::cpu());
        single.data_ptr<int64_t>()[0] = prompt[pos];
        cached_last_logits = model.forward_step(single, static_cast<int64_t>(pos), kv_caches);
    }

    // Verify exact parity between parallel sequence forward and cached step generation
    const float* p_par = parallel_logits.data_ptr<float>() + 2 * cfg.vocab_size; // last token logits
    const float* p_cac = cached_last_logits.data_ptr<float>();

    for (int64_t v = 0; v < cfg.vocab_size; ++v) {
        ASSERT_NEAR(p_par[v], p_cac[v], 1e-4f);
    }
}

AIC_TEST(TransformerLMTest, GPUExecutionAndParity) {
    if (!is_cuda_available()) return;

    TransformerLMConfig cfg;
    cfg.vocab_size = 48;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;

    TransformerLM model_cpu(cfg, Device::cpu());
    Device gpu = Device::cuda(0);

    Tensor input_ids = Tensor::empty(Shape({2, 4}), DType::Int64, Device::cpu());
    for (int64_t i = 0; i < 8; ++i) {
        input_ids.data_ptr<int64_t>()[i] = (i * 5) % cfg.vocab_size;
    }

    Tensor logits_cpu = model_cpu.forward(input_ids);

    // Move model to GPU
    model_cpu.to(gpu);
    Tensor input_ids_gpu = input_ids.cuda();
    Tensor logits_gpu = model_cpu.forward(input_ids_gpu);

    ASSERT_TRUE(logits_gpu.is_cuda());
    ASSERT_EQ(logits_gpu.shape(), Shape({2, 4, cfg.vocab_size}));

    Tensor logits_gpu_cpu = logits_gpu.cpu();
    const float* p_c = logits_cpu.data_ptr<float>();
    const float* p_g = logits_gpu_cpu.data_ptr<float>();

    for (int64_t i = 0; i < logits_cpu.numel(); ++i) {
        ASSERT_NEAR(p_c[i], p_g[i], 1e-4f);
    }
}

AIC_TEST(TransformerLMTest, AutoregressiveGeneration) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;

    TransformerLM model(cfg, Device::cpu());

    std::vector<int64_t> prompt = {1, 10, 20};
    int64_t max_new = 8;
    auto generated = model.generate(prompt, max_new, /*temperature=*/0.8f, /*top_k=*/10, /*top_p=*/0.9f);

    ASSERT_EQ(generated.size(), prompt.size() + static_cast<size_t>(max_new));
    for (size_t i = 0; i < prompt.size(); ++i) {
        ASSERT_EQ(generated[i], prompt[i]);
    }
    for (int64_t token : generated) {
        ASSERT_TRUE(token >= 0 && token < cfg.vocab_size);
    }
}
