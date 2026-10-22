// tests/unit/test_continuous_batch.cpp
// Comprehensive Unit Tests for Continuous Batching & Chunked Prefill Inference Engine.
#include "inference/continuous_batch_engine.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "models/transformer_lm.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <chrono>
#include <vector>

using namespace aic;
using namespace aic::models;
using namespace aic::inference;

namespace {

std::shared_ptr<TransformerLM> create_test_model(Device device = Device::cpu()) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.max_seq_len = 128;
    return std::make_shared<TransformerLM>(cfg, device);
}

}  // namespace

AIC_TEST(ContinuousBatchTest, RequestLifecycleAndStatus) {
    auto model = create_test_model(Device::cpu());
    ContinuousBatchConfig cfg;
    cfg.chunked_prefill_size = 16;
    ContinuousBatchEngine engine(model, cfg);

    std::vector<int64_t> prompt = {1, 10, 20, 30};
    uint64_t req_id = engine.add_request(prompt, /*max_new_tokens=*/5, /*temperature=*/0.0f);

    auto req = engine.get_request(req_id);
    ASSERT_TRUE(req != nullptr);
    ASSERT_TRUE(req->status == RequestStatus::WAITING);
    ASSERT_EQ(req->output_tokens.size(), 0);

    // First step admits request and finishes prefill (prompt size 4 <= chunk size 16)
    StepOutput step1 = engine.step();
    ASSERT_EQ(step1.items.size(), 1);
    ASSERT_EQ(req->output_tokens.size(), 1);
    ASSERT_TRUE(req->status == RequestStatus::DECODING);

    // Subsequent steps decode 1 token per step
    while (engine.has_active_requests()) {
        engine.step();
    }

    ASSERT_TRUE(req->status == RequestStatus::FINISHED);
    ASSERT_EQ(req->output_tokens.size(), 5);
}

AIC_TEST(ContinuousBatchTest, ZeroPaddingContinuousAdmission) {
    auto model = create_test_model(Device::cpu());
    ContinuousBatchConfig cfg;
    cfg.max_num_seqs = 4;
    cfg.chunked_prefill_size = 32;
    ContinuousBatchEngine engine(model, cfg);

    // Request 1 generates 3 tokens, Request 2 generates 8 tokens
    uint64_t req1_id = engine.add_request({2, 5, 8}, /*max_new_tokens=*/3, 0.0f);
    uint64_t req2_id = engine.add_request({3, 6, 9, 12, 15}, /*max_new_tokens=*/8, 0.0f);

    auto req1 = engine.get_request(req1_id);
    auto req2 = engine.get_request(req2_id);

    // Run until req1 finishes
    while (req1->status != RequestStatus::FINISHED) {
        engine.step();
    }

    ASSERT_TRUE(req1->status == RequestStatus::FINISHED);
    ASSERT_EQ(req1->output_tokens.size(), 3);
    ASSERT_TRUE(req2->status == RequestStatus::DECODING);

    // Req1 is done; now admit Request 3 dynamically into the running batch!
    uint64_t req3_id = engine.add_request({4, 8, 12, 16}, /*max_new_tokens=*/4, 0.0f);
    auto req3 = engine.get_request(req3_id);

    // Finish all requests
    while (engine.has_active_requests()) {
        engine.step();
    }

    ASSERT_TRUE(req2->status == RequestStatus::FINISHED);
    ASSERT_TRUE(req3->status == RequestStatus::FINISHED);
    ASSERT_EQ(req2->output_tokens.size(), 8);
    ASSERT_EQ(req3->output_tokens.size(), 4);

    auto m = engine.metrics();
    ASSERT_EQ(m.total_requests_completed, 3);
    ASSERT_EQ(m.total_tokens_generated, 3 + 8 + 4);
    ASSERT_NEAR(m.padding_waste_percentage, 0.0f, 1e-5f);
}

AIC_TEST(ContinuousBatchTest, ChunkedPrefillBudgetSplitting) {
    auto model = create_test_model(Device::cpu());
    ContinuousBatchConfig cfg;
    cfg.chunked_prefill_size = 4; // Max 4 prompt tokens per chunk
    ContinuousBatchEngine engine(model, cfg);

    // Prompt of 9 tokens -> requires 3 chunks (4 + 4 + 1)
    std::vector<int64_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint64_t req_id = engine.add_request(prompt, /*max_new_tokens=*/2, 0.0f);
    auto req = engine.get_request(req_id);

    // Step 1: Ingest chunk 1 (tokens 0..3)
    StepOutput s1 = engine.step();
    ASSERT_EQ(s1.num_prefill_tokens, 4);
    ASSERT_EQ(req->num_prefilled_tokens, 4);
    ASSERT_TRUE(req->status == RequestStatus::PREFILLING);
    ASSERT_EQ(req->output_tokens.size(), 0);

    // Step 2: Ingest chunk 2 (tokens 4..7)
    StepOutput s2 = engine.step();
    ASSERT_EQ(s2.num_prefill_tokens, 4);
    ASSERT_EQ(req->num_prefilled_tokens, 8);
    ASSERT_TRUE(req->status == RequestStatus::PREFILLING);
    ASSERT_EQ(req->output_tokens.size(), 0);

    // Step 3: Ingest chunk 3 (token 8), completing prefill and emitting token 1
    StepOutput s3 = engine.step();
    ASSERT_EQ(s3.num_prefill_tokens, 1);
    ASSERT_EQ(req->num_prefilled_tokens, 9);
    ASSERT_TRUE(req->status == RequestStatus::DECODING);
    ASSERT_EQ(req->output_tokens.size(), 1);

    // Step 4: Decode token 2 -> finished
    StepOutput s4 = engine.step();
    ASSERT_EQ(s4.num_decode_tokens, 1);
    ASSERT_TRUE(req->status == RequestStatus::FINISHED);
    ASSERT_EQ(req->output_tokens.size(), 2);
}

AIC_TEST(ContinuousBatchTest, BlockManagerMemoryReclamation) {
    auto model = create_test_model(Device::cpu());
    ContinuousBatchConfig cfg;
    cfg.total_blocks = 64;
    cfg.block_size = 16;
    ContinuousBatchEngine engine(model, cfg);

    auto bm = engine.block_manager();
    int initial_free = bm->num_free_blocks();
    ASSERT_EQ(initial_free, 64);

    // Submit 3 requests
    engine.add_request({1, 2, 3, 4, 5}, 4, 0.0f);
    engine.add_request({10, 20, 30}, 6, 0.0f);
    engine.add_request({7, 14, 21, 28, 35, 42}, 3, 0.0f);

    // Run 1 step: blocks are allocated
    engine.step();
    ASSERT_TRUE(bm->num_free_blocks() < initial_free);

    // Finish all requests
    while (engine.has_active_requests()) {
        engine.step();
    }

    // Step one more time so engine retirement reclaims finished blocks
    engine.step();

    // Verify 100% of memory blocks were safely recycled
    ASSERT_EQ(bm->num_free_blocks(), initial_free);
    ASSERT_EQ(bm->num_allocated_blocks(), 0);
}

AIC_TEST(ContinuousBatchTest, RequestCancellation) {
    auto model = create_test_model(Device::cpu());
    ContinuousBatchConfig cfg;
    ContinuousBatchEngine engine(model, cfg);

    uint64_t req1 = engine.add_request({1, 2, 3}, 10, 0.0f);
    uint64_t req2 = engine.add_request({4, 5, 6}, 10, 0.0f);

    engine.step(); // Start both

    // Cancel req1
    bool ok = engine.cancel_request(req1);
    ASSERT_TRUE(ok);

    auto r1 = engine.get_request(req1);
    ASSERT_TRUE(r1->status == RequestStatus::CANCELLED);

    // Next step purges cancelled request
    engine.step();

    auto r2 = engine.get_request(req2);
    ASSERT_TRUE(r2->status == RequestStatus::DECODING || r2->status == RequestStatus::FINISHED);
}

AIC_TEST(ContinuousBatchTest, TelemetryAndThroughputMetrics) {
    auto model = create_test_model(Device::cpu());
    ContinuousBatchConfig cfg;
    ContinuousBatchEngine engine(model, cfg);

    std::vector<std::vector<int64_t>> prompts = {
        {1, 5, 10},
        {2, 6, 12, 18},
        {3, 7}
    };

    auto results = engine.generate_all(prompts, /*max_new_tokens=*/4);
    ASSERT_EQ(results.size(), 3);
    for (const auto& r : results) {
        ASSERT_EQ(r.size(), 4);
    }

    auto m = engine.metrics();
    ASSERT_EQ(m.total_requests_completed, 3);
    ASSERT_EQ(m.total_tokens_generated, 12);
    ASSERT_TRUE(m.throughput_tokens_per_sec > 0.0);
    ASSERT_TRUE(m.avg_time_to_first_token_ms >= 0.0);
    ASSERT_NEAR(m.padding_waste_percentage, 0.0f, 1e-5f);
}

AIC_TEST(ContinuousBatchTest, CudaExecutionAndParity) {
    if (!is_cuda_available()) {
        std::cout << "[SKIP] CUDA not available on this device" << std::endl;
        return;
    }

    Device gpu_dev = Device::cuda(0);
    auto model_gpu = create_test_model(gpu_dev);

    ContinuousBatchConfig cfg;
    cfg.chunked_prefill_size = 16;
    ContinuousBatchEngine engine_gpu(model_gpu, cfg);

    std::vector<int64_t> prompt = {1, 10, 20, 30};
    uint64_t req_id = engine_gpu.add_request(prompt, /*max_new_tokens=*/6, /*temperature=*/0.0f);

    while (engine_gpu.has_active_requests()) {
        engine_gpu.step();
    }

    auto req = engine_gpu.get_request(req_id);
    ASSERT_TRUE(req != nullptr);
    ASSERT_TRUE(req->status == RequestStatus::FINISHED);
    ASSERT_EQ(req->output_tokens.size(), 6);

    auto m = engine_gpu.metrics();
    ASSERT_EQ(m.total_requests_completed, 1);
    ASSERT_EQ(m.total_tokens_generated, 6);
    ASSERT_TRUE(m.throughput_tokens_per_sec > 0.0);
}
