// include/inference/continuous_batch_engine.hpp
// Continuous Batching & Chunked Prefill High-Throughput Inference Engine.
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "memory/block_manager.hpp"
#include "models/transformer_lm.hpp"
#include "tensor/tensor.hpp"

namespace aic::data {
class Tokenizer;
}

namespace aic::inference {

class GrammarMatcher;
class VocabularyTrie;

enum class RequestStatus {
    WAITING,       // Queued, waiting for admission
    PREFILLING,    // Ingesting prompt tokens across iteration chunks
    DECODING,      // Generating tokens autoregressively 1 token per step
    FINISHED,      // Hit EOS or reached max_new_tokens
    CANCELLED      // Explicitly aborted by user/client
};

struct SequenceRequest {
    uint64_t request_id = 0;
    std::vector<int64_t> prompt_tokens;
    std::vector<int64_t> output_tokens;
    int64_t num_prefilled_tokens = 0;   // Number of prompt tokens already processed
    int max_new_tokens = 64;
    float temperature = 0.8f;
    int top_k = 20;
    float top_p = 0.9f;
    int64_t eos_token_id = -1;
    RequestStatus status = RequestStatus::WAITING;
    memory::BlockTable block_table;
    std::shared_ptr<GrammarMatcher> grammar_matcher = nullptr;

    // Per-sequence KV caches (for layers)
    std::vector<nn::KVCache> kv_caches;

    // Telemetry timestamps
    std::chrono::high_resolution_clock::time_point arrival_time;
    std::chrono::high_resolution_clock::time_point first_token_time;
    std::chrono::high_resolution_clock::time_point finish_time;
    bool has_emitted_first_token = false;

    int64_t total_length() const noexcept {
        return static_cast<int64_t>(prompt_tokens.size() + output_tokens.size());
    }

    int64_t remaining_prefill_tokens() const noexcept {
        return static_cast<int64_t>(prompt_tokens.size()) - num_prefilled_tokens;
    }

    bool is_prefill_complete() const noexcept {
        return num_prefilled_tokens >= static_cast<int64_t>(prompt_tokens.size());
    }
};

struct StepOutput {
    struct GenerationItem {
        uint64_t request_id = 0;
        int64_t token_id = 0;
        bool is_finished = false;
    };

    std::vector<GenerationItem> items;
    int num_active_requests = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;
    double step_latency_ms = 0.0;
};

struct ContinuousBatchMetrics {
    uint64_t total_requests_completed = 0;
    uint64_t total_tokens_generated = 0;
    uint64_t total_prefill_tokens = 0;
    double avg_time_to_first_token_ms = 0.0;
    double avg_inter_token_latency_ms = 0.0;
    double total_elapsed_ms = 0.0;
    double throughput_tokens_per_sec = 0.0;
    float padding_waste_percentage = 0.0f; // Strictly 0.0% in Continuous Batching
};

struct ContinuousBatchConfig {
    int max_num_seqs = 32;                  // Max concurrent active sequences
    int max_num_batched_tokens = 256;       // Max tokens processed per iteration budget
    int chunked_prefill_size = 64;          // Chunk size for prompt ingestion
    int block_size = memory::DEFAULT_BLOCK_SIZE; // 16 tokens/block
    int total_blocks = 512;
    uint64_t seed = 42;
    std::shared_ptr<data::Tokenizer> tokenizer = nullptr;
    std::shared_ptr<VocabularyTrie> vocab_trie = nullptr;
};

class ContinuousBatchEngine {
public:
    ContinuousBatchEngine(
        std::shared_ptr<models::TransformerLM> model,
        ContinuousBatchConfig config = {}
    );

    // Submits a new inference request to the queue
    uint64_t add_request(
        const std::vector<int64_t>& prompt_tokens,
        int max_new_tokens = 64,
        float temperature = 0.8f,
        int top_k = 20,
        float top_p = 0.9f,
        int64_t eos_token_id = -1,
        std::shared_ptr<GrammarMatcher> grammar_matcher = nullptr
    );

    // Associate a tokenizer (and optional trie) for grammar-guided sampling
    void set_tokenizer(
        std::shared_ptr<data::Tokenizer> tokenizer,
        std::shared_ptr<VocabularyTrie> vocab_trie = nullptr
    );

    // Cancels an ongoing or queued request
    bool cancel_request(uint64_t request_id);

    // Executes one iteration step: schedules, forwards, samples, updates blocks
    StepOutput step();

    // Checks if there are active (running or waiting) requests
    bool has_active_requests() const;

    // Retrieves status of a specific request
    std::shared_ptr<SequenceRequest> get_request(uint64_t request_id) const;

    // Synchronous helper: process batch of prompts to completion
    std::vector<std::vector<int64_t>> generate_all(
        const std::vector<std::vector<int64_t>>& batch_prompts,
        int max_new_tokens = 64
    );

    // Engine performance telemetry
    ContinuousBatchMetrics metrics() const;

    // Reset telemetry metrics
    void reset_metrics();

    std::shared_ptr<memory::BlockManager> block_manager() const noexcept { return block_mgr_; }
    const ContinuousBatchConfig& config() const noexcept { return config_; }

private:
    int64_t sample_next_token(const Tensor& logits, float temperature, int top_k, float top_p);

    std::shared_ptr<models::TransformerLM> model_;
    ContinuousBatchConfig config_;
    std::shared_ptr<memory::BlockManager> block_mgr_;

    uint64_t next_request_id_ = 1;
    std::deque<std::shared_ptr<SequenceRequest>> waiting_queue_;
    std::vector<std::shared_ptr<SequenceRequest>> active_requests_;
    std::unordered_map<uint64_t, std::shared_ptr<SequenceRequest>> all_requests_;

    std::mt19937_64 rng_;

    // Performance tracking
    uint64_t total_requests_completed_ = 0;
    uint64_t total_tokens_generated_ = 0;
    uint64_t total_prefill_tokens_ = 0;
    double sum_ttft_ms_ = 0.0;
    double sum_itl_ms_ = 0.0;
    uint64_t itl_count_ = 0;
    std::chrono::high_resolution_clock::time_point engine_start_time_;
};

}  // namespace aic::inference
