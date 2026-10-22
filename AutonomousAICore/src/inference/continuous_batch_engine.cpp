// src/inference/continuous_batch_engine.cpp
// Continuous Batching & Chunked Prefill High-Throughput Inference Engine.
#include "inference/continuous_batch_engine.hpp"
#include "inference/grammar.hpp"
#include "data/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

#include "core/error.hpp"

namespace aic::inference {

ContinuousBatchEngine::ContinuousBatchEngine(
    std::shared_ptr<models::TransformerLM> model,
    ContinuousBatchConfig config
) : model_(std::move(model)), config_(config), rng_(config.seed) {
    AIC_CHECK(model_ != nullptr, InvalidArgument, "ContinuousBatchEngine requires non-null model");

    if (config_.tokenizer && !config_.vocab_trie) {
        config_.vocab_trie = std::make_shared<VocabularyTrie>(VocabularyTrie::build(*config_.tokenizer));
    }

    memory::BlockManagerConfig b_cfg;
    b_cfg.num_blocks = config_.total_blocks;
    b_cfg.block_size = config_.block_size;
    b_cfg.num_heads = static_cast<int32_t>(model_->config().num_heads);
    b_cfg.head_dim = static_cast<int32_t>(model_->config().hidden_dim / model_->config().num_heads);
    b_cfg.device = model_->device();
    b_cfg.dtype = DType::Float32;

    block_mgr_ = std::make_shared<memory::BlockManager>(b_cfg);
    engine_start_time_ = std::chrono::high_resolution_clock::now();
}

uint64_t ContinuousBatchEngine::add_request(
    const std::vector<int64_t>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    int64_t eos_token_id,
    std::shared_ptr<GrammarMatcher> grammar_matcher
) {
    AIC_CHECK(!prompt_tokens.empty(), InvalidArgument, "Prompt tokens cannot be empty");

    auto req = std::make_shared<SequenceRequest>();
    req->request_id = next_request_id_++;
    req->prompt_tokens = prompt_tokens;
    req->max_new_tokens = max_new_tokens;
    req->temperature = temperature;
    req->top_k = top_k;
    req->top_p = top_p;
    req->eos_token_id = eos_token_id;
    req->grammar_matcher = std::move(grammar_matcher);
    req->status = RequestStatus::WAITING;
    req->arrival_time = std::chrono::high_resolution_clock::now();

    waiting_queue_.push_back(req);
    all_requests_[req->request_id] = req;

    return req->request_id;
}

void ContinuousBatchEngine::set_tokenizer(
    std::shared_ptr<data::Tokenizer> tokenizer,
    std::shared_ptr<VocabularyTrie> vocab_trie
) {
    config_.tokenizer = std::move(tokenizer);
    if (vocab_trie) {
        config_.vocab_trie = std::move(vocab_trie);
    } else if (config_.tokenizer) {
        config_.vocab_trie = std::make_shared<VocabularyTrie>(VocabularyTrie::build(*config_.tokenizer));
    }
}

bool ContinuousBatchEngine::cancel_request(uint64_t request_id) {
    auto it = all_requests_.find(request_id);
    if (it == all_requests_.end()) return false;

    auto req = it->second;
    if (req->status == RequestStatus::FINISHED || req->status == RequestStatus::CANCELLED) {
        return false;
    }

    req->status = RequestStatus::CANCELLED;
    if (!req->block_table.empty()) {
        block_mgr_->free(req->block_table);
        req->block_table.clear();
    }
    return true;
}

bool ContinuousBatchEngine::has_active_requests() const {
    if (!waiting_queue_.empty()) return true;
    for (const auto& req : active_requests_) {
        if (req->status != RequestStatus::FINISHED && req->status != RequestStatus::CANCELLED) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<SequenceRequest> ContinuousBatchEngine::get_request(uint64_t request_id) const {
    auto it = all_requests_.find(request_id);
    if (it != all_requests_.end()) {
        return it->second;
    }
    return nullptr;
}

int64_t ContinuousBatchEngine::sample_next_token(const Tensor& logits, float temperature, int top_k, float top_p) {
    Tensor cpu_logits = logits.cpu().contiguous();
    int64_t V = cpu_logits.numel();
    const float* l_ptr = cpu_logits.data_ptr<float>();

    if (temperature <= 0.001f || top_k == 1) {
        return static_cast<int64_t>(std::max_element(l_ptr, l_ptr + V) - l_ptr);
    }

    std::vector<float> scaled_logits(static_cast<size_t>(V));
    float max_l = l_ptr[0] / temperature;
    for (int64_t v = 0; v < V; ++v) {
        float val = l_ptr[v] / temperature;
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

    // Top-K filtering
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

    // Top-P (nucleus) filtering
    if (top_p < 1.0f && top_p > 0.0f) {
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

        float cumsum = 0.0f;
        int cutoff = 0;
        for (size_t i = 0; i < p_idx.size(); ++i) {
            cumsum += p_idx[i].first;
            cutoff = static_cast<int>(i);
            if (cumsum >= top_p) break;
        }

        std::vector<float> nuc_probs(static_cast<size_t>(V), 0.0f);
        float nuc_sum = 0.0f;
        for (int i = 0; i <= cutoff; ++i) {
            nuc_probs[static_cast<size_t>(p_idx[static_cast<size_t>(i)].second)] = p_idx[static_cast<size_t>(i)].first;
            nuc_sum += p_idx[static_cast<size_t>(i)].first;
        }
        for (int i = 0; i <= cutoff; ++i) {
            nuc_probs[static_cast<size_t>(p_idx[static_cast<size_t>(i)].second)] /= nuc_sum;
        }
        probs = std::move(nuc_probs);
    }

    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}

StepOutput ContinuousBatchEngine::step() {
    auto step_start = std::chrono::high_resolution_clock::now();
    StepOutput out;

    // 1. Retirement: Free virtual memory blocks and prune finished/cancelled requests
    for (auto it = active_requests_.begin(); it != active_requests_.end();) {
        auto& req = *it;
        if (req->status == RequestStatus::FINISHED || req->status == RequestStatus::CANCELLED) {
            if (!req->block_table.empty()) {
                block_mgr_->free(req->block_table);
                req->block_table.clear();
            }
            it = active_requests_.erase(it);
        } else {
            ++it;
        }
    }

    // 2. Scheduling: Allocate token budget for decode tokens and chunked prefill tokens
    int budget = config_.max_num_batched_tokens;

    // A: Prioritize active decoding requests (1 token per active sequence)
    std::vector<std::shared_ptr<SequenceRequest>> decode_batch;
    for (auto& req : active_requests_) {
        if (req->status == RequestStatus::DECODING && budget > 0) {
            decode_batch.push_back(req);
            budget -= 1;
        }
    }

    // B: Schedule ongoing prefilling requests
    std::vector<std::pair<std::shared_ptr<SequenceRequest>, int>> prefill_batch;
    for (auto& req : active_requests_) {
        if (req->status == RequestStatus::PREFILLING && budget > 0) {
            int chunk = std::min({
                budget,
                config_.chunked_prefill_size,
                static_cast<int>(req->remaining_prefill_tokens())
            });
            if (chunk > 0) {
                prefill_batch.push_back({req, chunk});
                budget -= chunk;
            }
        }
    }

    // C: Admit new requests from waiting queue if capacity and block pool permit
    while (!waiting_queue_.empty() &&
           static_cast<int>(active_requests_.size()) < config_.max_num_seqs &&
           budget > 0) {
        auto req = waiting_queue_.front();

        int needed_blocks = (static_cast<int>(req->prompt_tokens.size()) + config_.block_size - 1) / config_.block_size;
        if (block_mgr_->num_free_blocks() < needed_blocks + 1) {
            break; // Pause admission until blocks free up
        }

        waiting_queue_.pop_front();

        // Allocate initial physical blocks
        req->block_table = block_mgr_->allocate(static_cast<int>(req->prompt_tokens.size()));
        req->status = RequestStatus::PREFILLING;
        req->kv_caches.resize(model_->layers().size());

        active_requests_.push_back(req);

        int chunk = std::min({
            budget,
            config_.chunked_prefill_size,
            static_cast<int>(req->remaining_prefill_tokens())
        });
        if (chunk > 0) {
            prefill_batch.push_back({req, chunk});
            budget -= chunk;
        }
    }

    out.num_active_requests = static_cast<int>(active_requests_.size());

    // 3. Execution: Forward pass for Prefill Chunks
    for (auto& [req, chunk] : prefill_batch) {
        int64_t start_pos = req->num_prefilled_tokens;
        std::vector<int64_t> chunk_tokens(
            req->prompt_tokens.begin() + start_pos,
            req->prompt_tokens.begin() + start_pos + chunk
        );

        Tensor input_ids = Tensor::from_blob(
            chunk_tokens.data(),
            Shape({1, chunk}),
            DType::Int64,
            Device::cpu()
        ).to(model_->device());

        Tensor logits = model_->forward_step(input_ids, start_pos, req->kv_caches);
        req->num_prefilled_tokens += chunk;
        out.num_prefill_tokens += chunk;
        total_prefill_tokens_ += chunk;

        if (req->is_prefill_complete()) {
            // Prompt prefill complete: sample the first generated token
            Tensor last_logits = logits.slice(1, chunk - 1, chunk).squeeze(1).squeeze(0);
            if (req->grammar_matcher && config_.vocab_trie && config_.tokenizer) {
                std::vector<bool> valid_mask;
                req->grammar_matcher->get_valid_tokens(*config_.vocab_trie, *config_.tokenizer, valid_mask);
                GrammarLogitProcessor::apply_mask(last_logits, valid_mask);
            }

            int64_t next_token = sample_next_token(last_logits, req->temperature, req->top_k, req->top_p);
            if (req->grammar_matcher && config_.tokenizer) {
                req->grammar_matcher->consume_token(next_token, *config_.tokenizer);
            }
            req->output_tokens.push_back(next_token);
            req->status = RequestStatus::DECODING;

            auto now = std::chrono::high_resolution_clock::now();
            if (!req->has_emitted_first_token) {
                req->first_token_time = now;
                req->has_emitted_first_token = true;
                double ttft = std::chrono::duration<double, std::milli>(now - req->arrival_time).count();
                sum_ttft_ms_ += ttft;
            }

            bool is_finished = (next_token == req->eos_token_id) ||
                               (static_cast<int>(req->output_tokens.size()) >= req->max_new_tokens) ||
                               (req->grammar_matcher && req->grammar_matcher->is_terminated());
            if (is_finished) {
                req->status = RequestStatus::FINISHED;
                req->finish_time = now;
                total_requests_completed_++;
            }
            total_tokens_generated_++;
            out.items.push_back({req->request_id, next_token, is_finished});
        }
    }

    // 4. Execution: Forward pass for Decode Tokens
    for (auto& req : decode_batch) {
        int64_t last_token = req->output_tokens.back();
        int64_t pos = req->num_prefilled_tokens + static_cast<int64_t>(req->output_tokens.size()) - 1;

        // Ensure block capacity is available
        block_mgr_->append_slots(req->block_table, static_cast<int>(pos), 1);

        Tensor input_ids = Tensor::from_blob(&last_token, Shape({1, 1}), DType::Int64, Device::cpu()).to(model_->device());
        Tensor logits = model_->forward_step(input_ids, pos, req->kv_caches);
        Tensor token_logits = logits.squeeze(1).squeeze(0);

        if (req->grammar_matcher && config_.vocab_trie && config_.tokenizer) {
            std::vector<bool> valid_mask;
            req->grammar_matcher->get_valid_tokens(*config_.vocab_trie, *config_.tokenizer, valid_mask);
            GrammarLogitProcessor::apply_mask(token_logits, valid_mask);
        }

        int64_t next_token = sample_next_token(token_logits, req->temperature, req->top_k, req->top_p);
        if (req->grammar_matcher && config_.tokenizer) {
            req->grammar_matcher->consume_token(next_token, *config_.tokenizer);
        }
        req->output_tokens.push_back(next_token);
        out.num_decode_tokens += 1;
        total_tokens_generated_++;

        auto now = std::chrono::high_resolution_clock::now();
        bool is_finished = (next_token == req->eos_token_id) ||
                           (static_cast<int>(req->output_tokens.size()) >= req->max_new_tokens) ||
                           (req->grammar_matcher && req->grammar_matcher->is_terminated());
        if (is_finished) {
            req->status = RequestStatus::FINISHED;
            req->finish_time = now;
            total_requests_completed_++;
        }
        out.items.push_back({req->request_id, next_token, is_finished});
    }

    auto step_end = std::chrono::high_resolution_clock::now();
    out.step_latency_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

    if (out.num_decode_tokens > 0) {
        sum_itl_ms_ += out.step_latency_ms;
        itl_count_++;
    }

    return out;
}

std::vector<std::vector<int64_t>> ContinuousBatchEngine::generate_all(
    const std::vector<std::vector<int64_t>>& batch_prompts,
    int max_new_tokens
) {
    std::vector<uint64_t> ids;
    ids.reserve(batch_prompts.size());

    for (const auto& p : batch_prompts) {
        ids.push_back(add_request(p, max_new_tokens));
    }

    while (has_active_requests()) {
        step();
    }

    std::vector<std::vector<int64_t>> results;
    results.reserve(ids.size());
    for (auto id : ids) {
        auto req = get_request(id);
        results.push_back(req ? req->output_tokens : std::vector<int64_t>{});
    }

    return results;
}

ContinuousBatchMetrics ContinuousBatchEngine::metrics() const {
    ContinuousBatchMetrics m;
    m.total_requests_completed = total_requests_completed_;
    m.total_tokens_generated = total_tokens_generated_;
    m.total_prefill_tokens = total_prefill_tokens_;

    auto now = std::chrono::high_resolution_clock::now();
    m.total_elapsed_ms = std::chrono::duration<double, std::milli>(now - engine_start_time_).count();

    if (total_requests_completed_ > 0) {
        m.avg_time_to_first_token_ms = sum_ttft_ms_ / static_cast<double>(total_requests_completed_);
    }
    if (itl_count_ > 0) {
        m.avg_inter_token_latency_ms = sum_itl_ms_ / static_cast<double>(itl_count_);
    }
    if (m.total_elapsed_ms > 0.0) {
        m.throughput_tokens_per_sec = (static_cast<double>(total_tokens_generated_) / (m.total_elapsed_ms / 1000.0));
    }
    m.padding_waste_percentage = 0.0f; // Continuous batching eliminates padding waste
    return m;
}

void ContinuousBatchEngine::reset_metrics() {
    total_requests_completed_ = 0;
    total_tokens_generated_ = 0;
    total_prefill_tokens_ = 0;
    sum_ttft_ms_ = 0.0;
    sum_itl_ms_ = 0.0;
    itl_count_ = 0;
    engine_start_time_ = std::chrono::high_resolution_clock::now();
}

}  // namespace aic::inference
