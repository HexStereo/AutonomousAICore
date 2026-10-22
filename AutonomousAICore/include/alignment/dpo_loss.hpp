// include/alignment/dpo_loss.hpp
// Direct Preference Optimization (DPO) Loss & Sequence Log-Probability Extraction.
#pragma once

#include <cstdint>
#include <vector>
#include "core/dtype.hpp"
#include "nn/module.hpp"
#include "tensor/tensor.hpp"

namespace aic::alignment {

struct DPOMetrics {
    float loss = 0.0f;
    float chosen_reward = 0.0f;
    float rejected_reward = 0.0f;
    float reward_margin = 0.0f;
    float accuracy = 0.0f;  // 1.0 if chosen_reward > rejected_reward else 0.0
};

// Computes sum of log-probabilities for target label tokens:
// input_ids: [Batch, SeqLen]
// labels: [Batch, SeqLen] with ignore_index (default -100)
// Returns 1D Tensor [Batch] of sequence log-probabilities
Tensor compute_sequence_log_probs(
    nn::Module& model,
    const Tensor& input_ids,
    const Tensor& labels,
    int64_t ignore_index = -100
);

// Computes DPO loss and scalar metrics:
// policy_chosen_logps:   [Batch]
// policy_rejected_logps: [Batch]
// ref_chosen_logps:      [Batch]
// ref_rejected_logps:    [Batch]
// beta: Temperature hyperparameter (typically 0.1)
// label_smoothing: Conservative DPO regularization (default 0.0)
std::pair<Tensor, DPOMetrics> compute_dpo_loss(
    const Tensor& policy_chosen_logps,
    const Tensor& policy_rejected_logps,
    const Tensor& ref_chosen_logps,
    const Tensor& ref_rejected_logps,
    float beta = 0.1f,
    float label_smoothing = 0.0f
);

}  // namespace aic::alignment
