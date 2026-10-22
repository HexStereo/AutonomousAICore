// src/alignment/dpo_loss.cpp
// Direct Preference Optimization (DPO) Loss & Sequence Log-Probability extraction.
#include "alignment/dpo_loss.hpp"
#include "autograd/autograd.hpp"
#include "autograd/node.hpp"
#include "core/error.hpp"
#include "loss/loss.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace aic::alignment {

namespace {

inline autograd::Edge make_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return autograd::Edge{nullptr, 0};
    }
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

class DPOBackwardNode : public autograd::Node {
public:
    DPOBackwardNode(
        Tensor policy_chosen,
        Tensor policy_rejected,
        std::vector<float> grad_w,
        float beta,
        int64_t batch_size
    ) : saved_chosen_(policy_chosen),
        saved_rejected_(policy_rejected),
        grad_w_(std::move(grad_w)),
        beta_(beta),
        batch_size_(batch_size) {}

    std::string name() const override { return "DPOBackwardNode"; }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        float upstream = grad_outputs.empty() ? 1.0f : grad_outputs[0].cpu().item<float>();
        float scale = (batch_size_ > 0) ? (upstream / static_cast<float>(batch_size_)) : upstream;

        Tensor chosen = saved_chosen_.unpack("DPOBackwardNode");
        Tensor rejected = saved_rejected_.unpack("DPOBackwardNode");

        Tensor grad_chosen = Tensor::empty(chosen.shape(), DType::Float32, chosen.device());
        Tensor grad_rejected = Tensor::empty(rejected.shape(), DType::Float32, rejected.device());

        std::vector<float> h_grad_chosen(static_cast<size_t>(batch_size_));
        std::vector<float> h_grad_rejected(static_cast<size_t>(batch_size_));

        for (int64_t i = 0; i < batch_size_; ++i) {
            float w = grad_w_[static_cast<size_t>(i)];
            h_grad_chosen[static_cast<size_t>(i)] = scale * beta_ * w;
            h_grad_rejected[static_cast<size_t>(i)] = -scale * beta_ * w;
        }

        if (chosen.is_cpu()) {
            std::memcpy(grad_chosen.data_ptr<float>(), h_grad_chosen.data(), batch_size_ * sizeof(float));
            std::memcpy(grad_rejected.data_ptr<float>(), h_grad_rejected.data(), batch_size_ * sizeof(float));
        } else {
            Tensor t_c = Tensor::empty(chosen.shape(), DType::Float32, Device::cpu());
            Tensor t_r = Tensor::empty(rejected.shape(), DType::Float32, Device::cpu());
            std::memcpy(t_c.data_ptr<float>(), h_grad_chosen.data(), batch_size_ * sizeof(float));
            std::memcpy(t_r.data_ptr<float>(), h_grad_rejected.data(), batch_size_ * sizeof(float));
            grad_chosen = t_c.to(chosen.device());
            grad_rejected = t_r.to(rejected.device());
        }

        return {grad_chosen, grad_rejected};
    }

private:
    autograd::SavedVariable saved_chosen_;
    autograd::SavedVariable saved_rejected_;
    std::vector<float> grad_w_;
    float beta_;
    int64_t batch_size_;
};

}  // namespace

Tensor compute_sequence_log_probs(
    nn::Module& model,
    const Tensor& input_ids,
    const Tensor& labels,
    int64_t ignore_index
) {
    AIC_CHECK(input_ids.rank() == 2, InvalidArgument, "input_ids must be 2D [Batch, SeqLen]");
    AIC_CHECK(labels.rank() == 2, InvalidArgument, "labels must be 2D [Batch, SeqLen]");
    AIC_CHECK(input_ids.shape() == labels.shape(), InvalidArgument, "input_ids and labels shapes must match");

    Tensor logits = model.forward(input_ids);

    // Negative cross entropy with reduction="none" gives per-token log-probabilities
    Tensor ce = cross_entropy(logits, labels, ignore_index, "none");
    Tensor logps = neg(ce);

    // Sum over sequence dimension (dim 1) to get per-sequence log-probabilities: [Batch]
    return sum(logps, 1, /*keepdim=*/false);
}

std::pair<Tensor, DPOMetrics> compute_dpo_loss(
    const Tensor& policy_chosen_logps,
    const Tensor& policy_rejected_logps,
    const Tensor& ref_chosen_logps,
    const Tensor& ref_rejected_logps,
    float beta,
    float label_smoothing
) {
    AIC_CHECK(policy_chosen_logps.numel() == policy_rejected_logps.numel(),
              InvalidArgument, "policy chosen and rejected must have same numel");
    AIC_CHECK(ref_chosen_logps.numel() == ref_rejected_logps.numel(),
              InvalidArgument, "ref chosen and rejected must have same numel");
    AIC_CHECK(policy_chosen_logps.numel() == ref_chosen_logps.numel(),
              InvalidArgument, "policy and ref must have same numel");

    int64_t B = policy_chosen_logps.numel();
    AIC_CHECK(B > 0, InvalidArgument, "Batch size must be positive");

    Tensor pi_w_cpu = policy_chosen_logps.cpu().contiguous();
    Tensor pi_l_cpu = policy_rejected_logps.cpu().contiguous();
    Tensor ref_w_cpu = ref_chosen_logps.cpu().contiguous();
    Tensor ref_l_cpu = ref_rejected_logps.cpu().contiguous();

    const float* pi_w = pi_w_cpu.data_ptr<float>();
    const float* pi_l = pi_l_cpu.data_ptr<float>();
    const float* ref_w = ref_w_cpu.data_ptr<float>();
    const float* ref_l = ref_l_cpu.data_ptr<float>();

    std::vector<float> grad_w(static_cast<size_t>(B));
    float total_loss = 0.0f;
    float total_chosen_reward = 0.0f;
    float total_rejected_reward = 0.0f;
    float total_acc = 0.0f;

    for (int64_t i = 0; i < B; ++i) {
        float r_w = beta * (pi_w[i] - ref_w[i]);
        float r_l = beta * (pi_l[i] - ref_l[i]);
        float u = r_w - r_l;

        // Numerically stable softplus(-u) for -log(sigmoid(u)):
        // -log(sigmoid(u)) = log(1 + exp(-u))
        float loss_i;
        if (u >= 0.0f) {
            loss_i = std::log1p(std::exp(-u));
        } else {
            loss_i = -u + std::log1p(std::exp(u));
        }

        // Numerically stable sigmoid(-u) for derivative:
        // sigma(-u) = 1 / (1 + exp(u))
        float sig_neg_u;
        if (u >= 0.0f) {
            float exp_neg_u = std::exp(-u);
            sig_neg_u = exp_neg_u / (1.0f + exp_neg_u);
        } else {
            sig_neg_u = 1.0f / (1.0f + std::exp(u));
        }

        float sig_pos_u = 1.0f - sig_neg_u;

        if (label_smoothing > 0.0f) {
            float loss_neg_i;
            if (-u >= 0.0f) {
                loss_neg_i = std::log1p(std::exp(u));
            } else {
                loss_neg_i = u + std::log1p(std::exp(-u));
            }
            loss_i = (1.0f - label_smoothing) * loss_i + label_smoothing * loss_neg_i;
            grad_w[static_cast<size_t>(i)] = -(1.0f - label_smoothing) * sig_neg_u + label_smoothing * sig_pos_u;
        } else {
            grad_w[static_cast<size_t>(i)] = -sig_neg_u;
        }

        total_loss += loss_i;
        total_chosen_reward += r_w;
        total_rejected_reward += r_l;
        if (u > 0.0f) {
            total_acc += 1.0f;
        }
    }

    float mean_loss = total_loss / static_cast<float>(B);

    DPOMetrics metrics;
    metrics.loss = mean_loss;
    metrics.chosen_reward = total_chosen_reward / static_cast<float>(B);
    metrics.rejected_reward = total_rejected_reward / static_cast<float>(B);
    metrics.reward_margin = metrics.chosen_reward - metrics.rejected_reward;
    metrics.accuracy = total_acc / static_cast<float>(B);

    Tensor loss_tensor = Tensor::from_blob(&mean_loss, Shape({}), DType::Float32, Device::cpu())
                            .clone()
                            .to(policy_chosen_logps.device());

    if (autograd::GradMode::is_enabled() &&
        (policy_chosen_logps.requires_grad() || policy_rejected_logps.requires_grad())) {
        loss_tensor.set_requires_grad(true);
        loss_tensor.set_leaf(false);

        auto node = std::make_shared<DPOBackwardNode>(
            policy_chosen_logps,
            policy_rejected_logps,
            std::move(grad_w),
            beta,
            B
        );

        node->add_next_edge(make_edge(policy_chosen_logps));
        node->add_next_edge(make_edge(policy_rejected_logps));
        loss_tensor.set_grad_fn(std::move(node));
    }

    return {loss_tensor, metrics};
}

}  // namespace aic::alignment
