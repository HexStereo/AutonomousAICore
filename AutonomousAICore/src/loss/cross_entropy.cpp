// src/loss/cross_entropy.cpp
// Fused LogSumExp Cross-Entropy loss implementation with analytical backward pass.
#include "loss/cross_entropy.hpp"
#include "autograd/autograd.hpp"
#include "autograd/node.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_loss.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace aic {

namespace {

inline autograd::Edge make_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return autograd::Edge{nullptr, 0};
    }
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

class CrossEntropyBackward : public autograd::Node {
public:
    CrossEntropyBackward(
        Tensor input,
        Tensor target,
        Tensor lse,
        int64_t ignore_index,
        std::string reduction,
        int64_t valid_count,
        bool is_index_target
    ) : saved_input_(input),
        saved_target_(target),
        lse_(std::move(lse)),
        ignore_index_(ignore_index),
        reduction_(std::move(reduction)),
        valid_count_(valid_count),
        is_index_target_(is_index_target) {}

    std::string name() const override { return "CrossEntropyBackward"; }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& g = grad_outputs[0];
        Tensor input = saved_input_.unpack("CrossEntropyBackward");
        Tensor target = saved_target_.unpack("CrossEntropyBackward");

        int64_t C = input.dim(-1);
        int64_t N = input.numel() / C;

        Tensor grad_input = Tensor::empty(input.shape(), DType::Float32, input.device());

        if (input.is_cuda()) {
#if AIC_WITH_CUDA
            const float* g_ptr = g.defined() ? g.contiguous().data_ptr<float>() : nullptr;
            cuda::cuda_cross_entropy_backward(
                g_ptr,
                input.contiguous().data_ptr<float>(),
                is_index_target_ ? target.contiguous().data_ptr<int64_t>() : nullptr,
                is_index_target_ ? nullptr : target.contiguous().data_ptr<float>(),
                is_index_target_,
                lse_.contiguous().data_ptr<float>(),
                N, C, ignore_index_,
                reduction_.c_str(),
                valid_count_,
                grad_input.data_ptr<float>()
            );
#else
            AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
        } else {
            // CPU backward pass
            const float* z = input.contiguous().data_ptr<float>();
            const float* lse_ptr = lse_.contiguous().data_ptr<float>();
            float* gz = grad_input.data_ptr<float>();

            const int64_t* y_idx = is_index_target_ ? target.contiguous().data_ptr<int64_t>() : nullptr;
            const float* y_dist = is_index_target_ ? nullptr : target.contiguous().data_ptr<float>();

            float g_scalar = 1.0f;
            if (reduction_ == "mean" || reduction_ == "sum") {
                if (g.defined()) {
                    g_scalar = g.cpu().item<float>();
                }
            }

            for (int64_t i = 0; i < N; ++i) {
                if (is_index_target_ && y_idx[i] == ignore_index_) {
                    std::memset(gz + i * C, 0, sizeof(float) * static_cast<size_t>(C));
                    continue;
                }

                float scale = 1.0f;
                if (reduction_ == "mean") {
                    scale = valid_count_ > 0 ? (g_scalar / static_cast<float>(valid_count_)) : 0.0f;
                } else if (reduction_ == "sum") {
                    scale = g_scalar;
                } else {
                    scale = g.defined() ? g.contiguous().data_ptr<float>()[i] : 1.0f;
                }

                float row_lse = lse_ptr[i];
                const float* row_z = z + i * C;
                float* row_gz = gz + i * C;

                if (is_index_target_) {
                    int64_t target_idx = y_idx[i];
                    for (int64_t j = 0; j < C; ++j) {
                        float prob = std::exp(row_z[j] - row_lse);
                        float ind = (j == target_idx) ? 1.0f : 0.0f;
                        row_gz[j] = (prob - ind) * scale;
                    }
                } else {
                    const float* row_dist = y_dist + i * C;
                    for (int64_t j = 0; j < C; ++j) {
                        float prob = std::exp(row_z[j] - row_lse);
                        row_gz[j] = (prob - row_dist[j]) * scale;
                    }
                }
            }
        }

        std::vector<Tensor> grads;
        grads.push_back(grad_input);
        if (target.requires_grad()) {
            grads.push_back(Tensor());
        }
        return grads;
    }

private:
    autograd::SavedVariable saved_input_;
    autograd::SavedVariable saved_target_;
    Tensor lse_;
    int64_t ignore_index_;
    std::string reduction_;
    int64_t valid_count_;
    bool is_index_target_;
};

}  // namespace

Tensor cross_entropy(const Tensor& input, const Tensor& target,
                     int64_t ignore_index,
                     const std::string& reduction) {
    AIC_CHECK(input.defined() && target.defined(), InvalidArgument,
              "cross_entropy inputs must be defined tensors");
    AIC_CHECK(input.device() == target.device(), DeviceMismatch,
              "input (", input.device().str(), ") and target (", target.device().str(),
              ") must be on the same device");
    AIC_CHECK(input.rank() >= 1, InvalidArgument,
              "input logits must have rank >= 1, got rank ", input.rank());
    AIC_CHECK(reduction == "mean" || reduction == "sum" || reduction == "none",
              InvalidArgument, "reduction must be 'mean', 'sum', or 'none', got '", reduction, "'");

    int64_t C = input.dim(-1);
    int64_t N = input.numel() / C;

    bool is_index_target = false;
    Tensor target_processed = target;

    if (target.dtype() == DType::Int64 || target.dtype() == DType::Int32) {
        is_index_target = true;
        if (target.dtype() == DType::Int32) {
            target_processed = target.to(DType::Int64);
        }
        AIC_CHECK(target_processed.numel() == N, ShapeMismatch,
                  "Index target numel (", target_processed.numel(),
                  ") must match batch numel (", N, ")");
    } else if (target.dtype() == DType::Float32) {
        is_index_target = false;
        AIC_CHECK(target.shape() == input.shape(), ShapeMismatch,
                  "Soft target shape ", target.shape().str(),
                  " must match logits shape ", input.shape().str());
    } else {
        AIC_THROW(DTypeMismatch, "Unsupported target dtype for cross_entropy: ",
                  dtype_name(target.dtype()));
    }

    Tensor sample_losses = Tensor::empty(Shape({N}), DType::Float32, input.device());
    Tensor lse_tensor = Tensor::empty(Shape({N}), DType::Float32, input.device());
    int64_t valid_count = 0;

    if (input.is_cuda()) {
#if AIC_WITH_CUDA
        cuda::cuda_cross_entropy_forward(
            input.contiguous().data_ptr<float>(),
            is_index_target ? target_processed.contiguous().data_ptr<int64_t>() : nullptr,
            is_index_target ? nullptr : target_processed.contiguous().data_ptr<float>(),
            is_index_target,
            N, C, ignore_index,
            sample_losses.data_ptr<float>(),
            lse_tensor.data_ptr<float>(),
            &valid_count
        );
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    } else {
        // CPU Forward Pass
        const float* z = input.contiguous().data_ptr<float>();
        float* loss_ptr = sample_losses.data_ptr<float>();
        float* lse_ptr = lse_tensor.data_ptr<float>();

        const int64_t* y_idx = is_index_target ? target_processed.contiguous().data_ptr<int64_t>() : nullptr;
        const float* y_dist = is_index_target ? nullptr : target_processed.contiguous().data_ptr<float>();

        for (int64_t i = 0; i < N; ++i) {
            const float* row_z = z + i * C;

            // 1. Numerically stabilized LogSumExp
            float max_val = row_z[0];
            for (int64_t j = 1; j < C; ++j) {
                if (row_z[j] > max_val) max_val = row_z[j];
            }

            double sum_exp = 0.0;
            for (int64_t j = 0; j < C; ++j) {
                sum_exp += std::exp(static_cast<double>(row_z[j] - max_val));
            }
            float lse = static_cast<float>(static_cast<double>(max_val) + std::log(std::max(sum_exp, 1e-12)));
            lse_ptr[i] = lse;

            // 2. Sample loss calculation
            if (is_index_target) {
                int64_t target_idx = y_idx[i];
                if (target_idx == ignore_index) {
                    loss_ptr[i] = 0.0f;
                } else {
                    AIC_CHECK(target_idx >= 0 && target_idx < C, OutOfRange,
                              "Target index ", target_idx, " is out of range for classes [0, ", C, ")");
                    loss_ptr[i] = lse - row_z[target_idx];
                    valid_count++;
                }
            } else {
                const float* row_dist = y_dist + i * C;
                float dot = 0.0f;
                for (int64_t j = 0; j < C; ++j) {
                    dot += row_dist[j] * row_z[j];
                }
                loss_ptr[i] = lse - dot;
                valid_count++;
            }
        }
    }

    // Apply reduction
    Tensor result;
    if (reduction == "none") {
        if (is_index_target) {
            result = sample_losses.reshape(target.shape());
        } else {
            std::vector<int64_t> out_dims;
            for (int d = 0; d < input.rank() - 1; ++d) {
                out_dims.push_back(input.dim(d));
            }
            result = sample_losses.reshape(Shape(out_dims));
        }
    } else if (reduction == "sum") {
        result = sum(sample_losses);
    } else { // "mean"
        Tensor total = sum(sample_losses);
        if (valid_count > 0) {
            result = total / static_cast<float>(valid_count);
        } else {
            result = Tensor::zeros(Shape({}), DType::Float32, input.device());
        }
    }

    if (autograd::GradMode::is_enabled() && input.requires_grad()) {
        result.set_requires_grad(true);
        result.set_leaf(false);
        auto node = std::make_shared<CrossEntropyBackward>(
            input, target_processed, lse_tensor, ignore_index, reduction, valid_count, is_index_target
        );
        node->add_next_edge(make_edge(input));
        if (target.requires_grad()) {
            node->add_next_edge(make_edge(target));
        }
        result.set_grad_fn(std::move(node));
    }

    return result;
}

}  // namespace aic
