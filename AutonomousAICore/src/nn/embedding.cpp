// src/nn/embedding.cpp
// Embedding layer implementation with backward scatter-add gradient computation.
#include "nn/embedding.hpp"
#include "autograd/autograd.hpp"
#include "autograd/node.hpp"
#include "core/error.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_embedding.cuh"
#endif

#include <cstring>
#include <vector>

namespace aic::nn {

namespace {

inline autograd::Edge make_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return autograd::Edge{nullptr, 0};
    }
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

class EmbeddingBackward : public autograd::Node {
public:
    EmbeddingBackward(Tensor input_ids, Shape weight_shape, int64_t num_embeddings, int64_t embedding_dim)
        : saved_ids_(std::move(input_ids)),
          weight_shape_(std::move(weight_shape)),
          num_embeddings_(num_embeddings),
          embedding_dim_(embedding_dim) {}

    std::string name() const override { return "EmbeddingBackward"; }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& g = grad_outputs[0];
        Tensor ids = saved_ids_.unpack("EmbeddingBackward");

        Tensor grad_weight = Tensor::zeros(weight_shape_, DType::Float32, g.device());
        int64_t N = ids.numel();
        int64_t D = embedding_dim_;
        int64_t V = num_embeddings_;

        if (g.is_cuda()) {
#if AIC_WITH_CUDA
            cuda::cuda_embedding_backward(
                g.contiguous().data_ptr<float>(),
                ids.contiguous().data_ptr<int64_t>(),
                grad_weight.data_ptr<float>(),
                N, D, V
            );
#else
            AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
        } else {
            // CPU scatter-add
            const float* g_ptr = g.contiguous().data_ptr<float>();
            const int64_t* id_ptr = ids.contiguous().data_ptr<int64_t>();
            float* gw_ptr = grad_weight.data_ptr<float>();

            for (int64_t i = 0; i < N; ++i) {
                int64_t token_id = id_ptr[i];
                if (token_id >= 0 && token_id < V) {
                    float* row = gw_ptr + token_id * D;
                    const float* g_row = g_ptr + i * D;
                    for (int64_t d = 0; d < D; ++d) {
                        row[d] += g_row[d];
                    }
                }
            }
        }

        return {grad_weight};
    }

private:
    autograd::SavedVariable saved_ids_;
    Shape weight_shape_;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
};

}  // namespace

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    AIC_CHECK(num_embeddings_ > 0 && embedding_dim_ > 0, InvalidArgument,
              "num_embeddings and embedding_dim must be positive");

    float std = 1.0f / std::sqrt(static_cast<float>(embedding_dim_));
    Tensor w = Tensor::randn(Shape({num_embeddings_, embedding_dim_}), 0.0f, std, DType::Float32, device);
    weight_ = register_parameter("weight", std::move(w));
}

Tensor Embedding::forward(const Tensor& input_ids) {
    AIC_CHECK(input_ids.defined(), InvalidArgument, "input_ids must be a defined tensor");
    AIC_CHECK(input_ids.dtype() == DType::Int64 || input_ids.dtype() == DType::Int32, DTypeMismatch,
              "input_ids must be Int64 or Int32, got ", dtype_name(input_ids.dtype()));

    Tensor ids = input_ids;
    if (ids.dtype() == DType::Int32) {
        ids = ids.to(DType::Int64);
    }
    AIC_CHECK(ids.device() == weight_->device(), DeviceMismatch,
              "input_ids device (", ids.device().str(), ") must match weight device (", weight_->device().str(), ")");

    int64_t N = ids.numel();
    int64_t D = embedding_dim_;
    int64_t V = num_embeddings_;

    std::vector<int64_t> out_dims;
    for (int i = 0; i < ids.rank(); ++i) {
        out_dims.push_back(ids.dim(i));
    }
    out_dims.push_back(D);
    Shape out_shape(out_dims);

    Tensor out = Tensor::empty(out_shape, DType::Float32, weight_->device());

    if (weight_->is_cuda()) {
#if AIC_WITH_CUDA
        cuda::cuda_embedding_forward(
            ids.contiguous().data_ptr<int64_t>(),
            weight_->contiguous().data_ptr<float>(),
            out.data_ptr<float>(),
            N, D, V
        );
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    } else {
        const int64_t* id_ptr = ids.contiguous().data_ptr<int64_t>();
        const float* w_ptr = weight_->contiguous().data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        for (int64_t i = 0; i < N; ++i) {
            int64_t token_id = id_ptr[i];
            if (token_id >= 0 && token_id < V) {
                std::memcpy(out_ptr + i * D, w_ptr + token_id * D, sizeof(float) * static_cast<size_t>(D));
            } else {
                std::memset(out_ptr + i * D, 0, sizeof(float) * static_cast<size_t>(D));
            }
        }
    }

    if (autograd::GradMode::is_enabled() && weight_->requires_grad()) {
        out.set_requires_grad(true);
        out.set_leaf(false);
        auto node = std::make_shared<EmbeddingBackward>(ids, weight_->shape(), V, D);
        node->add_next_edge(make_edge(*weight_));
        out.set_grad_fn(std::move(node));
    }

    return out;
}

}  // namespace aic::nn
