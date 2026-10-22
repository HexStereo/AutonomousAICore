// include/nn/embedding.hpp
// Differentiable Token Embedding lookup table.
#pragma once

#include "nn/module.hpp"
#include <cmath>

namespace aic::nn {

class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device = Device::cpu());

    Tensor forward(const Tensor& input_ids) override;

    Tensor operator()(const Tensor& input_ids) {
        return forward(input_ids);
    }

    std::shared_ptr<Tensor> weight() const noexcept { return weight_; }
    int64_t num_embeddings() const noexcept { return num_embeddings_; }
    int64_t embedding_dim() const noexcept { return embedding_dim_; }

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    std::shared_ptr<Tensor> weight_;
};

}  // namespace aic::nn
