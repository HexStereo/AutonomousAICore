// include/nn/rope.hpp
// Rotary Position Embedding (RoPE) with complex rotation and sequence offset support.
#pragma once

#include "nn/module.hpp"
#include "ops/ops.hpp"
#include "autograd/autograd.hpp"
#include "autograd/node.hpp"
#include <cmath>
#include <vector>

namespace aic::nn {

namespace {

class RoPEBackward : public autograd::Node {
public:
    RoPEBackward(Tensor cos_b, Tensor sin_b, int64_t head_dim, int rank)
        : cos_b_(std::move(cos_b)), sin_b_(std::move(sin_b)), head_dim_(head_dim), rank_(rank) {}

    std::string name() const override { return "RoPEBackward"; }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& g = grad_outputs[0];
        int64_t half_d = head_dim_ / 2;
        int r = rank_;

        Tensor g1 = g.slice(r - 1, 0, half_d);
        Tensor g2 = g.slice(r - 1, half_d, head_dim_);

        // Inverse (adjoint) rotation:
        // dx1 = g1 * cos + g2 * sin
        // dx2 = -g1 * sin + g2 * cos = g2 * cos - g1 * sin
        Tensor dx1 = add(mul(g1, cos_b_), mul(g2, sin_b_));
        Tensor dx2 = sub(mul(g2, cos_b_), mul(g1, sin_b_));

        Tensor dx = cat({dx1, dx2}, r - 1);
        return {dx};
    }

private:
    Tensor cos_b_;
    Tensor sin_b_;
    int64_t head_dim_;
    int rank_;
};

}  // namespace

class RotaryPositionEmbedding : public Module {
public:
    RotaryPositionEmbedding(int64_t head_dim, int64_t max_seq_len = 4096, float base = 10000.0f, Device device = Device::cpu())
        : head_dim_(head_dim), max_seq_len_(max_seq_len), base_(base), device_(device) {
        AIC_CHECK(head_dim_ % 2 == 0, InvalidArgument, "head_dim must be even for RoPE, got ", head_dim_);
        build_cache(device_);
    }

    void to(const Device& device) override {
        Module::to(device);
        device_ = device;
        cos_cache_ = cos_cache_.to(device_);
        sin_cache_ = sin_cache_.to(device_);
    }

    // Rotates input tensor x: [B, H, S, head_dim] or [B, S, head_dim]
    Tensor forward(const Tensor& x, int64_t start_pos = 0) {
        int r = x.rank();
        AIC_CHECK(r >= 2, InvalidArgument, "RoPE requires rank >= 2, got ", r);
        int64_t seq_len = x.dim(r - 2);
        int64_t d = x.dim(r - 1);
        AIC_CHECK(d == head_dim_, ShapeMismatch, "RoPE head_dim mismatch: expected ", head_dim_, " got ", d);
        AIC_CHECK(start_pos + seq_len <= max_seq_len_, OutOfRange,
                  "RoPE position out of bounds: ", start_pos + seq_len, " > ", max_seq_len_);

        int64_t half_d = head_dim_ / 2;

        // Slice precomputed cos and sin: [seq_len, half_d]
        Tensor cos_slice = cos_cache_.slice(0, start_pos, start_pos + seq_len);
        Tensor sin_slice = sin_cache_.slice(0, start_pos, start_pos + seq_len);

        // Reshape cos and sin to broadcast with x: [1, 1, seq_len, half_d] if rank 4
        std::vector<int64_t> broadcast_shape(static_cast<size_t>(r), 1);
        broadcast_shape[static_cast<size_t>(r - 2)] = seq_len;
        broadcast_shape[static_cast<size_t>(r - 1)] = half_d;
        Tensor cos_b = cos_slice.reshape(Shape(broadcast_shape));
        Tensor sin_b = sin_slice.reshape(Shape(std::move(broadcast_shape)));

        // Split x into first and second half along last dimension
        Tensor x1 = x.slice(r - 1, 0, half_d);
        Tensor x2 = x.slice(r - 1, half_d, head_dim_);

        // Apply complex rotation:
        // x1_rot = x1 * cos - x2 * sin
        // x2_rot = x1 * sin + x2 * cos
        Tensor x1_rot = sub(mul(x1, cos_b), mul(x2, sin_b));
        Tensor x2_rot = add(mul(x1, sin_b), mul(x2, cos_b));

        // Concatenate halves back along last dimension
        Tensor res = cat({x1_rot, x2_rot}, r - 1);

        if (autograd::GradMode::is_enabled() && x.requires_grad()) {
            res.set_requires_grad(true);
            res.set_leaf(false);
            auto node = std::make_shared<RoPEBackward>(cos_b, sin_b, head_dim_, r);
            node->add_next_edge(autograd::Edge{
                x.grad_fn() ? x.grad_fn() : std::make_shared<autograd::AccumulateGrad>(x), 0
            });
            res.set_grad_fn(std::move(node));
        }

        return res;
    }

    Tensor operator()(const Tensor& x, int64_t start_pos = 0) {
        return forward(x, start_pos);
    }

    int64_t head_dim() const noexcept { return head_dim_; }
    int64_t max_seq_len() const noexcept { return max_seq_len_; }

private:
    void build_cache(const Device& dev) {
        int64_t half_d = head_dim_ / 2;
        std::vector<float> cos_vals(static_cast<size_t>(max_seq_len_ * half_d));
        std::vector<float> sin_vals(static_cast<size_t>(max_seq_len_ * half_d));

        std::vector<float> freqs(static_cast<size_t>(half_d));
        for (int64_t i = 0; i < half_d; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim_);
            freqs[static_cast<size_t>(i)] = 1.0f / std::pow(base_, exponent);
        }

        for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
            for (int64_t i = 0; i < half_d; ++i) {
                float angle = static_cast<float>(pos) * freqs[static_cast<size_t>(i)];
                size_t idx = static_cast<size_t>(pos * half_d + i);
                cos_vals[idx] = std::cos(angle);
                sin_vals[idx] = std::sin(angle);
            }
        }

        cos_cache_ = Tensor::from_blob(cos_vals.data(), Shape({max_seq_len_, half_d})).clone().to(dev);
        sin_cache_ = Tensor::from_blob(sin_vals.data(), Shape({max_seq_len_, half_d})).clone().to(dev);
    }

    int64_t head_dim_;
    int64_t max_seq_len_;
    float base_;
    Device device_;
    Tensor cos_cache_;
    Tensor sin_cache_;
};

}  // namespace aic::nn
