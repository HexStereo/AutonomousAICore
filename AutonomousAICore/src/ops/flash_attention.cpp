// src/ops/flash_attention.cpp
// FlashAttention-2 CPU implementation, Autograd backward node, and device dispatcher.
#include "ops/flash_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "autograd/autograd.hpp"
#include "autograd/functions.hpp"
#include "core/error.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_flash_attention.cuh"
#endif

namespace aic {

namespace {

inline autograd::Edge make_flash_edge(const Tensor& t) {
    if (!t.requires_grad()) return autograd::Edge{nullptr, 0};
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

}  // namespace

namespace cpu {

ops::FlashAttentionForwardResult flash_attention_forward_cpu(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    bool is_causal,
    float scale
) {
    int64_t B = q.dim(0);
    int64_t H = q.dim(1);
    int64_t Sq = q.dim(2);
    int64_t D = q.dim(3);
    int64_t Sk = k.dim(2);

    Tensor out = Tensor::zeros(Shape({B, H, Sq, D}), q.dtype(), Device::cpu());
    Tensor logsumexp = Tensor::zeros(Shape({B, H, Sq}), q.dtype(), Device::cpu());

    const float* q_data = q.contiguous().data_ptr<float>();
    const float* k_data = k.contiguous().data_ptr<float>();
    const float* v_data = v.contiguous().data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    float* lse_data = logsumexp.data_ptr<float>();

    constexpr int64_t Bc = 32;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t bh_offset_q = (b * H + h) * Sq * D;
            int64_t bh_offset_k = (b * H + h) * Sk * D;
            int64_t bh_offset_lse = (b * H + h) * Sq;

            const float* q_bh = q_data + bh_offset_q;
            const float* k_bh = k_data + bh_offset_k;
            const float* v_bh = v_data + bh_offset_k;
            float* out_bh = out_data + bh_offset_q;
            float* lse_bh = lse_data + bh_offset_lse;

            for (int64_t r = 0; r < Sq; ++r) {
                const float* q_vec = q_bh + r * D;
                float* out_row = out_bh + r * D;

                float m_i = -std::numeric_limits<float>::infinity();
                float l_i = 0.0f;
                std::vector<float> o_acc(static_cast<size_t>(D), 0.0f);

                for (int64_t j_b = 0; j_b < Sk; j_b += Bc) {
                    int64_t c_end = std::min(j_b + Bc, Sk);
                    if (is_causal && j_b > r) {
                        break;
                    }

                    int64_t block_len = c_end - j_b;
                    std::vector<float> s_block(static_cast<size_t>(block_len));
                    float block_max = -std::numeric_limits<float>::infinity();

                    for (int64_t c_idx = 0; c_idx < block_len; ++c_idx) {
                        int64_t c = j_b + c_idx;
                        if (is_causal && c > r) {
                            s_block[static_cast<size_t>(c_idx)] = -std::numeric_limits<float>::infinity();
                            continue;
                        }

                        const float* k_vec = k_bh + c * D;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < D; ++d) {
                            dot += q_vec[d] * k_vec[d];
                        }
                        dot *= scale;
                        s_block[static_cast<size_t>(c_idx)] = dot;
                        if (dot > block_max) {
                            block_max = dot;
                        }
                    }

                    if (block_max == -std::numeric_limits<float>::infinity()) {
                        continue;
                    }

                    float new_m = std::max(m_i, block_max);
                    float alpha = (m_i == -std::numeric_limits<float>::infinity())
                                      ? 0.0f
                                      : std::exp(m_i - new_m);

                    for (int64_t d = 0; d < D; ++d) {
                        o_acc[static_cast<size_t>(d)] *= alpha;
                    }

                    float block_l = 0.0f;
                    for (int64_t c_idx = 0; c_idx < block_len; ++c_idx) {
                        float s_val = s_block[static_cast<size_t>(c_idx)];
                        if (s_val > -1e30f) {
                            float p = std::exp(s_val - new_m);
                            block_l += p;

                            int64_t c = j_b + c_idx;
                            const float* v_vec = v_bh + c * D;
                            for (int64_t d = 0; d < D; ++d) {
                                o_acc[static_cast<size_t>(d)] += p * v_vec[d];
                            }
                        }
                    }

                    l_i = l_i * alpha + block_l;
                    m_i = new_m;
                }

                if (l_i > 0.0f) {
                    float inv_l = 1.0f / l_i;
                    for (int64_t d = 0; d < D; ++d) {
                        out_row[d] = o_acc[static_cast<size_t>(d)] * inv_l;
                    }
                    lse_bh[r] = m_i + std::log(l_i);
                } else {
                    for (int64_t d = 0; d < D; ++d) {
                        out_row[d] = 0.0f;
                    }
                    lse_bh[r] = 0.0f;
                }
            }
        }
    }

    return {out, logsumexp};
}

ops::FlashAttentionBackwardResult flash_attention_backward_cpu(
    const Tensor& grad_out,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& logsumexp,
    bool is_causal,
    float scale
) {
    int64_t B = q.dim(0);
    int64_t H = q.dim(1);
    int64_t Sq = q.dim(2);
    int64_t D = q.dim(3);
    int64_t Sk = k.dim(2);

    Tensor grad_q = Tensor::zeros(q.shape(), q.dtype(), Device::cpu());
    Tensor grad_k = Tensor::zeros(k.shape(), k.dtype(), Device::cpu());
    Tensor grad_v = Tensor::zeros(v.shape(), v.dtype(), Device::cpu());

    const float* dO_data = grad_out.contiguous().data_ptr<float>();
    const float* q_data = q.contiguous().data_ptr<float>();
    const float* k_data = k.contiguous().data_ptr<float>();
    const float* v_data = v.contiguous().data_ptr<float>();
    const float* out_data = out.contiguous().data_ptr<float>();
    const float* lse_data = logsumexp.contiguous().data_ptr<float>();

    float* dQ_data = grad_q.data_ptr<float>();
    float* dK_data = grad_k.data_ptr<float>();
    float* dV_data = grad_v.data_ptr<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t bh_offset_q = (b * H + h) * Sq * D;
            int64_t bh_offset_k = (b * H + h) * Sk * D;
            int64_t bh_offset_lse = (b * H + h) * Sq;

            const float* dO_bh = dO_data + bh_offset_q;
            const float* q_bh = q_data + bh_offset_q;
            const float* k_bh = k_data + bh_offset_k;
            const float* v_bh = v_data + bh_offset_k;
            const float* out_bh = out_data + bh_offset_q;
            const float* lse_bh = lse_data + bh_offset_lse;

            float* dQ_bh = dQ_data + bh_offset_q;
            float* dK_bh = dK_data + bh_offset_k;
            float* dV_bh = dV_data + bh_offset_k;

            // 1. Precompute D_i = sum_d (dO_i,d * O_i,d) for each row i
            std::vector<float> D_val(static_cast<size_t>(Sq), 0.0f);
            for (int64_t r = 0; r < Sq; ++r) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    dot += dO_bh[r * D + d] * out_bh[r * D + d];
                }
                D_val[static_cast<size_t>(r)] = dot;
            }

            // 2. Compute gradients recomputing attention P on-the-fly from L
            for (int64_t r = 0; r < Sq; ++r) {
                const float* q_vec = q_bh + r * D;
                const float* dO_vec = dO_bh + r * D;
                float* dQ_vec = dQ_bh + r * D;
                float l_val = lse_bh[r];
                float d_val = D_val[static_cast<size_t>(r)];

                int64_t max_c = is_causal ? std::min(r + 1, Sk) : Sk;
                for (int64_t c = 0; c < max_c; ++c) {
                    const float* k_vec = k_bh + c * D;
                    const float* v_vec = v_bh + c * D;
                    float* dK_vec = dK_bh + c * D;
                    float* dV_vec = dV_bh + c * D;

                    // Recompute dot product S_rc = scale * (Q_r . K_c)
                    float s_rc = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        s_rc += q_vec[d] * k_vec[d];
                    }
                    s_rc *= scale;

                    // P_rc = exp(S_rc - L_r)
                    float p_rc = std::exp(s_rc - l_val);

                    // dP_rc = dO_r . V_c
                    float dp_rc = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dp_rc += dO_vec[d] * v_vec[d];
                    }

                    // dS_rc = P_rc * (dP_rc - D_r) * scale
                    float ds_rc = p_rc * (dp_rc - d_val) * scale;

                    // Accumulate into dQ, dK, dV
                    for (int64_t d = 0; d < D; ++d) {
                        dQ_vec[d] += ds_rc * k_vec[d];
                        dK_vec[d] += ds_rc * q_vec[d];
                        dV_vec[d] += p_rc * dO_vec[d];
                    }
                }
            }
        }
    }

    return {grad_q, grad_k, grad_v};
}

}  // namespace cpu

namespace ops {

FlashAttentionForwardResult flash_attention_forward(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    bool is_causal,
    float scale
) {
    AIC_CHECK(q.rank() == 4 && k.rank() == 4 && v.rank() == 4, InvalidArgument,
              "FlashAttention inputs must be 4D tensors [B, H, S, D]");
    AIC_CHECK(q.device() == k.device() && q.device() == v.device(), DeviceMismatch,
              "All FlashAttention inputs must reside on the same device");
    AIC_CHECK(q.dim(0) == k.dim(0) && q.dim(0) == v.dim(0), ShapeMismatch,
              "Batch dimension mismatch");
    AIC_CHECK(q.dim(1) == k.dim(1) && q.dim(1) == v.dim(1), ShapeMismatch,
              "Head dimension mismatch");
    AIC_CHECK(k.dim(2) == v.dim(2), ShapeMismatch,
              "Key and Value sequence length mismatch");
    AIC_CHECK(q.dim(3) == k.dim(3) && q.dim(3) == v.dim(3), ShapeMismatch,
              "Feature dimension mismatch");

    int64_t D = q.dim(3);
    float effective_scale = (scale <= 0.0f) ? (1.0f / std::sqrt(static_cast<float>(D))) : scale;

    if (q.is_cpu()) {
        return cpu::flash_attention_forward_cpu(q, k, v, is_causal, effective_scale);
    }
#if AIC_WITH_CUDA
    else if (q.is_cuda()) {
        int64_t B = q.dim(0);
        int64_t H = q.dim(1);
        int64_t Sq = q.dim(2);
        int64_t Sk = k.dim(2);

        Tensor out = Tensor::zeros(Shape({B, H, Sq, D}), q.dtype(), q.device());
        Tensor lse = Tensor::zeros(Shape({B, H, Sq}), q.dtype(), q.device());

        cuda::launch_flash_attention_forward_gpu(
            out.data_ptr<float>(),
            lse.data_ptr<float>(),
            q.contiguous().data_ptr<float>(),
            k.contiguous().data_ptr<float>(),
            v.contiguous().data_ptr<float>(),
            static_cast<int>(B),
            static_cast<int>(H),
            static_cast<int>(Sq),
            static_cast<int>(Sk),
            static_cast<int>(D),
            is_causal,
            effective_scale
        );
        return {out, lse};
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "FlashAttention not supported on device ", q.device().str());
    }
}

FlashAttentionBackwardResult flash_attention_backward(
    const Tensor& grad_out,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& logsumexp,
    bool is_causal,
    float scale
) {
    int64_t D = q.dim(3);
    float effective_scale = (scale <= 0.0f) ? (1.0f / std::sqrt(static_cast<float>(D))) : scale;

    if (q.is_cpu()) {
        return cpu::flash_attention_backward_cpu(
            grad_out, q, k, v, out, logsumexp, is_causal, effective_scale
        );
    }
#if AIC_WITH_CUDA
    else if (q.is_cuda()) {
        int64_t B = q.dim(0);
        int64_t H = q.dim(1);
        int64_t Sq = q.dim(2);
        int64_t Sk = k.dim(2);

        Tensor grad_q = Tensor::zeros(q.shape(), q.dtype(), q.device());
        Tensor grad_k = Tensor::zeros(k.shape(), k.dtype(), q.device());
        Tensor grad_v = Tensor::zeros(v.shape(), v.dtype(), q.device());

        cuda::launch_flash_attention_backward_gpu(
            grad_q.data_ptr<float>(),
            grad_k.data_ptr<float>(),
            grad_v.data_ptr<float>(),
            grad_out.contiguous().data_ptr<float>(),
            q.contiguous().data_ptr<float>(),
            k.contiguous().data_ptr<float>(),
            v.contiguous().data_ptr<float>(),
            out.contiguous().data_ptr<float>(),
            logsumexp.contiguous().data_ptr<float>(),
            static_cast<int>(B),
            static_cast<int>(H),
            static_cast<int>(Sq),
            static_cast<int>(Sk),
            static_cast<int>(D),
            is_causal,
            effective_scale
        );
        return {grad_q, grad_k, grad_v};
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "FlashAttention backward not supported on device ", q.device().str());
    }
}

Tensor flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    bool is_causal,
    float scale
) {
    auto fwd = flash_attention_forward(q, k, v, is_causal, scale);
    Tensor out = fwd.output;

    if (autograd::GradMode::is_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad())) {
        out.set_requires_grad(true);
        out.set_leaf(false);

        int64_t D = q.dim(3);
        float effective_scale = (scale <= 0.0f) ? (1.0f / std::sqrt(static_cast<float>(D))) : scale;

        auto node = std::make_shared<autograd::FlashAttentionBackwardNode>(
            q, k, v, out, fwd.logsumexp, is_causal, effective_scale
        );
        node->add_next_edge(make_flash_edge(q));
        node->add_next_edge(make_flash_edge(k));
        node->add_next_edge(make_flash_edge(v));
        out.set_grad_fn(std::move(node));
    }

    return out;
}

}  // namespace ops

namespace autograd {

FlashAttentionBackwardNode::FlashAttentionBackwardNode(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& logsumexp,
    bool is_causal,
    float scale
) : saved_q_(q),
    saved_k_(k),
    saved_v_(v),
    saved_out_(out),
    saved_lse_(logsumexp),
    is_causal_(is_causal),
    scale_(scale) {}

std::vector<Tensor> FlashAttentionBackwardNode::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_out = grad_outputs[0];
    Tensor q = saved_q_.unpack("FlashAttentionBackward.q");
    Tensor k = saved_k_.unpack("FlashAttentionBackward.k");
    Tensor v = saved_v_.unpack("FlashAttentionBackward.v");
    Tensor out = saved_out_.unpack("FlashAttentionBackward.out");
    Tensor lse = saved_lse_.unpack("FlashAttentionBackward.lse");

    auto res = ops::flash_attention_backward(grad_out, q, k, v, out, lse, is_causal_, scale_);
    return {res.grad_q, res.grad_k, res.grad_v};
}

}  // namespace autograd

}  // namespace aic
