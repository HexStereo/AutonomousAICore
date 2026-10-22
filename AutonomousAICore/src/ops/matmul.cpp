// src/ops/matmul.cpp
// Top-level matmul / gemm dispatcher with CPU, CUDA, and Autograd integration.
#include "ops/matmul.hpp"
#include "ops/cpu_gemm.hpp"
#include "ops/ops.hpp"
#include "autograd/autograd.hpp"
#include "autograd/functions.hpp"
#include "core/error.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_gemm.cuh"
#endif

namespace aic {

namespace {

inline autograd::Edge make_matmul_edge(const Tensor& t) {
    if (!t.requires_grad()) return autograd::Edge{nullptr, 0};
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    AIC_CHECK(a.device() == b.device(), DeviceMismatch,
              "matmul: tensors must be on the same device (got ", a.device().str(), " and ", b.device().str(), ")");

    Tensor out;
    if (a.is_cpu()) {
        out = cpu::matmul(a, b);
    }
#if AIC_WITH_CUDA
    else if (a.is_cuda()) {
        out = cuda::matmul(a, b);
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    }

    if (autograd::GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);
        out.set_leaf(false);
        auto node = std::make_shared<autograd::MatMulBackward>(a, b);
        node->add_next_edge(make_matmul_edge(a));
        node->add_next_edge(make_matmul_edge(b));
        out.set_grad_fn(std::move(node));
    }

    return out;
}

Tensor gemm(const Tensor& a, const Tensor& b,
            float alpha, float beta,
            const Tensor& bias,
            [[maybe_unused]] GemmAlgorithm algorithm) {
    Tensor prod;
    if (a.is_cpu()) {
        prod = cpu::matmul(a, b);
    }
#if AIC_WITH_CUDA
    else if (a.is_cuda()) {
        prod = cuda::matmul(a, b, algorithm);
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    }

    if (alpha != 1.0f) {
        prod = prod * alpha;
    }
    if (beta != 0.0f && bias.defined()) {
        prod = prod + (beta * bias);
    } else if (bias.defined()) {
        prod = prod + bias;
    }

    return prod;
}

}  // namespace aic
