// src/ops/ops.cpp
// Top-level tensor mathematical operations dispatcher with Autograd graph integration.
#include "ops/ops.hpp"
#include "ops/cpu_ops.hpp"
#include "autograd/autograd.hpp"
#include "autograd/functions.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_elementwise.cuh"
#include "cuda/cuda_reductions.cuh"
#include <cuda_runtime.h>
#endif

#include <cstring>

namespace aic {

namespace {

void check_device_match(const Tensor& a, const Tensor& b) {
    AIC_CHECK(a.device() == b.device(), DeviceMismatch,
              "Tensors must be on the same device (got ", a.device().str(), " and ", b.device().str(), ")");
}

inline autograd::Edge make_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return autograd::Edge{nullptr, 0};
    }
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}

template <typename NodeT, typename... Args>
inline void attach_autograd_binary(Tensor& out, const Tensor& a, const Tensor& b, Args&&... args) {
    if (autograd::GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);
        out.set_leaf(false);
        auto node = std::make_shared<NodeT>(std::forward<Args>(args)...);
        node->add_next_edge(make_edge(a));
        node->add_next_edge(make_edge(b));
        out.set_grad_fn(std::move(node));
    }
}

template <typename NodeT, typename... Args>
inline void attach_autograd_unary(Tensor& out, const Tensor& a, Args&&... args) {
    if (autograd::GradMode::is_enabled() && a.requires_grad()) {
        out.set_requires_grad(true);
        out.set_leaf(false);
        auto node = std::make_shared<NodeT>(std::forward<Args>(args)...);
        node->add_next_edge(make_edge(a));
        out.set_grad_fn(std::move(node));
    }
}

}  // namespace

// ============================================================================
// Elementwise Binary Operations
// ============================================================================
Tensor add(const Tensor& a, const Tensor& b) {
    check_device_match(a, b);
    Tensor res;
    if (a.is_cpu()) {
        res = cpu::add(a, b);
    }
#if AIC_WITH_CUDA
    else if (a.is_cuda()) {
        res = cuda::add(a, b);
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "No available backend for device ", a.device().str());
    }
    attach_autograd_binary<autograd::AddBackward>(res, a, b, a.shape(), b.shape());
    return res;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    check_device_match(a, b);
    Tensor res;
    if (a.is_cpu()) {
        res = cpu::sub(a, b);
    }
#if AIC_WITH_CUDA
    else if (a.is_cuda()) {
        res = cuda::sub(a, b);
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "No available backend for device ", a.device().str());
    }
    attach_autograd_binary<autograd::SubBackward>(res, a, b, a.shape(), b.shape());
    return res;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    check_device_match(a, b);
    Tensor res;
    if (a.is_cpu()) {
        res = cpu::mul(a, b);
    }
#if AIC_WITH_CUDA
    else if (a.is_cuda()) {
        res = cuda::mul(a, b);
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "No available backend for device ", a.device().str());
    }
    attach_autograd_binary<autograd::MulBackward>(res, a, b, a, b);
    return res;
}

Tensor div(const Tensor& a, const Tensor& b) {
    check_device_match(a, b);
    Tensor res;
    if (a.is_cpu()) {
        res = cpu::div(a, b);
    }
#if AIC_WITH_CUDA
    else if (a.is_cuda()) {
        res = cuda::div(a, b);
    }
#endif
    else {
        AIC_THROW(BackendUnavailable, "No available backend for device ", a.device().str());
    }
    attach_autograd_binary<autograd::DivBackward>(res, a, b, a, b);
    return res;
}

// In-place operations (with graph version tracking and leaf protection)
Tensor& add_(Tensor& a, const Tensor& b) {
    if (a.requires_grad() && a.is_leaf()) {
        AIC_THROW(RuntimeError, "A leaf Variable that requires grad cannot be modified in-place");
    }
    a.bump_version();

    Tensor res = add(a, b);
    AIC_CHECK(res.shape() == a.shape(), ShapeMismatch,
              "In-place add_ result shape ", res.shape().str(), " does not match target ", a.shape().str());
    if (a.is_cpu()) {
        std::memcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes());
    } else {
#if AIC_WITH_CUDA
        cudaMemcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes(), cudaMemcpyDeviceToDevice);
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    }
    return a;
}

Tensor& sub_(Tensor& a, const Tensor& b) {
    if (a.requires_grad() && a.is_leaf()) {
        AIC_THROW(RuntimeError, "A leaf Variable that requires grad cannot be modified in-place");
    }
    a.bump_version();

    Tensor res = sub(a, b);
    AIC_CHECK(res.shape() == a.shape(), ShapeMismatch,
              "In-place sub_ result shape ", res.shape().str(), " does not match target ", a.shape().str());
    if (a.is_cpu()) {
        std::memcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes());
    } else {
#if AIC_WITH_CUDA
        cudaMemcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes(), cudaMemcpyDeviceToDevice);
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    }
    return a;
}

Tensor& mul_(Tensor& a, const Tensor& b) {
    if (a.requires_grad() && a.is_leaf()) {
        AIC_THROW(RuntimeError, "A leaf Variable that requires grad cannot be modified in-place");
    }
    a.bump_version();

    Tensor res = mul(a, b);
    AIC_CHECK(res.shape() == a.shape(), ShapeMismatch,
              "In-place mul_ result shape ", res.shape().str(), " does not match target ", a.shape().str());
    if (a.is_cpu()) {
        std::memcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes());
    } else {
#if AIC_WITH_CUDA
        cudaMemcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes(), cudaMemcpyDeviceToDevice);
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    }
    return a;
}

Tensor& div_(Tensor& a, const Tensor& b) {
    if (a.requires_grad() && a.is_leaf()) {
        AIC_THROW(RuntimeError, "A leaf Variable that requires grad cannot be modified in-place");
    }
    a.bump_version();

    Tensor res = div(a, b);
    AIC_CHECK(res.shape() == a.shape(), ShapeMismatch,
              "In-place div_ result shape ", res.shape().str(), " does not match target ", a.shape().str());
    if (a.is_cpu()) {
        std::memcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes());
    } else {
#if AIC_WITH_CUDA
        cudaMemcpy(a.raw_data(), res.contiguous().raw_data(), a.nbytes(), cudaMemcpyDeviceToDevice);
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    }
    return a;
}

Tensor& add_(Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return add_(a, s);
}

Tensor& sub_(Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return sub_(a, s);
}

Tensor& mul_(Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return mul_(a, s);
}

Tensor& div_(Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return div_(a, s);
}

// Scalar elementwise operations
Tensor add(const Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return add(a, s);
}

Tensor add(float scalar, const Tensor& a) {
    return add(a, scalar);
}

Tensor sub(const Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return sub(a, s);
}

Tensor sub(float scalar, const Tensor& a) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return sub(s, a);
}

Tensor mul(const Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return mul(a, s);
}

Tensor mul(float scalar, const Tensor& a) {
    return mul(a, scalar);
}

Tensor div(const Tensor& a, float scalar) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return div(a, s);
}

Tensor div(float scalar, const Tensor& a) {
    Tensor s = Tensor::full(Shape({}), scalar, a.dtype(), a.device());
    return div(s, a);
}

// ============================================================================
// Unary and Activation Functions
// ============================================================================
Tensor neg(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::neg(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::neg(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::NegBackward>(res, a);
    return res;
}

Tensor abs(const Tensor& a) {
    if (a.is_cpu()) return cpu::abs(a);
#if AIC_WITH_CUDA
    if (a.is_cuda()) return cuda::abs(a);
#endif
    AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
}

Tensor exp(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::exp(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::exp(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::ExpBackward>(res, a, res);
    return res;
}

Tensor log(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::log(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::log(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::LogBackward>(res, a, a);
    return res;
}

Tensor sqrt(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::sqrt(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::sqrt(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::SqrtBackward>(res, a, res);
    return res;
}

Tensor tanh(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::tanh(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::tanh(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::TanhBackward>(res, a, res);
    return res;
}

Tensor sigmoid(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::sigmoid(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::sigmoid(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::SigmoidBackward>(res, a, res);
    return res;
}

Tensor gelu(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::gelu(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::gelu(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::GeluBackward>(res, a, a);
    return res;
}

Tensor silu(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::silu(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::silu(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::SiluBackward>(res, a, a);
    return res;
}

// ============================================================================
// Full Reductions
// ============================================================================
Tensor sum(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::sum(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::sum(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::SumBackward>(res, a, a.shape());
    return res;
}

Tensor mean(const Tensor& a) {
    Tensor res;
    if (a.is_cpu()) res = cpu::mean(a);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::mean(a);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::MeanBackward>(res, a, a.shape());
    return res;
}

Tensor max(const Tensor& a) {
    if (a.is_cpu()) return cpu::max(a);
#if AIC_WITH_CUDA
    if (a.is_cuda()) return cuda::max(a);
#endif
    AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
}

Tensor min(const Tensor& a) {
    if (a.is_cpu()) return cpu::min(a);
#if AIC_WITH_CUDA
    if (a.is_cuda()) return cuda::min(a);
#endif
    AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
}

// ============================================================================
// Dimensional Reductions
// ============================================================================
Tensor sum(const Tensor& a, int dim, bool keepdim) {
    Tensor res;
    if (a.is_cpu()) res = cpu::sum(a, dim, keepdim);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::sum(a, dim, keepdim);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::SumBackward>(res, a, a.shape(), dim, keepdim, true);
    return res;
}

Tensor mean(const Tensor& a, int dim, bool keepdim) {
    Tensor res;
    if (a.is_cpu()) res = cpu::mean(a, dim, keepdim);
#if AIC_WITH_CUDA
    else if (a.is_cuda()) res = cuda::mean(a, dim, keepdim);
#endif
    else AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
    attach_autograd_unary<autograd::MeanBackward>(res, a, a.shape(), dim, keepdim, true);
    return res;
}

Tensor max(const Tensor& a, int dim, bool keepdim) {
    if (a.is_cpu()) return cpu::max(a, dim, keepdim);
#if AIC_WITH_CUDA
    if (a.is_cuda()) return cuda::max(a, dim, keepdim);
#endif
    AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
}

Tensor min(const Tensor& a, int dim, bool keepdim) {
    if (a.is_cpu()) return cpu::min(a, dim, keepdim);
#if AIC_WITH_CUDA
    if (a.is_cuda()) return cuda::min(a, dim, keepdim);
#endif
    AIC_THROW(BackendUnavailable, "No backend for ", a.device().str());
}

// ============================================================================
// Sequence and Tensor Manipulation Operations
// ============================================================================
Tensor softmax(const Tensor& a, int dim) {
    int r = a.rank();
    if (dim < 0) dim += r;
    AIC_CHECK(dim >= 0 && dim < r, OutOfRange, "softmax dim out of range");

    // Numerically stable softmax: e^(x - max(x)) / sum(e^(x - max(x)))
    Tensor m = max(a, dim, /*keepdim=*/true);
    Tensor shifted = sub(a, m);
    Tensor ex = exp(shifted);
    Tensor s = sum(ex, dim, /*keepdim=*/true);
    return div(ex, s);
}

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    AIC_CHECK(!tensors.empty(), InvalidArgument, "cat expects a non-empty list of tensors");
    if (tensors.size() == 1) return tensors[0].clone();

    const Tensor& first = tensors[0];
    int r = first.rank();
    if (dim < 0) dim += r;
    AIC_CHECK(dim >= 0 && dim < r, OutOfRange, "cat dim out of range");

    int64_t total_mid = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor& t = tensors[i];
        AIC_CHECK(t.rank() == r, ShapeMismatch, "All tensors in cat must have the same rank");
        AIC_CHECK(t.device() == first.device(), DeviceMismatch, "All tensors in cat must be on the same device");
        AIC_CHECK(t.dtype() == first.dtype(), DTypeMismatch, "All tensors in cat must have the same dtype");
        for (int d = 0; d < r; ++d) {
            if (d != dim) {
                AIC_CHECK(t.dim(d) == first.dim(d), ShapeMismatch,
                          "Dimension mismatch in cat at non-concatenated dim ", d);
            }
        }
        total_mid += t.dim(dim);
    }

    std::vector<int64_t> out_dims(static_cast<size_t>(r));
    for (int d = 0; d < r; ++d) {
        out_dims[static_cast<size_t>(d)] = (d == dim) ? total_mid : first.dim(d);
    }

    Tensor out = Tensor::empty(Shape(std::move(out_dims)), first.dtype(), first.device());
    if (out.numel() == 0) return out;

    int64_t n_outer = 1;
    for (int d = 0; d < dim; ++d) n_outer *= first.dim(d);
    int64_t n_inner = 1;
    for (int d = dim + 1; d < r; ++d) n_inner *= first.dim(d);

    size_t itemsize = first.itemsize();
    char* dst_base = reinterpret_cast<char*>(out.raw_data());

    int64_t offset_mid = 0;
    for (const Tensor& t : tensors) {
        int64_t mid_k = t.dim(dim);
        if (mid_k == 0) continue;

        Tensor t_contig = t.contiguous();
        const char* src_base = reinterpret_cast<const char*>(t_contig.raw_data());
        size_t chunk_bytes = static_cast<size_t>(mid_k * n_inner) * itemsize;

        for (int64_t o = 0; o < n_outer; ++o) {
            const char* src = src_base + static_cast<size_t>(o * mid_k * n_inner) * itemsize;
            char* dst = dst_base + static_cast<size_t>((o * total_mid + offset_mid) * n_inner) * itemsize;

            if (first.is_cpu()) {
                std::memcpy(dst, src, chunk_bytes);
            } else {
#if AIC_WITH_CUDA
                cudaMemcpy(dst, src, chunk_bytes, cudaMemcpyDeviceToDevice);
#else
                AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
            }
        }
        offset_mid += mid_k;
    }

    return out;
}

}  // namespace aic
