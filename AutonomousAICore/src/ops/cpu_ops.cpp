// src/ops/cpu_ops.cpp
#include "ops/cpu_ops.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace aic::cpu {

namespace {

// Broadcast Binary Loop helper
template <typename Op>
Tensor binary_broadcast_cpu(const Tensor& a, const Tensor& b, Op&& op) {
    AIC_CHECK(a.dtype() == DType::Float32 && b.dtype() == DType::Float32,
              DTypeMismatch, "CPU binary ops currently require Float32 tensors");

    Shape out_shape = Shape::broadcast(a.shape(), b.shape());
    Tensor out = Tensor::empty(out_shape, DType::Float32, Device::cpu());
    int64_t total = out_shape.numel();
    if (total == 0) return out;

    // Fast-path: exactly identical shapes and both contiguous
    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        const float* p_a = a.data_ptr<float>();
        const float* p_b = b.data_ptr<float>();
        float* p_out = out.data_ptr<float>();
        for (int64_t i = 0; i < total; ++i) {
            p_out[i] = op(p_a[i], p_b[i]);
        }
        return out;
    }

    // General broadcast and non-contiguous path
    int r = out_shape.rank();
    auto strides_a = Shape::broadcast_strides(a.shape(), a.strides(), out_shape);
    auto strides_b = Shape::broadcast_strides(b.shape(), b.strides(), out_shape);

    const float* base_a = static_cast<const float*>(a.storage()->data()) + a.storage_offset();
    const float* base_b = static_cast<const float*>(b.storage()->data()) + b.storage_offset();
    float* p_out = out.data_ptr<float>();

    std::vector<int64_t> coords(static_cast<size_t>(r), 0);
    for (int64_t i = 0; i < total; ++i) {
        int64_t off_a = 0;
        int64_t off_b = 0;
        for (int k = 0; k < r; ++k) {
            off_a += coords[static_cast<size_t>(k)] * strides_a[static_cast<size_t>(k)];
            off_b += coords[static_cast<size_t>(k)] * strides_b[static_cast<size_t>(k)];
        }
        p_out[i] = op(base_a[off_a], base_b[off_b]);

        // Increment coords
        for (int k = r - 1; k >= 0; --k) {
            coords[static_cast<size_t>(k)]++;
            if (coords[static_cast<size_t>(k)] < out_shape[k]) {
                break;
            }
            coords[static_cast<size_t>(k)] = 0;
        }
    }
    return out;
}

// Unary Loop helper
template <typename Op>
Tensor unary_cpu(const Tensor& a, Op&& op) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CPU unary ops require Float32 tensors");
    Tensor contig = a.contiguous();
    Tensor out = Tensor::empty(contig.shape(), DType::Float32, Device::cpu());
    int64_t total = contig.numel();
    const float* src = contig.data_ptr<float>();
    float* dst = out.data_ptr<float>();
    for (int64_t i = 0; i < total; ++i) {
        dst[i] = op(src[i]);
    }
    return out;
}

}  // namespace

// ============================================================================
// Elementwise Binary Implementations
// ============================================================================

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_broadcast_cpu(a, b, [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_broadcast_cpu(a, b, [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_broadcast_cpu(a, b, [](float x, float y) { return x * y; });
}

Tensor div(const Tensor& a, const Tensor& b) {
    return binary_broadcast_cpu(a, b, [](float x, float y) { return x / y; });
}

// ============================================================================
// Unary Functions
// ============================================================================

Tensor neg(const Tensor& a) {
    return unary_cpu(a, [](float x) { return -x; });
}

Tensor abs(const Tensor& a) {
    return unary_cpu(a, [](float x) { return std::abs(x); });
}

Tensor exp(const Tensor& a) {
    return unary_cpu(a, [](float x) { return std::exp(x); });
}

Tensor log(const Tensor& a) {
    return unary_cpu(a, [](float x) { return std::log(x); });
}

Tensor sqrt(const Tensor& a) {
    return unary_cpu(a, [](float x) { return std::sqrt(x); });
}

Tensor tanh(const Tensor& a) {
    return unary_cpu(a, [](float x) { return std::tanh(x); });
}

Tensor sigmoid(const Tensor& a) {
    return unary_cpu(a, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

Tensor gelu(const Tensor& a) {
    // Fast tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float kSqrt2OverPi = 0.7978845608f;
    return unary_cpu(a, [=](float x) {
        float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + std::tanh(inner));
    });
}

Tensor silu(const Tensor& a) {
    // x * sigmoid(x)
    return unary_cpu(a, [](float x) { return x / (1.0f + std::exp(-x)); });
}

// ============================================================================
// Reductions
// ============================================================================

Tensor sum(const Tensor& a) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CPU sum requires Float32 tensor");
    Tensor contig = a.contiguous();
    int64_t total = contig.numel();
    const float* src = contig.data_ptr<float>();
    double acc = 0.0;
    for (int64_t i = 0; i < total; ++i) {
        acc += static_cast<double>(src[i]);
    }
    Tensor out = Tensor::empty(Shape({}), DType::Float32, Device::cpu());
    *out.data_ptr<float>() = static_cast<float>(acc);
    return out;
}

Tensor mean(const Tensor& a) {
    AIC_CHECK(a.numel() > 0, InvalidArgument, "Cannot compute mean of empty tensor");
    Tensor total_sum = sum(a);
    *total_sum.data_ptr<float>() /= static_cast<float>(a.numel());
    return total_sum;
}

Tensor max(const Tensor& a) {
    AIC_CHECK(a.numel() > 0, InvalidArgument, "Cannot compute max of empty tensor");
    Tensor contig = a.contiguous();
    int64_t total = contig.numel();
    const float* src = contig.data_ptr<float>();
    float m = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < total; ++i) {
        if (src[i] > m) m = src[i];
    }
    Tensor out = Tensor::empty(Shape({}), DType::Float32, Device::cpu());
    *out.data_ptr<float>() = m;
    return out;
}

Tensor min(const Tensor& a) {
    AIC_CHECK(a.numel() > 0, InvalidArgument, "Cannot compute min of empty tensor");
    Tensor contig = a.contiguous();
    int64_t total = contig.numel();
    const float* src = contig.data_ptr<float>();
    float m = std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < total; ++i) {
        if (src[i] < m) m = src[i];
    }
    Tensor out = Tensor::empty(Shape({}), DType::Float32, Device::cpu());
    *out.data_ptr<float>() = m;
    return out;
}

namespace {

enum class ReductionType { Sum, Mean, Max, Min };

Tensor reduce_dim_cpu(const Tensor& a, int dim, bool keepdim, ReductionType type) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CPU reduction requires Float32");
    int r = a.rank();
    int norm_dim = dim < 0 ? dim + r : dim;
    AIC_CHECK(norm_dim >= 0 && norm_dim < r, OutOfRange, "Reduction dim ", dim, " out of range for rank ", r);

    int64_t reduce_size = a.dim(norm_dim);
    AIC_CHECK(reduce_size > 0, InvalidArgument, "Cannot reduce along dimension of size 0");

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<size_t>(keepdim ? r : r - 1));
    for (int i = 0; i < r; ++i) {
        if (i == norm_dim) {
            if (keepdim) out_dims.push_back(1);
        } else {
            out_dims.push_back(a.dim(i));
        }
    }

    Tensor out = Tensor::empty(Shape(std::move(out_dims)), DType::Float32, Device::cpu());
    int64_t out_numel = out.numel();
    if (out_numel == 0) return out;

    Tensor a_contig = a.contiguous();
    const float* src = a_contig.data_ptr<float>();
    float* dst = out.data_ptr<float>();

    // Calculate stride of the reduced dimension
    int64_t stride_reduce = a_contig.stride(norm_dim);

    // Number of elements before and after the reduced dimension
    int64_t outer_size = 1;
    for (int i = 0; i < norm_dim; ++i) outer_size *= a_contig.dim(i);
    int64_t inner_size = 1;
    for (int i = norm_dim + 1; i < r; ++i) inner_size *= a_contig.dim(i);

    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t base_idx = outer * (reduce_size * inner_size) + inner;
            int64_t out_idx = outer * inner_size + inner;

            double acc = 0.0;
            float max_val = -std::numeric_limits<float>::infinity();
            float min_val = std::numeric_limits<float>::infinity();

            for (int64_t k = 0; k < reduce_size; ++k) {
                float val = src[base_idx + k * stride_reduce];
                if (type == ReductionType::Sum || type == ReductionType::Mean) {
                    acc += static_cast<double>(val);
                } else if (type == ReductionType::Max) {
                    if (val > max_val) max_val = val;
                } else if (type == ReductionType::Min) {
                    if (val < min_val) min_val = val;
                }
            }

            if (type == ReductionType::Sum) dst[out_idx] = static_cast<float>(acc);
            else if (type == ReductionType::Mean) dst[out_idx] = static_cast<float>(acc / reduce_size);
            else if (type == ReductionType::Max) dst[out_idx] = max_val;
            else if (type == ReductionType::Min) dst[out_idx] = min_val;
        }
    }

    return out;
}

}  // namespace

Tensor sum(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cpu(a, dim, keepdim, ReductionType::Sum);
}

Tensor mean(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cpu(a, dim, keepdim, ReductionType::Mean);
}

Tensor max(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cpu(a, dim, keepdim, ReductionType::Max);
}

Tensor min(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cpu(a, dim, keepdim, ReductionType::Min);
}

}  // namespace aic::cpu
