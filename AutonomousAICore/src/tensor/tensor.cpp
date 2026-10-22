// src/tensor/tensor.cpp
#include "tensor/tensor.hpp"
#include "autograd/autograd_meta.hpp"
#include "autograd/autograd.hpp"
#include "autograd/functions.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#define AIC_CUDA_CHECK_TENSOR(call)                                                        \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            AIC_THROW(CudaError, "CUDA error in Tensor: ", cudaGetErrorString(err__));     \
        }                                                                                  \
    } while (false)
#endif

namespace aic {

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : shape_(std::move(shape)),
      strides_(shape_.default_strides()),
      dtype_(dtype),
      device_(device) {
    size_t bytes = static_cast<size_t>(shape_.numel()) * dtype_size(dtype_);
    storage_ = std::make_shared<Storage>(bytes, device_);
}

Tensor::Tensor(std::shared_ptr<Storage> storage, int64_t storage_offset,
               Shape shape, std::vector<int64_t> strides, DType dtype)
    : storage_(std::move(storage)),
      storage_offset_(storage_offset),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      dtype_(dtype),
      device_(storage_ ? storage_->device() : Device::cpu()) {
    AIC_CHECK(storage_offset_ >= 0, InvalidArgument, "storage_offset must be non-negative");
    AIC_CHECK(shape_.rank() == static_cast<int>(strides_.size()), InvalidArgument,
              "Rank and strides count mismatch: ", shape_.rank(), " vs ", strides_.size());
}

int64_t Tensor::stride(int i) const {
    int r = rank();
    if (r == 0) {
        AIC_THROW(OutOfRange, "Cannot index stride of a 0-D tensor");
    }
    int orig_i = i;
    if (i < 0) i += r;
    if (i < 0 || i >= r) {
        AIC_THROW(OutOfRange, "Stride index ", orig_i, " out of range for rank ", r);
    }
    return strides_[static_cast<size_t>(i)];
}

bool Tensor::is_contiguous() const noexcept {
    return Shape::is_contiguous(shape_, strides_);
}

void* Tensor::raw_data() noexcept {
    if (!storage_ || !storage_->data()) return nullptr;
    return static_cast<char*>(storage_->data()) + (storage_offset_ * itemsize());
}

const void* Tensor::raw_data() const noexcept {
    if (!storage_ || !storage_->data()) return nullptr;
    return static_cast<const char*>(storage_->data()) + (storage_offset_ * itemsize());
}

Tensor Tensor::view(const Shape& target_shape) const {
    AIC_CHECK(target_shape.numel() == numel(), ShapeMismatch,
              "view shape ", target_shape.str(), " numel (", target_shape.numel(),
              ") must match tensor numel (", numel(), ")");
    AIC_CHECK(is_contiguous(), InvalidArgument,
              "view() requires a contiguous tensor. Use .reshape() or .contiguous().view()");
    return Tensor(storage_, storage_offset_, target_shape, target_shape.default_strides(), dtype_);
}

namespace {
inline autograd::Edge make_tensor_edge(const Tensor& t) {
    if (!t.requires_grad()) return autograd::Edge{nullptr, 0};
    return autograd::Edge{t.grad_fn() ? t.grad_fn() : std::make_shared<autograd::AccumulateGrad>(t), 0};
}
}  // namespace

Tensor Tensor::reshape(const Shape& target_shape) const {
    AIC_CHECK(target_shape.numel() == numel(), ShapeMismatch,
              "reshape target shape ", target_shape.str(), " numel (", target_shape.numel(),
              ") must match tensor numel (", numel(), ")");
    Tensor res = is_contiguous() ? view(target_shape) : contiguous().view(target_shape);
    if (autograd::GradMode::is_enabled() && requires_grad()) {
        res.set_requires_grad(true);
        res.set_leaf(false);
        auto node = std::make_shared<autograd::ReshapeBackward>(shape_);
        node->add_next_edge(make_tensor_edge(*this));
        res.set_grad_fn(std::move(node));
    }
    return res;
}

Tensor Tensor::squeeze(int dim) const {
    int r = rank();
    int normalized_dim = dim < 0 ? dim + r : dim;
    if (normalized_dim < 0 || normalized_dim >= r) {
        AIC_THROW(OutOfRange, "squeeze dim ", dim, " out of range for rank ", r);
    }
    if (shape_[normalized_dim] != 1) {
        return *this;
    }

    Shape new_shape = shape_.squeeze(normalized_dim);
    std::vector<int64_t> new_strides;
    new_strides.reserve(static_cast<size_t>(r - 1));
    for (int i = 0; i < r; ++i) {
        if (i != normalized_dim) {
            new_strides.push_back(strides_[static_cast<size_t>(i)]);
        }
    }
    Tensor res(storage_, storage_offset_, std::move(new_shape), std::move(new_strides), dtype_);
    if (autograd::GradMode::is_enabled() && requires_grad()) {
        res.set_requires_grad(true);
        res.set_leaf(false);
        auto node = std::make_shared<autograd::ReshapeBackward>(shape_);
        node->add_next_edge(make_tensor_edge(*this));
        res.set_grad_fn(std::move(node));
    }
    return res;
}

Tensor Tensor::squeeze() const {
    Tensor t = *this;
    for (int i = rank() - 1; i >= 0; --i) {
        if (t.dim(i) == 1) {
            t = t.squeeze(i);
        }
    }
    return t;
}

Tensor Tensor::unsqueeze(int dim) const {
    int r = rank();
    int normalized_dim = dim < 0 ? dim + r + 1 : dim;
    if (normalized_dim < 0 || normalized_dim > r) {
        AIC_THROW(OutOfRange, "unsqueeze dim ", dim, " out of range for rank ", r);
    }

    Shape new_shape = shape_.unsqueeze(normalized_dim);
    std::vector<int64_t> new_strides;
    new_strides.reserve(static_cast<size_t>(r + 1));
    for (int i = 0; i < normalized_dim; ++i) {
        new_strides.push_back(strides_[static_cast<size_t>(i)]);
    }
    int64_t new_stride = 1;
    if (normalized_dim < r) {
        new_stride = strides_[static_cast<size_t>(normalized_dim)] * shape_[static_cast<size_t>(normalized_dim)];
    }
    new_strides.push_back(new_stride);
    for (int i = normalized_dim; i < r; ++i) {
        new_strides.push_back(strides_[static_cast<size_t>(i)]);
    }
    Tensor res(storage_, storage_offset_, std::move(new_shape), std::move(new_strides), dtype_);
    if (autograd::GradMode::is_enabled() && requires_grad()) {
        res.set_requires_grad(true);
        res.set_leaf(false);
        auto node = std::make_shared<autograd::ReshapeBackward>(shape_);
        node->add_next_edge(make_tensor_edge(*this));
        res.set_grad_fn(std::move(node));
    }
    return res;
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    int r = rank();
    AIC_CHECK(static_cast<int>(dims.size()) == r, InvalidArgument,
              "permute requires exactly ", r, " dimensions, got ", dims.size());

    std::vector<bool> seen(static_cast<size_t>(r), false);
    std::vector<int64_t> new_dims(static_cast<size_t>(r));
    std::vector<int64_t> new_strides(static_cast<size_t>(r));

    for (int i = 0; i < r; ++i) {
        int d = dims[static_cast<size_t>(i)];
        if (d < 0) d += r;
        AIC_CHECK(d >= 0 && d < r, OutOfRange, "permute dimension ", dims[static_cast<size_t>(i)], " out of range");
        AIC_CHECK(!seen[static_cast<size_t>(d)], InvalidArgument, "duplicate dimension in permute: ", d);
        seen[static_cast<size_t>(d)] = true;
        new_dims[static_cast<size_t>(i)] = shape_[d];
        new_strides[static_cast<size_t>(i)] = strides_[static_cast<size_t>(d)];
    }

    Tensor res(storage_, storage_offset_, Shape(std::move(new_dims)), std::move(new_strides), dtype_);
    if (autograd::GradMode::is_enabled() && requires_grad()) {
        res.set_requires_grad(true);
        res.set_leaf(false);
        auto node = std::make_shared<autograd::PermuteBackward>(dims);
        node->add_next_edge(make_tensor_edge(*this));
        res.set_grad_fn(std::move(node));
    }
    return res;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    int r = rank();
    if (dim0 < 0) dim0 += r;
    if (dim1 < 0) dim1 += r;
    AIC_CHECK(dim0 >= 0 && dim0 < r, OutOfRange, "transpose dim0 out of range");
    AIC_CHECK(dim1 >= 0 && dim1 < r, OutOfRange, "transpose dim1 out of range");

    std::vector<int> p(static_cast<size_t>(r));
    std::iota(p.begin(), p.end(), 0);
    std::swap(p[static_cast<size_t>(dim0)], p[static_cast<size_t>(dim1)]);
    return permute(p);
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end, int64_t step) const {
    int r = rank();
    if (dim < 0) dim += r;
    AIC_CHECK(dim >= 0 && dim < r, OutOfRange, "slice dim out of range");
    AIC_CHECK(step > 0, InvalidArgument, "slice step must be positive");

    int64_t dim_size = shape_[dim];
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    if (start < 0) start = 0;
    if (start > dim_size) start = dim_size;
    if (end < start) end = start;
    if (end > dim_size) end = dim_size;

    int64_t new_dim_size = (end - start + step - 1) / step;
    std::vector<int64_t> new_dims(static_cast<size_t>(r));
    std::vector<int64_t> new_strides = strides_;
    for (int i = 0; i < r; ++i) {
        new_dims[static_cast<size_t>(i)] = shape_[i];
    }
    new_dims[static_cast<size_t>(dim)] = new_dim_size;
    new_strides[static_cast<size_t>(dim)] = strides_[static_cast<size_t>(dim)] * step;

    int64_t new_offset = storage_offset_ + start * strides_[static_cast<size_t>(dim)];
    return Tensor(storage_, new_offset, Shape(std::move(new_dims)), std::move(new_strides), dtype_);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return clone();
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_, device_);
    int64_t total = numel();
    if (total == 0) return result;

    if (is_contiguous()) {
        size_t bytes = nbytes();
        if (is_cpu()) {
            std::memcpy(result.raw_data(), raw_data(), bytes);
        } else {
#if AIC_WITH_CUDA
            AIC_CUDA_CHECK_TENSOR(cudaMemcpy(result.raw_data(), raw_data(), bytes, cudaMemcpyDeviceToDevice));
#else
            AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
        }
        return result;
    }

    // Non-contiguous copy: iterate multi-index
    if (is_cpu()) {
        const char* src_base = static_cast<const char*>(storage_->data());
        char* dst_base = static_cast<char*>(result.raw_data());
        size_t sz = itemsize();
        int r = rank();

        std::vector<int64_t> coords(static_cast<size_t>(r), 0);
        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t src_offset = storage_offset_;
            for (int i = 0; i < r; ++i) {
                src_offset += coords[static_cast<size_t>(i)] * strides_[static_cast<size_t>(i)];
            }
            std::memcpy(dst_base + linear * sz, src_base + src_offset * sz, sz);

            // Increment coordinates
            for (int i = r - 1; i >= 0; --i) {
                coords[static_cast<size_t>(i)]++;
                if (coords[static_cast<size_t>(i)] < shape_[i]) {
                    break;
                }
                coords[static_cast<size_t>(i)] = 0;
            }
        }
    } else {
        // For GPU non-contiguous: transfer to CPU (which flattens without recursion) and transfer back
        Tensor cpu_tensor = this->to(Device::cpu());
        return cpu_tensor.to(device_);
    }
    return result;
}

Tensor Tensor::to(Device target_device) const {
    if (device_ == target_device) {
        return *this;
    }

    if (!is_contiguous()) {
        if (is_cpu()) {
            return clone().to(target_device);
        }
#if AIC_WITH_CUDA
        else if (is_cuda()) {
            size_t bytes = storage_->nbytes();
            auto cpu_storage = std::make_shared<Storage>(bytes, Device::cpu());
            AIC_CUDA_CHECK_TENSOR(cudaSetDevice(device_.index));
            AIC_CUDA_CHECK_TENSOR(cudaMemcpy(cpu_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToHost));
            Tensor cpu_view(cpu_storage, storage_offset_, shape_, strides_, dtype_);
            Tensor cpu_contig = cpu_view.contiguous();
            return cpu_contig.to(target_device);
        }
#endif
    }

    Tensor result(shape_, dtype_, target_device);
    size_t bytes = nbytes();
    if (bytes == 0) return result;

    if (is_cpu() && target_device.is_cpu()) {
        std::memcpy(result.raw_data(), raw_data(), bytes);
    }
#if AIC_WITH_CUDA
    else if (is_cpu() && target_device.is_cuda()) {
        AIC_CUDA_CHECK_TENSOR(cudaSetDevice(target_device.index));
        AIC_CUDA_CHECK_TENSOR(cudaMemcpy(result.raw_data(), raw_data(), bytes, cudaMemcpyHostToDevice));
    } else if (is_cuda() && target_device.is_cpu()) {
        AIC_CUDA_CHECK_TENSOR(cudaSetDevice(device_.index));
        AIC_CUDA_CHECK_TENSOR(cudaMemcpy(result.raw_data(), raw_data(), bytes, cudaMemcpyDeviceToHost));
    } else if (is_cuda() && target_device.is_cuda()) {
        if (device_.index == target_device.index) {
            AIC_CUDA_CHECK_TENSOR(cudaMemcpy(result.raw_data(), raw_data(), bytes, cudaMemcpyDeviceToDevice));
        } else {
            AIC_CUDA_CHECK_TENSOR(cudaMemcpyPeer(result.raw_data(), target_device.index,
                                                 raw_data(), device_.index, bytes));
        }
    }
#else
    else {
        AIC_THROW(BackendUnavailable, "CUDA not enabled in build to transfer between ",
                  device_.str(), " and ", target_device.str());
    }
#endif
    return result;
}

Tensor Tensor::to(DType target_dtype) const {
    if (dtype_ == target_dtype) {
        return *this;
    }

    // Cast on CPU (convert to CPU if on GPU, cast, and transfer back)
    if (is_cuda()) {
        return this->to(Device::cpu()).to(target_dtype).to(device_);
    }

    Tensor contig = this->contiguous();
    Tensor result(contig.shape(), target_dtype, contig.device());
    int64_t total = contig.numel();
    if (total == 0) return result;

    // Helper casting lambda
    auto cast_loop = [&](auto from_tag, auto to_tag) {
        using FromT = decltype(from_tag);
        using ToT = decltype(to_tag);
        const FromT* src = contig.data_ptr<FromT>();
        ToT* dst = result.data_ptr<ToT>();
        for (int64_t i = 0; i < total; ++i) {
            if constexpr (std::is_same_v<FromT, Half>) {
                float f = half_to_float(src[i].bits);
                if constexpr (std::is_same_v<ToT, Half>) dst[i].bits = src[i].bits;
                else if constexpr (std::is_same_v<ToT, BFloat16>) dst[i].bits = float_to_bf16(f);
                else dst[i] = static_cast<ToT>(f);
            } else if constexpr (std::is_same_v<FromT, BFloat16>) {
                float f = bf16_to_float(src[i].bits);
                if constexpr (std::is_same_v<ToT, Half>) dst[i].bits = float_to_half(f);
                else if constexpr (std::is_same_v<ToT, BFloat16>) dst[i].bits = src[i].bits;
                else dst[i] = static_cast<ToT>(f);
            } else {
                if constexpr (std::is_same_v<ToT, Half>) dst[i].bits = float_to_half(static_cast<float>(src[i]));
                else if constexpr (std::is_same_v<ToT, BFloat16>) dst[i].bits = float_to_bf16(static_cast<float>(src[i]));
                else dst[i] = static_cast<ToT>(src[i]);
            }
        }
    };

    // Dispatch table for common types
    if (dtype_ == DType::Float32 && target_dtype == DType::Float16) cast_loop(float{}, Half{});
    else if (dtype_ == DType::Float32 && target_dtype == DType::BFloat16) cast_loop(float{}, BFloat16{});
    else if (dtype_ == DType::Float16 && target_dtype == DType::Float32) cast_loop(Half{}, float{});
    else if (dtype_ == DType::BFloat16 && target_dtype == DType::Float32) cast_loop(BFloat16{}, float{});
    else if (dtype_ == DType::Float32 && target_dtype == DType::Int64) cast_loop(float{}, int64_t{});
    else if (dtype_ == DType::Int64 && target_dtype == DType::Float32) cast_loop(int64_t{}, float{});
    else if (dtype_ == DType::Float32 && target_dtype == DType::Int32) cast_loop(float{}, int32_t{});
    else if (dtype_ == DType::Int32 && target_dtype == DType::Float32) cast_loop(int32_t{}, float{});
    else if (dtype_ == DType::Int64 && target_dtype == DType::Int32) cast_loop(int64_t{}, int32_t{});
    else if (dtype_ == DType::Int32 && target_dtype == DType::Int64) cast_loop(int32_t{}, int64_t{});
    else {
        AIC_THROW(NotImplemented, "Type cast from ", dtype_name(dtype_), " to ",
                  dtype_name(target_dtype), " not implemented yet");
    }

    return result;
}

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
    return Tensor(std::move(shape), dtype, device);
}

Tensor Tensor::zeros(Shape shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    if (t.nbytes() == 0) return t;

    if (t.is_cpu()) {
        std::memset(t.raw_data(), 0, t.nbytes());
    } else {
#if AIC_WITH_CUDA
        AIC_CUDA_CHECK_TENSOR(cudaSetDevice(t.device().index));
        AIC_CUDA_CHECK_TENSOR(cudaMemset(t.raw_data(), 0, t.nbytes()));
#else
        AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
    }
    return t;
}

Tensor Tensor::ones(Shape shape, DType dtype, Device device) {
    return full(std::move(shape), 1.0f, dtype, device);
}

Tensor Tensor::full(Shape shape, float value, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    if (t.numel() == 0) return t;

    Tensor cpu_t = t.is_cpu() ? t : Tensor::empty(t.shape(), dtype, Device::cpu());
    int64_t n = cpu_t.numel();

    if (dtype == DType::Float32) {
        float* p = cpu_t.data_ptr<float>();
        std::fill_n(p, n, value);
    } else if (dtype == DType::Float16) {
        Half* p = cpu_t.data_ptr<Half>();
        Half h{float_to_half(value)};
        std::fill_n(p, n, h);
    } else if (dtype == DType::BFloat16) {
        BFloat16* p = cpu_t.data_ptr<BFloat16>();
        BFloat16 b{float_to_bf16(value)};
        std::fill_n(p, n, b);
    } else if (dtype == DType::Int32) {
        int32_t* p = cpu_t.data_ptr<int32_t>();
        std::fill_n(p, n, static_cast<int32_t>(value));
    } else if (dtype == DType::Int64) {
        int64_t* p = cpu_t.data_ptr<int64_t>();
        std::fill_n(p, n, static_cast<int64_t>(value));
    } else {
        AIC_THROW(NotImplemented, "full() not implemented for dtype ", dtype_name(dtype));
    }

    if (t.is_cuda()) {
        return cpu_t.to(device);
    }
    return cpu_t;
}

Tensor Tensor::arange(float start, float end, float step, DType dtype, Device device) {
    AIC_CHECK(step != 0.0f, InvalidArgument, "step must not be zero");
    int64_t count = static_cast<int64_t>(std::ceil((end - start) / step));
    if (count < 0) count = 0;

    Tensor t = Tensor::empty(Shape({count}), dtype, Device::cpu());
    if (dtype == DType::Float32) {
        float* p = t.data_ptr<float>();
        for (int64_t i = 0; i < count; ++i) {
            p[i] = start + static_cast<float>(i) * step;
        }
    } else if (dtype == DType::Int64) {
        int64_t* p = t.data_ptr<int64_t>();
        for (int64_t i = 0; i < count; ++i) {
            p[i] = static_cast<int64_t>(start + static_cast<float>(i) * step);
        }
    } else if (dtype == DType::Int32) {
        int32_t* p = t.data_ptr<int32_t>();
        for (int64_t i = 0; i < count; ++i) {
            p[i] = static_cast<int32_t>(start + static_cast<float>(i) * step);
        }
    } else {
        AIC_THROW(NotImplemented, "arange() not implemented for dtype ", dtype_name(dtype));
    }

    if (device.is_cuda()) {
        return t.to(device);
    }
    return t;
}

Tensor Tensor::randn(Shape shape, float mean, float std, DType dtype, Device device) {
    AIC_CHECK(dtype == DType::Float32, NotImplemented, "randn currently supports Float32");
    Tensor t = Tensor::empty(shape, dtype, Device::cpu());
    int64_t n = shape.numel();
    float* p = t.data_ptr<float>();

    static thread_local std::mt19937 gen(1337);
    std::normal_distribution<float> dist(mean, std);
    for (int64_t i = 0; i < n; ++i) {
        p[i] = dist(gen);
    }
    if (device.is_cuda()) {
        return t.to(device);
    }
    return t;
}

Tensor Tensor::uniform(Shape shape, float low, float high, DType dtype, Device device) {
    AIC_CHECK(dtype == DType::Float32, NotImplemented, "uniform currently supports Float32");
    Tensor t = Tensor::empty(shape, dtype, Device::cpu());
    int64_t n = shape.numel();
    float* p = t.data_ptr<float>();

    static thread_local std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(low, high);
    for (int64_t i = 0; i < n; ++i) {
        p[i] = dist(gen);
    }
    if (device.is_cuda()) {
        return t.to(device);
    }
    return t;
}

Tensor Tensor::from_blob(void* data, Shape shape, DType dtype, Device device) {
    AIC_CHECK(data != nullptr, InvalidArgument, "data pointer must not be null in from_blob");
    size_t bytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);
    auto storage = std::make_shared<Storage>(data, bytes, device, /*owns_memory=*/false);
    std::vector<int64_t> strides = shape.default_strides();
    return Tensor(std::move(storage), 0, std::move(shape), std::move(strides), dtype);
}

std::string Tensor::summary() const {
    std::ostringstream os;
    os << "Tensor(shape=" << shape_.str()
       << ", dtype=" << dtype_name(dtype_)
       << ", device=" << device_.str()
       << ", contiguous=" << (is_contiguous() ? "true" : "false")
       << ", numel=" << numel()
       << ")";
    return os.str();
}

std::string Tensor::to_string(int64_t max_elements) const {
    std::ostringstream os;
    os << summary() << "\n[";
    if (numel() == 0) {
        os << "]";
        return os.str();
    }

    Tensor cpu_t = this->to(Device::cpu()).contiguous();
    int64_t print_count = std::min(numel(), max_elements);

    for (int64_t i = 0; i < print_count; ++i) {
        if (i > 0) os << ", ";
        if (dtype_ == DType::Float32) {
            os << cpu_t.data_ptr<float>()[i];
        } else if (dtype_ == DType::Float16) {
            os << half_to_float(cpu_t.data_ptr<Half>()[i].bits);
        } else if (dtype_ == DType::BFloat16) {
            os << bf16_to_float(cpu_t.data_ptr<BFloat16>()[i].bits);
        } else if (dtype_ == DType::Int32) {
            os << cpu_t.data_ptr<int32_t>()[i];
        } else if (dtype_ == DType::Int64) {
            os << cpu_t.data_ptr<int64_t>()[i];
        } else if (dtype_ == DType::UInt8) {
            os << static_cast<int>(cpu_t.data_ptr<uint8_t>()[i]);
        }
    }
    if (print_count < numel()) {
        os << ", ... (" << (numel() - print_count) << " more)";
    }
    os << "]";
    return os.str();
}

void Tensor::ensure_autograd_meta() const {
    if (!autograd_meta_) {
        autograd_meta_ = std::make_shared<autograd::AutogradMeta>();
    }
}

bool Tensor::requires_grad() const noexcept {
    return autograd_meta_ ? autograd_meta_->requires_grad : false;
}

void Tensor::set_requires_grad(bool req) const {
    if (req) {
        ensure_autograd_meta();
        autograd_meta_->requires_grad = true;
    } else if (autograd_meta_) {
        autograd_meta_->requires_grad = false;
    }
}

bool Tensor::is_leaf() const noexcept {
    return autograd_meta_ ? autograd_meta_->is_leaf : true;
}

void Tensor::set_leaf(bool leaf) const noexcept {
    if (leaf) {
        if (autograd_meta_) autograd_meta_->is_leaf = true;
    } else {
        ensure_autograd_meta();
        autograd_meta_->is_leaf = false;
    }
}

Tensor Tensor::grad() const {
    if (autograd_meta_ && autograd_meta_->grad) {
        return *autograd_meta_->grad;
    }
    return Tensor();
}

void Tensor::set_grad(const Tensor& g) const {
    ensure_autograd_meta();
    autograd_meta_->grad = std::make_shared<Tensor>(g);
}

void Tensor::zero_grad() const {
    if (autograd_meta_) {
        autograd_meta_->grad = nullptr;
    }
}

std::shared_ptr<autograd::Node> Tensor::grad_fn() const noexcept {
    return autograd_meta_ ? autograd_meta_->grad_fn : nullptr;
}

void Tensor::set_grad_fn(std::shared_ptr<autograd::Node> fn) const {
    ensure_autograd_meta();
    autograd_meta_->grad_fn = std::move(fn);
}

uint64_t Tensor::version() const noexcept {
    return autograd_meta_ ? autograd_meta_->version : 0;
}

void Tensor::bump_version() const noexcept {
    if (autograd_meta_) {
        autograd_meta_->version++;
    }
}

Tensor Tensor::detach() const {
    Tensor t = *this;
    t.autograd_meta_ = nullptr;
    return t;
}

void Tensor::backward(const Tensor& gradient, bool retain_graph) const {
    autograd::backward(*this, gradient, retain_graph);
}

}  // namespace aic
