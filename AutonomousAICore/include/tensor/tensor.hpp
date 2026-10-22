// include/tensor/tensor.hpp
// Core Tensor abstraction: multidimensional array with strided views, device abstraction, and RAII memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/error.hpp"
#include "core/shape.hpp"
#include "memory/storage.hpp"

namespace aic {

namespace autograd {
class Node;
struct AutogradMeta;
}

class Tensor {
public:
    Tensor() = default;

    // Construct a tensor with newly allocated storage
    Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());

    // Construct a view into existing storage
    Tensor(std::shared_ptr<Storage> storage, int64_t storage_offset,
           Shape shape, std::vector<int64_t> strides, DType dtype);

    ~Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Metadata queries
    const Shape& shape() const noexcept { return shape_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }
    int rank() const noexcept { return shape_.rank(); }
    int64_t dim(int i) const { return shape_[i]; }
    int64_t stride(int i) const;
    int64_t numel() const noexcept { return shape_.numel(); }
    size_t itemsize() const noexcept { return dtype_size(dtype_); }
    size_t nbytes() const noexcept { return static_cast<size_t>(numel()) * itemsize(); }

    DType dtype() const noexcept { return dtype_; }
    Device device() const noexcept { return device_; }
    bool is_cpu() const noexcept { return device_.is_cpu(); }
    bool is_cuda() const noexcept { return device_.is_cuda(); }

    bool is_contiguous() const noexcept;
    int64_t storage_offset() const noexcept { return storage_offset_; }
    std::shared_ptr<Storage> storage() const noexcept { return storage_; }
    bool defined() const noexcept { return storage_ != nullptr; }

    // Direct data access (with bounds/device verification)
    void* raw_data() noexcept;
    const void* raw_data() const noexcept;

    template <typename T>
    T* data_ptr() {
        check_dtype_match<T>();
        return reinterpret_cast<T*>(raw_data());
    }

    template <typename T>
    const T* data_ptr() const {
        check_dtype_match<T>();
        return reinterpret_cast<const T*>(raw_data());
    }

    // Scalar extraction (0-D or 1-element tensor)
    template <typename T>
    T item() const {
        AIC_CHECK(numel() == 1, InvalidArgument, "item() can only be called on tensors with 1 element, got ", numel());
        AIC_CHECK(is_cpu(), DeviceMismatch, "item() can only be called on CPU tensors, call .cpu() first");
        return *data_ptr<T>();
    }

    // View operations (zero-copy, returns a new Tensor sharing storage)
    Tensor view(const Shape& target_shape) const;
    Tensor reshape(const Shape& target_shape) const;
    Tensor squeeze(int dim) const;
    Tensor squeeze() const;
    Tensor unsqueeze(int dim) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;

    // Makes sure the tensor is contiguous; if already contiguous, returns *this, otherwise clones
    Tensor contiguous() const;

    // Deep copy
    Tensor clone() const;

    // Device and type casting
    Tensor to(Device target_device) const;
    Tensor to(DType target_dtype) const;
    Tensor cpu() const { return to(Device::cpu()); }
    Tensor cuda(int device_index = 0) const { return to(Device::cuda(device_index)); }

    // Factory methods
    static Tensor empty(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor zeros(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor ones(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor full(Shape shape, float value, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor arange(float start, float end, float step = 1.0f,
                         DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor randn(Shape shape, float mean = 0.0f, float std = 1.0f,
                        DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor uniform(Shape shape, float low = 0.0f, float high = 1.0f,
                          DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor from_blob(void* data, Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());

    // Autograd interface
    bool requires_grad() const noexcept;
    void set_requires_grad(bool req) const;
    bool is_leaf() const noexcept;
    void set_leaf(bool leaf) const noexcept;
    Tensor grad() const;
    void set_grad(const Tensor& g) const;
    void zero_grad() const;
    std::shared_ptr<autograd::Node> grad_fn() const noexcept;
    void set_grad_fn(std::shared_ptr<autograd::Node> fn) const;
    uint64_t version() const noexcept;
    void bump_version() const noexcept;
    Tensor detach() const;
    void backward(const Tensor& gradient = Tensor(), bool retain_graph = false) const;

    // Printing and inspection
    std::string summary() const;
    std::string to_string(int64_t max_elements = 20) const;

private:
    void ensure_autograd_meta() const;
    mutable std::shared_ptr<autograd::AutogradMeta> autograd_meta_;
    template <typename T>
    void check_dtype_match() const {
        AIC_CHECK(DTypeOf<T>::value == dtype_, DTypeMismatch,
                  "Requested data_ptr<", dtype_name(DTypeOf<T>::value),
                  "> does not match tensor dtype ", dtype_name(dtype_));
    }

    std::shared_ptr<Storage> storage_;
    int64_t storage_offset_ = 0;
    Shape shape_;
    std::vector<int64_t> strides_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::cpu();
};

}  // namespace aic
