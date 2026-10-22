// src/core/shape.cpp
#include "core/shape.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace aic {

Shape::Shape(std::initializer_list<int64_t> dims) : dims_(dims) {
    compute_numel();
}

Shape::Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
    compute_numel();
}

Shape::Shape(std::span<const int64_t> dims) : dims_(dims.begin(), dims.end()) {
    compute_numel();
}

void Shape::compute_numel() {
    if (dims_.empty()) {
        numel_ = 1;  // 0-D scalar tensor has 1 element
        return;
    }
    int64_t prod = 1;
    for (int i = 0; i < rank(); ++i) {
        int64_t d = dims_[i];
        if (d < 0) {
            AIC_THROW(InvalidArgument, "Negative dimension size in shape at index ", i, ": ", d);
        }
        if (d == 0) {
            numel_ = 0;
            return;
        }
        // Checked multiplication against 64-bit overflow
        if (prod > std::numeric_limits<int64_t>::max() / d) {
            AIC_THROW(Overflow, "Integer overflow occurred computing numel for shape: ", str());
        }
        prod *= d;
    }
    numel_ = prod;
}

int64_t Shape::operator[](int index) const {
    int r = rank();
    if (r == 0) {
        AIC_THROW(OutOfRange, "Cannot index a 0-D (scalar) shape");
    }
    int orig_index = index;
    if (index < 0) {
        index += r;
    }
    if (index < 0 || index >= r) {
        AIC_THROW(OutOfRange, "Shape index ", orig_index, " out of range for rank ", r);
    }
    return dims_[static_cast<size_t>(index)];
}

std::vector<int64_t> Shape::default_strides() const {
    int r = rank();
    if (r == 0) {
        return {};
    }
    std::vector<int64_t> strides(static_cast<size_t>(r), 1);
    int64_t s = 1;
    for (int i = r - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = s;
        int64_t d = dims_[static_cast<size_t>(i)];
        // If dimension is 0, stride stays 1
        if (d > 0) {
            s *= d;
        }
    }
    return strides;
}

std::string Shape::str() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) os << ", ";
        os << dims_[i];
    }
    os << "]";
    return os.str();
}

Shape Shape::squeeze(int dim) const {
    int r = rank();
    int normalized_dim = dim < 0 ? dim + r : dim;
    if (normalized_dim < 0 || normalized_dim >= r) {
        AIC_THROW(OutOfRange, "squeeze dim ", dim, " out of range for rank ", r);
    }
    if (dims_[static_cast<size_t>(normalized_dim)] != 1) {
        return *this;
    }
    std::vector<int64_t> new_dims;
    new_dims.reserve(static_cast<size_t>(r - 1));
    for (int i = 0; i < r; ++i) {
        if (i != normalized_dim) {
            new_dims.push_back(dims_[static_cast<size_t>(i)]);
        }
    }
    return Shape(std::move(new_dims));
}

Shape Shape::squeeze_all() const {
    std::vector<int64_t> new_dims;
    new_dims.reserve(dims_.size());
    for (int64_t d : dims_) {
        if (d != 1) {
            new_dims.push_back(d);
        }
    }
    return Shape(std::move(new_dims));
}

Shape Shape::unsqueeze(int dim) const {
    int r = rank();
    int normalized_dim = dim < 0 ? dim + r + 1 : dim;
    if (normalized_dim < 0 || normalized_dim > r) {
        AIC_THROW(OutOfRange, "unsqueeze dim ", dim, " out of range for rank ", r);
    }
    std::vector<int64_t> new_dims;
    new_dims.reserve(static_cast<size_t>(r + 1));
    for (int i = 0; i < normalized_dim; ++i) {
        new_dims.push_back(dims_[static_cast<size_t>(i)]);
    }
    new_dims.push_back(1);
    for (int i = normalized_dim; i < r; ++i) {
        new_dims.push_back(dims_[static_cast<size_t>(i)]);
    }
    return Shape(std::move(new_dims));
}

bool Shape::can_broadcast(const Shape& a, const Shape& b) noexcept {
    int ra = a.rank();
    int rb = b.rank();
    int max_r = std::max(ra, rb);
    for (int i = 0; i < max_r; ++i) {
        int64_t da = (i < ra) ? a.dims()[static_cast<size_t>(ra - 1 - i)] : 1;
        int64_t db = (i < rb) ? b.dims()[static_cast<size_t>(rb - 1 - i)] : 1;
        if (da != db && da != 1 && db != 1) {
            return false;
        }
    }
    return true;
}

Shape Shape::broadcast(const Shape& a, const Shape& b) {
    if (!can_broadcast(a, b)) {
        AIC_THROW(ShapeMismatch, "Cannot broadcast shapes ", a.str(), " and ", b.str());
    }
    int ra = a.rank();
    int rb = b.rank();
    int max_r = std::max(ra, rb);
    std::vector<int64_t> out_dims(static_cast<size_t>(max_r));
    for (int i = 0; i < max_r; ++i) {
        int64_t da = (i < ra) ? a.dims()[static_cast<size_t>(ra - 1 - i)] : 1;
        int64_t db = (i < rb) ? b.dims()[static_cast<size_t>(rb - 1 - i)] : 1;
        out_dims[static_cast<size_t>(max_r - 1 - i)] = std::max(da, db);
    }
    return Shape(std::move(out_dims));
}

std::vector<int64_t> Shape::broadcast_strides(const Shape& orig_shape,
                                              const std::vector<int64_t>& orig_strides,
                                              const Shape& target_shape) {
    int orig_r = orig_shape.rank();
    int target_r = target_shape.rank();
    if (orig_r > target_r) {
        AIC_THROW(ShapeMismatch, "Original rank ", orig_r, " exceeds target rank ", target_r);
    }
    std::vector<int64_t> new_strides(static_cast<size_t>(target_r), 0);
    for (int i = 0; i < target_r; ++i) {
        int orig_idx = orig_r - 1 - i;
        int target_idx = target_r - 1 - i;
        if (orig_idx >= 0) {
            int64_t orig_dim = orig_shape.dims()[static_cast<size_t>(orig_idx)];
            int64_t target_dim = target_shape.dims()[static_cast<size_t>(target_idx)];
            if (orig_dim == target_dim) {
                new_strides[static_cast<size_t>(target_idx)] = orig_strides[static_cast<size_t>(orig_idx)];
            } else if (orig_dim == 1) {
                new_strides[static_cast<size_t>(target_idx)] = 0;  // Repeat along broadcast dim
            } else {
                AIC_THROW(ShapeMismatch, "Incompatible broadcast dimension: ", orig_dim, " vs ", target_dim);
            }
        } else {
            new_strides[static_cast<size_t>(target_idx)] = 0;
        }
    }
    return new_strides;
}

bool Shape::is_contiguous(const Shape& shape, const std::vector<int64_t>& strides) noexcept {
    int r = shape.rank();
    if (r == 0) return true;
    if (static_cast<int>(strides.size()) != r) return false;
    int64_t expected_stride = 1;
    for (int i = r - 1; i >= 0; --i) {
        int64_t dim = shape.dims()[static_cast<size_t>(i)];
        if (dim == 0) return true;
        if (dim != 1) {
            if (strides[static_cast<size_t>(i)] != expected_stride) {
                return false;
            }
            expected_stride *= dim;
        }
    }
    return true;
}

}  // namespace aic
