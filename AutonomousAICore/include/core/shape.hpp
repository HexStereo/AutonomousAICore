// include/core/shape.hpp
// Multi-dimensional tensor shape and stride management.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <span>

#include "core/error.hpp"

namespace aic {

class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims);
    explicit Shape(std::vector<int64_t> dims);
    explicit Shape(std::span<const int64_t> dims);

    int rank() const noexcept { return static_cast<int>(dims_.size()); }
    bool empty() const noexcept { return dims_.empty(); }
    int64_t numel() const noexcept { return numel_; }

    int64_t operator[](int index) const;
    int64_t at(int index) const { return (*this)[index]; }
    const std::vector<int64_t>& dims() const noexcept { return dims_; }

    const int64_t* data() const noexcept { return dims_.data(); }
    auto begin() const noexcept { return dims_.begin(); }
    auto end() const noexcept { return dims_.end(); }

    std::vector<int64_t> default_strides() const;

    bool operator==(const Shape& other) const noexcept { return dims_ == other.dims_; }
    bool operator!=(const Shape& other) const noexcept { return dims_ != other.dims_; }

    std::string str() const;

    friend std::ostream& operator<<(std::ostream& os, const Shape& s) {
        return os << s.str();
    }

    // Reshaping transformations
    Shape squeeze(int dim) const;
    Shape squeeze_all() const;
    Shape unsqueeze(int dim) const;

    // Broadcasting: returns broadcasted shape or throws ShapeMismatch
    static Shape broadcast(const Shape& a, const Shape& b);
    static bool can_broadcast(const Shape& a, const Shape& b) noexcept;

    // Compute broadcasted strides for tensor with original strides
    static std::vector<int64_t> broadcast_strides(const Shape& orig_shape,
                                                  const std::vector<int64_t>& orig_strides,
                                                  const Shape& target_shape);

    static bool is_contiguous(const Shape& shape, const std::vector<int64_t>& strides) noexcept;

private:
    void compute_numel();

    std::vector<int64_t> dims_;
    int64_t numel_ = 1;
};

}  // namespace aic
