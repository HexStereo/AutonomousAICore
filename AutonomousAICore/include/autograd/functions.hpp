// include/autograd/functions.hpp
// Differentiable computational graph backward functions for arithmetic, activations, and transformations.
#pragma once

#include <vector>

#include "autograd/node.hpp"
#include "core/shape.hpp"
#include "tensor/tensor.hpp"

namespace aic::autograd {

class AddBackward : public Node {
public:
    AddBackward(Shape shape_a, Shape shape_b)
        : shape_a_(std::move(shape_a)), shape_b_(std::move(shape_b)) {}

    std::string name() const override { return "AddBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    Shape shape_a_;
    Shape shape_b_;
};

class SubBackward : public Node {
public:
    SubBackward(Shape shape_a, Shape shape_b)
        : shape_a_(std::move(shape_a)), shape_b_(std::move(shape_b)) {}

    std::string name() const override { return "SubBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    Shape shape_a_;
    Shape shape_b_;
};

class MulBackward : public Node {
public:
    MulBackward(const Tensor& a, const Tensor& b)
        : saved_a_(a), saved_b_(b), shape_a_(a.shape()), shape_b_(b.shape()) {}

    std::string name() const override { return "MulBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_a_;
    SavedVariable saved_b_;
    Shape shape_a_;
    Shape shape_b_;
};

class DivBackward : public Node {
public:
    DivBackward(const Tensor& a, const Tensor& b)
        : saved_a_(a), saved_b_(b), shape_a_(a.shape()), shape_b_(b.shape()) {}

    std::string name() const override { return "DivBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_a_;
    SavedVariable saved_b_;
    Shape shape_a_;
    Shape shape_b_;
};

class PowBackward : public Node {
public:
    PowBackward(const Tensor& a, float exponent)
        : saved_a_(a), exponent_(exponent) {}

    std::string name() const override { return "PowBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_a_;
    float exponent_;
};

class NegBackward : public Node {
public:
    std::string name() const override { return "NegBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
};

class ReluBackward : public Node {
public:
    explicit ReluBackward(const Tensor& x) : saved_x_(x) {}

    std::string name() const override { return "ReluBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_x_;
};

class TanhBackward : public Node {
public:
    explicit TanhBackward(const Tensor& out) : saved_out_(out, true) {}

    std::string name() const override { return "TanhBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_out_;
};

class SigmoidBackward : public Node {
public:
    explicit SigmoidBackward(const Tensor& out) : saved_out_(out, true) {}

    std::string name() const override { return "SigmoidBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_out_;
};

class GeluBackward : public Node {
public:
    explicit GeluBackward(const Tensor& x) : saved_x_(x) {}

    std::string name() const override { return "GeluBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_x_;
};

class SiluBackward : public Node {
public:
    explicit SiluBackward(const Tensor& x) : saved_x_(x) {}

    std::string name() const override { return "SiluBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_x_;
};

class ExpBackward : public Node {
public:
    explicit ExpBackward(const Tensor& out) : saved_out_(out, true) {}

    std::string name() const override { return "ExpBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_out_;
};

class LogBackward : public Node {
public:
    explicit LogBackward(const Tensor& x) : saved_x_(x) {}

    std::string name() const override { return "LogBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_x_;
};

class SqrtBackward : public Node {
public:
    explicit SqrtBackward(const Tensor& out) : saved_out_(out, true) {}

    std::string name() const override { return "SqrtBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_out_;
};

class SumBackward : public Node {
public:
    SumBackward(Shape input_shape, int dim = -1, bool keepdim = false, bool is_dim_reduce = false)
        : input_shape_(std::move(input_shape)), dim_(dim), keepdim_(keepdim), is_dim_reduce_(is_dim_reduce) {}

    std::string name() const override { return "SumBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    Shape input_shape_;
    int dim_;
    bool keepdim_;
    bool is_dim_reduce_;
};

class MeanBackward : public Node {
public:
    MeanBackward(Shape input_shape, int dim = -1, bool keepdim = false, bool is_dim_reduce = false)
        : input_shape_(std::move(input_shape)), dim_(dim), keepdim_(keepdim), is_dim_reduce_(is_dim_reduce) {}

    std::string name() const override { return "MeanBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    Shape input_shape_;
    int dim_;
    bool keepdim_;
    bool is_dim_reduce_;
};

class ReshapeBackward : public Node {
public:
    explicit ReshapeBackward(Shape original_shape) : original_shape_(std::move(original_shape)) {}

    std::string name() const override { return "ReshapeBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    Shape original_shape_;
};

class TransposeBackward : public Node {
public:
    TransposeBackward(int dim0, int dim1) : dim0_(dim0), dim1_(dim1) {}

    std::string name() const override { return "TransposeBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    int dim0_;
    int dim1_;
};

class PermuteBackward : public Node {
public:
    explicit PermuteBackward(std::vector<int> original_dims) : original_dims_(std::move(original_dims)) {}

    std::string name() const override { return "PermuteBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    std::vector<int> original_dims_;
};

class MatMulBackward : public Node {
public:
    MatMulBackward(const Tensor& a, const Tensor& b)
        : saved_a_(a), saved_b_(b) {}

    std::string name() const override { return "MatMulBackward"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

private:
    SavedVariable saved_a_;
    SavedVariable saved_b_;
};

}  // namespace aic::autograd
