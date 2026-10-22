// include/autograd/node.hpp
// Computational DAG Node, Edge, and SavedVariable abstractions for reverse-mode autograd.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor/tensor.hpp"

namespace aic::autograd {

class Node;

struct Edge {
    std::shared_ptr<Node> function;
    uint32_t input_nr = 0;

    bool is_valid() const noexcept { return function != nullptr; }
};

class SavedVariable {
public:
    SavedVariable() = default;
    SavedVariable(const Tensor& tensor, bool is_output = false);

    Tensor unpack(const std::string& op_name = "") const;

private:
    Tensor tensor_;
    uint64_t saved_version_ = 0;
    bool was_default_ = true;
};

class Node : public std::enable_shared_from_this<Node> {
public:
    virtual ~Node() = default;

    virtual std::string name() const = 0;

    // Backward pass: given incoming gradients for each output, returns gradients for each input
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;

    const std::vector<Edge>& next_edges() const noexcept { return next_edges_; }
    std::vector<Edge>& next_edges() noexcept { return next_edges_; }

    void add_next_edge(Edge edge) { next_edges_.push_back(std::move(edge)); }
    void set_next_edges(std::vector<Edge> edges) { next_edges_ = std::move(edges); }

    uint32_t topological_nr() const noexcept { return topological_nr_; }
    void set_topological_nr(uint32_t nr) noexcept { topological_nr_ = nr; }

private:
    std::vector<Edge> next_edges_;
    uint32_t topological_nr_ = 0;
};

// AccumulateGrad is the backward leaf node that accumulates gradients into leaf variables
class AccumulateGrad : public Node {
public:
    explicit AccumulateGrad(Tensor variable);

    std::string name() const override { return "AccumulateGrad"; }
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    const Tensor& variable() const noexcept { return variable_; }

private:
    Tensor variable_;
};

}  // namespace aic::autograd
