// src/autograd/engine.cpp
// Topological sort and graph execution engine for reverse-mode automatic differentiation.
#include "autograd/autograd.hpp"
#include "autograd/functions.hpp"
#include "autograd/node.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace aic::autograd {



namespace {

thread_local bool g_grad_enabled = true;

// Discovers all nodes reachable from root and computes backward graph in-degrees
void compute_dependencies(Node* root,
                          std::unordered_set<Node*>& all_nodes,
                          std::unordered_map<Node*, int>& in_degree) {
    std::queue<Node*> q;
    q.push(root);
    all_nodes.insert(root);
    in_degree[root] = 0;

    while (!q.empty()) {
        Node* curr = q.front();
        q.pop();

        for (const auto& edge : curr->next_edges()) {
            if (edge.is_valid()) {
                Node* next_node = edge.function.get();
                in_degree[next_node]++;
                if (all_nodes.find(next_node) == all_nodes.end()) {
                    all_nodes.insert(next_node);
                    q.push(next_node);
                }
            }
        }
    }
}

// Topological ordering via Kahn's algorithm
std::vector<std::shared_ptr<Node>> topological_sort(const std::shared_ptr<Node>& root) {
    std::unordered_set<Node*> all_nodes;
    std::unordered_map<Node*, int> in_degree;

    compute_dependencies(root.get(), all_nodes, in_degree);

    // Keep shared_ptrs alive during sort
    std::unordered_map<Node*, std::shared_ptr<Node>> node_sp_map;
    std::queue<std::shared_ptr<Node>> sp_q;
    sp_q.push(root);
    node_sp_map[root.get()] = root;

    while (!sp_q.empty()) {
        auto curr = sp_q.front();
        sp_q.pop();

        for (const auto& edge : curr->next_edges()) {
            if (edge.is_valid()) {
                Node* child = edge.function.get();
                if (node_sp_map.find(child) == node_sp_map.end()) {
                    node_sp_map[child] = edge.function;
                    sp_q.push(edge.function);
                }
            }
        }
    }

    std::queue<Node*> zero_in_degree;
    for (Node* n : all_nodes) {
        if (in_degree[n] == 0) {
            zero_in_degree.push(n);
        }
    }

    std::vector<std::shared_ptr<Node>> sorted;
    sorted.reserve(all_nodes.size());

    while (!zero_in_degree.empty()) {
        Node* u = zero_in_degree.front();
        zero_in_degree.pop();
        sorted.push_back(node_sp_map[u]);

        for (const auto& edge : u->next_edges()) {
            if (edge.is_valid()) {
                Node* v = edge.function.get();
                in_degree[v]--;
                if (in_degree[v] == 0) {
                    zero_in_degree.push(v);
                }
            }
        }
    }

    AIC_CHECK(sorted.size() == all_nodes.size(), RuntimeError,
              "Cycle detected in computational graph during backward topological sort");
    return sorted;
}

}  // namespace

bool GradMode::is_enabled() noexcept {
    return g_grad_enabled;
}

void GradMode::set_enabled(bool enabled) noexcept {
    g_grad_enabled = enabled;
}

void backward(const Tensor& loss, const Tensor& gradient, bool retain_graph) {
    AIC_CHECK(loss.defined(), InvalidArgument, "Cannot run backward on undefined tensor");

    auto root = loss.grad_fn();
    if (!root) {
        // If loss is a leaf that requires grad, accumulate directly
        if (loss.requires_grad()) {
            Tensor g = gradient.defined() ? gradient : Tensor::ones(loss.shape(), loss.dtype(), loss.device());
            Tensor cur = loss.grad();
            if (cur.defined()) {
                loss.set_grad(cur + g);
            } else {
                loss.set_grad(g.clone());
            }
        }
        return;
    }

    // Determine initial gradient
    Tensor root_grad = gradient;
    if (!root_grad.defined()) {
        AIC_CHECK(loss.numel() == 1, InvalidArgument,
                  "Grad can only be implicitly created for scalar outputs (got numel=", loss.numel(), ")");
        root_grad = Tensor::ones(loss.shape(), loss.dtype(), loss.device());
    } else {
        AIC_CHECK(root_grad.shape() == loss.shape(), ShapeMismatch,
                  "Initial gradient shape ", root_grad.shape().str(), " must match loss shape ", loss.shape().str());
    }

    // Perform topological sort
    auto sorted_nodes = topological_sort(root);

    // Map: Node* -> vector of incoming gradient tensors
    std::unordered_map<Node*, std::vector<Tensor>> grad_map;
    grad_map[root.get()] = {root_grad};

    // Execute backward pass in topological order
    for (const auto& node : sorted_nodes) {
        auto it = grad_map.find(node.get());
        if (it == grad_map.end() || it->second.empty()) {
            continue;
        }

        const auto& grad_outputs = it->second;
        // Compute input gradients
        std::vector<Tensor> input_grads = node->backward(grad_outputs);

        // Dispatch gradients to predecessor nodes along edges
        const auto& edges = node->next_edges();
        for (size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
            const auto& edge = edges[i];
            if (edge.is_valid() && input_grads[i].defined()) {
                Node* next_node = edge.function.get();
                size_t input_nr = edge.input_nr;

                auto& next_grads = grad_map[next_node];
                if (next_grads.size() <= input_nr) {
                    next_grads.resize(input_nr + 1);
                }

                if (next_grads[input_nr].defined()) {
                    next_grads[input_nr] = next_grads[input_nr] + input_grads[i];
                } else {
                    next_grads[input_nr] = input_grads[i];
                }
            }
        }

        // Clean up gradients for processed node to free memory
        if (!retain_graph) {
            grad_map.erase(node.get());
        }
    }
}

}  // namespace aic::autograd
