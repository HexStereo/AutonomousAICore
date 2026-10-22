// include/autograd/autograd_meta.hpp
// Internal metadata holding gradient accumulator, computational graph links, and in-place versioning.
#pragma once

#include <cstdint>
#include <memory>

namespace aic {
class Tensor;
}

namespace aic::autograd {

class Node;

struct AutogradMeta {
    bool requires_grad = false;
    bool is_leaf = true;
    std::shared_ptr<Node> grad_fn;
    std::shared_ptr<Tensor> grad;
    uint64_t version = 0;
};

}  // namespace aic::autograd
