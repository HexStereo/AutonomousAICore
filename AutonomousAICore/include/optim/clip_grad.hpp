// include/optim/clip_grad.hpp
// Gradient clipping by global L2 norm across model parameters.
#pragma once

#include <memory>
#include <vector>
#include "tensor/tensor.hpp"

namespace aic::optim {

// Clips gradients of an iterable of parameters to a maximum L2 norm.
// Returns the total norm of the gradients (viewed as a single vector).
float clip_grad_norm_(const std::vector<std::shared_ptr<Tensor>>& params,
                      float max_norm,
                      float norm_type = 2.0f);

}  // namespace aic::optim
