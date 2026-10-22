// src/autograd/node.cpp
// Implementation of SavedVariable version verification and AccumulateGrad node.
#include "autograd/node.hpp"
#include "ops/ops.hpp"

namespace aic::autograd {

SavedVariable::SavedVariable(const Tensor& tensor, bool is_output)
    : tensor_(tensor),
      saved_version_(tensor.defined() ? tensor.version() : 0),
      was_default_(!tensor.defined()) {
    (void)is_output;
}

Tensor SavedVariable::unpack(const std::string& op_name) const {
    if (was_default_ || !tensor_.defined()) {
        AIC_THROW(RuntimeError, "Attempted to unpack an undefined SavedVariable in ", op_name);
    }
    if (tensor_.version() != saved_version_) {
        AIC_THROW(RuntimeError,
                  "One of the variables needed for gradient computation has been modified by an in-place operation "
                  "(expected version ", saved_version_, ", but current version is ", tensor_.version(),
                  " during backward of ", op_name.empty() ? "an operation" : op_name, ")");
    }
    return tensor_;
}

AccumulateGrad::AccumulateGrad(Tensor variable)
    : variable_(std::move(variable)) {}

std::vector<Tensor> AccumulateGrad::backward(const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty() || !grad_outputs[0].defined()) {
        return {};
    }

    const Tensor& incoming_grad = grad_outputs[0];
    Tensor current_grad = variable_.grad();

    if (current_grad.defined()) {
        variable_.set_grad(current_grad + incoming_grad);
    } else {
        variable_.set_grad(incoming_grad.clone());
    }

    return {};
}

}  // namespace aic::autograd
