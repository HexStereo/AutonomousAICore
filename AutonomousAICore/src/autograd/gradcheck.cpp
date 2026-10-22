// src/autograd/gradcheck.cpp
// Numerical gradient checking using two-sided centered finite differences.
#include "autograd/autograd.hpp"
#include "core/logging.hpp"
#include "ops/ops.hpp"

#include <cmath>
#include <iostream>

namespace aic::autograd {

bool gradcheck(
    const std::function<Tensor(const std::vector<Tensor>&)>& func,
    const std::vector<Tensor>& inputs,
    float eps,
    float atol,
    float rtol) {

    // 1. Run forward pass with autograd enabled
    for (auto& inp : inputs) {
        inp.zero_grad();
    }

    Tensor output = func(inputs);
    AIC_CHECK(output.numel() == 1, InvalidArgument,
              "gradcheck function must return a scalar tensor, got numel=", output.numel());

    backward(output);

    // 2. Compute numerical gradient for each requires_grad input using finite differences
    for (size_t k = 0; k < inputs.size(); ++k) {
        auto& inp = inputs[k];
        if (!inp.requires_grad()) {
            continue;
        }

        Tensor analytic_grad = inp.grad();
        AIC_CHECK(analytic_grad.defined(), RuntimeError,
                  "Analytic gradient not computed for input ", k, " in gradcheck");

        Tensor analytic_cpu = analytic_grad.cpu().contiguous();
        const float* p_analytic = analytic_cpu.data_ptr<float>();

        // Ensure input is on CPU for direct element access during finite differences
        Device orig_device = inp.device();
        Tensor inp_cpu = inp.cpu().contiguous().clone();
        float* p_data = inp_cpu.data_ptr<float>();
        int64_t n = inp.numel();

        for (int64_t i = 0; i < n; ++i) {
            float orig_val = p_data[i];

            // Evaluate f(x + eps)
            p_data[i] = orig_val + eps;
            std::vector<Tensor> perturbed_plus = inputs;
            perturbed_plus[k] = inp_cpu.to(orig_device);
            float y_plus = 0.0f;
            {
                NoGradGuard guard;
                y_plus = func(perturbed_plus).item<float>();
            }

            // Evaluate f(x - eps)
            p_data[i] = orig_val - eps;
            std::vector<Tensor> perturbed_minus = inputs;
            perturbed_minus[k] = inp_cpu.to(orig_device);
            float y_minus = 0.0f;
            {
                NoGradGuard guard;
                y_minus = func(perturbed_minus).item<float>();
            }

            // Restore original input value
            p_data[i] = orig_val;

            // Two-sided finite difference
            float num_grad = (y_plus - y_minus) / (2.0f * eps);
            float ana_grad = p_analytic[i];

            float diff = std::abs(num_grad - ana_grad);
            float max_val = std::max({std::abs(num_grad), std::abs(ana_grad), 1.0f});
            float rel_err = diff / max_val;

            if (diff > atol && rel_err > rtol) {
                AIC_LOG_ERROR("gradcheck failed for input ", k, " at index ", i,
                              ": numerical=", num_grad, ", analytic=", ana_grad,
                              ", diff=", diff, ", rel_err=", rel_err);
                return false;
            }
        }
    }

    return true;
}

}  // namespace aic::autograd
