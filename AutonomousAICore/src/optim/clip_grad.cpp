// src/optim/clip_grad.cpp
// Gradient clipping implementation using global L2 norm.
#include "optim/clip_grad.hpp"
#include "core/error.hpp"
#include "ops/ops.hpp"

#include <cmath>

namespace aic::optim {

float clip_grad_norm_(const std::vector<std::shared_ptr<Tensor>>& params,
                      float max_norm,
                      float norm_type) {
    AIC_CHECK(norm_type == 2.0f, InvalidArgument,
              "Currently only L2 gradient norm (norm_type = 2.0) is supported, got ", norm_type);
    AIC_CHECK(max_norm > 0.0f, InvalidArgument, "max_norm must be positive, got ", max_norm);

    double total_norm_sq = 0.0;

    for (const auto& p : params) {
        if (!p || !p->defined()) continue;
        Tensor g = p->grad();
        if (!g.defined()) continue;

        if (g.is_cpu()) {
            const float* data = g.contiguous().data_ptr<float>();
            int64_t n = g.numel();
            double sum_sq = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                sum_sq += static_cast<double>(data[i]) * static_cast<double>(data[i]);
            }
            total_norm_sq += sum_sq;
        } else if (g.is_cuda()) {
            Tensor sq = mul(g, g);
            Tensor s = sum(sq);
            total_norm_sq += static_cast<double>(s.cpu().item<float>());
        }
    }

    float total_norm = static_cast<float>(std::sqrt(total_norm_sq));
    float clip_coef = max_norm / (total_norm + 1e-6f);

    if (clip_coef < 1.0f) {
        for (const auto& p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            mul_(g, clip_coef);
        }
    }

    return total_norm;
}

}  // namespace aic::optim
