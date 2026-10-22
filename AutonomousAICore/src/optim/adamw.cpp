// src/optim/adamw.cpp
// AdamW optimizer implementation with decoupled weight decay and fused updates.
#include "optim/adamw.hpp"
#include "core/error.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_adamw.cuh"
#endif

#include <cmath>

namespace aic::optim {

void AdamW::step() {
    for (auto& p : params_) {
        if (!p || !p->defined()) continue;
        Tensor g = p->grad();
        if (!g.defined()) continue;

        AdamWParamState& st = state_[p.get()];
        st.step += 1;

        if (!st.exp_avg.defined()) {
            st.exp_avg = Tensor::zeros(p->shape(), DType::Float32, p->device());
            st.exp_avg_sq = Tensor::zeros(p->shape(), DType::Float32, p->device());
        }

        float bc1 = 1.0f - std::pow(options_.beta1, static_cast<float>(st.step));
        float bc2 = 1.0f - std::pow(options_.beta2, static_cast<float>(st.step));

        if (p->is_cuda()) {
#if AIC_WITH_CUDA
            cuda::cuda_adamw_step(
                p->data_ptr<float>(),
                g.data_ptr<float>(),
                st.exp_avg.data_ptr<float>(),
                st.exp_avg_sq.data_ptr<float>(),
                p->numel(),
                options_.lr,
                options_.beta1,
                options_.beta2,
                options_.eps,
                options_.weight_decay,
                bc1,
                bc2
            );
#else
            AIC_THROW(BackendUnavailable, "CUDA AdamW requested but CUDA is not enabled");
#endif
        } else {
            // CPU parameter update
            float* p_ptr = p->data_ptr<float>();
            const float* g_ptr = g.data_ptr<float>();
            float* m_ptr = st.exp_avg.data_ptr<float>();
            float* v_ptr = st.exp_avg_sq.data_ptr<float>();

            int64_t n = p->numel();
            float lr = options_.lr;
            float beta1 = options_.beta1;
            float beta2 = options_.beta2;
            float eps = options_.eps;
            float wd = options_.weight_decay;

            for (int64_t i = 0; i < n; ++i) {
                float grad = g_ptr[i];
                float param = p_ptr[i];

                // 1. Decoupled weight decay
                param -= lr * wd * param;

                // 2. Exponential moving averages of gradient and square gradient
                float m = beta1 * m_ptr[i] + (1.0f - beta1) * grad;
                float v = beta2 * v_ptr[i] + (1.0f - beta2) * grad * grad;

                m_ptr[i] = m;
                v_ptr[i] = v;

                // 3. Bias correction
                float m_hat = m / bc1;
                float v_hat = v / bc2;

                // 4. Update parameter
                param -= lr * (m_hat / (std::sqrt(v_hat) + eps));
                p_ptr[i] = param;
            }
        }

        p->bump_version();
    }
}

}  // namespace aic::optim
