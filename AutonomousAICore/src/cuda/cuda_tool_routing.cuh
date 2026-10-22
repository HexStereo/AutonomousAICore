// src/cuda/cuda_tool_routing.cuh
// Blackwell sm_120 CUDA Kernel for Fast Tool Embedding Routing & Scoring (Aşama 24)
#pragma once

#include <cstdint>

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace aic::cuda {

#if AIC_WITH_CUDA
// Computes normalized cosine similarity between query embedding [embed_dim]
// and a bank of tool embeddings [num_tools, embed_dim] on Blackwell GPU.
void launch_tool_embedding_similarity(
    const float* query_embed,
    const float* tool_embeds,
    float* scores,
    int num_tools,
    int embed_dim,
    cudaStream_t stream = nullptr
);
#endif

// CPU reference implementation for dual-backend parity
inline void cpu_tool_embedding_similarity(
    const float* query_embed,
    const float* tool_embeds,
    float* scores,
    int num_tools,
    int embed_dim
) {
    if (num_tools <= 0 || embed_dim <= 0) return;

    float q_norm2 = 0.0f;
    for (int d = 0; d < embed_dim; ++d) {
        q_norm2 += query_embed[d] * query_embed[d];
    }
    float q_norm = std::sqrt(q_norm2);

    for (int i = 0; i < num_tools; ++i) {
        const float* t_ptr = tool_embeds + i * embed_dim;
        float dot = 0.0f;
        float t_norm2 = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
            dot += query_embed[d] * t_ptr[d];
            t_norm2 += t_ptr[d] * t_ptr[d];
        }
        float t_norm = std::sqrt(t_norm2);
        float denom = (q_norm * t_norm) + 1e-8f;
        scores[i] = dot / denom;
    }
}

} // namespace aic::cuda
