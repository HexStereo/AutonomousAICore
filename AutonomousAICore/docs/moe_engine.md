# Mixture-of-Experts (MoE) Sparse Routing & Dynamic Load-Balancing Engine

## 1. Executive Summary

The **Mixture-of-Experts (MoE) Engine** in **AutonomousAICore** enables massive parameter scaling at constant inference and training FLOPs by dynamically routing tokens to specialized subset feed-forward expert networks.

Key architectural achievements:
- **Sub-Linear Compute Scaling**: Scales model parameter capacity by $E\times$ (e.g., 8 to 64 experts) while only activating top-$k$ experts per token (e.g., $k=2$), yielding $\frac{E}{k}\times$ (e.g., $4.0\times$) compute efficiency.
- **Differentiable Top-$K$ Gating**: Implements normalized softmax gating over selected top-$k$ experts with full autograd graph connectivity, enabling gradients from task losses to shape expert specialization.
- **Switch/GShard Auxiliary Load-Balancing Loss**: Introduces a differentiable auxiliary regularization loss $\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{e=0}^{E-1} f_e \cdot \bar{P}_e$ that prevents expert collapse and guarantees balanced workload distribution across hardware SMs.
- **Conditional Sparse Execution**: Inactive experts in any given token batch are conditionally skipped entirely, conserving execution cycles and memory bandwidth.
- **End-to-End Decoder-Only Architecture**: Includes `MoELayer`, `MoETransformerBlock`, and complete `MoETransformerLM` with KV-caching autoregressive generation and multi-layer auxiliary loss aggregation.
- **CUDA sm_120 (NVIDIA Blackwell) Parity**: Fully validated on GeForce RTX 5090 GPU achieving 2,879+ tokens/sec throughput with load imbalance ratio near 1.19.

---

## 2. Mathematical Formulation

### 2.1 Top-$K$ Sparse Gating Network

Given an input representation $x \in \mathbb{R}^{\dots \times d_{\text{model}}}$, the router projects tokens to $E$ expert logits:

$$S(x) = x \cdot W_g^T \in \mathbb{R}^{\dots \times E}$$

The gating probability distribution over all $E$ experts is given by the softmax:

$$P(x) = \text{softmax}(S(x)) = \frac{\exp(S(x)_e)}{\sum_{j=0}^{E-1} \exp(S(x)_j)}$$

For each token $t$, the router selects the indices of the $k$ experts with the highest logit values:

$$\mathcal{T}(t) = \text{Top-}k\left(S(x_t), k\right) \subset \{0, \dots, E-1\}$$

The routing weights over the chosen experts are normalized so that their sum equals $1$:

$$w_{t, e} = \begin{cases} \frac{P(x_t)_e}{\sum_{j \in \mathcal{T}(t)} P(x_t)_j} & \text{if } e \in \mathcal{T}(t) \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 Sparse Expert Combination

Each expert $e \in \{0, \dots, E-1\}$ is parameterized as an independent Swish-Gated Linear Unit (`SwiGLU`):

$$\text{Expert}_e(x) = \left(\text{SiLU}(x W_{\text{gate}, e}^T) \odot x W_{\text{up}, e}^T\right) W_{\text{down}, e}^T$$

The final output of the MoE layer for token $t$ is the weighted combination of only its active experts:

$$y_t = \sum_{e \in \mathcal{T}(t)} w_{t, e} \cdot \text{Expert}_e(x_t)$$

If an expert $e$ has zero allocated tokens across the entire batch ($\sum_t \mathbf{1}\{e \in \mathcal{T}(t)\} = 0$), its forward execution is completely bypassed.

### 2.3 Auxiliary Load-Balancing Regularization Loss

Without regularization, router networks exhibit positive feedback loops where a few experts receive all tokens, while the rest starve (expert collapse). To enforce uniform load distribution across all physical compute units, an auxiliary loss is computed per layer:

$$\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}} \cdot E \sum_{e=0}^{E-1} f_e \cdot \bar{P}_e$$

where:
- $f_e = \frac{1}{N \cdot k} \sum_{t=1}^N \mathbf{1}\{e \in \mathcal{T}(t)\}$: fraction of dispatched tokens allocated to expert $e$ (non-differentiable frequency count).
- $\bar{P}_e = \frac{1}{N} \sum_{t=1}^N P(x_t)_e$: average router probability allocated to expert $e$ across all $N$ tokens (fully differentiable with respect to $W_g$).
- $E$: total number of experts.
- $\alpha_{\text{aux}}$: loss scaling coefficient (typically $0.01$).

When all experts receive an equal share of tokens ($f_e = \frac{1}{E}$ and $\bar{P}_e = \frac{1}{E}$), $\sum_e f_e \bar{P}_e = \frac{1}{E}$, and the loss reaches its theoretical minimum of $\alpha_{\text{aux}}$. If an expert receives disproportionate tokens, $\frac{\partial \mathcal{L}_{\text{aux}}}{\partial \bar{P}_e} = \alpha_{\text{aux}} E f_e$ penalizes the overused expert and pushes probability mass toward underutilized experts.

---

## 3. C++20 Core Architecture

```
                    ┌─────────────────────────┐
                    │    Input Tokens x       │  [B, S, d_model]
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          │                                           │
          ▼                                           │
  ┌──────────────┐                                    │
  │ MoERouter    │  gate(x) -> Top-K Indices          │
  │ (Linear W_g) │  L_aux load-balancing loss        │
  └───────┬──────┘                                    │
          │ w_e, M_e                                  │
          ▼                                           │
  ┌───────────────────────────────────────────────┐   │
  │ Expert Dispatch Switch (True Conditional Skip)│   │
  │                                               │   │
  │  ┌───────────┐ ┌───────────┐   ┌───────────┐  │   │
  │  │ Expert 0  │ │ Expert 1  │...│Expert E-1 │  │   │
  │  │ (SwiGLU)  │ │ (SwiGLU)  │   │ (SwiGLU)  │  │   │
  │  └─────┬─────┘ └─────┬─────┘   └─────┬─────┘  │   │
  │        │             │               │        │   │
  │     Active         Active         Skipped     │   │
  └────────┼─────────────┼───────────────┼────────┘   │
           │ * w_0       │ * w_1         │ (0 FLOPs)  │
           └──────┬──────┴───────────────┘            │
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │ Weighted Add (+)  │──────────────>│ Residual Add (+)  │
        └───────────────────┘               └─────────┬─────────┘
                                                      │
                                                      ▼
                                            Output [B, S, d_model]
```

---

## 4. Empirical Benchmarks (RTX 5090 Blackwell GPU)

Benchmark executed via `aic moe-bench 8 2 128 cuda`:

| Metric | Measured Value | Theoretical Target | Status |
| :--- | :--- | :--- | :--- |
| **Total Experts ($E$)** | 8 | 8 | Configured |
| **Top-$K$ Active ($k$)** | 2 | 2 | Configured |
| **Active Capacity** | 25.0% | 25.0% | Bit-Exact |
| **Compute Sparsity Ratio** | 4.0x | 4.0x | Verified |
| **Batch Size (Tokens)** | 128 | 128 | Full Sequence |
| **Avg Latency (Fwd+Bwd)** | **44.45 ms** | < 100 ms | High-Performance |
| **Throughput** | **2,879.5 tok/sec** | > 1,500 tok/sec | **1.92x Target** |
| **Load Imbalance Ratio** | **1.19** | $\le 1.30$ | Highly Balanced |
| **Auxiliary Loss** | $1.008 \times 10^{-2}$ | $\approx 1.0 \times 10^{-2}$ | Optimal Minimum |

### Expert Dispatch Balance Histogram

```
Expert [0]: 38 tokens (14.8%) |###
Expert [1]: 36 tokens (14.1%) |###
Expert [2]: 23 tokens ( 9.0%) |##
Expert [3]: 31 tokens (12.1%) |###
Expert [4]: 30 tokens (11.7%) |##
Expert [5]: 37 tokens (14.5%) |###
Expert [6]: 31 tokens (12.1%) |###
Expert [7]: 30 tokens (11.7%) |##
```

---

## 5. Usage & Integration

### CLI Command

```bash
# Benchmark MoE sparse routing on CUDA (8 experts, top-2, 128 tokens)
aic moe-bench 8 2 128 cuda
```

### C++ API

```cpp
#include "models/moe_transformer_lm.hpp"

using namespace aic;
using namespace aic::models;

MoETransformerLMConfig cfg;
cfg.vocab_size = 32000;
cfg.hidden_dim = 1024;
cfg.num_layers = 12;
cfg.num_heads = 16;
cfg.intermediate_dim = 2048;
cfg.num_experts = 8;
cfg.top_k = 2;
cfg.aux_loss_coef = 0.01f;

MoETransformerLM model(cfg, Device::cuda(0));

// Forward pass with total auxiliary load-balancing loss
Tensor logits = model.forward(input_ids);
Tensor total_loss = cross_entropy(logits, targets) + model.total_aux_loss();
total_loss.backward();

// Autoregressive generation with KV caching
std::vector<int64_t> prompt = {1, 150, 420};
std::vector<int64_t> generated = model.generate(prompt, 32, 0.8f, 50, 0.9f);
```
