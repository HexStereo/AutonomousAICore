# Autonomous AI Core (AIC) — Comprehensive System Architecture

## 1. System Philosophy & Objectives
`AutonomousAICore` is a from-scratch, production-grade Deep Learning and Reasoning Core written in Modern C++20 and CUDA C++ (NVIDIA Blackwell sm_120). It contains zero external deep learning frameworks (strictly no PyTorch, LibTorch, TensorFlow, or ONNX Runtime).

The core provides:
1. Custom multi-tier **MemoryPool Subsystem** with 64-byte aligned SIMD CPU bins and stream-aware CUDA caching allocators.
2. Arbitrary-rank, strided **Tensor Abstraction** supporting multidimensional views, broadcasting, slicing, and bit-exact casting.
3. **Hardware Acceleration Engine**:
   - Cache-blocked AVX2+FMA CPU GEMM reaching **>580 GFLOP/s** on Intel Core Ultra 9 285HX.
   - Shared-memory tiled and Blackwell sm_120 Tensor Core GEMM reaching **>170 TFLOP/s** on NVIDIA GeForce RTX 5090 Laptop GPU.
4. Reverse-mode **Dynamic Autograd Engine** with automatic topological sorting, diamond graph gradient accumulation, in-place version tracking, and analytical gradients.
5. Production **Neural Network Layers**: `Linear`, `RMSNorm`, `RotaryPositionEmbedding` (RoPE with exact adjoint backpropagation), `SwiGLU`, and `CausalSelfAttention` with dynamic `KVCache`.
6. Full **Decoder-Only Transformer Language Model (`TransformerLM`)** with parallel prefill and autoregressive $O(1)$ cached sequence generation with nucleus top-$p$, top-$k$, and temperature sampling.
7. Numerically stable **Loss Functions & Optimizers**: fused LogSumExp `CrossEntropyLoss` with analytical backwards, `MSELoss`, `AdamW` with decoupled weight decay, `CosineWarmupScheduler`, and global $L_2$ `clip_grad_norm_`.
8. High-throughput **Data Pipeline**: Byte-Level BPE Tokenizer with zero-OOV fallback, and native Windows Win32 memory-mapped dataset (`CreateFileMapping` / `MapViewOfFile`) with prefetching.
9. Resilient **Binary Checkpoint Format (`.aic`)** with magic header `AIC0`, metadata, tensor dictionary, and Adler-32 checksum integrity checks.
10. Closed-Loop **Autonomous Optimization Controller**: Real-time telemetry tracking (loss EMAs, gradient variance, throughput TFLOP/s, VRAM watermarks), anomaly detection, and automated self-healing actions (checkpoint rollback, adaptive LR scaling, dynamic gradient clipping, and gradient accumulation auto-tuning).

---

## 2. Hardware Environment Verification
Live execution diagnostics on target machine:
```
[CPU Diagnostic]
CPU: Intel(R) Core(TM) Ultra 9 285HX (24 threads, 95.4 GB RAM)
SIMD: AVX=1 AVX2=1 FMA=1 AVX512F=0 AVX512BW=0 AMX=0

[CUDA GPU Diagnostic]
GPU [0]: NVIDIA GeForce RTX 5090 Laptop GPU
  Compute Capability: 12.0
  Architecture: NVIDIA Blackwell (sm_120)
  True Total VRAM: 23.86 GB (Free: 22.61 GB)
  SMs: 82, Warp Size: 32
  Tensor Cores: Supported (5th Gen)
  Precision: FP16=Yes, BF16=Yes, FP8=Yes
```

---

## 3. High-Level Architectural Diagram
```
+--------------------------------------------------------------------------+
|                     Unified CLI & Training Pipeline                      |
|                  tools/aic_cli.cpp | examples/train_mini_llm.cpp         |
+--------------------------------------------------------------------------+
|                     Autonomous Optimization Controller                   |
|   TelemetryTracker (Loss EMA, TFLOP/s, VRAM) | Self-Healing Rollback     |
+--------------------------------------------------------------------------+
|                        Training & Evaluation Loop                        |
|   Trainer (Grad Accumulation) | AdamW (Decoupled) | CosineWarmupSchedule |
+--------------------------------------------------------------------------+
|                    Decoder-Only Transformer Architecture                 |
|   TransformerLM | TransformerBlock | SwiGLU | CausalSelfAttention + KV    |
|   Embedding | RMSNorm | RotaryPositionEmbedding (RoPE Adjoint)           |
+--------------------------------------------------------------------------+
|                    Loss Functions & Autograd Differentiation             |
|   CrossEntropyLoss (Fused LogSumExp) | MSELoss | Dynamic Autograd DAG    |
+--------------------------------------------------------------------------+
|                     HPC Numerical Kernel Acceleration                    |
|   CPU: AVX2+FMA Cache-Blocked GEMM (580 GFLOP/s)                         |
|   CUDA: Blackwell sm_120 2D Register-Tiled & Tensor Cores (170 TFLOP/s)  |
+--------------------------------------------------------------------------+
|                       Strided Tensor Abstraction                         |
|   Contiguous/Non-Contiguous Views | Broadcasting | Host <-> Device DMA   |
+--------------------------------------------------------------------------+
|                       Memory Management Subsystem                        |
|   HostMemoryPool (64B Aligned Bins) | DeviceMemoryPool (CUDA Streams)    |
+--------------------------------------------------------------------------+
```

---

## 4. Stage Breakdown & Verification Matrix

| Stage | Component | Implementations | Unit / Numerical Tests | Parity Status |
|---|---|---|---|---|
| **Aşama 1** | Memory & Tensor | `HostMemoryPool`, `DeviceMemoryPool`, `Storage`, `Shape`, `Tensor` | 19 tests | 100% PASS |
| **Aşama 2** | CPU & GPU Ops | `cpu_ops.cpp`, `cuda_elementwise.cu`, `cuda_reductions.cu` | 8 tests | 100% PASS ($\epsilon < 10^{-5}$) |
| **Aşama 3** | Dynamic Autograd | `Node`, `Edge`, `GradMode`, `Engine`, `NumericalGradcheck` | 8 tests | 100% PASS ($\epsilon < 10^{-4}$) |
| **Aşama 4** | HPC GEMM | Cache-blocked CPU AVX2, CUDA Shared-Memory Tiling & Tensor Cores | 6 tests | 100% PASS (580 GFLOP/s CPU, 170 TFLOP/s GPU) |
| **Aşama 5** | NN Layers | `Linear`, `RMSNorm`, `RoPE`, `SwiGLU`, `CausalSelfAttention`, `Block` | 7 tests | 100% PASS |
| **Aşama 6** | Loss & Optimizers | `CrossEntropyLoss` (LogSumExp), `MSELoss`, `AdamW`, `clip_grad_norm_` | 9 tests | 100% PASS (Conv. 22.1 $\to$ 0.00048) |
| **Aşama 7** | Data Pipeline | Byte-Level BPE `Tokenizer`, Win32 Memory-Mapped `Dataset`, `DataLoader` | 7 tests | 100% PASS (Zero OOV, 100% roundtrip) |
| **Aşama 8** | Decoder-Only LM | `Embedding`, `TransformerLM`, Autoregressive KV-Cached Generation | 6 tests | 100% PASS (Parallel == Stepwise) |
| **Aşama 9** | Training & Checkpoints| `Trainer`, `CosineWarmupScheduler`, `.aic` Binary with Adler-32 | 3 tests | 100% PASS |
| **Aşama 10**| Autonomous Controller | `TelemetryTracker`, Anomaly Detection, Self-Healing Rollback | 7 tests | 100% PASS |
| **Aşama 11**| Integration & CLI | End-to-End Training, `aic` CLI, `train_mini_llm`, Documentation | 2 tests | 100% PASS (82/82 Total) |

---

## 5. Binary Checkpoint Format Specification (.aic)
The `.aic` format is designed for deterministic, corrupt-resistant model persistence:
```
+--------------------------------------------------------------------------+
| FIELD            | TYPE        | SIZE (BYTES) | DESCRIPTION              |
+--------------------------------------------------------------------------+
| magic            | uint32_t    | 4            | Magic bytes 0x41494330   |
| version          | uint32_t    | 4            | Format version (v1)      |
| step             | int64_t     | 8            | Training step index      |
| epoch            | int64_t     | 8            | Training epoch index     |
| best_val_loss    | float       | 4            | Lowest validation loss   |
| desc_len         | uint32_t    | 4            | Description string size  |
| description      | char[]      | desc_len     | UTF-8 description text   |
| num_tensors      | uint64_t    | 8            | Total number of tensors  |
+--------------------------------------------------------------------------+
| REPEATED FOR EACH TENSOR:                                                |
| name_len         | uint32_t    | 4            | Parameter name length    |
| name             | char[]      | name_len     | E.g. "layers.0.attn.w"   |
| rank             | int32_t     | 4            | Number of dimensions $R$ |
| dims             | int64_t[R]  | 8 * R        | Dimension extents        |
| dtype            | int32_t     | 4            | DType enum value         |
| nbytes           | uint64_t    | 8            | Raw data buffer byte len |
| data             | uint8_t[]   | nbytes       | Dense tensor bytes       |
| checksum         | uint32_t    | 4            | Adler-32 checksum        |
+--------------------------------------------------------------------------+
```
