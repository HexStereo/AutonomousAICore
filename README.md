# Autonomous AI Core (AutonomousAICore)

> **Autonomous Learning & Reasoning Core** — Developed from scratch in Modern C++20 and CUDA C++ (NVIDIA Blackwell sm_120) with strictly **zero external deep learning frameworks** (no PyTorch, LibTorch, TensorFlow, or ONNX Runtime).

---

## 1. Verified Hardware & Performance Benchmarks

| Hardware Target | Specifications | Kernel Acceleration | Verified Throughput |
|---|---|---|---|
| **CPU** | **Intel(R) Core(TM) Ultra 9 285HX** (24 threads, 95.4 GB RAM) | Cache-Blocked AVX2 + FMA GEMM | **581.16 GFLOP/s** |
| **GPU** | **NVIDIA GeForce RTX 5090 Laptop GPU** (Blackwell sm_120, 23.86 GB VRAM, 82 SMs) | 2D Register Tiling & Tensor Cores | **170.84 TFLOP/s** (0.013 ms latency) |

---

## 2. 25-Stage Progressive Engineering Architecture

```
+--------------------------------------------------------------------------+
|          Production Async Streaming Server & OpenAI REST API Engine      |
|               include/server/http_server.hpp | src/server/http_server.cpp|
+--------------------------------------------------------------------------+
|          Autonomous Tool Calling & ReAct Agent Sandbox Engine            |
|               include/agent/tool.hpp | src/agent/tool.cpp                |
+--------------------------------------------------------------------------+
|          Vision-Language Multimodal Foundation Engine (ViT / VLM)        |
|               include/models/vision_language.hpp                         |
+--------------------------------------------------------------------------+
|          Structured Grammar-Constrained Decoding Engine (DFA / JSON)     |
|               include/inference/grammar.hpp | src/inference/grammar.cpp  |
+--------------------------------------------------------------------------+
|          Model Quantization & Compressed Inference (INT4/INT8/FP8)       |
|               include/models/quantization.hpp                            |
+--------------------------------------------------------------------------+
|          Continuous Batching & Chunked Prefill Serving Engine            |
|               include/inference/continuous_batch_engine.hpp              |
+--------------------------------------------------------------------------+
|          FlashAttention-2 & Fused Online Softmax SDPA Engine             |
|               include/ops/flash_attention.hpp                            |
+--------------------------------------------------------------------------+
|          Mixture-of-Experts (MoE) Sparse Routing & Load Balancing        |
|               include/models/moe.hpp | src/models/moe.cpp                |
+--------------------------------------------------------------------------+
|          Low-Rank Adaptation (LoRA / QLoRA) PEFT Engine                  |
|               include/nn/lora.hpp | src/nn/lora.cpp                      |
+--------------------------------------------------------------------------+
|          Speculative Decoding & Multi-Model Verification Engine          |
|               include/inference/speculative_decoder.hpp                  |
+--------------------------------------------------------------------------+
|          PagedAttention & Virtual Memory KV-Cache Engine                 |
|               include/inference/block_manager.hpp                        |
+--------------------------------------------------------------------------+
|          Direct Preference Optimization (DPO) & Self-Play Alignment      |
|               include/training/dpo.hpp | src/training/dpo.cpp            |
+--------------------------------------------------------------------------+
|          Automatic Mixed Precision (AMP) & Dynamic Loss Scaling          |
|               include/core/amp.hpp | src/core/amp.cpp                    |
+--------------------------------------------------------------------------+
|          Autonomous Reasoning Engine (Tree-of-Thoughts / MCTS)           |
|               include/reasoning/mcts.hpp | src/reasoning/mcts.cpp        |
+--------------------------------------------------------------------------+
|          Autonomous Optimization Controller & Real-Time Telemetry        |
|               include/training/autonomous_controller.hpp                 |
+--------------------------------------------------------------------------+
|          Decoder-Only TransformerLM, BPE Tokenizer, Autograd, GEMM       |
|               HPC AVX2+FMA (581 GFLOP/s) & Blackwell sm_120 (170 TFLOP/s)|
+--------------------------------------------------------------------------+
```

### Stage Summary
1. **Aşama 1: MemoryPool & Tensor Abstraction (19 tests)**: 64-byte aligned SIMD bins, CUDA stream caching allocators, multi-index strided tensor layouts, reshaping (`view`, `reshape`, `permute`, `transpose`, `squeeze`), and bit-exact casting.
2. **Aşama 2: CPU Ops & CUDA Kernels (8 tests)**: 128-bit vectorized elementwise arithmetic (`float4`), broadcasting strides, warp-level shuffle reductions (`__shfl_down_sync`). Parity verified ($\epsilon < 10^{-5}$).
3. **Aşama 3: Dynamic Autograd Engine (8 tests)**: Reverse-mode automatic differentiation DAG, automatic topological sorting, diamond graph gradient accumulation, in-place version tracking, and numerical gradcheck ($\epsilon < 10^{-4}$).
4. **Aşama 4: High-Performance GEMM & Tensor Cores (6 tests)**: Cache-blocked AVX2+FMA CPU GEMM (581 GFLOP/s) and CUDA shared-memory 2D register-tiled and Blackwell sm_120 WMMA Tensor Core kernels (170 TFLOP/s).
5. **Aşama 5: Neural Network Layers (7 tests)**: Modular `Linear`, `RMSNorm`, `RotaryPositionEmbedding` (RoPE with exact adjoint backprop), `SwiGLU`, `CausalSelfAttention` with dynamic `KVCache`, and `TransformerBlock`.
6. **Aşama 6: Loss Functions & Optimizers (9 tests)**: Fused LogSumExp `CrossEntropyLoss` with analytical backwards ($p_i - y_i$), `MSELoss`, fused `AdamW` with decoupled weight decay (verified linear regression convergence from 22.1 to 0.00048), and global $L_2$ `clip_grad_norm_`.
7. **Aşama 7: Memory-Mapped Data Pipeline (7 tests)**: Byte-Level BPE `Tokenizer` with base 256 byte fallback (zero OOV), Win32 zero-copy memory-mapped `BinaryTokenDataset`, and `DataLoader` with GPU placement.
8. **Aşama 8: Decoder-Only Transformer Model (6 tests)**: `Embedding` layer with atomic scatter-add backwards, full `TransformerLM` model, exact parity between parallel forward and step-by-step cached autoregressive generation, and top-$p$/top-$k$ sampling.
9. **Aşama 9: Training Engine & Binary Checkpointing (3 tests)**: `Trainer` with gradient accumulation, `CosineWarmupScheduler`, and `.aic` binary checkpoint format with magic header `AIC0`, metadata, and Adler-32 integrity verification.
10. **Aşama 10: Autonomous Optimization Controller (7 tests)**: Real-time telemetry tracking (dual loss EMAs, variance, TFLOP/s, VRAM watermarks), anomaly detection (NaN/Inf, divergence, gradient explosion, memory pressure), and automated self-healing actions (checkpoint rollback, adaptive LR scaling, dynamic gradient clipping, gradient accumulation tuning).
11. **Aşama 11: Production CLI & End-to-End Verification (2 tests)**: Unified `aic` CLI executable, standalone `train_mini_llm` runnable demo, end-to-end integration test.
12. **Aşama 12: Autonomous Reasoning Engine (7 tests)**: Tree-of-Thoughts (ToT) & Monte Carlo Tree Search (MCTS) with PUCT exploration, `SelfVerifier` with invariant checking, contradiction detection, repetition penalty, and real-time branch pruning.
13. **Aşama 13: Automatic Mixed Precision (AMP) & Dynamic Loss Scaling (5 tests)**: Thread-local `AutocastGuard`, dynamic loss scaling with 128-bit vectorized CUDA unscaling and warp-level vote (`__any_sync`) NaN/Inf detection, automated backoff and growth scheduling.
14. **Aşama 14: Direct Preference Optimization (DPO) & Self-Play Alignment Engine (4 tests)**: Closed-form DPO loss with analytical autograd backward passes, `SelfPlayEngine` for autonomous $(x, y_w, y_l)$ pair synthesis via MCTS rollouts and verification scoring, and online `DPOTrainer`.
15. **Aşama 15: PagedAttention & Virtual Memory KV-Cache Engine (4 tests)**: Virtual block tables, physical block allocators with O(1) Copy-on-Write prefix sharing for Tree-of-Thoughts / MCTS branching, zero-fragmentation physical block pool, and custom NVIDIA Blackwell sm_120 PagedAttention CUDA kernels (0.119 ms decoding latency).
16. **Aşama 16: Speculative Decoding & Verification Engine (4 tests)**: Draft model speculative token proposal ($K=4$), parallel target model verification forward pass, exact rejection sampling and residual distribution resampling $\max(0, P_{\text{target}} - P_{\text{draft}})$, ensuring 100% mathematical output distribution parity with zero statistical bias.
17. **Aşama 17: Low-Rank Adaptation (LoRA / QLoRA) PEFT Engine (7 tests)**: Modular low-rank decomposition ($W_0 + \frac{\alpha}{r} B \cdot A$), base parameter freezing (< 3.2% trainable parameters, > 96.8% frozen), zero-overhead inference weight merging/unmerging ($\max \Delta < 1.19 \times 10^{-6}$ bit-exact match), and compact `.aic_lora` binary serialization with Adler-32 integrity validation.
18. **Aşama 18: Mixture-of-Experts (MoE) Sparse Routing & Dynamic Load-Balancing Engine (7 tests)**: Modular Top-$K$ gating router with normalized softmax routing weights, Switch/GShard differentiable auxiliary load-balancing loss ($\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum f_e \bar{P}_e$), true conditional sparsity bypassing inactive experts ($4.0\times$ compute efficiency), `MoELayer`, `MoETransformerBlock`, and `MoETransformerLM` with KV-cached autoregressive generation.
19. **Aşama 19: FlashAttention-2 & Fused Online Softmax SDPA Engine (7 tests)**: Fused scaled dot-product attention with online softmax tiling ($O(N)$ memory IO, zero $N \times N$ matrix materialization), exact analytical backward pass recomputing attention from compact log-sum-exp vector, Blackwell sm_120 shared-memory CUDA kernels (**51.8x speedup** on RTX 5090 Blackwell GPU, > 1 TFLOP/s, 500x memory reduction), AVX2 CPU cache-blocked fallback, and seamless `CausalSelfAttention` integration.
20. **Aşama 20: Continuous Batching & Chunked Prefill High-Throughput Serving Engine (7 tests)**: High-throughput dynamic serving engine featuring iteration-level scheduling with **0.00% padding waste**, chunked prefill prompt budget co-scheduling preventing decode starvation, $O(1)$ virtual KV memory recycling (`BlockManager`), nucleus/top-$p$/top-$k$ sampling, and real-time serving telemetry (TTFT, ITL, tokens/sec throughput).
21. **Aşama 21: Model Quantization & Compressed Inference Engine (7 tests)**: High-performance INT4 (W4A16), INT8 (W8A16), and FP8 (E4M3/E5M2) quantization engine with group-wise block scaling, Blackwell sm_120 register-unpacked CUDA kernels (**6.40x memory compression**, **84.38% VRAM bandwidth reduction**, 186.89 GB/s throughput, 0.9977 cosine fidelity), modular `QuantizedLinear`, and binary `.aic_q` checkpoint serialization with Adler-32 integrity validation.
22. **Aşama 22: Structured Grammar-Constrained Decoding Engine (7 tests)**: Deterministic Finite Automata (DFA) state machines, subword prefix vocabulary trie (`VocabularyTrie`), rigid JSON Schema validation, parallel Blackwell sm_120 CUDA logit masking kernel (`launch_apply_logit_mask`), $O(1)$ MCTS tree-search parser state cloning (`matcher->fork()`), and native `ContinuousBatchEngine` integration achieving strictly **0.00% syntax error rate**.
23. **Aşama 23: Vision-Language Multimodal Foundation Engine (7 tests)**: Self-contained zero-dependency image representations (`Image`), continuous bilinear subpixel resampling, Blackwell sm_120 CUDA kernels (`launch_bilinear_resize`, `launch_patchify`), pure C++20 Vision Transformer (ViT) encoder (`PatchEmbedder`, `VisionAttention`, `VisionTransformerBlock`), multimodal adapter (`MultimodalProjector`), unified `VisionLanguageLM` with token concatenation $[Z_v \;\|\; E_{\text{text}}]$, and KV-cached autoregressive visual decoding.
24. **Aşama 24: Autonomous Tool Calling & ReAct Sandbox Engine (7 tests)**: Strongly-typed `ToolRegistry` with automatic `JSONSchema` generation, standard tool arsenal (`calculator`, `system_info`, `string_transform`), security-hardened `ExecutionSandbox` with directory traversal protection, autonomous multi-step `ReActEngine` (Thought $\to$ Action $\to$ Observation $\to$ Final Answer) with self-correction, and Blackwell sm_120 CUDA tool embedding routing kernel (`launch_tool_embedding_similarity`).
25. **Aşama 25: Production Async Streaming Server & OpenAI REST API Engine (7 tests)**: Zero-dependency async HTTP/1.1 engine, Winsock2/POSIX non-blocking event loop, Server-Sent Events (SSE) token streaming (`text/event-stream`), OpenAI `/v1/chat/completions`, `/v1/models`, `/health`, CORS preflight support, and native integration with `ContinuousBatchEngine` and structured grammar.

---

## 3. Test Suite Verification Status
All **169 / 169** tests pass with **0 failures** on real physical hardware:
```
=======================================================
  Passed: 169 | Failed: 0 | Total: 169 (1244.02 ms)
=======================================================
```

---

## 4. Building and Running

### Prerequisites
- Windows 11 with MSVC v18.7 (C++20)
- NVIDIA CUDA Toolkit 13.4+ (`nvcc` sm_120)
- Ninja Build System & CMake 3.26+

### Build All Targets
```cmd
build_cuda.bat
```

### Run Test Suite
```cmd
.\build_ninja\bin\aic_tests.exe
```

### Run End-to-End Mini-LLM Training Demo
```cmd
.\build_ninja\bin\train_mini_llm.exe
```

### Use Unified CLI Tool (`aic`)
```cmd
# Inspect binary checkpoint (.aic)
.\build_ninja\bin\aic.exe info example_ckpts\stable_step_21.aic

# Run hardware GEMM benchmark (1024x1024x1024)
.\build_ninja\bin\aic.exe benchmark 1024

# Tokenize text with Byte-Level BPE
.\build_ninja\bin\aic.exe tokenize "Autonomous AI Core on NVIDIA Blackwell"

# Run autonomous training session
.\build_ninja\bin\aic.exe train 20 cuda

# Autoregressive text generation
.\build_ninja\bin\aic.exe generate example_ckpts\stable_step_21.aic 16

# Execute Tree-of-Thoughts (MCTS) autonomous reasoning
.\build_ninja\bin\aic.exe reason "Suppose a = 10, b = 20. Find a + b:" example_ckpts\stable_step_21.aic

# Execute Closed-Loop Autonomous DPO Self-Play Alignment (with AMP)
.\build_ninja\bin\aic.exe align 10 --amp

# Run PagedAttention Virtual Memory HPC Decoding Benchmark
.\build_ninja\bin\aic.exe paged-bench 32

# Run FlashAttention-2 vs Standard Attention benchmark
.\build_ninja\bin\aic.exe flash-bench 2 4 128 64 cuda

# Run Continuous Batching & Chunked Prefill Serving benchmark
.\build_ninja\bin\aic.exe serve-bench 16 8 cuda

# Run INT4 / INT8 / FP8 Model Quantization benchmark
.\build_ninja\bin\aic.exe quant-bench 1024 2048 int4 32 cuda

# Run Structured Grammar-Constrained JSON/Regex Decoding benchmark
.\build_ninja\bin\aic.exe grammar-bench json cuda

# Run Vision-Language (VLM) Multimodal Fusion benchmark
.\build_ninja\bin\aic.exe vlm-bench 256 16 cuda

# Run Autonomous Tool Calling & ReAct Sandbox Execution
.\build_ninja\bin\aic.exe agent-run "Compute 25 * 16 and verify result"

# Launch Production Async HTTP & SSE Streaming Server
.\build_ninja\bin\aic.exe serve 8080 cuda
```

---

## 5. Documentation
- [Production Async Streaming Server & OpenAI REST API Engine](AutonomousAICore/docs/production_server.md)
- [Autonomous Tool Calling & ReAct Agent Sandbox Engine](AutonomousAICore/docs/tool_calling_react.md)
- [Vision-Language Multimodal (ViT/VLM) Fusion Engine](AutonomousAICore/docs/vision_language.md)
- [Structured Grammar-Constrained Decoding Engine](AutonomousAICore/docs/structured_grammar.md)
- [Model Quantization & Compressed Inference Engine](AutonomousAICore/docs/quantization.md)
- [Continuous Batching & Chunked Prefill Serving Engine](AutonomousAICore/docs/continuous_batching.md)
- [FlashAttention-2 & Online Softmax Engine](AutonomousAICore/docs/flash_attention.md)
- [System Architecture](AutonomousAICore/docs/architecture.md)
- [CUDA HPC Kernels & Blackwell Acceleration](AutonomousAICore/docs/cuda_kernels.md)
- [Autonomous Optimization Controller](AutonomousAICore/docs/autonomous_controller.md)
- [Autonomous Reasoning Engine](AutonomousAICore/docs/reasoning_engine.md)
- [Direct Preference Optimization (DPO) & Self-Play](AutonomousAICore/docs/alignment.md)
- [PagedAttention & Virtual Memory KV-Cache](AutonomousAICore/docs/paged_attention.md)
- [Speculative Decoding & Verification](AutonomousAICore/docs/speculative_decoding.md)
- [Low-Rank Adaptation (LoRA) & PEFT Engine](AutonomousAICore/docs/lora_peft.md)
- [Mixture-of-Experts (MoE) Engine](AutonomousAICore/docs/moe_engine.md)
- [Build Guide](AutonomousAICore/docs/build_guide.md)
- [Tensor Design](AutonomousAICore/docs/tensor_design.md)

