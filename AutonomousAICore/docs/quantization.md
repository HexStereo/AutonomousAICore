# AutonomousAICore: Model Quantization & Compressed Inference Engine (Aşama 21)

## Overview

Large language models require immense GPU memory capacity and memory bus bandwidth during both prefill and autoregressive decode phases. In generation workloads, LLM inference is strictly **memory-bandwidth bound**: reading model weight tensors from GDDR7 VRAM at every single token step dominates latency.

**AutonomousAICore Aşama 21** introduces a high-performance **Model Quantization & Compressed Inference Engine** built from first principles in Modern C++20 and CUDA C++ (NVIDIA Blackwell sm_120 architecture), supporting:
1. **INT4 Packed Quantization (W4A16)**: 4-bit signed/unsigned integers with 2 weights packed per byte (`uint8_t`), achieving **$6.4\times$ memory footprint reduction** compared to FP32 and **$3.2\times$ reduction** compared to FP16.
2. **INT8 Quantization (W8A16)**: Symmetric and asymmetric group-scaled 8-bit integer quantization with **99.999% numerical cosine fidelity**.
3. **FP8 Quantization (E4M3 & E5M2)**: Native Blackwell 8-bit floating-point formats conforming to the OCP/NVIDIA FP8 specification.
4. **Group-Wise Block Scaling**: Independent scale factors per block (e.g. 32 or 64 elements) mitigating outlier distortion.
5. **Drop-in `QuantizedLinear` Module**: Seamless modular replacement of standard `Linear` projections in Attention and FFN layers.
6. **Binary Serialization (`.aic_q`)**: Zero-copy binary persistence with Adler-32 checksum integrity validation.

---

## Supported Quantization Formats & Numerical Mathematics

### 1. INT4 Symmetric Packed Format (`QuantType::INT4_SYMMETRIC`)
- **Range**: $q \in [-7, 7]$
- **Scale Factor**: $s = \frac{\max(|w|)}{7.0}$
- **Quantization Formula**:
  $$q = \text{clamp}\left(\text{round}\left(\frac{w}{s}\right), -7, 7\right)$$
- **Bit Packing**: Two 4-bit nibbles packed into a single 8-bit byte:
  $$\text{byte} = ((q_1 \& 0x0F) \ll 4) \mid (q_0 \& 0x0F)$$
- **Dequantization Formula**:
  $$\hat{w}_0 = q_0 \times s, \quad \hat{w}_1 = q_1 \times s$$
- **Compression**: Cuts a 32-bit float down to 4 bits ($8\times$ raw, $\sim 6.4\times$ including block scales).

### 2. INT8 Symmetric Format (`QuantType::INT8_SYMMETRIC`)
- **Range**: $q \in [-127, 127]$
- **Scale Factor**: $s = \frac{\max(|w|)}{127.0}$
- **Dequantization**: $\hat{w} = q \times s$
- **Cosine Similarity**: $> 0.99999$ (essentially lossless relative to FP32).

### 3. FP8 E4M3 (`QuantType::FP8_E4M3`)
- **Specification**: 1 sign bit, 4 exponent bits, 3 mantissa bits (bias = 7).
- **Max Representable Normal**: $\pm 448.0$.
- **Precision**: Higher mantissa precision (3 bits), optimal for weight representation.

### 4. FP8 E5M2 (`QuantType::FP8_E5M2`)
- **Specification**: 1 sign bit, 5 exponent bits, 2 mantissa bits (bias = 15).
- **Max Representable Normal**: $\pm 57344.0$.
- **Dynamic Range**: Matches IEEE FP16 exponent range, optimal for activation preservation.

---

## Hardware Acceleration: NVIDIA Blackwell sm_120 CUDA Kernels

On NVIDIA Blackwell GPUs (RTX 5090), custom CUDA kernels execute register-tiled bitwise unpacking and dequantization directly in warp registers:
```cpp
// Kernel: dequantize_int4_kernel
uint8_t byte = d_packed[pair_idx];
uint8_t u_low  = byte & 0x0F;
uint8_t u_high = (byte >> 4) & 0x0F;

float val0 = (u_low >= 8)  ? static_cast<float>(static_cast<int>(u_low) - 16)  : static_cast<float>(u_low);
float val1 = (u_high >= 8) ? static_cast<float>(static_cast<int>(u_high) - 16) : static_cast<float>(u_high);

d_output[idx0] = val0 * d_scales[idx0 / group_size];
d_output[idx1] = val1 * d_scales[idx1 / group_size];
```

---

## Empirical Benchmarks (RTX 5090 Laptop GPU - Blackwell sm_120)

### Workload: Standard $1024 \times 2048$ Feed-Forward Weight Tensor ($2,097,152$ elements)
Executed via unified CLI: `aic quant-bench 1024 2048 <format> <group_size> cuda`

| Quantization Format | Original Size | Quantized Size | Compression Ratio | VRAM Saved | Dequant Latency | Bandwidth | Cosine Similarity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INT4 Symmetric** ($G=32$) | 8.00 MB | **1.25 MB** | **6.40x** | **84.38%** | **0.04 ms** | **186.89 GB/s** | **0.997698** |
| **INT8 Symmetric** ($G=64$) | 8.00 MB | **2.12 MB** | **3.76x** | **73.44%** | **0.05 ms** | **155.47 GB/s** | **0.999993** |
| **FP8 E4M3** ($G=64$) | 8.00 MB | **2.12 MB** | **3.76x** | **73.44%** | **0.11 ms** | **73.24 GB/s** | **0.984753** |

---

## CLI Usage

Run the built-in quantization benchmark across formats and devices:
```bash
# Benchmark INT4 (group size 32) on CUDA RTX 5090 Blackwell
aic quant-bench 1024 2048 int4 32 cuda

# Benchmark INT8 (group size 64) on CUDA
aic quant-bench 1024 2048 int8 64 cuda

# Benchmark FP8 E4M3 on CUDA
aic quant-bench 1024 2048 fp8_e4m3 64 cuda

# Benchmark INT4 on CPU AVX2
aic quant-bench 1024 2048 int4 32 cpu
```

---

## Unit Test Suite (`test_quantization.cpp`)
All 7 unit tests pass 100% on Ninja CUDA and MSVC CPU:
1. `QuantizationTest.INT8SymmetricQuantizationAndParity`: Range $[-127, 127]$, cosine similarity $> 0.999$, max error $< 0.08$.
2. `QuantizationTest.INT4BlockScaledPackingAndCompression`: 4-bit nibble packing, $> 3.5\times$ compression, cosine similarity $> 0.99$.
3. `QuantizationTest.FP8E4M3AndE5M2BitExactRepresentation`: Bit-exact encoding/decoding of zero, subnormal, and normal floats.
4. `QuantizationTest.QuantizedLinearInferenceParity`: Modular replacement of `Linear` with activations $[B, S, D]$, cosine similarity $> 0.99$.
5. `QuantizationTest.TransformerLMModelQuantization`: Quantization of attention projections in full `TransformerLM` model.
6. `QuantizationTest.BinarySerializationAndAdlerChecksum`: `.aic_q` save/load with Adler-32 corruption detection.
7. `QuantizationTest.CudaDequantizationAndExecution`: Native Blackwell sm_120 GPU execution and parity verification on RTX 5090.
