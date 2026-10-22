// tests/unit/test_quantization.cpp
// Unit tests for Model Quantization & Compressed Inference Engine (Aşama 21)
#include "tests/test_framework.hpp"
#include "nn/quantization.hpp"
#include "nn/linear.hpp"
#include "models/transformer_lm.hpp"
#include "core/device.hpp"
#include "runtime/cuda_info.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace aic;
using namespace aic::nn;
using namespace aic::models;

AIC_TEST(QuantizationTest, INT8SymmetricQuantizationAndParity) {
    int64_t rows = 32;
    int64_t cols = 64;
    Tensor orig = Tensor::uniform(Shape({rows, cols}), -4.0f, 4.0f, DType::Float32, Device::cpu());

    QuantizedTensor q = QuantizedTensor::quantize(orig, QuantType::INT8_SYMMETRIC, 32);
    ASSERT_TRUE(q.quant_type() == QuantType::INT8_SYMMETRIC);
    ASSERT_EQ(q.group_size(), 32);
    ASSERT_TRUE(q.original_shape() == Shape({rows, cols}));
    ASSERT_EQ(q.packed_data().numel(), rows * cols);

    Tensor dequant = q.dequantize();
    ASSERT_TRUE(dequant.shape() == orig.shape());

    float sim = compute_cosine_similarity(orig, dequant);
    ASSERT_TRUE(sim > 0.999f);

    // Verify quantization bounds
    const float* p_orig = orig.data_ptr<float>();
    const float* p_deq = dequant.data_ptr<float>();
    float max_err = 0.0f;
    for (int64_t i = 0; i < orig.numel(); ++i) {
        float err = std::abs(p_orig[i] - p_deq[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err < 0.08f);
}

AIC_TEST(QuantizationTest, INT4BlockScaledPackingAndCompression) {
    int64_t rows = 64;
    int64_t cols = 128; // 8192 elements = 32,768 bytes in FP32
    Tensor orig = Tensor::uniform(Shape({rows, cols}), -2.5f, 2.5f, DType::Float32, Device::cpu());

    QuantizedTensor q = QuantizedTensor::quantize(orig, QuantType::INT4_SYMMETRIC, 32);
    ASSERT_TRUE(q.quant_type() == QuantType::INT4_SYMMETRIC);
    ASSERT_EQ(q.group_size(), 32);

    // 8192 elements packed 2 per byte -> 4096 bytes
    ASSERT_EQ(q.packed_data().numel(), 4096);
    size_t packed_bytes = static_cast<size_t>(q.packed_data().numel()) * dtype_size(q.packed_data().dtype());
    ASSERT_EQ(packed_bytes, 4096);

    // Original = 32768 bytes -> Compression Ratio > 3.5x
    ASSERT_TRUE(q.compression_ratio() > 3.5);

    Tensor dequant = q.dequantize();
    float sim = compute_cosine_similarity(orig, dequant);
    ASSERT_TRUE(sim > 0.99f);
}

AIC_TEST(QuantizationTest, FP8E4M3AndE5M2BitExactRepresentation) {
    // Test E4M3 encoding/decoding on key values
    std::vector<float> test_vals = {0.0f, -0.0f, 0.125f, -0.125f, 1.0f, -1.0f, 16.0f, -16.0f, 128.0f, 448.0f};
    for (float val : test_vals) {
        uint8_t b = float_to_fp8_e4m3(val);
        float rec = fp8_e4m3_to_float(b);
        if (val == 0.0f) {
            ASSERT_EQ(rec, 0.0f);
        } else {
            float rel_err = std::abs(val - rec) / std::abs(val);
            ASSERT_TRUE(rel_err < 0.15f); // E4M3 has 3 bits of mantissa (~12.5% precision)
        }
    }

    // Test tensor-level FP8 quantization
    Tensor orig = Tensor::uniform(Shape({32, 64}), -10.0f, 10.0f, DType::Float32, Device::cpu());
    QuantizedTensor q_e4m3 = QuantizedTensor::quantize(orig, QuantType::FP8_E4M3, 64);
    Tensor deq_e4m3 = q_e4m3.dequantize();
    float sim_e4m3 = compute_cosine_similarity(orig, deq_e4m3);
    ASSERT_TRUE(sim_e4m3 > 0.980f);

    QuantizedTensor q_e5m2 = QuantizedTensor::quantize(orig, QuantType::FP8_E5M2, 64);
    Tensor deq_e5m2 = q_e5m2.dequantize();
    float sim_e5m2 = compute_cosine_similarity(orig, deq_e5m2);
    ASSERT_TRUE(sim_e5m2 > 0.990f);
}

AIC_TEST(QuantizationTest, QuantizedLinearInferenceParity) {
    int64_t in_features = 64;
    int64_t out_features = 128;
    Linear linear(in_features, out_features, true, Device::cpu());

    QuantConfig cfg;
    cfg.type = QuantType::INT4_SYMMETRIC;
    cfg.group_size = 32;

    auto qlinear = QuantizedLinear::from_linear(linear, cfg);
    ASSERT_EQ(qlinear->in_features(), in_features);
    ASSERT_EQ(qlinear->out_features(), out_features);
    ASSERT_TRUE(qlinear->has_bias());

    // Input activation batch: [2, 8, 64]
    Tensor x = Tensor::uniform(Shape({2, 8, in_features}), -1.0f, 1.0f, DType::Float32, Device::cpu());

    Tensor y_orig = linear(x);
    Tensor y_quant = (*qlinear)(x);

    ASSERT_TRUE(y_quant.shape() == y_orig.shape());
    float sim = compute_cosine_similarity(y_orig, y_quant);
    ASSERT_TRUE(sim > 0.990f);
}

AIC_TEST(QuantizationTest, TransformerLMModelQuantization) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.max_seq_len = 32;
    cfg.intermediate_dim = 128;

    TransformerLM model(cfg, Device::cpu());

    // Extract one linear projection and convert to QuantizedLinear
    Linear& q_proj = *model.layers()[0]->attn()->q_proj();
    QuantConfig q_cfg;
    q_cfg.type = QuantType::INT4_SYMMETRIC;
    q_cfg.group_size = 32;

    auto q_proj_quant = QuantizedLinear::from_linear(q_proj, q_cfg);
    ASSERT_TRUE(q_proj_quant->quantized_weight().compression_ratio() > 3.5);

    Tensor token_acts = Tensor::uniform(Shape({1, 4, 64}), -0.5f, 0.5f, DType::Float32, Device::cpu());
    Tensor out_orig = q_proj(token_acts);
    Tensor out_quant = (*q_proj_quant)(token_acts);

    float sim = compute_cosine_similarity(out_orig, out_quant);
    ASSERT_TRUE(sim > 0.990f);
}

AIC_TEST(QuantizationTest, BinarySerializationAndAdlerChecksum) {
    Tensor orig = Tensor::uniform(Shape({48, 64}), -3.0f, 3.0f, DType::Float32, Device::cpu());
    QuantizedTensor q = QuantizedTensor::quantize(orig, QuantType::INT4_SYMMETRIC, 32);

    std::string path = "test_quant_checkpoint.aic_q";
    q.save(path);

    QuantizedTensor loaded = QuantizedTensor::load(path, Device::cpu());
    ASSERT_TRUE(loaded.quant_type() == q.quant_type());
    ASSERT_EQ(loaded.group_size(), q.group_size());
    ASSERT_TRUE(loaded.original_shape() == q.original_shape());
    ASSERT_EQ(loaded.packed_data().numel(), q.packed_data().numel());

    Tensor deq_loaded = loaded.dequantize();
    ASSERT_TRUE(compute_cosine_similarity(orig, deq_loaded) > 0.99f);

    // Corrupt one byte in the packed payload (offset 60) and verify checksum failure
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(60);
        char bad_byte = 0x5A;
        file.write(&bad_byte, 1);
    }

    bool caught = false;
    try {
        QuantizedTensor::load(path, Device::cpu());
    } catch (const Error&) {
        caught = true;
    }
    ASSERT_TRUE(caught);

    std::remove(path.c_str());
}

AIC_TEST(QuantizationTest, CudaDequantizationAndExecution) {
    if (!aic::is_cuda_available()) {
        std::cout << "[SKIP] CUDA not available on this device\n";
        return;
    }

    Device gpu = Device::cuda(0);
    int64_t rows = 64;
    int64_t cols = 128;
    Tensor orig_cpu = Tensor::uniform(Shape({rows, cols}), -3.0f, 3.0f, DType::Float32, Device::cpu());

    // 1. INT4 GPU test
    QuantizedTensor q_int4 = QuantizedTensor::quantize(orig_cpu, QuantType::INT4_SYMMETRIC, 32);
    QuantizedTensor q_int4_gpu = q_int4.to(gpu);
    ASSERT_TRUE(q_int4_gpu.device().is_cuda());

    Tensor deq_gpu = q_int4_gpu.dequantize();
    ASSERT_TRUE(deq_gpu.is_cuda());
    Tensor deq_gpu_cpu = deq_gpu.to(Device::cpu());
    Tensor deq_cpu = q_int4.dequantize();

    float sim_int4 = compute_cosine_similarity(deq_cpu, deq_gpu_cpu);
    ASSERT_TRUE(sim_int4 > 0.9999f);

    // 2. FP8 E4M3 GPU test
    QuantizedTensor q_fp8 = QuantizedTensor::quantize(orig_cpu, QuantType::FP8_E4M3, 32);
    QuantizedTensor q_fp8_gpu = q_fp8.to(gpu);
    Tensor deq_fp8_gpu = q_fp8_gpu.dequantize();
    Tensor deq_fp8_cpu = deq_fp8_gpu.to(Device::cpu());
    float sim_fp8 = compute_cosine_similarity(q_fp8.dequantize(), deq_fp8_cpu);
    ASSERT_TRUE(sim_fp8 > 0.9999f);

    // 3. QuantizedLinear on CUDA
    Linear lin_cpu(cols, rows, true, Device::cpu());
    QuantConfig cfg;
    cfg.type = QuantType::INT4_SYMMETRIC;
    cfg.group_size = 32;

    auto qlinear = QuantizedLinear::from_linear(lin_cpu, cfg);
    // Move weight to GPU
    qlinear->set_quantized_weight(qlinear->quantized_weight().to(gpu));
    if (qlinear->bias()) {
        qlinear->set_bias(std::make_shared<Tensor>(qlinear->bias()->to(gpu)));
    }

    Tensor x_gpu = Tensor::uniform(Shape({2, 4, cols}), -1.0f, 1.0f, DType::Float32, gpu);
    Tensor y_gpu = (*qlinear)(x_gpu);
    ASSERT_TRUE(y_gpu.is_cuda());
    ASSERT_TRUE(y_gpu.shape() == Shape({2, 4, rows}));
}
