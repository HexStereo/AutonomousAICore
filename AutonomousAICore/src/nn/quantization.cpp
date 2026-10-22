// src/nn/quantization.cpp
// Implementation of Model Quantization & Compressed Inference Engine (Aşama 21)
#include "nn/quantization.hpp"
#include "core/logging.hpp"
#include "ops/matmul.hpp"
#include "ops/ops.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_quantize.cuh"
#include "runtime/cuda_info.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <stdexcept>

namespace aic::nn {

const char* quant_type_to_string(QuantType type) noexcept {
    switch (type) {
        case QuantType::INT8_SYMMETRIC:  return "INT8_SYMMETRIC";
        case QuantType::INT8_ASYMMETRIC: return "INT8_ASYMMETRIC";
        case QuantType::INT4_SYMMETRIC:  return "INT4_SYMMETRIC";
        case QuantType::INT4_ASYMMETRIC: return "INT4_ASYMMETRIC";
        case QuantType::FP8_E4M3:        return "FP8_E4M3";
        case QuantType::FP8_E5M2:        return "FP8_E5M2";
    }
    return "UNKNOWN";
}

bool parse_quant_type(const std::string& str, QuantType& out) noexcept {
    if (str == "int8" || str == "INT8" || str == "int8_sym" || str == "INT8_SYMMETRIC") {
        out = QuantType::INT8_SYMMETRIC;
        return true;
    }
    if (str == "int8_asym" || str == "INT8_ASYMMETRIC") {
        out = QuantType::INT8_ASYMMETRIC;
        return true;
    }
    if (str == "int4" || str == "INT4" || str == "int4_sym" || str == "INT4_SYMMETRIC") {
        out = QuantType::INT4_SYMMETRIC;
        return true;
    }
    if (str == "int4_asym" || str == "INT4_ASYMMETRIC") {
        out = QuantType::INT4_ASYMMETRIC;
        return true;
    }
    if (str == "fp8_e4m3" || str == "FP8_E4M3" || str == "fp8") {
        out = QuantType::FP8_E4M3;
        return true;
    }
    if (str == "fp8_e5m2" || str == "FP8_E5M2") {
        out = QuantType::FP8_E5M2;
        return true;
    }
    return false;
}

uint8_t float_to_fp8_e4m3(float val) noexcept {
    if (std::isnan(val)) return 0x7F; // Standard NaN in E4M3
    uint32_t x;
    std::memcpy(&x, &val, sizeof(x));
    uint8_t sign = (x >> 31) & 0x01;
    float abs_val = std::abs(val);

    if (abs_val < 0.001953125f) { // Underflow to zero
        return sign << 7;
    }
    if (abs_val > 448.0f) { // Saturation
        abs_val = 448.0f;
    }

    int exp;
    std::frexp(abs_val, &exp); // abs_val = mant * 2^exp, mant in [0.5, 1)
    // In IEEE standard: mantissa has implicit 1.0 (range [1, 2)), so unbiased_exp = exp - 1
    int unbiased_exp = exp - 1;
    int biased_exp = unbiased_exp + 7;

    if (biased_exp <= 0) {
        // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
        float subnormal_val = abs_val * 64.0f * 8.0f;
        int m = static_cast<int>(std::round(subnormal_val));
        if (m > 7) m = 7;
        return static_cast<uint8_t>((sign << 7) | m);
    }
    if (biased_exp >= 15) {
        // Clamp to max normal (exp = 15, mant = 6 -> 448.0)
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    // Normal: 1.mant_bits * 2^unbiased_exp
    float frac = (abs_val / std::ldexp(1.0f, unbiased_exp)) - 1.0f;
    int m = static_cast<int>(std::round(frac * 8.0f));
    if (m >= 8) {
        m = 0;
        biased_exp++;
        if (biased_exp >= 15) return static_cast<uint8_t>((sign << 7) | 0x7E);
    }
    return static_cast<uint8_t>((sign << 7) | ((biased_exp & 0x0F) << 3) | (m & 0x07));
}

float fp8_e4m3_to_float(uint8_t byte) noexcept {
    uint32_t sign = (byte >> 7) & 0x01;
    uint32_t exp  = (byte >> 3) & 0x0F;
    uint32_t mant = byte & 0x07;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = static_cast<float>(mant) / 8.0f * (1.0f / 64.0f);
        return sign ? -val : val;
    }
    if (exp == 0x0F && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    int unbiased_exp = static_cast<int>(exp) - 7;
    float scale = std::ldexp(1.0f, unbiased_exp);
    float val = scale * (1.0f + static_cast<float>(mant) / 8.0f);
    return sign ? -val : val;
}

uint8_t float_to_fp8_e5m2(float val) noexcept {
    if (std::isnan(val)) return 0x7F;
    if (std::isinf(val)) return val > 0 ? 0x7C : 0xFC;
    uint32_t x;
    std::memcpy(&x, &val, sizeof(x));
    uint8_t sign = (x >> 31) & 0x01;
    float abs_val = std::abs(val);

    if (abs_val < 0.0000152587890625f) { // Underflow
        return sign << 7;
    }
    if (abs_val > 57344.0f) {
        abs_val = 57344.0f;
    }

    int exp;
    std::frexp(abs_val, &exp);
    int unbiased_exp = exp - 1;
    int biased_exp = unbiased_exp + 15;

    if (biased_exp <= 0) {
        // Subnormal
        float subnormal_val = abs_val * 16384.0f * 4.0f;
        int m = static_cast<int>(std::round(subnormal_val));
        if (m > 3) m = 3;
        return static_cast<uint8_t>((sign << 7) | m);
    }
    if (biased_exp >= 31) {
        return static_cast<uint8_t>((sign << 7) | 0x7B); // Max representable normal
    }

    float frac = (abs_val / std::ldexp(1.0f, unbiased_exp)) - 1.0f;
    int m = static_cast<int>(std::round(frac * 4.0f));
    if (m >= 4) {
        m = 0;
        biased_exp++;
        if (biased_exp >= 31) return static_cast<uint8_t>((sign << 7) | 0x7B);
    }
    return static_cast<uint8_t>((sign << 7) | ((biased_exp & 0x1F) << 2) | (m & 0x03));
}

float fp8_e5m2_to_float(uint8_t byte) noexcept {
    uint32_t sign = (byte >> 7) & 0x01;
    uint32_t exp  = (byte >> 2) & 0x1F;
    uint32_t mant = byte & 0x03;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = static_cast<float>(mant) / 4.0f * (1.0f / 16384.0f);
        return sign ? -val : val;
    }
    if (exp == 0x1F) {
        if (mant == 0) return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }
    int unbiased_exp = static_cast<int>(exp) - 15;
    float scale = std::ldexp(1.0f, unbiased_exp);
    float val = scale * (1.0f + static_cast<float>(mant) / 4.0f);
    return sign ? -val : val;
}

QuantizedTensor QuantizedTensor::quantize(const Tensor& input, QuantType type, int32_t group_size) {
    if (input.dtype() != DType::Float32) {
        AIC_THROW(DTypeMismatch, "QuantizedTensor::quantize currently requires Float32 input tensor");
    }

    Tensor cpu_input = input.is_cuda() ? input.to(Device::cpu()) : input;
    if (!cpu_input.is_contiguous()) {
        cpu_input = cpu_input.contiguous();
    }

    const float* in_ptr = cpu_input.data_ptr<float>();
    int64_t numel = cpu_input.numel();
    if (numel == 0) {
        AIC_THROW(InvalidArgument, "QuantizedTensor::quantize cannot quantize an empty tensor");
    }

    if (group_size <= 0) {
        // Per-tensor group
        group_size = static_cast<int32_t>(numel);
    }

    int64_t num_groups = (numel + group_size - 1) / group_size;

    QuantizedTensor result;
    result.type_ = type;
    result.group_size_ = group_size;
    result.original_shape_ = input.shape();

    // Allocate scales and zero_points on CPU
    Tensor cpu_scales = Tensor::zeros(Shape({num_groups}), DType::Float32, Device::cpu());
    float* scales_ptr = cpu_scales.data_ptr<float>();

    Tensor cpu_zero_points;
    float* zp_ptr = nullptr;
    if (type == QuantType::INT8_ASYMMETRIC || type == QuantType::INT4_ASYMMETRIC) {
        result.has_zero_points_ = true;
        cpu_zero_points = Tensor::zeros(Shape({num_groups}), DType::Float32, Device::cpu());
        zp_ptr = cpu_zero_points.data_ptr<float>();
    }

    // Allocate packed byte storage on CPU
    int64_t packed_numel = (type == QuantType::INT4_SYMMETRIC || type == QuantType::INT4_ASYMMETRIC)
                           ? (numel + 1) / 2
                           : numel;
    Tensor cpu_packed = Tensor::zeros(Shape({packed_numel}), DType::UInt8, Device::cpu());
    uint8_t* packed_ptr = cpu_packed.data_ptr<uint8_t>();

    // Perform quantization per group
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t start = g * group_size;
        int64_t end = std::min<int64_t>(start + group_size, numel);

        float min_val = in_ptr[start];
        float max_val = in_ptr[start];
        float abs_max = std::abs(in_ptr[start]);

        for (int64_t i = start + 1; i < end; ++i) {
            float v = in_ptr[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            float av = std::abs(v);
            if (av > abs_max) abs_max = av;
        }

        switch (type) {
            case QuantType::INT8_SYMMETRIC: {
                float s = abs_max / 127.0f;
                if (s < 1e-8f) s = 1e-8f;
                scales_ptr[g] = s;
                for (int64_t i = start; i < end; ++i) {
                    float q = std::round(in_ptr[i] / s);
                    q = std::clamp(q, -127.0f, 127.0f);
                    packed_ptr[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
                }
                break;
            }
            case QuantType::INT8_ASYMMETRIC: {
                float range = max_val - min_val;
                float s = range / 255.0f;
                if (s < 1e-8f) s = 1e-8f;
                float zp = std::round(-min_val / s);
                zp = std::clamp(zp, 0.0f, 255.0f);
                scales_ptr[g] = s;
                zp_ptr[g] = zp;
                for (int64_t i = start; i < end; ++i) {
                    float q = std::round(in_ptr[i] / s) + zp;
                    q = std::clamp(q, 0.0f, 255.0f);
                    packed_ptr[i] = static_cast<uint8_t>(q);
                }
                break;
            }
            case QuantType::INT4_SYMMETRIC: {
                float s = abs_max / 7.0f;
                if (s < 1e-8f) s = 1e-8f;
                scales_ptr[g] = s;
                for (int64_t i = start; i < end; i += 2) {
                    float q0 = std::round(in_ptr[i] / s);
                    q0 = std::clamp(q0, -7.0f, 7.0f);
                    int8_t low = static_cast<int8_t>(q0);

                    int8_t high = 0;
                    if (i + 1 < end) {
                        float q1 = std::round(in_ptr[i + 1] / s);
                        q1 = std::clamp(q1, -7.0f, 7.0f);
                        high = static_cast<int8_t>(q1);
                    }
                    packed_ptr[i / 2] = pack_int4_byte(low, high);
                }
                break;
            }
            case QuantType::INT4_ASYMMETRIC: {
                float range = max_val - min_val;
                float s = range / 15.0f;
                if (s < 1e-8f) s = 1e-8f;
                float zp = std::round(-min_val / s);
                zp = std::clamp(zp, 0.0f, 15.0f);
                scales_ptr[g] = s;
                zp_ptr[g] = zp;
                for (int64_t i = start; i < end; i += 2) {
                    float q0 = std::round(in_ptr[i] / s) + zp;
                    q0 = std::clamp(q0, 0.0f, 15.0f);
                    int8_t low = static_cast<int8_t>(q0);

                    int8_t high = 0;
                    if (i + 1 < end) {
                        float q1 = std::round(in_ptr[i + 1] / s) + zp;
                        q1 = std::clamp(q1, 0.0f, 15.0f);
                        high = static_cast<int8_t>(q1);
                    }
                    packed_ptr[i / 2] = pack_int4_byte(low, high);
                }
                break;
            }
            case QuantType::FP8_E4M3: {
                float s = abs_max / 448.0f;
                if (s < 1e-8f) s = 1e-8f;
                scales_ptr[g] = s;
                for (int64_t i = start; i < end; ++i) {
                    packed_ptr[i] = float_to_fp8_e4m3(in_ptr[i] / s);
                }
                break;
            }
            case QuantType::FP8_E5M2: {
                float s = abs_max / 57344.0f;
                if (s < 1e-8f) s = 1e-8f;
                scales_ptr[g] = s;
                for (int64_t i = start; i < end; ++i) {
                    packed_ptr[i] = float_to_fp8_e5m2(in_ptr[i] / s);
                }
                break;
            }
        }
    }

    // Move to target device if needed
    Device target_dev = input.device();
    if (target_dev.is_cuda()) {
        result.packed_data_ = cpu_packed.to(target_dev);
        result.scales_ = cpu_scales.to(target_dev);
        if (result.has_zero_points_) {
            result.zero_points_ = cpu_zero_points.to(target_dev);
        }
    } else {
        result.packed_data_ = std::move(cpu_packed);
        result.scales_ = std::move(cpu_scales);
        if (result.has_zero_points_) {
            result.zero_points_ = std::move(cpu_zero_points);
        }
    }

    return result;
}

Tensor QuantizedTensor::dequantize() const {
    int64_t numel = original_shape_.numel();
    if (numel == 0) {
        return Tensor::zeros(original_shape_, DType::Float32, device());
    }

#if AIC_WITH_CUDA
    if (device().is_cuda()) {
        Tensor output = Tensor::empty(original_shape_, DType::Float32, device());
        const uint8_t* d_packed = packed_data_.data_ptr<uint8_t>();
        const float* d_scales = scales_.data_ptr<float>();
        const float* d_zp = has_zero_points_ ? zero_points_.data_ptr<float>() : nullptr;
        float* d_out = output.data_ptr<float>();

        switch (type_) {
            case QuantType::INT4_SYMMETRIC:
                cuda::launch_dequantize_int4(d_packed, d_scales, nullptr, d_out, numel, group_size_, true);
                break;
            case QuantType::INT4_ASYMMETRIC:
                cuda::launch_dequantize_int4(d_packed, d_scales, d_zp, d_out, numel, group_size_, false);
                break;
            case QuantType::INT8_SYMMETRIC:
                cuda::launch_dequantize_int8(d_packed, d_scales, nullptr, d_out, numel, group_size_, true);
                break;
            case QuantType::INT8_ASYMMETRIC:
                cuda::launch_dequantize_int8(d_packed, d_scales, d_zp, d_out, numel, group_size_, false);
                break;
            case QuantType::FP8_E4M3:
                cuda::launch_dequantize_fp8(d_packed, d_scales, d_out, numel, group_size_, true);
                break;
            case QuantType::FP8_E5M2:
                cuda::launch_dequantize_fp8(d_packed, d_scales, d_out, numel, group_size_, false);
                break;
        }
        cudaDeviceSynchronize();
        return output;
    }
#endif

    // CPU execution path
    Tensor cpu_packed = packed_data_.is_cuda() ? packed_data_.to(Device::cpu()) : packed_data_;
    Tensor cpu_scales = scales_.is_cuda() ? scales_.to(Device::cpu()) : scales_;
    Tensor cpu_zp;
    if (has_zero_points_) {
        cpu_zp = zero_points_.is_cuda() ? zero_points_.to(Device::cpu()) : zero_points_;
    }

    Tensor output = Tensor::empty(original_shape_, DType::Float32, Device::cpu());
    const uint8_t* p_packed = cpu_packed.data_ptr<uint8_t>();
    const float* p_scales = cpu_scales.data_ptr<float>();
    const float* p_zp = has_zero_points_ ? cpu_zp.data_ptr<float>() : nullptr;
    float* p_out = output.data_ptr<float>();

    int64_t num_groups = (numel + group_size_ - 1) / group_size_;

    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t start = g * group_size_;
        int64_t end = std::min<int64_t>(start + group_size_, numel);
        float s = p_scales[g];
        float zp = p_zp ? p_zp[g] : 0.0f;

        switch (type_) {
            case QuantType::INT8_SYMMETRIC: {
                for (int64_t i = start; i < end; ++i) {
                    p_out[i] = static_cast<float>(static_cast<int8_t>(p_packed[i])) * s;
                }
                break;
            }
            case QuantType::INT8_ASYMMETRIC: {
                for (int64_t i = start; i < end; ++i) {
                    p_out[i] = (static_cast<float>(p_packed[i]) - zp) * s;
                }
                break;
            }
            case QuantType::INT4_SYMMETRIC: {
                for (int64_t i = start; i < end; i += 2) {
                    int8_t low, high;
                    unpack_int4_byte(p_packed[i / 2], low, high, true);
                    p_out[i] = static_cast<float>(low) * s;
                    if (i + 1 < end) {
                        p_out[i + 1] = static_cast<float>(high) * s;
                    }
                }
                break;
            }
            case QuantType::INT4_ASYMMETRIC: {
                for (int64_t i = start; i < end; i += 2) {
                    int8_t low, high;
                    unpack_int4_byte(p_packed[i / 2], low, high, false);
                    p_out[i] = (static_cast<float>(low) - zp) * s;
                    if (i + 1 < end) {
                        p_out[i + 1] = (static_cast<float>(high) - zp) * s;
                    }
                }
                break;
            }
            case QuantType::FP8_E4M3: {
                for (int64_t i = start; i < end; ++i) {
                    p_out[i] = fp8_e4m3_to_float(p_packed[i]) * s;
                }
                break;
            }
            case QuantType::FP8_E5M2: {
                for (int64_t i = start; i < end; ++i) {
                    p_out[i] = fp8_e5m2_to_float(p_packed[i]) * s;
                }
                break;
            }
        }
    }

    if (device().is_cuda()) {
        return output.to(device());
    }
    return output;
}

QuantizedTensor QuantizedTensor::to(Device dev) const {
    if (dev == device()) return *this;
    QuantizedTensor result;
    result.type_ = type_;
    result.group_size_ = group_size_;
    result.original_shape_ = original_shape_;
    result.has_zero_points_ = has_zero_points_;
    result.packed_data_ = packed_data_.to(dev);
    result.scales_ = scales_.to(dev);
    if (has_zero_points_) {
        result.zero_points_ = zero_points_.to(dev);
    }
    return result;
}

// Binary Adler-32 implementation
static uint32_t compute_adler32(const uint8_t* data, size_t len) {
    const uint32_t MOD_ADLER = 65521;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    return (b << 16) | a;
}

void QuantizedTensor::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        AIC_THROW(IOError, "QuantizedTensor::save failed to open file: ", filepath);
    }

    // Header: Magic "AICQ", version 1
    const char magic[4] = {'A', 'I', 'C', 'Q'};
    out.write(magic, 4);
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint8_t q_type = static_cast<uint8_t>(type_);
    out.write(reinterpret_cast<const char*>(&q_type), sizeof(q_type));
    out.write(reinterpret_cast<const char*>(&group_size_), sizeof(group_size_));

    uint32_t ndim = static_cast<uint32_t>(original_shape_.rank());
    out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int d = 0; d < original_shape_.rank(); ++d) {
        int64_t dim_val = original_shape_[d];
        out.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
    }

    uint8_t has_zp = has_zero_points_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&has_zp), sizeof(has_zp));

    // Ensure CPU buffers for writing
    Tensor cpu_packed = packed_data_.is_cuda() ? packed_data_.to(Device::cpu()) : packed_data_;
    Tensor cpu_scales = scales_.is_cuda() ? scales_.to(Device::cpu()) : scales_;

    uint64_t packed_bytes = static_cast<uint64_t>(cpu_packed.numel() * dtype_size(cpu_packed.dtype()));
    uint64_t scales_bytes = static_cast<uint64_t>(cpu_scales.numel() * dtype_size(cpu_scales.dtype()));
    out.write(reinterpret_cast<const char*>(&packed_bytes), sizeof(packed_bytes));
    out.write(reinterpret_cast<const char*>(&scales_bytes), sizeof(scales_bytes));

    out.write(reinterpret_cast<const char*>(cpu_packed.data_ptr<uint8_t>()), packed_bytes);
    out.write(reinterpret_cast<const char*>(cpu_scales.data_ptr<float>()), scales_bytes);

    if (has_zero_points_) {
        Tensor cpu_zp = zero_points_.is_cuda() ? zero_points_.to(Device::cpu()) : zero_points_;
        uint64_t zp_bytes = static_cast<uint64_t>(cpu_zp.numel() * dtype_size(cpu_zp.dtype()));
        out.write(reinterpret_cast<const char*>(&zp_bytes), sizeof(zp_bytes));
        out.write(reinterpret_cast<const char*>(cpu_zp.data_ptr<float>()), zp_bytes);
    }

    // Compute and append Adler32
    uint32_t checksum = compute_adler32(cpu_packed.data_ptr<uint8_t>(), static_cast<size_t>(packed_bytes));
    out.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
}

QuantizedTensor QuantizedTensor::load(const std::string& filepath, Device device) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        AIC_THROW(IOError, "QuantizedTensor::load failed to open file: ", filepath);
    }

    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, "AICQ", 4) != 0) {
        AIC_THROW(FormatError, "QuantizedTensor::load invalid magic header");
    }

    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        AIC_THROW(FormatError, "QuantizedTensor::load unsupported version: ", version);
    }

    uint8_t q_type;
    in.read(reinterpret_cast<char*>(&q_type), sizeof(q_type));
    int32_t group_size;
    in.read(reinterpret_cast<char*>(&group_size), sizeof(group_size));

    uint32_t ndim;
    in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<int64_t> dims(ndim);
    for (uint32_t d = 0; d < ndim; ++d) {
        in.read(reinterpret_cast<char*>(&dims[d]), sizeof(dims[d]));
    }

    uint8_t has_zp;
    in.read(reinterpret_cast<char*>(&has_zp), sizeof(has_zp));

    uint64_t packed_bytes, scales_bytes;
    in.read(reinterpret_cast<char*>(&packed_bytes), sizeof(packed_bytes));
    in.read(reinterpret_cast<char*>(&scales_bytes), sizeof(scales_bytes));

    Tensor cpu_packed = Tensor::zeros(Shape({static_cast<int64_t>(packed_bytes)}), DType::UInt8, Device::cpu());
    in.read(reinterpret_cast<char*>(cpu_packed.data_ptr<uint8_t>()), packed_bytes);

    int64_t num_scales = static_cast<int64_t>(scales_bytes / sizeof(float));
    Tensor cpu_scales = Tensor::zeros(Shape({num_scales}), DType::Float32, Device::cpu());
    in.read(reinterpret_cast<char*>(cpu_scales.data_ptr<float>()), scales_bytes);

    Tensor cpu_zp;
    if (has_zp) {
        uint64_t zp_bytes;
        in.read(reinterpret_cast<char*>(&zp_bytes), sizeof(zp_bytes));
        int64_t num_zp = static_cast<int64_t>(zp_bytes / sizeof(float));
        cpu_zp = Tensor::zeros(Shape({num_zp}), DType::Float32, Device::cpu());
        in.read(reinterpret_cast<char*>(cpu_zp.data_ptr<float>()), zp_bytes);
    }

    uint32_t file_checksum;
    in.read(reinterpret_cast<char*>(&file_checksum), sizeof(file_checksum));

    uint32_t computed_checksum = compute_adler32(cpu_packed.data_ptr<uint8_t>(), static_cast<size_t>(packed_bytes));
    if (file_checksum != computed_checksum) {
        AIC_THROW(FormatError, "QuantizedTensor::load checksum mismatch! File corrupted.");
    }

    QuantizedTensor result;
    result.type_ = static_cast<QuantType>(q_type);
    result.group_size_ = group_size;
    result.original_shape_ = Shape(dims);
    result.has_zero_points_ = (has_zp != 0);

    if (device.is_cuda()) {
        result.packed_data_ = cpu_packed.to(device);
        result.scales_ = cpu_scales.to(device);
        if (result.has_zero_points_) {
            result.zero_points_ = cpu_zp.to(device);
        }
    } else {
        result.packed_data_ = std::move(cpu_packed);
        result.scales_ = std::move(cpu_scales);
        if (result.has_zero_points_) {
            result.zero_points_ = std::move(cpu_zp);
        }
    }

    return result;
}

// ----------------------------------------------------------------------------
// QuantizedLinear Implementation
// ----------------------------------------------------------------------------
QuantizedLinear::QuantizedLinear(int64_t in_features, int64_t out_features,
                                 const QuantConfig& config,
                                 bool has_bias, Device device)
    : in_features_(in_features), out_features_(out_features),
      has_bias_(has_bias), config_(config) {
    
    // Create random weight and quantize it
    float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
    Tensor w = Tensor::uniform(Shape({out_features_, in_features_}), -bound, bound, DType::Float32, device);
    weight_q_ = QuantizedTensor::quantize(w, config_.type, config_.group_size);

    if (has_bias_) {
        Tensor b = Tensor::zeros(Shape({out_features_}), DType::Float32, device);
        bias_ = register_parameter("bias", std::move(b));
    }
}

std::shared_ptr<QuantizedLinear> QuantizedLinear::from_linear(const Linear& linear, const QuantConfig& config) {
    auto qlinear = std::make_shared<QuantizedLinear>(
        linear.in_features(), linear.out_features(), config, linear.has_bias(), linear.weight()->device()
    );
    
    // Quantize existing weight
    qlinear->weight_q_ = QuantizedTensor::quantize(*linear.weight(), config.type, config.group_size);
    if (linear.has_bias() && linear.bias()) {
        qlinear->bias_ = std::make_shared<Tensor>(*linear.bias());
    }
    return qlinear;
}

Tensor QuantizedLinear::forward(const Tensor& x) {
    // x: [..., in_features]
    // Dequantize weight on-the-fly and perform matmul
    Tensor w_dequant = weight_q_.dequantize();
    Tensor w_t = w_dequant.transpose(0, 1);
    Tensor y = matmul(x, w_t);
    if (has_bias_ && bias_) {
        y = y + *bias_;
    }
    return y;
}

// ----------------------------------------------------------------------------
// Cosine Similarity Utility
// ----------------------------------------------------------------------------
float compute_cosine_similarity(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) {
        AIC_THROW(ShapeMismatch, "compute_cosine_similarity requires identical numel");
    }

    Tensor cpu_a = a.is_cuda() ? a.to(Device::cpu()) : a;
    Tensor cpu_b = b.is_cuda() ? b.to(Device::cpu()) : b;
    if (!cpu_a.is_contiguous()) cpu_a = cpu_a.contiguous();
    if (!cpu_b.is_contiguous()) cpu_b = cpu_b.contiguous();

    const float* p_a = cpu_a.data_ptr<float>();
    const float* p_b = cpu_b.data_ptr<float>();
    int64_t n = cpu_a.numel();

    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        double va = static_cast<double>(p_a[i]);
        double vb = static_cast<double>(p_b[i]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-12) return 1.0f;
    return static_cast<float>(dot / denom);
}

} // namespace aic::nn
