// include/nn/quantization.hpp
// Model Quantization & Compressed Inference Engine (Aşama 21)
// Supports INT4 (W4A16), INT8 (W8A16), and FP8 (E4M3 / E5M2) with group-wise scaling.
#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/error.hpp"
#include "core/shape.hpp"
#include "nn/module.hpp"
#include "nn/linear.hpp"
#include "tensor/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aic::nn {

enum class QuantType : uint8_t {
    INT8_SYMMETRIC = 0,    // 8-bit signed integer [-127, 127]
    INT8_ASYMMETRIC = 1,   // 8-bit unsigned integer [0, 255] with zero-point
    INT4_SYMMETRIC = 2,    // 4-bit signed integer [-7, 7], packed 2 per byte
    INT4_ASYMMETRIC = 3,   // 4-bit unsigned integer [0, 15], packed 2 per byte
    FP8_E4M3 = 4,          // OCP FP8 (1 sign, 4 exponent, 3 mantissa)
    FP8_E5M2 = 5           // IEEE-like FP8 (1 sign, 5 exponent, 2 mantissa)
};

const char* quant_type_to_string(QuantType type) noexcept;
bool parse_quant_type(const std::string& str, QuantType& out) noexcept;

// Bit-level helpers for FP8 formats
uint8_t float_to_fp8_e4m3(float val) noexcept;
float fp8_e4m3_to_float(uint8_t byte) noexcept;

uint8_t float_to_fp8_e5m2(float val) noexcept;
float fp8_e5m2_to_float(uint8_t byte) noexcept;

// Packed 4-bit helpers (low nibble: index 0, high nibble: index 1)
inline uint8_t pack_int4_byte(int8_t low, int8_t high) noexcept {
    return static_cast<uint8_t>((static_cast<uint8_t>(high & 0x0F) << 4) | 
                                (static_cast<uint8_t>(low & 0x0F)));
}

inline void unpack_int4_byte(uint8_t packed, int8_t& low, int8_t& high, bool is_signed = true) noexcept {
    uint8_t u_low = packed & 0x0F;
    uint8_t u_high = (packed >> 4) & 0x0F;
    if (is_signed) {
        low = (u_low >= 8) ? static_cast<int8_t>(u_low - 16) : static_cast<int8_t>(u_low);
        high = (u_high >= 8) ? static_cast<int8_t>(u_high - 16) : static_cast<int8_t>(u_high);
    } else {
        low = static_cast<int8_t>(u_low);
        high = static_cast<int8_t>(u_high);
    }
}

// Configuration for quantization
struct QuantConfig {
    QuantType type = QuantType::INT4_SYMMETRIC;
    int32_t group_size = 32; // Group-wise scaling size (e.g. 32, 64, 128); 0 = per-channel
};

// QuantizedTensor: Encapsulates packed weights, group scale factors, and optional zero-points
class QuantizedTensor {
public:
    QuantizedTensor() = default;

    // Factory method to quantize an unquantized FP32 tensor
    static QuantizedTensor quantize(const Tensor& input, QuantType type, int32_t group_size = 32);

    // Dequantize back to standard FP32 tensor
    Tensor dequantize() const;

    // Properties
    QuantType quant_type() const noexcept { return type_; }
    int32_t group_size() const noexcept { return group_size_; }
    Shape original_shape() const noexcept { return original_shape_; }
    int64_t numel() const noexcept { return original_shape_.numel(); }
    Device device() const noexcept { return packed_data_.device(); }

    const Tensor& packed_data() const noexcept { return packed_data_; }
    const Tensor& scales() const noexcept { return scales_; }
    const Tensor& zero_points() const noexcept { return zero_points_; }
    bool has_zero_points() const noexcept { return has_zero_points_; }

    // Memory consumption analysis
    size_t original_bytes() const noexcept {
        return static_cast<size_t>(original_shape_.numel()) * sizeof(float);
    }

    size_t quantized_bytes() const noexcept {
        size_t bytes = static_cast<size_t>(packed_data_.numel()) * dtype_size(packed_data_.dtype());
        bytes += static_cast<size_t>(scales_.numel()) * dtype_size(scales_.dtype());
        if (has_zero_points_) {
            bytes += static_cast<size_t>(zero_points_.numel()) * dtype_size(zero_points_.dtype());
        }
        return bytes;
    }

    double compression_ratio() const noexcept {
        size_t q_bytes = quantized_bytes();
        if (q_bytes == 0) return 1.0;
        return static_cast<double>(original_bytes()) / static_cast<double>(q_bytes);
    }

    // Move to target device
    QuantizedTensor to(Device dev) const;

    // Serialization
    void save(const std::string& filepath) const;
    static QuantizedTensor load(const std::string& filepath, Device device = Device::cpu());

private:
    QuantType type_ = QuantType::INT4_SYMMETRIC;
    int32_t group_size_ = 32;
    Shape original_shape_;
    Tensor packed_data_;       // Byte storage (UInt8)
    Tensor scales_;            // Scale factors (Float32)
    Tensor zero_points_;       // Zero points (Float32, if asymmetric)
    bool has_zero_points_ = false;
};

// QuantizedLinear: Compressed Linear layer for inference
class QuantizedLinear : public Module {
public:
    QuantizedLinear(int64_t in_features, int64_t out_features, 
                    const QuantConfig& config = QuantConfig{},
                    bool has_bias = true, Device device = Device::cpu());

    // Construct directly from an existing unquantized Linear layer
    static std::shared_ptr<QuantizedLinear> from_linear(const Linear& linear, const QuantConfig& config);

    virtual ~QuantizedLinear() = default;

    virtual Tensor forward(const Tensor& x) override;

    Tensor operator()(const Tensor& x) { return forward(x); }

    int64_t in_features() const noexcept { return in_features_; }
    int64_t out_features() const noexcept { return out_features_; }
    bool has_bias() const noexcept { return has_bias_; }
    const QuantizedTensor& quantized_weight() const noexcept { return weight_q_; }
    std::shared_ptr<Tensor> bias() const noexcept { return bias_; }

    void set_quantized_weight(QuantizedTensor w_q) { weight_q_ = std::move(w_q); }
    void set_bias(std::shared_ptr<Tensor> b) { bias_ = std::move(b); has_bias_ = (bias_ != nullptr); }

private:
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    bool has_bias_ = false;
    QuantConfig config_;
    QuantizedTensor weight_q_;
    std::shared_ptr<Tensor> bias_;
};

// High-level utility to compute Cosine Similarity between two tensors
float compute_cosine_similarity(const Tensor& a, const Tensor& b);

} // namespace aic::nn
