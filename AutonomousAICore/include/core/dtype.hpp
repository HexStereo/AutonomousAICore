// include/core/dtype.hpp
// Data type policy:
//   * Float32  : reference type; all ops supported on CPU and CUDA, compute in FP32.
//   * Float16 / BFloat16 : storage types. Elementwise/copy/cast/matmul/softmax/norm kernels load them,
//                compute in FP32 and store back. Tensor-core GEMM path on CUDA.
//   * Int32 / Int64 : indices (token ids, targets). UInt8 : masks. UInt16 : compact token storage.
//   * FP8 is NOT implemented as a tensor dtype; hardware capability is only reported by the runtime.
// This header is safe to include from CUDA translation units.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>

namespace aic {

enum class DType : std::uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int64 = 4,
    UInt8 = 5,
    UInt16 = 6
};

constexpr int kNumDTypes = 7;

constexpr std::size_t dtype_size(DType d) noexcept {
    switch (d) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::UInt8: return 1;
        case DType::UInt16: return 2;
    }
    return 0;
}

constexpr bool is_floating_point(DType d) noexcept {
    return d == DType::Float32 || d == DType::Float16 || d == DType::BFloat16;
}

constexpr bool is_integral(DType d) noexcept { return !is_floating_point(d); }

const char* dtype_name(DType d) noexcept;
bool parse_dtype(const char* text, DType& out) noexcept;

inline std::ostream& operator<<(std::ostream& os, DType d) {
    return os << dtype_name(d);
}

// Storage-compatible 16-bit float types for host code (bit patterns only).
struct Half {
    std::uint16_t bits = 0;
};
struct BFloat16 {
    std::uint16_t bits = 0;
};

// IEEE-754 binary16 conversion, round-to-nearest-even, handles subnormals/inf/nan.
inline std::uint16_t float_to_half(float f) noexcept {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t mant = x & 0x007FFFFFu;
    const std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu);
    if (exp == 0xFF) {
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? (0x200u | (mant >> 13)) : 0u));
    }
    const std::int32_t e = exp - 127 + 15;
    if (e >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x00800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - e);
        std::uint32_t half_mant = mant >> shift;
        const std::uint32_t rem = mant & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) ++half_mant;
        return static_cast<std::uint16_t>(sign | half_mant);
    }
    std::uint32_t h = sign | (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
    const std::uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(h);
}

inline float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::uint16_t float_to_bf16(float f) noexcept {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x007FFFFFu) != 0) {
        return static_cast<std::uint16_t>(((x >> 16) & 0x8000u) | 0x7FC0u);
    }
    const std::uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>((x + rounding) >> 16);
}

inline float bf16_to_float(std::uint16_t b) noexcept {
    const std::uint32_t x = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

template <typename T>
struct DTypeOf;
template <>
struct DTypeOf<float> {
    static constexpr DType value = DType::Float32;
};
template <>
struct DTypeOf<Half> {
    static constexpr DType value = DType::Float16;
};
template <>
struct DTypeOf<BFloat16> {
    static constexpr DType value = DType::BFloat16;
};
template <>
struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::Int32;
};
template <>
struct DTypeOf<std::int64_t> {
    static constexpr DType value = DType::Int64;
};
template <>
struct DTypeOf<std::uint8_t> {
    static constexpr DType value = DType::UInt8;
};
template <>
struct DTypeOf<std::uint16_t> {
    static constexpr DType value = DType::UInt16;
};

}  // namespace aic
