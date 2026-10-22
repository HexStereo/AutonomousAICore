// src/core/dtype.cpp
#include "core/dtype.hpp"
#include <string_view>

namespace aic {

const char* dtype_name(DType d) noexcept {
    switch (d) {
        case DType::Float32:  return "Float32";
        case DType::Float16:  return "Float16";
        case DType::BFloat16: return "BFloat16";
        case DType::Int32:    return "Int32";
        case DType::Int64:    return "Int64";
        case DType::UInt8:    return "UInt8";
        case DType::UInt16:   return "UInt16";
    }
    return "UnknownDType";
}

bool parse_dtype(const char* text, DType& out) noexcept {
    if (!text) return false;
    std::string_view sv(text);
    if (sv == "float32" || sv == "Float32" || sv == "fp32" || sv == "float") {
        out = DType::Float32;
        return true;
    }
    if (sv == "float16" || sv == "Float16" || sv == "fp16" || sv == "half") {
        out = DType::Float16;
        return true;
    }
    if (sv == "bfloat16" || sv == "BFloat16" || sv == "bf16") {
        out = DType::BFloat16;
        return true;
    }
    if (sv == "int32" || sv == "Int32" || sv == "int") {
        out = DType::Int32;
        return true;
    }
    if (sv == "int64" || sv == "Int64" || sv == "long") {
        out = DType::Int64;
        return true;
    }
    if (sv == "uint8" || sv == "UInt8" || sv == "byte") {
        out = DType::UInt8;
        return true;
    }
    if (sv == "uint16" || sv == "UInt16") {
        out = DType::UInt16;
        return true;
    }
    return false;
}

}  // namespace aic
