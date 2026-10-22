// src/core/error.cpp
#include "core/error.hpp"

#include <iostream>

namespace aic {

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::InvalidArgument:   return "InvalidArgument";
        case ErrorCode::OutOfRange:        return "OutOfRange";
        case ErrorCode::ShapeMismatch:     return "ShapeMismatch";
        case ErrorCode::DTypeMismatch:     return "DTypeMismatch";
        case ErrorCode::DeviceMismatch:    return "DeviceMismatch";
        case ErrorCode::OutOfMemory:       return "OutOfMemory";
        case ErrorCode::CudaError:         return "CudaError";
        case ErrorCode::NotImplemented:    return "NotImplemented";
        case ErrorCode::BackendUnavailable:return "BackendUnavailable";
        case ErrorCode::IOError:           return "IOError";
        case ErrorCode::FormatError:       return "FormatError";
        case ErrorCode::Overflow:          return "Overflow";
        case ErrorCode::InvalidState:      return "InvalidState";
        case ErrorCode::RuntimeError:      return "RuntimeError";
        case ErrorCode::Internal:          return "Internal";
    }
    return "UnknownError";
}

Error::Error(ErrorCode code, std::string message, const char* file, int line)
    : std::runtime_error(format(code, message, file, line)),
      code_(code),
      message_(std::move(message)),
      file_(file ? file : ""),
      line_(line) {}

std::string Error::format(ErrorCode code, const std::string& message, const char* file, int line) {
    std::ostringstream os;
    os << "[" << error_code_name(code) << "] " << message;
    if (file && file[0] != '\0') {
        os << " (" << file << ":" << line << ")";
    }
    return os.str();
}

namespace detail {

[[noreturn]] void throw_error(ErrorCode code, const std::string& message, const char* file, int line) {
    throw Error(code, message, file, line);
}

}  // namespace detail
}  // namespace aic
