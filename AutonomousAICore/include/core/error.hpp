// include/core/error.hpp
// Single, consistent error policy for the whole project: exceptions derived from aic::Error.
// Every error carries an ErrorCode, a message and the throw site (file:line).
// Destructors and noexcept paths never throw; they log and degrade safely instead.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace aic {

enum class ErrorCode : int {
    InvalidArgument = 1,
    OutOfRange,
    ShapeMismatch,
    DTypeMismatch,
    DeviceMismatch,
    OutOfMemory,
    CudaError,
    NotImplemented,
    BackendUnavailable,
    IOError,
    FormatError,
    Overflow,
    InvalidState,
    RuntimeError,
    Internal
};

const char* error_code_name(ErrorCode code) noexcept;

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message, const char* file, int line);

    ErrorCode code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }
    const std::string& file() const noexcept { return file_; }
    int line() const noexcept { return line_; }

private:
    static std::string format(ErrorCode code, const std::string& message, const char* file, int line);

    ErrorCode code_;
    std::string message_;
    std::string file_;
    int line_;
};

namespace detail {

template <typename... Args>
std::string str_cat(const Args&... args) {
    std::ostringstream os;
    (os << ... << args);
    return os.str();
}

[[noreturn]] void throw_error(ErrorCode code, const std::string& message, const char* file, int line);

}  // namespace detail
}  // namespace aic

#define AIC_THROW(code, ...) \
    ::aic::detail::throw_error(::aic::ErrorCode::code, ::aic::detail::str_cat(__VA_ARGS__), __FILE__, __LINE__)

#define AIC_CHECK(cond, code, ...)                                                                   \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            ::aic::detail::throw_error(::aic::ErrorCode::code,                                       \
                                       ::aic::detail::str_cat("check failed: (" #cond ") ", __VA_ARGS__), \
                                       __FILE__, __LINE__);                                          \
        }                                                                                            \
    } while (false)
