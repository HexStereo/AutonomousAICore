// include/core/logging.hpp
// Minimal thread-safe logger. Level can be set programmatically or via env AIC_LOG_LEVEL
// (trace|debug|info|warn|error|off). Optional log file mirror.
#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include "core/error.hpp"

namespace aic {

enum class LogLevel : int { Trace = 0, Debug, Info, Warn, Error, Off };

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level) noexcept;
    LogLevel level() const noexcept;
    bool enabled(LogLevel level) const noexcept;
    void set_file(const std::string& path);
    void log(LogLevel level, const std::string& message) noexcept;

private:
    Logger();
    mutable std::mutex mu_;
    LogLevel level_;
    std::ofstream file_;
};

const char* log_level_name(LogLevel level) noexcept;
bool parse_log_level(const std::string& text, LogLevel& out) noexcept;

}  // namespace aic

#define AIC_LOG(level, ...)                                                   \
    do {                                                                      \
        auto& aic_logger_ = ::aic::Logger::instance();                        \
        if (aic_logger_.enabled(level)) {                                     \
            aic_logger_.log(level, ::aic::detail::str_cat(__VA_ARGS__));      \
        }                                                                     \
    } while (false)

#define AIC_LOG_DEBUG(...) AIC_LOG(::aic::LogLevel::Debug, __VA_ARGS__)
#define AIC_LOG_INFO(...) AIC_LOG(::aic::LogLevel::Info, __VA_ARGS__)
#define AIC_LOG_WARN(...) AIC_LOG(::aic::LogLevel::Warn, __VA_ARGS__)
#define AIC_LOG_ERROR(...) AIC_LOG(::aic::LogLevel::Error, __VA_ARGS__)
