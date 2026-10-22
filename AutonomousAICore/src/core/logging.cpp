// src/core/logging.cpp
#include "core/logging.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace aic {

const char* log_level_name(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Off:   return "OFF";
    }
    return "UNKNOWN";
}

bool parse_log_level(const std::string& text, LogLevel& out) noexcept {
    if (text == "trace" || text == "TRACE") { out = LogLevel::Trace; return true; }
    if (text == "debug" || text == "DEBUG") { out = LogLevel::Debug; return true; }
    if (text == "info"  || text == "INFO")  { out = LogLevel::Info;  return true; }
    if (text == "warn"  || text == "WARN")  { out = LogLevel::Warn;  return true; }
    if (text == "error" || text == "ERROR") { out = LogLevel::Error; return true; }
    if (text == "off"   || text == "OFF")   { out = LogLevel::Off;   return true; }
    return false;
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger() : level_(LogLevel::Info) {
    const char* env_lvl = std::getenv("AIC_LOG_LEVEL");
    if (env_lvl) {
        LogLevel lvl;
        if (parse_log_level(env_lvl, lvl)) {
            level_ = lvl;
        }
    }
}

void Logger::set_level(LogLevel level) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    level_ = level;
}

LogLevel Logger::level() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return level_;
}

bool Logger::enabled(LogLevel level) const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(level) >= static_cast<int>(level_) && level_ != LogLevel::Off;
}

void Logger::set_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);
    file_.close();
    file_.open(path, std::ios::out | std::ios::app);
}

void Logger::log(LogLevel level, const std::string& message) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mu_);
        if (static_cast<int>(level) < static_cast<int>(level_) || level_ == LogLevel::Off) {
            return;
        }

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::tm tm_buf;
#if defined(_WIN32)
        localtime_s(&tm_buf, &in_time_t);
#else
        localtime_r(&in_time_t, &tm_buf);
#endif

        std::ostringstream os;
        os << "[" << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
           << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ["
           << log_level_name(level) << "] "
           << message << "\n";

        std::string s = os.str();
        if (level >= LogLevel::Warn) {
            std::cerr << s;
        } else {
            std::cout << s;
        }
        if (file_.is_open()) {
            file_ << s;
            file_.flush();
        }
    } catch (...) {
        // Logging must never propagate exceptions.
    }
}

}  // namespace aic
