// src/core/device.cpp
#include "core/device.hpp"

#include <charconv>
#include <sstream>

#include "core/error.hpp"

namespace aic {

std::string Device::str() const {
    if (is_cpu()) {
        return "cpu";
    }
    std::ostringstream os;
    os << "cuda:" << index;
    return os.str();
}

Device Device::parse(std::string_view text) {
    if (text.empty() || text == "cpu" || text == "CPU") {
        return Device::cpu();
    }
    if (text == "cuda" || text == "CUDA" || text == "gpu" || text == "GPU") {
        return Device::cuda(0);
    }
    if (text.starts_with("cuda:") || text.starts_with("CUDA:") ||
        text.starts_with("gpu:")  || text.starts_with("GPU:")) {
        auto colon = text.find(':');
        std::string_view num_str = text.substr(colon + 1);
        int idx = 0;
        auto res = std::from_chars(num_str.data(), num_str.data() + num_str.size(), idx);
        if (res.ec == std::errc() && idx >= 0) {
            return Device::cuda(idx);
        }
    }
    AIC_THROW(InvalidArgument, "Failed to parse device string: '", text, "'. Expected 'cpu', 'cuda', or 'cuda:N'");
}

}  // namespace aic
