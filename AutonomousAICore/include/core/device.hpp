// include/core/device.hpp
// Device abstraction: CPU or CUDA:<index>. Value type, cheap to copy.
#pragma once

#include <string>
#include <string_view>

namespace aic {

enum class DeviceType : unsigned char { CPU = 0, CUDA = 1 };

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;

    constexpr Device() = default;
    constexpr Device(DeviceType t, int i = 0) : type(t), index(t == DeviceType::CPU ? 0 : i) {}

    static constexpr Device cpu() { return Device(DeviceType::CPU, 0); }
    static constexpr Device cuda(int i = 0) { return Device(DeviceType::CUDA, i); }

    constexpr bool is_cpu() const { return type == DeviceType::CPU; }
    constexpr bool is_cuda() const { return type == DeviceType::CUDA; }

    friend constexpr bool operator==(const Device& a, const Device& b) {
        return a.type == b.type && a.index == b.index;
    }
    friend constexpr bool operator!=(const Device& a, const Device& b) { return !(a == b); }

    std::string str() const;
    // Accepts "cpu", "cuda", "cuda:N", "gpu", "gpu:N". Throws InvalidArgument otherwise.
    static Device parse(std::string_view text);

    friend std::ostream& operator<<(std::ostream& os, const Device& d) {
        return os << d.str();
    }
};

}  // namespace aic
