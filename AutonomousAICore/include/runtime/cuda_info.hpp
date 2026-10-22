// include/runtime/cuda_info.hpp
// Live GPU runtime discovery, VRAM inspection and architecture capability queries.
#pragma once

#include <string>
#include <vector>

namespace aic {

struct GpuProperties {
    int device_index = 0;
    std::string name;
    size_t total_vram_bytes = 0;
    size_t free_vram_bytes = 0;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    int warp_size = 32;

    bool is_blackwell = false;
    bool has_tensor_cores = false;
    bool has_fp16 = false;
    bool has_bf16 = false;
    bool has_fp8 = false;

    std::string summary() const;
};

bool is_cuda_available() noexcept;
int get_cuda_device_count() noexcept;
GpuProperties get_gpu_properties(int device_index = 0);

}  // namespace aic
