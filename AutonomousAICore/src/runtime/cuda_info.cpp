// src/runtime/cuda_info.cpp
#include "runtime/cuda_info.hpp"

#include <iomanip>
#include <sstream>

#include "core/error.hpp"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace aic {

bool is_cuda_available() noexcept {
#if AIC_WITH_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

int get_cuda_device_count() noexcept {
#if AIC_WITH_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
#else
    return 0;
#endif
}

GpuProperties get_gpu_properties(int device_index) {
#if !AIC_WITH_CUDA
    (void)device_index;
    AIC_THROW(BackendUnavailable, "CUDA runtime is not enabled in this build.");
#else
    int count = get_cuda_device_count();
    AIC_CHECK(device_index >= 0 && device_index < count, InvalidArgument,
              "Invalid CUDA device index ", device_index, ", count is ", count);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        AIC_THROW(CudaError, "cudaGetDeviceProperties failed for device ", device_index, ": ",
                  cudaGetErrorString(err));
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaSetDevice(device_index);
    cudaMemGetInfo(&free_bytes, &total_bytes);

    GpuProperties p;
    p.device_index = device_index;
    p.name = prop.name;
    p.total_vram_bytes = total_bytes;  // Live hardware query, not assumed config value!
    p.free_vram_bytes = free_bytes;
    p.compute_major = prop.major;
    p.compute_minor = prop.minor;
    p.multiprocessor_count = prop.multiProcessorCount;
    p.warp_size = prop.warpSize;

    // Blackwell architectures have compute capability sm_100, sm_120, etc.
    p.is_blackwell = (prop.major >= 12 || (prop.major == 10 && prop.minor >= 0));
    p.has_tensor_cores = (prop.major >= 7);
    p.has_fp16 = (prop.major >= 6);
    p.has_bf16 = (prop.major >= 8);
    p.has_fp8 = (prop.major >= 8 && prop.minor >= 9) || (prop.major >= 9);

    return p;
#endif
}

std::string GpuProperties::summary() const {
    std::ostringstream os;
    auto to_gb = [](size_t bytes) -> double {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    };

    os << "GPU [" << device_index << "]: " << name << "\n"
       << "  Compute Capability: " << compute_major << "." << compute_minor << "\n"
       << "  Architecture: " << (is_blackwell ? "NVIDIA Blackwell" : "NVIDIA Pre-Blackwell") << "\n"
       << std::fixed << std::setprecision(2)
       << "  True Total VRAM: " << to_gb(total_vram_bytes) << " GB (Free: " << to_gb(free_vram_bytes) << " GB)\n"
       << "  SMs: " << multiprocessor_count << ", Warp Size: " << warp_size << "\n"
       << "  Tensor Cores: " << (has_tensor_cores ? "Supported" : "No") << "\n"
       << "  FP16: " << (has_fp16 ? "Yes" : "No")
       << ", BF16: " << (has_bf16 ? "Yes" : "No")
       << ", FP8: " << (has_fp8 ? "Yes" : "No");
    return os.str();
}

}  // namespace aic
