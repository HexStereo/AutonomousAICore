// tests/unit/test_runtime.cpp
#include "runtime/cpu_info.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

using namespace aic;

AIC_TEST(RuntimeTest, CpuDetection) {
    const auto& cpu = get_cpu_features();
    ASSERT_TRUE(cpu.logical_cores > 0);
    ASSERT_TRUE(cpu.total_ram_bytes > 0);
    std::cout << "\n[CPU Diagnostic]\n" << cpu.summary() << "\n";
}

AIC_TEST(RuntimeTest, CudaDetection) {
    bool available = is_cuda_available();
    int dev_count = get_cuda_device_count();
    std::cout << "\n[CUDA Diagnostic]\nCUDA available: " << (available ? "YES" : "NO")
              << " | Device count: " << dev_count << "\n";
    if (available && dev_count > 0) {
        auto prop = get_gpu_properties(0);
        std::cout << prop.summary() << "\n";
        ASSERT_TRUE(prop.total_vram_bytes > 0);
    }
}
