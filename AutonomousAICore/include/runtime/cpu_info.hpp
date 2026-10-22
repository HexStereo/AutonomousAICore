// include/runtime/cpu_info.hpp
// CPU feature and SIMD discovery using CPUID.
#pragma once

#include <string>

namespace aic {

struct CpuFeatures {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512dq = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool amx_tile = false;
    bool amx_bf16 = false;

    int logical_cores = 0;
    size_t total_ram_bytes = 0;
    std::string brand_string;

    std::string summary() const;
};

const CpuFeatures& get_cpu_features();

}  // namespace aic
