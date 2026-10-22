// src/runtime/cpu_info.cpp
#include "runtime/cpu_info.hpp"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#include <windows.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <unistd.h>
#endif

namespace aic {

namespace {

void query_cpuid(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

CpuFeatures detect_features() {
    CpuFeatures f;
    f.logical_cores = static_cast<int>(std::thread::hardware_concurrency());

#if defined(_WIN32)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        f.total_ram_bytes = memInfo.ullTotalPhys;
    }
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        f.total_ram_bytes = static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
#endif

    int regs[4];
    query_cpuid(0, 0, regs);
    int max_leaf = regs[0];

    // CPU Brand string
    char brand[49] = {0};
    int ext_regs[4];
    query_cpuid(0x80000000, 0, ext_regs);
    unsigned int max_ext = static_cast<unsigned int>(ext_regs[0]);
    if (max_ext >= 0x80000004) {
        query_cpuid(0x80000002, 0, reinterpret_cast<int*>(brand));
        query_cpuid(0x80000003, 0, reinterpret_cast<int*>(brand + 16));
        query_cpuid(0x80000004, 0, reinterpret_cast<int*>(brand + 32));
        f.brand_string = brand;
    }

    if (max_leaf >= 1) {
        query_cpuid(1, 0, regs);
        int ecx = regs[2];
        (void)regs[3];  // edx feature bits
        f.sse42 = (ecx & (1 << 20)) != 0;
        f.fma   = (ecx & (1 << 12)) != 0;
        f.avx   = (ecx & (1 << 28)) != 0;
    }

    if (max_leaf >= 7) {
        query_cpuid(7, 0, regs);
        int ebx = regs[1];
        int edx = regs[3];
        f.avx2     = (ebx & (1 << 5)) != 0;
        f.avx512f  = (ebx & (1 << 16)) != 0;
        f.avx512dq = (ebx & (1 << 17)) != 0;
        f.avx512bw = (ebx & (1 << 30)) != 0;
        f.avx512vl = (ebx & (1 << 31)) != 0;
        f.amx_bf16 = (edx & (1 << 22)) != 0;
        f.amx_tile = (edx & (1 << 24)) != 0;
    }

    return f;
}

}  // namespace

const CpuFeatures& get_cpu_features() {
    static CpuFeatures features = detect_features();
    return features;
}

std::string CpuFeatures::summary() const {
    std::ostringstream os;
    os << "CPU: " << (brand_string.empty() ? "Generic x86_64" : brand_string)
       << " (" << logical_cores << " threads, "
       << std::fixed << std::setprecision(1) << (static_cast<double>(total_ram_bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB RAM)\n"
       << "SIMD: AVX=" << (avx ? "1" : "0")
       << " AVX2=" << (avx2 ? "1" : "0")
       << " FMA=" << (fma ? "1" : "0")
       << " AVX512F=" << (avx512f ? "1" : "0")
       << " AVX512BW=" << (avx512bw ? "1" : "0")
       << " AMX=" << (amx_tile ? "1" : "0");
    return os.str();
}

}  // namespace aic
