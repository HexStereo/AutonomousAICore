// include/core/macros.hpp
// Portable macros shared by host C++ and CUDA translation units.
#pragma once

#if defined(__CUDACC__)
#define AIC_HD __host__ __device__
#define AIC_DEVICE __device__
#define AIC_INLINE __forceinline__
#else
#define AIC_HD
#define AIC_DEVICE
#define AIC_INLINE inline
#endif

#ifndef AIC_WITH_CUDA
#define AIC_WITH_CUDA 0
#endif

#ifndef AIC_WITH_CUBLAS
#define AIC_WITH_CUBLAS 0
#endif

#if defined(_WIN32)
#define AIC_PLATFORM_WINDOWS 1
#else
#define AIC_PLATFORM_WINDOWS 0
#endif
