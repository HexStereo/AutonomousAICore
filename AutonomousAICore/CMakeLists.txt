# AutonomousAICore - root build script
# Builds a from-scratch tensor / autograd / NN / training stack with an optional CUDA backend.
cmake_minimum_required(VERSION 3.26)

project(AutonomousAICore
    VERSION 0.1.0
    DESCRIPTION "Autonomous Learning & Reasoning Core (from-scratch C++20/CUDA)"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(AIC_ENABLE_CUDA "Build the CUDA backend if a CUDA compiler is available" ON)
option(AIC_ENABLE_CUBLAS "Optional cuBLAS GEMM backend (benchmark comparison / kernel selection only)" OFF)
option(AIC_BUILD_TESTS "Build unit / numerical / integration / stress tests" ON)
option(AIC_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(AIC_BUILD_EXAMPLES "Build example programs" ON)
option(AIC_BUILD_TOOLS "Build command line tools" ON)
option(AIC_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(AIC_CUDA_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to nvcc" OFF)
option(AIC_ENABLE_CPU_AVX2_KERNELS "Compile AVX2/FMA CPU kernels (selected at runtime via CPUID)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(AicCompilerOptions)
include(AicCuda)

add_subdirectory(src)

if(AIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(AIC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(AIC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(AIC_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

message(STATUS "AutonomousAICore ${PROJECT_VERSION}")
message(STATUS "  CUDA backend      : ${AIC_WITH_CUDA}")
message(STATUS "  CUDA archs        : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  cuBLAS (optional) : ${AIC_WITH_CUBLAS}")
message(STATUS "  AVX2 CPU kernels  : ${AIC_ENABLE_CPU_AVX2_KERNELS}")
