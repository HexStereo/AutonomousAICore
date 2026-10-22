# Autonomous AI Core — Build & Verification Guide

## Requirements
- **C++ Compiler**: Modern C++20 compliant compiler (MSVC 19.51+ / VS 2026 or GCC 13+ / Clang 17+).
- **Build System**: CMake 3.26+ and Ninja (or Visual Studio Generator).
- **GPU Toolchain**: NVIDIA CUDA Toolkit 13.x (NVCC 13.4+) for GPU acceleration.

## Quick Build Commands (Windows)

### 1. Build with CUDA Acceleration (Recommended)
Run the automated build script:
```cmd
build_cuda.bat
```
Or manually configure and compile with Ninja:
```cmd
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.28000.0
set "CUDA_PATH=C:\Users\ardab\cuda"
set "PATH=C:\Users\ardab\cuda\bin;%PATH%"
cmake -B build_ninja -S AutonomousAICore -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Users/ardab/cuda/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120 -DAIC_ENABLE_CUDA=ON
cmake --build build_ninja --config Release
```

### 2. CPU-Only Build (No CUDA Required)
Using Visual Studio Generator:
```cmd
cmake -B build -S AutonomousAICore -G "Visual Studio 18 2026" -A x64 -DAIC_ENABLE_CUDA=OFF
cmake --build build --config Release
```

## Running the Automated Test Suite
After building, run the test runner executable:
```cmd
.\build_ninja\bin\aic_tests.exe
```
Or via CTest:
```cmd
ctest --test-dir build_ninja --output-on-failure
```
