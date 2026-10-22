@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.28000.0
set "CUDA_PATH=C:\Users\ardab\cuda"
set "PATH=C:\Users\ardab\cuda\bin;%PATH%"
if not exist build_ninja (
    cmake -B build_ninja -S AutonomousAICore -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Users/ardab/cuda/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120 -DAIC_ENABLE_CUDA=ON
)
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
cmake --build build_ninja --config Release
