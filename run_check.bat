@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.28000.0
ninja -C build_ninja src\CMakeFiles\aic_core.dir\ops\cpu_gemm.cpp.obj
