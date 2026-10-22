@echo off
title Autonomous AI Core (AIC) - Blackwell sm_120
color 0B
cd /d "%~dp0"

:MENU
cls
echo ======================================================================
echo    AUTONOMOUS AI CORE (AIC) - HIGH PERFORMANCE DEEP LEARNING CORE
echo    NVIDIA RTX 5090 Blackwell (sm_120) ^| Intel Core Ultra 9 285HX
echo ======================================================================
echo  [G] Launch Modern Desktop GUI Studio (Royal Purple / Pearl / Black)
echo  [1] Launch Production Async HTTP ^& SSE Streaming Server (Port 8080)
echo  [2] Run Hardware GEMM Benchmark (AVX2 vs Blackwell sm_120)
echo  [3] Run FlashAttention-2 vs Standard Attention Benchmark
echo  [4] Run Continuous Batching ^& Chunked Prefill Serving Benchmark
echo  [5] Run INT4 / INT8 / FP8 Quantization Benchmark
echo  [6] Execute Tree-of-Thoughts (MCTS) Reasoning Engine
echo  [7] Execute Autonomous Tool Calling ^& ReAct Sandbox Agent
echo  [8] Run Complete Verification Suite (169 / 169 Tests)
echo  [9] Run Interactive AIC Shell (Help ^& Custom Commands)
echo  [0] Exit
echo.
echo ======================================================================
set /p choice="Seciminizi yapin [G / 0-9]: "

if /i "%choice%"=="G" (
    start "" Launch_AIC_Studio.bat
    goto MENU
)

if "%choice%"=="1" (
    echo.
    echo [AIC] Starting Production Async Streaming Server on 127.0.0.1:8080...
    .\build_ninja\bin\aic.exe serve 8080 cuda
    pause
    goto MENU
)
if "%choice%"=="2" (
    echo.
    .\build_ninja\bin\aic.exe benchmark 1024
    pause
    goto MENU
)
if "%choice%"=="3" (
    echo.
    .\build_ninja\bin\aic.exe flash-bench 2 4 128 64 cuda
    pause
    goto MENU
)
if "%choice%"=="4" (
    echo.
    .\build_ninja\bin\aic.exe serve-bench 16 8 cuda
    pause
    goto MENU
)
if "%choice%"=="5" (
    echo.
    .\build_ninja\bin\aic.exe quant-bench 1024 2048 int4 32 cuda
    pause
    goto MENU
)
if "%choice%"=="6" (
    echo.
    .\build_ninja\bin\aic.exe reason "Suppose a = 10, b = 20. Find a + b:" example_ckpts\stable_step_21.aic
    pause
    goto MENU
)
if "%choice%"=="7" (
    echo.
    .\build_ninja\bin\aic.exe agent-run "Compute 25 * 16 and verify result"
    pause
    goto MENU
)
if "%choice%"=="8" (
    echo.
    .\build_ninja\bin\aic_tests.exe
    pause
    goto MENU
)
if "%choice%"=="9" (
    echo.
    .\build_ninja\bin\aic.exe --help
    cmd /k
    goto MENU
)
if "%choice%"=="0" exit /b 0
goto MENU
