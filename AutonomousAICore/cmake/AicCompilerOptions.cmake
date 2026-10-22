# Common compiler options exposed through the INTERFACE target `aic_compile_options`.
add_library(aic_compile_options INTERFACE)

if(MSVC)
    target_compile_options(aic_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /utf-8 /EHsc /bigobj /wd4324>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/Zc:__cplusplus,/EHsc,/wd4324>)
    target_compile_definitions(aic_compile_options INTERFACE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(AIC_WARNINGS_AS_ERRORS)
        target_compile_options(aic_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
    # AVX2 + FMA flag for the runtime-dispatched CPU micro kernels.
    set(AIC_AVX2_FLAGS /arch:AVX2)
else()
    target_compile_options(aic_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>)
    if(AIC_WARNINGS_AS_ERRORS)
        target_compile_options(aic_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
    set(AIC_AVX2_FLAGS -mavx2 -mfma -mf16c)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    set(AIC_X86_64 ON)
else()
    set(AIC_X86_64 OFF)
endif()
if(NOT AIC_X86_64)
    set(AIC_ENABLE_CPU_AVX2_KERNELS OFF)
endif()
