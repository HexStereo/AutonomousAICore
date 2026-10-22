# tools/CMakeLists.txt
add_executable(aic aic_cli.cpp)

target_include_directories(aic
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(aic
    PRIVATE
        aic_core
        aic_compile_options
)
