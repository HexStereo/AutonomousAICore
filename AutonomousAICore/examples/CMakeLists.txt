# examples/CMakeLists.txt
add_executable(train_mini_llm train_mini_llm.cpp)

target_include_directories(train_mini_llm
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(train_mini_llm
    PRIVATE
        aic_core
        aic_compile_options
)
