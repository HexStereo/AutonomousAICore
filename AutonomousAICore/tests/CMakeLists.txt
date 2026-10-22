# tests/CMakeLists.txt
add_executable(aic_tests
    main.cpp
    unit/test_shape.cpp
    unit/test_memory_pool.cpp
    unit/test_tensor.cpp
    unit/test_runtime.cpp
    unit/test_ops.cpp
    unit/test_autograd.cpp
    unit/test_matmul.cpp
    unit/test_nn.cpp
    unit/test_optim.cpp
    unit/test_data.cpp
    unit/test_model.cpp
    unit/test_train.cpp
    unit/test_autonomous.cpp
    unit/test_reasoning.cpp
    unit/test_amp.cpp
    unit/test_dpo.cpp
    unit/test_paged_attention.cpp
    unit/test_speculative.cpp
    unit/test_lora.cpp
    unit/test_moe.cpp
    unit/test_flash_attention.cpp
    unit/test_continuous_batch.cpp
    unit/test_quantization.cpp
    unit/test_grammar.cpp
    unit/test_vision_language.cpp
    unit/test_tool_calling.cpp
    unit/test_server.cpp
    integration/test_end_to_end.cpp
)

target_include_directories(aic_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(aic_tests
    PRIVATE
        aic_core
        aic_compile_options
)

add_test(NAME AllTests COMMAND aic_tests)
