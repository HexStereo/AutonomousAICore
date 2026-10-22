# src/CMakeLists.txt
set(AIC_CORE_SOURCES
    core/error.cpp
    core/logging.cpp
    core/dtype.cpp
    core/device.cpp
    core/shape.cpp
    memory/allocator.cpp
    memory/memory_pool.cpp
    memory/storage.cpp
    tensor/tensor.cpp
    runtime/cpu_info.cpp
    runtime/cuda_info.cpp
    ops/cpu_ops.cpp
    ops/ops.cpp
    ops/cpu_gemm.cpp
    ops/matmul.cpp
    autograd/node.cpp
    autograd/functions.cpp
    autograd/engine.cpp
    autograd/gradcheck.cpp
    loss/cross_entropy.cpp
    loss/mse.cpp
    optim/optimizer.cpp
    optim/adamw.cpp
    optim/clip_grad.cpp
    data/tokenizer.cpp
    data/dataset.cpp
    data/dataloader.cpp
    nn/embedding.cpp
    models/transformer_lm.cpp
    train/checkpoint.cpp
    train/trainer.cpp
    train/amp.cpp
    autonomous/telemetry.cpp
    autonomous/controller.cpp
    reasoning/thought_node.cpp
    reasoning/verifier.cpp
    reasoning/mcts_engine.cpp
    alignment/dpo_loss.cpp
    alignment/dpo_trainer.cpp
    alignment/self_play.cpp
    memory/block_manager.cpp
    ops/paged_attention.cpp
    inference/speculative_decoder.cpp
    inference/continuous_batch_engine.cpp
    nn/lora.cpp
    nn/moe.cpp
    models/moe_transformer_lm.cpp
    ops/flash_attention.cpp
    nn/quantization.cpp
    inference/grammar.cpp
    data/image.cpp
    models/vision_transformer.cpp
    models/vision_language_lm.cpp
    autonomous/tool.cpp
    autonomous/sandbox.cpp
    autonomous/react_engine.cpp
    server/http_server.cpp
)

set(AIC_CUDA_SOURCES)
if(AIC_WITH_CUDA)
    list(APPEND AIC_CUDA_SOURCES
        cuda/device_discovery.cu
        cuda/cuda_elementwise.cu
        cuda/cuda_reductions.cu
        cuda/cuda_gemm.cu
        cuda/cuda_loss.cu
        cuda/cuda_adamw.cu
        cuda/cuda_embedding.cu
        cuda/cuda_amp.cu
        cuda/cuda_paged_attention.cu
        cuda/cuda_flash_attention.cu
        cuda/cuda_quantize.cu
        cuda/cuda_grammar.cu
        cuda/cuda_vision.cu
        cuda/cuda_tool_routing.cu
    )
endif()

add_library(aic_core STATIC
    ${AIC_CORE_SOURCES}
    ${AIC_CUDA_SOURCES}
)

target_include_directories(aic_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(aic_core
    PUBLIC
        aic_compile_options
)

if(AIC_WITH_CUDA)
    target_link_libraries(aic_core PUBLIC CUDA::cudart_static)
    target_compile_definitions(aic_core PUBLIC AIC_WITH_CUDA=1)
    if(AIC_WITH_CUBLAS)
        target_link_libraries(aic_core PUBLIC CUDA::cublas)
        target_compile_definitions(aic_core PUBLIC AIC_WITH_CUBLAS=1)
    endif()
else()
    target_compile_definitions(aic_core PUBLIC AIC_WITH_CUDA=0 AIC_WITH_CUBLAS=0)
endif()

if(WIN32)
    target_link_libraries(aic_core PUBLIC OneCore ws2_32)
endif()
