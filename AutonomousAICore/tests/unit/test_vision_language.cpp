// tests/unit/test_vision_language.cpp
// Unit tests for Vision-Language Multimodal Engine (Aşama 23)
#include "tests/test_framework.hpp"
#include "data/image.hpp"
#include "models/vision_transformer.hpp"
#include "models/vision_language_lm.hpp"
#include "core/device.hpp"
#include "runtime/cuda_info.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace aic;
using namespace aic::data;
using namespace aic::models;

AIC_TEST(VisionLanguageTest, ImageResizingAndBilinearInterpolation) {
    Image grad = Image::create_gradient(32, 32, 3);
    ASSERT_EQ(grad.channels(), 3);
    ASSERT_EQ(grad.height(), 32);
    ASSERT_EQ(grad.width(), 32);

    // Upsample 32x32 -> 64x64
    Image up = grad.resize(64, 64);
    ASSERT_EQ(up.channels(), 3);
    ASSERT_EQ(up.height(), 64);
    ASSERT_EQ(up.width(), 64);
    ASSERT_EQ(up.data().size(), static_cast<size_t>(3 * 64 * 64));

    // Downsample 32x32 -> 16x16
    Image down = grad.resize(16, 16);
    ASSERT_EQ(down.height(), 16);
    ASSERT_EQ(down.width(), 16);

    // Verify corner preservation in normalized space
    const auto& orig_d = grad.data();
    const auto& up_d = up.data();
    ASSERT_NEAR(orig_d[0], up_d[0], 0.05f); // Top-left corner
}

AIC_TEST(VisionLanguageTest, ImagePatchExtractionAndDimensionality) {
    int H = 32, W = 32, C = 3, P = 8;
    int expected_np = (H / P) * (W / P); // 16 patches
    int expected_dp = C * P * P;         // 192 features per patch

    Image cb = Image::create_checkerboard(H, W, 8, C);
    Tensor patches = cb.patchify(P);

    ASSERT_EQ(patches.rank(), 2);
    ASSERT_EQ(patches.dim(0), expected_np);
    ASSERT_EQ(patches.dim(1), expected_dp);

    // Batched patchification: [2, 3, 32, 32] -> [2, 16, 192]
    Tensor batch_imgs = Tensor::zeros(Shape({2, C, H, W}), DType::Float32, Device::cpu());
    Tensor b_patches = Image::batch_patchify(batch_imgs, P);
    ASSERT_EQ(b_patches.rank(), 3);
    ASSERT_EQ(b_patches.dim(0), 2);
    ASSERT_EQ(b_patches.dim(1), expected_np);
    ASSERT_EQ(b_patches.dim(2), expected_dp);
}

AIC_TEST(VisionLanguageTest, PatchEmbedderProjectionAndPositionalBias) {
    ViTConfig cfg;
    cfg.image_size = 32;
    cfg.patch_size = 8;
    cfg.in_channels = 3;
    cfg.hidden_dim = 64;

    PatchEmbedder embedder(cfg, Device::cpu());
    ASSERT_EQ(embedder.num_patches(), 16);

    Tensor images = Tensor::uniform(Shape({2, 3, 32, 32}), -1.0f, 1.0f, DType::Float32, Device::cpu());
    Tensor embedded = embedder.forward(images);

    ASSERT_EQ(embedded.rank(), 3);
    ASSERT_EQ(embedded.dim(0), 2);
    ASSERT_EQ(embedded.dim(1), 16); // 16 patches
    ASSERT_EQ(embedded.dim(2), 64); // 64 hidden_dim
}

AIC_TEST(VisionLanguageTest, VisionTransformerBidirectionalAttention) {
    ViTConfig cfg;
    cfg.image_size = 32;
    cfg.patch_size = 8;
    cfg.in_channels = 3;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 128;

    VisionTransformer vit(cfg, Device::cpu());

    Tensor images = Tensor::uniform(Shape({2, 3, 32, 32}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    Tensor tokens = vit.forward(images);

    ASSERT_EQ(tokens.rank(), 3);
    ASSERT_EQ(tokens.dim(0), 2);
    ASSERT_EQ(tokens.dim(1), 16);
    ASSERT_EQ(tokens.dim(2), 64);

    // Verify finite numerical stability
    const float* ptr = tokens.data_ptr<float>();
    for (int64_t i = 0; i < tokens.numel(); ++i) {
        ASSERT_FALSE(std::isnan(ptr[i]));
        ASSERT_FALSE(std::isinf(ptr[i]));
    }
}

AIC_TEST(VisionLanguageTest, MultimodalProjectorDimensionAdaptation) {
    int64_t vit_dim = 64;
    int64_t lm_dim = 128;
    MultimodalProjector projector(vit_dim, lm_dim, Device::cpu());

    Tensor vit_tokens = Tensor::uniform(Shape({2, 16, vit_dim}), -1.0f, 1.0f, DType::Float32, Device::cpu());
    Tensor projected = projector.forward(vit_tokens);

    ASSERT_EQ(projected.rank(), 3);
    ASSERT_EQ(projected.dim(0), 2);
    ASSERT_EQ(projected.dim(1), 16);
    ASSERT_EQ(projected.dim(2), lm_dim);
}

AIC_TEST(VisionLanguageTest, EndToEndVisionLanguageGenerationAndMultimodalFusion) {
    VisionLanguageLMConfig cfg;
    cfg.vit_config.image_size = 32;
    cfg.vit_config.patch_size = 8;
    cfg.vit_config.in_channels = 3;
    cfg.vit_config.hidden_dim = 64;
    cfg.vit_config.num_layers = 2;
    cfg.vit_config.num_heads = 2;
    cfg.vit_config.intermediate_dim = 128;

    cfg.lm_config.vocab_size = 128;
    cfg.lm_config.hidden_dim = 64;
    cfg.lm_config.num_layers = 2;
    cfg.lm_config.num_heads = 2;
    cfg.lm_config.intermediate_dim = 128;
    cfg.lm_config.max_seq_len = 64;

    VisionLanguageLM vlm(cfg, Device::cpu());

    // 1. Multimodal Forward Pass
    Tensor images = Tensor::uniform(Shape({1, 3, 32, 32}), 0.0f, 1.0f, DType::Float32, Device::cpu());
    std::vector<int64_t> prompt = {10, 20, 30};
    Tensor text_ids = Tensor::from_blob(prompt.data(), Shape({1, 3}), DType::Int64, Device::cpu());

    Tensor logits = vlm.forward(images, text_ids);
    int64_t expected_seq = 16 + 3; // 16 vision patches + 3 text tokens
    ASSERT_EQ(logits.rank(), 3);
    ASSERT_EQ(logits.dim(0), 1);
    ASSERT_EQ(logits.dim(1), expected_seq);
    ASSERT_EQ(logits.dim(2), 128); // vocab_size

    // 2. Multimodal Autoregressive Generation
    std::vector<int64_t> generated = vlm.generate(images, prompt, /*max_new_tokens=*/8);
    ASSERT_FALSE(generated.empty());
    ASSERT_TRUE(generated.size() <= 8);
    for (int64_t token : generated) {
        ASSERT_TRUE(token >= 0 && token < 128);
    }
}

AIC_TEST(VisionLanguageTest, CudaVisionPatchifyAndParity) {
    if (!is_cuda_available()) {
        std::cout << "[VisionLanguageTest] CUDA not available, skipping GPU patchify test\n";
        return;
    }

    int64_t B = 2, C = 3, H = 32, W = 32, P = 8;
    Tensor cpu_imgs = Tensor::uniform(Shape({B, C, H, W}), -2.0f, 2.0f, DType::Float32, Device::cpu());
    Tensor cuda_imgs = cpu_imgs.to(Device::cuda(0));

    Tensor cpu_patches = Image::batch_patchify(cpu_imgs, static_cast<int>(P));
    Tensor cuda_patches = Image::batch_patchify(cuda_imgs, static_cast<int>(P));

    Tensor cuda_result = cuda_patches.cpu();
    ASSERT_TRUE(cpu_patches.shape() == cuda_result.shape());

    const float* p_cpu = cpu_patches.data_ptr<float>();
    const float* p_cuda = cuda_result.data_ptr<float>();

    for (int64_t i = 0; i < cpu_patches.numel(); ++i) {
        ASSERT_NEAR(p_cpu[i], p_cuda[i], 1e-5f);
    }
}
