// include/data/image.hpp
// Zero-dependency image representation, bilinear resampling, and patch extraction (Aşama 23)
#pragma once

#include "tensor/tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace aic::data {

struct ImageDimensions {
    int channels = 3;
    int height = 224;
    int width = 224;
};

class Image {
public:
    Image() = default;
    Image(int channels, int height, int width, std::vector<float> data);
    explicit Image(const Tensor& tensor);

    int channels() const noexcept { return dims_.channels; }
    int height() const noexcept { return dims_.height; }
    int width() const noexcept { return dims_.width; }
    const ImageDimensions& dims() const noexcept { return dims_; }
    const std::vector<float>& data() const noexcept { return data_; }
    size_t num_pixels() const noexcept { return static_cast<size_t>(dims_.channels * dims_.height * dims_.width); }

    // Bilinear interpolation resize: returns new Image of target dimensions
    Image resize(int target_h, int target_w) const;

    // Convert to Tensor [1, C, H, W]
    Tensor to_tensor(Device device = Device::cpu()) const;

    // Extract non-overlapping spatial patches of size (P x P)
    // Returns flattened patch Tensor of shape [num_patches, C * P * P]
    // where num_patches = (H / P) * (W / P)
    Tensor patchify(int patch_size, Device device = Device::cpu()) const;

    // Batched image patchification: [B, C, H, W] -> [B, num_patches, C * P * P]
    static Tensor batch_patchify(const Tensor& images, int patch_size);

    // Standard ImageNet normalization: (x - mean) / std
    void normalize(
        const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
        const std::vector<float>& std  = {0.229f, 0.224f, 0.225f}
    );

    // Synthetic test pattern generators
    static Image create_checkerboard(int height, int width, int tile_size = 16, int channels = 3);
    static Image create_gradient(int height, int width, int channels = 3);
    static Image create_solid(int height, int width, float r = 1.0f, float g = 0.5f, float b = 0.2f);

private:
    ImageDimensions dims_{0, 0, 0};
    std::vector<float> data_;
};

} // namespace aic::data
