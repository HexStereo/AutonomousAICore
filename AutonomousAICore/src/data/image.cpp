// src/data/image.cpp
// Implementation of Image representation, bilinear resampling, and patch extraction (Aşama 23)
#include "data/image.hpp"
#include "core/error.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_vision.cuh"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>

namespace aic::data {

Image::Image(int channels, int height, int width, std::vector<float> data)
    : dims_{channels, height, width}, data_(std::move(data)) {
    AIC_CHECK(channels > 0 && height > 0 && width > 0, InvalidArgument,
              "Image dimensions must be positive, got C=", channels, " H=", height, " W=", width);
    AIC_CHECK(data_.size() == static_cast<size_t>(channels * height * width), InvalidArgument,
              "Image data size (", data_.size(), ") does not match C*H*W (", channels * height * width, ")");
}

Image::Image(const Tensor& tensor) {
    Tensor cpu_t = tensor.cpu().contiguous();
    AIC_CHECK(cpu_t.dtype() == DType::Float32, InvalidArgument, "Image expects Float32 tensor");

    if (cpu_t.rank() == 3) {
        dims_.channels = static_cast<int>(cpu_t.dim(0));
        dims_.height = static_cast<int>(cpu_t.dim(1));
        dims_.width = static_cast<int>(cpu_t.dim(2));
    } else if (cpu_t.rank() == 4 && cpu_t.dim(0) == 1) {
        dims_.channels = static_cast<int>(cpu_t.dim(1));
        dims_.height = static_cast<int>(cpu_t.dim(2));
        dims_.width = static_cast<int>(cpu_t.dim(3));
    } else {
        AIC_THROW(ShapeMismatch, "Image expects 3D [C, H, W] or 4D [1, C, H, W] tensor, got rank ", cpu_t.rank());
    }

    size_t total = static_cast<size_t>(dims_.channels * dims_.height * dims_.width);
    data_.resize(total);
    const float* ptr = cpu_t.data_ptr<float>();
    std::copy(ptr, ptr + total, data_.begin());
}

Image Image::resize(int target_h, int target_w) const {
    AIC_CHECK(target_h > 0 && target_w > 0, InvalidArgument, "Target dimensions must be positive");
    if (dims_.height == target_h && dims_.width == target_w) {
        return *this;
    }

    std::vector<float> out_data(static_cast<size_t>(dims_.channels * target_h * target_w));

    float r_h = static_cast<float>(dims_.height) / static_cast<float>(target_h);
    float r_w = static_cast<float>(dims_.width) / static_cast<float>(target_w);

    for (int c = 0; c < dims_.channels; ++c) {
        const float* in_plane = data_.data() + c * (dims_.height * dims_.width);
        float* out_plane = out_data.data() + c * (target_h * target_w);

        for (int oh = 0; oh < target_h; ++oh) {
            float in_y = (static_cast<float>(oh) + 0.5f) * r_h - 0.5f;
            in_y = std::max(0.0f, std::min(in_y, static_cast<float>(dims_.height - 1)));
            int y0 = static_cast<int>(in_y);
            int y1 = (y0 + 1 < dims_.height) ? (y0 + 1) : y0;
            float dy = in_y - static_cast<float>(y0);

            for (int ow = 0; ow < target_w; ++ow) {
                float in_x = (static_cast<float>(ow) + 0.5f) * r_w - 0.5f;
                in_x = std::max(0.0f, std::min(in_x, static_cast<float>(dims_.width - 1)));
                int x0 = static_cast<int>(in_x);
                int x1 = (x0 + 1 < dims_.width) ? (x0 + 1) : x0;
                float dx = in_x - static_cast<float>(x0);

                float v00 = in_plane[y0 * dims_.width + x0];
                float v01 = in_plane[y0 * dims_.width + x1];
                float v10 = in_plane[y1 * dims_.width + x0];
                float v11 = in_plane[y1 * dims_.width + x1];

                float val = (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01) +
                            dy * ((1.0f - dx) * v10 + dx * v11);

                out_plane[oh * target_w + ow] = val;
            }
        }
    }

    return Image(dims_.channels, target_h, target_w, std::move(out_data));
}

Tensor Image::to_tensor(Device device) const {
    Tensor t = Tensor::from_blob(
        const_cast<float*>(data_.data()),
        Shape({1, dims_.channels, dims_.height, dims_.width}),
        DType::Float32,
        Device::cpu()
    ).clone();
    return t.to(device);
}

Tensor Image::patchify(int patch_size, Device device) const {
    Tensor img_tensor = to_tensor(device);
    Tensor patches = batch_patchify(img_tensor, patch_size);
    return patches.squeeze(0); // [num_patches, C * P * P]
}

Tensor Image::batch_patchify(const Tensor& images, int patch_size) {
    AIC_CHECK(images.rank() == 4, InvalidArgument, "batch_patchify expects 4D tensor [B, C, H, W], got rank ", images.rank());
    int64_t B = images.dim(0);
    int64_t C = images.dim(1);
    int64_t H = images.dim(2);
    int64_t W = images.dim(3);

    AIC_CHECK(patch_size > 0 && H % patch_size == 0 && W % patch_size == 0, InvalidArgument,
              "Image H=", H, ", W=", W, " must be divisible by patch_size=", patch_size);

    int64_t Hp = H / patch_size;
    int64_t Wp = W / patch_size;
    int64_t Np = Hp * Wp;
    int64_t Dp = C * patch_size * patch_size;

#if AIC_WITH_CUDA
    if (images.is_cuda()) {
        Tensor out = Tensor::zeros(Shape({B, Np, Dp}), DType::Float32, images.device());
        cuda::launch_patchify(
            images.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C, H, W, patch_size
        );
        cudaDeviceSynchronize();
        return out;
    }
#endif

    Tensor out = Tensor::zeros(Shape({B, Np, Dp}), DType::Float32, Device::cpu());
    const float* in_ptr = images.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t py = 0; py < Hp; ++py) {
            for (int64_t px = 0; px < Wp; ++px) {
                int64_t n = py * Wp + px;
                float* p_dest = out_ptr + (b * Np + n) * Dp;
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t y = 0; y < patch_size; ++y) {
                        for (int64_t x = 0; x < patch_size; ++x) {
                            int64_t in_idx = b * (C * H * W) + c * (H * W) + (py * patch_size + y) * W + (px * patch_size + x);
                            int64_t out_idx = c * (patch_size * patch_size) + y * patch_size + x;
                            p_dest[out_idx] = in_ptr[in_idx];
                        }
                    }
                }
            }
        }
    }

    return out;
}

void Image::normalize(const std::vector<float>& mean, const std::vector<float>& std_dev) {
    AIC_CHECK(mean.size() >= static_cast<size_t>(dims_.channels) &&
              std_dev.size() >= static_cast<size_t>(dims_.channels),
              InvalidArgument, "Mean and std vectors must have at least C elements");

    size_t plane_size = static_cast<size_t>(dims_.height * dims_.width);
    for (int c = 0; c < dims_.channels; ++c) {
        float m = mean[static_cast<size_t>(c)];
        float s = std_dev[static_cast<size_t>(c)];
        float* plane = data_.data() + c * plane_size;
        for (size_t i = 0; i < plane_size; ++i) {
            plane[i] = (plane[i] - m) / s;
        }
    }
}

Image Image::create_checkerboard(int height, int width, int tile_size, int channels) {
    std::vector<float> data(static_cast<size_t>(channels * height * width), 0.0f);
    for (int c = 0; c < channels; ++c) {
        float* plane = data.data() + c * (height * width);
        for (int y = 0; y < height; ++y) {
            int ty = y / tile_size;
            for (int x = 0; x < width; ++x) {
                int tx = x / tile_size;
                bool is_white = ((ty + tx) % 2 == 0);
                plane[y * width + x] = is_white ? 1.0f : 0.0f;
            }
        }
    }
    return Image(channels, height, width, std::move(data));
}

Image Image::create_gradient(int height, int width, int channels) {
    std::vector<float> data(static_cast<size_t>(channels * height * width), 0.0f);
    for (int c = 0; c < channels; ++c) {
        float* plane = data.data() + c * (height * width);
        for (int y = 0; y < height; ++y) {
            float vy = static_cast<float>(y) / static_cast<float>(height);
            for (int x = 0; x < width; ++x) {
                float vx = static_cast<float>(x) / static_cast<float>(width);
                plane[y * width + x] = (c == 0) ? vx : ((c == 1) ? vy : (vx + vy) * 0.5f);
            }
        }
    }
    return Image(channels, height, width, std::move(data));
}

Image Image::create_solid(int height, int width, float r, float g, float b) {
    std::vector<float> data(static_cast<size_t>(3 * height * width));
    size_t plane_size = static_cast<size_t>(height * width);
    std::fill(data.begin(), data.begin() + plane_size, r);
    std::fill(data.begin() + plane_size, data.begin() + 2 * plane_size, g);
    std::fill(data.begin() + 2 * plane_size, data.end(), b);
    return Image(3, height, width, std::move(data));
}

} // namespace aic::data
