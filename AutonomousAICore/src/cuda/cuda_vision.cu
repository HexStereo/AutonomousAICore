// src/cuda/cuda_vision.cu
// CUDA Vision Kernels for Blackwell sm_120: Bilinear Resizing and Patchification
#include "cuda_vision.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace aic::cuda {

__global__ void bilinear_resize_kernel(
    const float* __restrict__ d_in,
    float* __restrict__ d_out,
    int64_t B,
    int64_t C,
    int64_t in_H,
    int64_t in_W,
    int64_t out_H,
    int64_t out_W
) {
    int64_t total = B * C * out_H * out_W;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t ow = idx % out_W;
    int64_t tmp = idx / out_W;
    int64_t oh = tmp % out_H;
    tmp /= out_H;
    int64_t c = tmp % C;
    int64_t b = tmp / C;

    float r_h = static_cast<float>(in_H) / static_cast<float>(out_H);
    float r_w = static_cast<float>(in_W) / static_cast<float>(out_W);

    float in_y = (static_cast<float>(oh) + 0.5f) * r_h - 0.5f;
    float in_x = (static_cast<float>(ow) + 0.5f) * r_w - 0.5f;

    in_y = fmaxf(0.0f, fminf(in_y, static_cast<float>(in_H - 1)));
    in_x = fmaxf(0.0f, fminf(in_x, static_cast<float>(in_W - 1)));

    int64_t y0 = static_cast<int64_t>(in_y);
    int64_t y1 = (y0 + 1 < in_H) ? (y0 + 1) : y0;
    int64_t x0 = static_cast<int64_t>(in_x);
    int64_t x1 = (x0 + 1 < in_W) ? (x0 + 1) : x0;

    float dy = in_y - static_cast<float>(y0);
    float dx = in_x - static_cast<float>(x0);

    int64_t plane_offset = (b * C + c) * (in_H * in_W);
    float v00 = d_in[plane_offset + y0 * in_W + x0];
    float v01 = d_in[plane_offset + y0 * in_W + x1];
    float v10 = d_in[plane_offset + y1 * in_W + x0];
    float v11 = d_in[plane_offset + y1 * in_W + x1];

    float val = (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01) +
                dy * ((1.0f - dx) * v10 + dx * v11);

    d_out[idx] = val;
}

void launch_bilinear_resize(
    const float* d_in,
    float* d_out,
    int64_t B,
    int64_t C,
    int64_t in_H,
    int64_t in_W,
    int64_t out_H,
    int64_t out_W,
    cudaStream_t stream
) {
    int64_t total = B * C * out_H * out_W;
    if (total <= 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    bilinear_resize_kernel<<<blocks, threads, 0, stream>>>(
        d_in, d_out, B, C, in_H, in_W, out_H, out_W
    );
}

__global__ void patchify_kernel(
    const float* __restrict__ d_images,
    float* __restrict__ d_patches,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t P,
    int64_t Hp,
    int64_t Wp,
    int64_t Np,
    int64_t Dp
) {
    int64_t total = B * Np * Dp;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t d = idx % Dp;
    int64_t tmp = idx / Dp;
    int64_t n = tmp % Np;
    int64_t b = tmp / Np;

    int64_t p_y = n / Wp;
    int64_t p_x = n % Wp;

    int64_t p_sq = P * P;
    int64_t c = d / p_sq;
    int64_t rem = d % p_sq;
    int64_t py = rem / P;
    int64_t px = rem % P;

    int64_t h = p_y * P + py;
    int64_t w = p_x * P + px;

    int64_t in_idx = b * (C * H * W) + c * (H * W) + h * W + w;
    d_patches[idx] = d_images[in_idx];
}

void launch_patchify(
    const float* d_images,
    float* d_patches,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t P,
    cudaStream_t stream
) {
    int64_t Hp = H / P;
    int64_t Wp = W / P;
    int64_t Np = Hp * Wp;
    int64_t Dp = C * P * P;

    int64_t total = B * Np * Dp;
    if (total <= 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    patchify_kernel<<<blocks, threads, 0, stream>>>(
        d_images, d_patches, B, C, H, W, P, Hp, Wp, Np, Dp
    );
}

} // namespace aic::cuda
