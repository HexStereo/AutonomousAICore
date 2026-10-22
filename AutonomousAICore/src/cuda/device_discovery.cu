// src/cuda/device_discovery.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void aic_warmup_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42.0f;
    }
}

extern "C" void aic_cuda_warmup(int device_index) {
    cudaSetDevice(device_index);
    float* d_ptr = nullptr;
    if (cudaMalloc(&d_ptr, sizeof(float)) == cudaSuccess) {
        aic_warmup_kernel<<<1, 1>>>(d_ptr);
        cudaDeviceSynchronize();
        cudaFree(d_ptr);
    }
}
