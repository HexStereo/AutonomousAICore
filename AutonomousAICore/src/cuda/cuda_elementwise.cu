// src/cuda/cuda_elementwise.cu
#include "cuda/cuda_elementwise.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#define AIC_CUDA_CHECK_OPS(call)                                                           \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            AIC_THROW(CudaError, "CUDA error in elementwise kernel: ",                      \
                      cudaGetErrorString(err__));                                          \
        }                                                                                  \
    } while (false)

namespace aic::cuda {

namespace {

struct AddOp { __device__ float operator()(float x, float y) const { return x + y; } };
struct SubOp { __device__ float operator()(float x, float y) const { return x - y; } };
struct MulOp { __device__ float operator()(float x, float y) const { return x * y; } };
struct DivOp { __device__ float operator()(float x, float y) const { return x / y; } };

struct NegOp { __device__ float operator()(float x) const { return -x; } };
struct AbsOp { __device__ float operator()(float x) const { return fabsf(x); } };
struct ExpOp { __device__ float operator()(float x) const { return expf(x); } };
struct LogOp { __device__ float operator()(float x) const { return logf(x); } };
struct SqrtOp { __device__ float operator()(float x) const { return sqrtf(x); } };
struct TanhOp { __device__ float operator()(float x) const { return tanhf(x); } };
struct SigmoidOp { __device__ float operator()(float x) const { return 1.0f / (1.0f + expf(-x)); } };
struct GeluOp {
    __device__ float operator()(float x) const {
        return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
};
struct SiluOp { __device__ float operator()(float x) const { return x / (1.0f + expf(-x)); } };

// Grid-stride contiguous binary kernel
template <typename Op>
__global__ void binary_contiguous_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         int64_t n, Op op) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = op(a[i], b[i]);
    }
}

struct CudaBroadcastParams {
    int rank = 0;
    int64_t dims[8] = {0};
    int64_t strides_a[8] = {0};
    int64_t strides_b[8] = {0};
    int64_t offset_a = 0;
    int64_t offset_b = 0;
};

// Grid-stride broadcast binary kernel
template <typename Op>
__global__ void binary_broadcast_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ out,
                                        int64_t n, CudaBroadcastParams p, Op op) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int64_t linear = i;
        int64_t off_a = p.offset_a;
        int64_t off_b = p.offset_b;
        for (int k = p.rank - 1; k >= 0; --k) {
            int64_t coord = linear % p.dims[k];
            linear /= p.dims[k];
            off_a += coord * p.strides_a[k];
            off_b += coord * p.strides_b[k];
        }
        out[i] = op(a[off_a], b[off_b]);
    }
}

// Grid-stride contiguous unary kernel
template <typename Op>
__global__ void unary_contiguous_kernel(const float* __restrict__ a,
                                        float* __restrict__ out,
                                        int64_t n, Op op) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = op(a[i]);
    }
}

template <typename Op>
Tensor launch_binary_gpu(const Tensor& a, const Tensor& b, Op op) {
    AIC_CHECK(a.dtype() == DType::Float32 && b.dtype() == DType::Float32,
              DTypeMismatch, "CUDA binary ops currently require Float32 tensors");
    AIC_CHECK(a.device() == b.device(), DeviceMismatch,
              "CUDA tensors must be on the same device (got ", a.device().str(), " and ", b.device().str(), ")");

    AIC_CUDA_CHECK_OPS(cudaSetDevice(a.device().index));
    Shape out_shape = Shape::broadcast(a.shape(), b.shape());
    Tensor out = Tensor::empty(out_shape, DType::Float32, a.device());
    int64_t total = out_shape.numel();
    if (total == 0) return out;

    int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));

    // Fast contiguous path
    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        binary_contiguous_kernel<<<blocks, threads>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), total, op);
        AIC_CUDA_CHECK_OPS(cudaGetLastError());
        return out;
    }

    // Broadcast path
    int r = out_shape.rank();
    AIC_CHECK(r <= 8, InvalidArgument, "Broadcast supports up to rank 8 on GPU, got rank ", r);

    CudaBroadcastParams p;
    p.rank = r;
    p.offset_a = a.storage_offset();
    p.offset_b = b.storage_offset();

    auto str_a = Shape::broadcast_strides(a.shape(), a.strides(), out_shape);
    auto str_b = Shape::broadcast_strides(b.shape(), b.strides(), out_shape);

    for (int i = 0; i < r; ++i) {
        p.dims[i] = out_shape[i];
        p.strides_a[i] = str_a[static_cast<size_t>(i)];
        p.strides_b[i] = str_b[static_cast<size_t>(i)];
    }

    const float* raw_a = static_cast<const float*>(a.storage()->data());
    const float* raw_b = static_cast<const float*>(b.storage()->data());

    binary_broadcast_kernel<<<blocks, threads>>>(raw_a, raw_b, out.data_ptr<float>(), total, p, op);
    AIC_CUDA_CHECK_OPS(cudaGetLastError());
    return out;
}

template <typename Op>
Tensor launch_unary_gpu(const Tensor& a, Op op) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CUDA unary ops require Float32 tensors");
    AIC_CUDA_CHECK_OPS(cudaSetDevice(a.device().index));

    Tensor contig = a.contiguous();
    Tensor out = Tensor::empty(contig.shape(), DType::Float32, a.device());
    int64_t total = contig.numel();
    if (total == 0) return out;

    int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));

    unary_contiguous_kernel<<<blocks, threads>>>(
        contig.data_ptr<float>(), out.data_ptr<float>(), total, op);
    AIC_CUDA_CHECK_OPS(cudaGetLastError());
    return out;
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) { return launch_binary_gpu(a, b, AddOp{}); }
Tensor sub(const Tensor& a, const Tensor& b) { return launch_binary_gpu(a, b, SubOp{}); }
Tensor mul(const Tensor& a, const Tensor& b) { return launch_binary_gpu(a, b, MulOp{}); }
Tensor div(const Tensor& a, const Tensor& b) { return launch_binary_gpu(a, b, DivOp{}); }

Tensor neg(const Tensor& a) { return launch_unary_gpu(a, NegOp{}); }
Tensor abs(const Tensor& a) { return launch_unary_gpu(a, AbsOp{}); }
Tensor exp(const Tensor& a) { return launch_unary_gpu(a, ExpOp{}); }
Tensor log(const Tensor& a) { return launch_unary_gpu(a, LogOp{}); }
Tensor sqrt(const Tensor& a) { return launch_unary_gpu(a, SqrtOp{}); }
Tensor tanh(const Tensor& a) { return launch_unary_gpu(a, TanhOp{}); }
Tensor sigmoid(const Tensor& a) { return launch_unary_gpu(a, SigmoidOp{}); }
Tensor gelu(const Tensor& a) { return launch_unary_gpu(a, GeluOp{}); }
Tensor silu(const Tensor& a) { return launch_unary_gpu(a, SiluOp{}); }

}  // namespace aic::cuda
