// src/cuda/cuda_reductions.cu
// High-performance CUDA reduction kernels using warp shuffle primitives and shared memory.
#include "cuda/cuda_reductions.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#define AIC_CUDA_CHECK_REDUCE(call)                                                        \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            AIC_THROW(CudaError, "CUDA error in reduction kernel: ",                       \
                      cudaGetErrorString(err__));                                          \
        }                                                                                  \
    } while (false)

namespace aic::cuda {

namespace {

// ============================================================================
// Warp Reduction Primitives
// ============================================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceMin(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Atomic CAS for float Max / Min
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    }
}

__device__ __forceinline__ void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    }
}

// ============================================================================
// Block-level Full Reductions
// ============================================================================
__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    __shared__ float s_warp[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float acc = 0.0f;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        acc += in[i];
    }
    acc = warpReduceSum(acc);

    if (lane == 0) {
        s_warp[wid] = acc;
    }
    __syncthreads();

    if (wid == 0) {
        float b_acc = (lane < (blockDim.x / 32)) ? s_warp[lane] : 0.0f;
        b_acc = warpReduceSum(b_acc);
        if (lane == 0) {
            atomicAdd(out, b_acc);
        }
    }
}

__global__ void reduce_max_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    __shared__ float s_warp[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float acc = -1e38f;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        acc = fmaxf(acc, in[i]);
    }
    acc = warpReduceMax(acc);

    if (lane == 0) {
        s_warp[wid] = acc;
    }
    __syncthreads();

    if (wid == 0) {
        float b_acc = (lane < (blockDim.x / 32)) ? s_warp[lane] : -1e38f;
        b_acc = warpReduceMax(b_acc);
        if (lane == 0) {
            atomicMaxFloat(out, b_acc);
        }
    }
}

__global__ void reduce_min_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    __shared__ float s_warp[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float acc = 1e38f;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        acc = fminf(acc, in[i]);
    }
    acc = warpReduceMin(acc);

    if (lane == 0) {
        s_warp[wid] = acc;
    }
    __syncthreads();

    if (wid == 0) {
        float b_acc = (lane < (blockDim.x / 32)) ? s_warp[lane] : 1e38f;
        b_acc = warpReduceMin(b_acc);
        if (lane == 0) {
            atomicMinFloat(out, b_acc);
        }
    }
}

// In-place scale kernel for mean calculation
__global__ void scale_kernel(float* data, float factor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *data *= factor;
    }
}

// ============================================================================
// Dimensional Reductions
// ============================================================================
enum class ReduceOp { Sum, Mean, Max, Min };

__global__ void reduce_dim_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int64_t outer_size,
                                  int64_t reduce_size,
                                  int64_t inner_size,
                                  ReduceOp op) {
    int64_t total_out = outer_size * inner_size;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_out; i += blockDim.x * gridDim.x) {
        int64_t outer = i / inner_size;
        int64_t inner = i % inner_size;
        int64_t base_idx = outer * (reduce_size * inner_size) + inner;

        if (op == ReduceOp::Sum || op == ReduceOp::Mean) {
            float sum = 0.0f;
            for (int64_t k = 0; k < reduce_size; ++k) {
                sum += in[base_idx + k * inner_size];
            }
            out[i] = (op == ReduceOp::Mean) ? (sum / static_cast<float>(reduce_size)) : sum;
        } else if (op == ReduceOp::Max) {
            float max_val = -1e38f;
            for (int64_t k = 0; k < reduce_size; ++k) {
                max_val = fmaxf(max_val, in[base_idx + k * inner_size]);
            }
            out[i] = max_val;
        } else if (op == ReduceOp::Min) {
            float min_val = 1e38f;
            for (int64_t k = 0; k < reduce_size; ++k) {
                min_val = fminf(min_val, in[base_idx + k * inner_size]);
            }
            out[i] = min_val;
        }
    }
}

Tensor reduce_dim_cuda(const Tensor& a, int dim, bool keepdim, ReduceOp op) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CUDA reduction requires Float32");
    AIC_CUDA_CHECK_REDUCE(cudaSetDevice(a.device().index));

    int r = a.rank();
    int norm_dim = dim < 0 ? dim + r : dim;
    AIC_CHECK(norm_dim >= 0 && norm_dim < r, OutOfRange, "Reduction dim ", dim, " out of range for rank ", r);

    int64_t reduce_size = a.dim(norm_dim);
    AIC_CHECK(reduce_size > 0, InvalidArgument, "Cannot reduce along dimension of size 0");

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<size_t>(keepdim ? r : r - 1));
    for (int i = 0; i < r; ++i) {
        if (i == norm_dim) {
            if (keepdim) out_dims.push_back(1);
        } else {
            out_dims.push_back(a.dim(i));
        }
    }

    Tensor out = Tensor::empty(Shape(std::move(out_dims)), DType::Float32, a.device());
    int64_t total_out = out.numel();
    if (total_out == 0) return out;

    Tensor a_contig = a.contiguous();
    int64_t outer_size = 1;
    for (int i = 0; i < norm_dim; ++i) outer_size *= a_contig.dim(i);
    int64_t inner_size = 1;
    for (int i = norm_dim + 1; i < r; ++i) inner_size *= a_contig.dim(i);

    int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((total_out + threads - 1) / threads, 65535));

    reduce_dim_kernel<<<blocks, threads>>>(
        a_contig.data_ptr<float>(), out.data_ptr<float>(),
        outer_size, reduce_size, inner_size, op);
    AIC_CUDA_CHECK_REDUCE(cudaGetLastError());
    return out;
}

}  // namespace

Tensor sum(const Tensor& a) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CUDA sum requires Float32");
    AIC_CUDA_CHECK_REDUCE(cudaSetDevice(a.device().index));

    Tensor out = Tensor::zeros(Shape({}), DType::Float32, a.device());
    int64_t n = a.numel();
    if (n == 0) return out;

    Tensor contig = a.contiguous();
    int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 1024));

    reduce_sum_kernel<<<blocks, threads>>>(contig.data_ptr<float>(), out.data_ptr<float>(), n);
    AIC_CUDA_CHECK_REDUCE(cudaGetLastError());
    return out;
}

Tensor mean(const Tensor& a) {
    AIC_CHECK(a.numel() > 0, InvalidArgument, "Cannot compute mean of empty tensor");
    Tensor res = sum(a);
    float factor = 1.0f / static_cast<float>(a.numel());
    scale_kernel<<<1, 1>>>(res.data_ptr<float>(), factor);
    AIC_CUDA_CHECK_REDUCE(cudaGetLastError());
    return res;
}

Tensor max(const Tensor& a) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CUDA max requires Float32");
    AIC_CHECK(a.numel() > 0, InvalidArgument, "Cannot compute max of empty tensor");
    AIC_CUDA_CHECK_REDUCE(cudaSetDevice(a.device().index));

    Tensor out = Tensor::empty(Shape({}), DType::Float32, a.device());
    float init_val = -1e38f;
    AIC_CUDA_CHECK_REDUCE(cudaMemcpy(out.data_ptr<float>(), &init_val, sizeof(float), cudaMemcpyHostToDevice));

    Tensor contig = a.contiguous();
    int64_t n = contig.numel();
    int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 1024));

    reduce_max_kernel<<<blocks, threads>>>(contig.data_ptr<float>(), out.data_ptr<float>(), n);
    AIC_CUDA_CHECK_REDUCE(cudaGetLastError());
    return out;
}

Tensor min(const Tensor& a) {
    AIC_CHECK(a.dtype() == DType::Float32, DTypeMismatch, "CUDA min requires Float32");
    AIC_CHECK(a.numel() > 0, InvalidArgument, "Cannot compute min of empty tensor");
    AIC_CUDA_CHECK_REDUCE(cudaSetDevice(a.device().index));

    Tensor out = Tensor::empty(Shape({}), DType::Float32, a.device());
    float init_val = 1e38f;
    AIC_CUDA_CHECK_REDUCE(cudaMemcpy(out.data_ptr<float>(), &init_val, sizeof(float), cudaMemcpyHostToDevice));

    Tensor contig = a.contiguous();
    int64_t n = contig.numel();
    int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 1024));

    reduce_min_kernel<<<blocks, threads>>>(contig.data_ptr<float>(), out.data_ptr<float>(), n);
    AIC_CUDA_CHECK_REDUCE(cudaGetLastError());
    return out;
}

Tensor sum(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cuda(a, dim, keepdim, ReduceOp::Sum);
}

Tensor mean(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cuda(a, dim, keepdim, ReduceOp::Mean);
}

Tensor max(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cuda(a, dim, keepdim, ReduceOp::Max);
}

Tensor min(const Tensor& a, int dim, bool keepdim) {
    return reduce_dim_cuda(a, dim, keepdim, ReduceOp::Min);
}

}  // namespace aic::cuda
