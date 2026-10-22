// src/cuda/cuda_gemm.cu
// High-performance CUDA GEMM implementation: Shared Memory, 2D Register Tiling, and WMMA Tensor Cores.
#include "cuda/cuda_gemm.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <vector>

#define AIC_CUDA_CHECK_GEMM(call)                                                          \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            AIC_THROW(CudaError, "CUDA error in GEMM kernel: ",                            \
                      cudaGetErrorString(err__));                                          \
        }                                                                                  \
    } while (false)

namespace aic::cuda {

namespace {

// ============================================================================
// 1. Naive Baseline Kernel
// ============================================================================
__global__ void gemm_naive_kernel(
    int64_t M, int64_t N, int64_t K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p) {
            sum += A[row * K + p] * B[p * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 2. Shared Memory Tiled Kernel (32x32 Tile with Bank Conflict Padding)
// ============================================================================
constexpr int TILE_SIZE = 32;

__global__ void gemm_shared_tiled_kernel(
    int64_t M, int64_t N, int64_t K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    // 33 columns avoids 32-way shared memory bank conflicts
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================================
// 3. 2D Block Tiling + Register Tiling Kernel (64x64 Block, 4x4 Thread Tile)
// ============================================================================
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void gemm_register_tiled_kernel(
    int64_t M, int64_t N, int64_t K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    __shared__ float s_A[BM][BK + 1];
    __shared__ float s_B[BK][BN + 1];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    float reg_c[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    int linear_thread = ty * (BN / TN) + tx; // 0..255

    for (int k_step = 0; k_step < K; k_step += BK) {
        // Load A into shared memory (BM x BK = 64 x 8 = 512 elements; 2 elements per thread)
        int a_load_row0 = linear_thread / BK;
        int a_load_col0 = linear_thread % BK;
        int a_load_row1 = a_load_row0 + 32;

        if (block_row + a_load_row0 < M && k_step + a_load_col0 < K)
            s_A[a_load_row0][a_load_col0] = A[(block_row + a_load_row0) * K + (k_step + a_load_col0)];
        else
            s_A[a_load_row0][a_load_col0] = 0.0f;

        if (block_row + a_load_row1 < M && k_step + a_load_col0 < K)
            s_A[a_load_row1][a_load_col0] = A[(block_row + a_load_row1) * K + (k_step + a_load_col0)];
        else
            s_A[a_load_row1][a_load_col0] = 0.0f;

        // Load B into shared memory (BK x BN = 8 x 64 = 512 elements; 2 elements per thread)
        int b_load_row0 = linear_thread / BN;
        int b_load_col0 = linear_thread % BN;
        int b_load_row1 = b_load_row0 + 4;

        if (k_step + b_load_row0 < K && block_col + b_load_col0 < N)
            s_B[b_load_row0][b_load_col0] = B[(k_step + b_load_row0) * N + (block_col + b_load_col0)];
        else
            s_B[b_load_row0][b_load_col0] = 0.0f;

        if (k_step + b_load_row1 < K && block_col + b_load_col0 < N)
            s_B[b_load_row1][b_load_col0] = B[(k_step + b_load_row1) * N + (block_col + b_load_col0)];
        else
            s_B[b_load_row1][b_load_col0] = 0.0f;

        __syncthreads();

        // Accumulate outer product of BK steps
        #pragma unroll
        for (int p = 0; p < BK; ++p) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = s_A[ty * TM + i][p];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = s_B[p][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write out results
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = block_row + ty * TM + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = block_col + tx * TN + j;
                if (c < N) {
                    C[r * N + c] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// 4. Hardware Tensor Core WMMA Kernel (sm_120 Blackwell Compatible)
// ============================================================================
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void gemm_wmma_kernel(
    int64_t M, int64_t N, int64_t K,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C) {

    // Each block has 1 warp (32 threads) computing a 16x16 tile
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int a_row = warpM * WMMA_M;
    int b_col = warpN * WMMA_N;

    for (int k = 0; k < K; k += WMMA_K) {
        if (a_row < M && k < K && b_col < N) {
            wmma::load_matrix_sync(a_frag, A + a_row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + b_col, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    if (a_row < M && b_col < N) {
        wmma::store_matrix_sync(C + a_row * N + b_col, c_frag, N, wmma::mem_row_major);
    }
}

__global__ void float_to_half_kernel(const float* __restrict__ in, __half* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __float2half(in[i]);
    }
}

}  // namespace

Tensor gemm_2d(const Tensor& a, const Tensor& b, GemmAlgorithm algorithm) {
    AIC_CHECK(a.rank() == 2 && b.rank() == 2, InvalidArgument,
              "CUDA gemm_2d requires 2D tensors");
    AIC_CHECK(a.dim(1) == b.dim(0), ShapeMismatch,
              "CUDA Matrix multiplication dimension mismatch: [", a.dim(0), ", ", a.dim(1),
              "] vs [", b.dim(0), ", ", b.dim(1), "]");
    AIC_CHECK(a.device() == b.device(), DeviceMismatch,
              "CUDA GEMM tensors must be on the same device");

    AIC_CUDA_CHECK_GEMM(cudaSetDevice(a.device().index));

    int64_t M = a.dim(0);
    int64_t K = a.dim(1);
    int64_t N = b.dim(1);

    Tensor c = Tensor::zeros(Shape({M, N}), DType::Float32, a.device());
    if (M == 0 || K == 0 || N == 0) return c;

    Tensor a_contig = a.contiguous();
    Tensor b_contig = b.contiguous();

    // Select algorithm
    if (algorithm == GemmAlgorithm::Auto) {
        if (M >= 64 && N >= 64 && K >= 16) {
            algorithm = GemmAlgorithm::CudaRegTiled;
        } else {
            algorithm = GemmAlgorithm::CudaTiled;
        }
    }

    if (algorithm == GemmAlgorithm::CudaRegTiled) {
        dim3 block(BN / TN, BM / TM); // 16 x 16 = 256 threads
        dim3 grid((static_cast<unsigned int>(N) + BN - 1) / BN,
                  (static_cast<unsigned int>(M) + BM - 1) / BM);

        gemm_register_tiled_kernel<<<grid, block>>>(
            M, N, K,
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            c.data_ptr<float>());
        AIC_CUDA_CHECK_GEMM(cudaGetLastError());
    } else if (algorithm == GemmAlgorithm::CudaTensorCore && (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0)) {
        // Convert to half on device for Tensor Core execution
        __half* d_half_a = nullptr;
        __half* d_half_b = nullptr;
        AIC_CUDA_CHECK_GEMM(cudaMalloc(&d_half_a, a.numel() * sizeof(__half)));
        AIC_CUDA_CHECK_GEMM(cudaMalloc(&d_half_b, b.numel() * sizeof(__half)));

        int threads = 256;
        int blocks_a = static_cast<int>((a.numel() + threads - 1) / threads);
        int blocks_b = static_cast<int>((b.numel() + threads - 1) / threads);

        float_to_half_kernel<<<blocks_a, threads>>>(a_contig.data_ptr<float>(), d_half_a, a.numel());
        float_to_half_kernel<<<blocks_b, threads>>>(b_contig.data_ptr<float>(), d_half_b, b.numel());

        dim3 block(32, 1);
        dim3 grid(static_cast<unsigned int>(N / WMMA_N), static_cast<unsigned int>(M / WMMA_M));

        gemm_wmma_kernel<<<grid, block>>>(M, N, K, d_half_a, d_half_b, c.data_ptr<float>());
        AIC_CUDA_CHECK_GEMM(cudaGetLastError());

        cudaFree(d_half_a);
        cudaFree(d_half_b);
    } else {
        // Shared memory tiled (32x32)
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((static_cast<unsigned int>(N) + TILE_SIZE - 1) / TILE_SIZE,
                  (static_cast<unsigned int>(M) + TILE_SIZE - 1) / TILE_SIZE);

        gemm_shared_tiled_kernel<<<grid, block>>>(
            M, N, K,
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            c.data_ptr<float>());
        AIC_CUDA_CHECK_GEMM(cudaGetLastError());
    }

    return c;
}

Tensor matmul(const Tensor& a, const Tensor& b, GemmAlgorithm algorithm) {
    int ra = a.rank();
    int rb = b.rank();

    AIC_CHECK(ra >= 2 && rb >= 2, InvalidArgument,
              "CUDA matmul requires tensors with rank >= 2");

    if (ra == 2 && rb == 2) {
        return gemm_2d(a, b, algorithm);
    }

    // Batched matrix multiplication on CUDA
    int64_t M = a.dim(ra - 2);
    int64_t Ka = a.dim(ra - 1);
    int64_t Kb = b.dim(rb - 2);
    int64_t N = b.dim(rb - 1);

    AIC_CHECK(Ka == Kb, ShapeMismatch,
              "Inner dimensions must match in CUDA batched matmul: ", Ka, " vs ", Kb);

    std::vector<int64_t> batch_a(static_cast<size_t>(ra - 2));
    for (int i = 0; i < ra - 2; ++i) batch_a[static_cast<size_t>(i)] = a.dim(i);
    std::vector<int64_t> batch_b(static_cast<size_t>(rb - 2));
    for (int i = 0; i < rb - 2; ++i) batch_b[static_cast<size_t>(i)] = b.dim(i);

    Shape batch_shape = Shape::broadcast(Shape(std::move(batch_a)), Shape(std::move(batch_b)));
    int batch_rank = batch_shape.rank();
    int64_t num_batches = batch_shape.numel();

    std::vector<int64_t> out_dims(static_cast<size_t>(batch_rank + 2));
    for (int i = 0; i < batch_rank; ++i) out_dims[static_cast<size_t>(i)] = batch_shape[i];
    out_dims[static_cast<size_t>(batch_rank)] = M;
    out_dims[static_cast<size_t>(batch_rank + 1)] = N;

    Tensor out = Tensor::empty(Shape(std::move(out_dims)), a.dtype(), a.device());
    if (num_batches == 0 || M == 0 || N == 0) return out;

    int64_t a_mat_stride = M * Ka;
    int64_t b_mat_stride = Kb * N;
    int64_t out_mat_stride = M * N;

    Tensor a_contig = a.contiguous();
    Tensor b_contig = b.contiguous();

    for (int64_t b_idx = 0; b_idx < num_batches; ++b_idx) {
        int64_t linear = b_idx;
        int64_t off_a = 0;
        int64_t off_b = 0;
        int64_t stride_a = a_mat_stride;
        int64_t stride_b = b_mat_stride;

        for (int k = batch_rank - 1; k >= 0; --k) {
            int64_t coord = linear % batch_shape[k];
            linear /= batch_shape[k];

            int a_dim_k = (k < ra - 2) ? static_cast<int>(a.dim(k)) : 1;
            int b_dim_k = (k < rb - 2) ? static_cast<int>(b.dim(k)) : 1;

            if (a_dim_k > 1) off_a += coord * stride_a;
            if (b_dim_k > 1) off_b += coord * stride_b;

            stride_a *= a_dim_k;
            stride_b *= b_dim_k;
        }

        Tensor a_slice(a_contig.storage(), a_contig.storage_offset() + off_a, Shape({M, Ka}), {Ka, 1}, a.dtype());
        Tensor b_slice(b_contig.storage(), b_contig.storage_offset() + off_b, Shape({Kb, N}), {N, 1}, b.dtype());

        Tensor c_slice = gemm_2d(a_slice, b_slice, algorithm);
        AIC_CUDA_CHECK_GEMM(cudaMemcpy(
            static_cast<char*>(out.raw_data()) + b_idx * out_mat_stride * sizeof(float),
            c_slice.raw_data(),
            static_cast<size_t>(out_mat_stride) * sizeof(float),
            cudaMemcpyDeviceToDevice));
    }

    return out;
}

}  // namespace aic::cuda
