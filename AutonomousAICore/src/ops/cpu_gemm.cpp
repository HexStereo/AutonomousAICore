// src/ops/cpu_gemm.cpp
// Cache-blocked, AVX2+FMA vectorized, and multi-threaded CPU GEMM implementation.
#include "ops/cpu_gemm.hpp"
#include "core/error.hpp"

#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__restrict__)
#define __restrict__ __restrict
#endif
#include <cstring>
#include <future>
#include <thread>
#include <vector>

namespace aic::cpu {

namespace {

// Vectorized microkernel: accumulates into 4 rows of 16 output floats (using 8 YMM registers)
void gemm_microkernel_4x16_avx2(
    int64_t dim_k,
    const float* __restrict__ a, int64_t lda,
    const float* __restrict__ b, int64_t ldb,
    float* __restrict__ c, int64_t ldc) {

    __m256 c00 = _mm256_loadu_ps(c + 0 * ldc + 0);
    __m256 c01 = _mm256_loadu_ps(c + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(c + 1 * ldc + 0);
    __m256 c11 = _mm256_loadu_ps(c + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(c + 2 * ldc + 0);
    __m256 c21 = _mm256_loadu_ps(c + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(c + 3 * ldc + 0);
    __m256 c31 = _mm256_loadu_ps(c + 3 * ldc + 8);

    for (int64_t p = 0; p < dim_k; ++p) {
        __m256 b0 = _mm256_loadu_ps(b + p * ldb + 0);
        __m256 b1 = _mm256_loadu_ps(b + p * ldb + 8);

        __m256 a0 = _mm256_set1_ps(a[0 * lda + p]);
        __m256 a1 = _mm256_set1_ps(a[1 * lda + p]);
        __m256 a2 = _mm256_set1_ps(a[2 * lda + p]);
        __m256 a3 = _mm256_set1_ps(a[3 * lda + p]);

        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    _mm256_storeu_ps(c + 0 * ldc + 0, c00);
    _mm256_storeu_ps(c + 0 * ldc + 8, c01);
    _mm256_storeu_ps(c + 1 * ldc + 0, c10);
    _mm256_storeu_ps(c + 1 * ldc + 8, c11);
    _mm256_storeu_ps(c + 2 * ldc + 0, c20);
    _mm256_storeu_ps(c + 2 * ldc + 8, c21);
    _mm256_storeu_ps(c + 3 * ldc + 0, c30);
    _mm256_storeu_ps(c + 3 * ldc + 8, c31);
}

// Block-tiled 2D single-thread tile computation
void gemm_block_tiled(
    int64_t m_start, int64_t m_end,
    int64_t dim_n, int64_t dim_k,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c) {

    constexpr int64_t TILE_N = 64;
    constexpr int64_t TILE_K = 64;

    for (int64_t k_block = 0; k_block < dim_k; k_block += TILE_K) {
        int64_t k_len = (k_block + TILE_K <= dim_k) ? TILE_K : (dim_k - k_block);

        for (int64_t n_block = 0; n_block < dim_n; n_block += TILE_N) {
            int64_t n_len = (n_block + TILE_N <= dim_n) ? TILE_N : (dim_n - n_block);

            // Micro-tiling: 4 rows x 16 cols
            int64_t m = m_start;
            for (; m + 4 <= m_end; m += 4) {
                int64_t n = n_block;
                for (; n + 16 <= n_block + n_len; n += 16) {
                    gemm_microkernel_4x16_avx2(
                        k_len,
                        a + m * dim_k + k_block, dim_k,
                        b + k_block * dim_n + n, dim_n,
                        c + m * dim_n + n, dim_n);
                }
                // Remainder columns
                for (; n < n_block + n_len; ++n) {
                    for (int64_t sub_m = 0; sub_m < 4; ++sub_m) {
                        float sum = 0.0f;
                        for (int64_t p = 0; p < k_len; ++p) {
                            sum += a[(m + sub_m) * dim_k + (k_block + p)] * b[(k_block + p) * dim_n + n];
                        }
                        c[(m + sub_m) * dim_n + n] += sum;
                    }
                }
            }
            // Remainder rows
            for (; m < m_end; ++m) {
                for (int64_t n = n_block; n < n_block + n_len; ++n) {
                    float sum = 0.0f;
                    for (int64_t p = 0; p < k_len; ++p) {
                        sum += a[m * dim_k + (k_block + p)] * b[(k_block + p) * dim_n + n];
                    }
                    c[m * dim_n + n] += sum;
                }
            }
        }
    }
}

}  // namespace

Tensor gemm_2d(const Tensor& a, const Tensor& b) {
    AIC_CHECK(a.rank() == 2 && b.rank() == 2, InvalidArgument,
              "gemm_2d requires 2D tensors, got ranks ", a.rank(), " and ", b.rank());
    AIC_CHECK(a.dim(1) == b.dim(0), ShapeMismatch,
              "Matrix multiplication dimension mismatch: [", a.dim(0), ", ", a.dim(1),
              "] vs [", b.dim(0), ", ", b.dim(1), "]");
    AIC_CHECK(a.dtype() == DType::Float32 && b.dtype() == DType::Float32, DTypeMismatch,
              "gemm_2d currently supports Float32");

    int64_t dim_m = a.dim(0);
    int64_t dim_k = a.dim(1);
    int64_t dim_n = b.dim(1);

    Tensor c = Tensor::zeros(Shape({dim_m, dim_n}), DType::Float32, Device::cpu());
    if (dim_m == 0 || dim_k == 0 || dim_n == 0) return c;

    Tensor a_contig = a.contiguous();
    Tensor b_contig = b.contiguous();

    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    // Multi-threaded row partitioning
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    int num_threads = (hw <= 0) ? 4 : hw;
    if (num_threads > 24) num_threads = 24;
    if (num_threads < 1) num_threads = 1;

    int64_t work_size = dim_m * dim_n * dim_k;
    if (dim_m < num_threads * 4 || work_size < 64 * 64 * 64) {
        num_threads = 1;
    }

    if (num_threads == 1) {
        gemm_block_tiled(0, dim_m, dim_n, dim_k, a_ptr, b_ptr, c_ptr);
    } else {
        std::vector<std::future<void>> futures;
        futures.reserve(static_cast<size_t>(num_threads));

        int64_t rows_per_thread = (dim_m + num_threads - 1) / num_threads;
        rows_per_thread = ((rows_per_thread + 3) / 4) * 4;

        for (int t = 0; t < num_threads; ++t) {
            int64_t m_start = t * rows_per_thread;
            int64_t m_end = (m_start + rows_per_thread <= dim_m) ? (m_start + rows_per_thread) : dim_m;
            if (m_start >= dim_m) break;

            futures.push_back(std::async(std::launch::async, [=]() {
                gemm_block_tiled(m_start, m_end, dim_n, dim_k, a_ptr, b_ptr, c_ptr);
            }));
        }

        for (auto& f : futures) {
            f.get();
        }
    }

    return c;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    int ra = a.rank();
    int rb = b.rank();

    AIC_CHECK(ra >= 2 && rb >= 2, InvalidArgument,
              "matmul requires tensors with rank >= 2, got ", ra, " and ", rb);

    if (ra == 2 && rb == 2) {
        return gemm_2d(a, b);
    }

    int64_t dim_m = a.dim(ra - 2);
    int64_t dim_ka = a.dim(ra - 1);
    int64_t dim_kb = b.dim(rb - 2);
    int64_t dim_n = b.dim(rb - 1);

    AIC_CHECK(dim_ka == dim_kb, ShapeMismatch,
              "Inner dimensions must match in batched matmul: ", dim_ka, " vs ", dim_kb);

    std::vector<int64_t> batch_a(static_cast<size_t>(ra - 2));
    for (int i = 0; i < ra - 2; ++i) batch_a[static_cast<size_t>(i)] = a.dim(i);
    std::vector<int64_t> batch_b(static_cast<size_t>(rb - 2));
    for (int i = 0; i < rb - 2; ++i) batch_b[static_cast<size_t>(i)] = b.dim(i);

    Shape batch_shape = Shape::broadcast(Shape(std::move(batch_a)), Shape(std::move(batch_b)));
    int batch_rank = batch_shape.rank();
    int64_t num_batches = batch_shape.numel();

    std::vector<int64_t> out_dims(static_cast<size_t>(batch_rank + 2));
    for (int i = 0; i < batch_rank; ++i) out_dims[static_cast<size_t>(i)] = batch_shape[i];
    out_dims[static_cast<size_t>(batch_rank)] = dim_m;
    out_dims[static_cast<size_t>(batch_rank + 1)] = dim_n;

    Tensor out = Tensor::empty(Shape(std::move(out_dims)), a.dtype(), a.device());
    if (num_batches == 0 || dim_m == 0 || dim_n == 0) return out;

    int64_t a_mat_stride = dim_m * dim_ka;
    int64_t b_mat_stride = dim_kb * dim_n;
    int64_t out_mat_stride = dim_m * dim_n;

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

        Tensor a_slice(a_contig.storage(), a_contig.storage_offset() + off_a, Shape({dim_m, dim_ka}), {dim_ka, 1}, a.dtype());
        Tensor b_slice(b_contig.storage(), b_contig.storage_offset() + off_b, Shape({dim_kb, dim_n}), {dim_n, 1}, b.dtype());

        Tensor c_slice = gemm_2d(a_slice, b_slice);
        std::memcpy(static_cast<char*>(out.raw_data()) + b_idx * out_mat_stride * sizeof(float),
                    c_slice.raw_data(),
                    static_cast<size_t>(out_mat_stride) * sizeof(float));
    }

    return out;
}

}  // namespace aic::cpu
