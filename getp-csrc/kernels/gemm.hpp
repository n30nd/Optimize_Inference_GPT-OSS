// Matrix multiplies C[m, n] = sum_k A[m, k] * W[n, k] (+ bias[n]) with fp32
// activations A (M, K) and bf16 or MXFP4 weights W (N, K).
//
// The *_exact kernels reproduce matmul() in run.cpp bit for bit: every output
// is summed over k in increasing order, as fp32 products added into an fp64
// total, then rounded to fp32; the bias is added afterwards in fp32.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "misc.hpp"

#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 32

__constant__ float kFp4Lut[16] = {+0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f,
                                  +4.0f, +6.0f, -0.0f, -0.5f, -1.0f, -1.5f,
                                  -2.0f, -3.0f, -4.0f, -6.0f};

// W stored as bf16, row-major (N, K).
struct Bf16Rows {
  const uint16_t *w;
  int K;
  __device__ float operator()(int n, int k) const {
    return bf16_to_float(w[(size_t)n * K + k]);
  }
};

// W stored as MXFP4: blocks (N, K/32, 16) holding two 4-bit codes per byte
// (low nibble first) and scales (N, K/32), one power of two per 32 values.
struct Mxfp4Rows {
  const uint8_t *blocks;
  const uint8_t *scales;
  int K;
  __device__ float operator()(int n, int k) const {
    const size_t g = (size_t)n * (K / 32) + k / 32;
    const uint8_t byte = blocks[g * 16 + (k % 32) / 2];
    const uint8_t code = (k & 1) ? (byte >> 4) : (byte & 0x0F);
    return __fmul_rn(kFp4Lut[code], ldexpf(1.0f, (int)scales[g] - 127));
  }
};

// One 64x64 output tile at (m0, n0), 256 threads, each owning 4x4 outputs.
// Rows >= M and columns >= N are padded with zeros and not written.
template <class W>
__device__ void gemm_tile_exact(const float *__restrict__ A, int M, W w,
                                const float *__restrict__ bias,
                                float *__restrict__ C, int N, int K, int m0,
                                int n0) {
  __shared__ float As[GEMM_BK][GEMM_BM + 1];
  __shared__ float Ws[GEMM_BK][GEMM_BN + 1];
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  double acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      acc[i][j] = 0.0;

  for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
    for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += 256) {
      const int r = idx / GEMM_BK, kk = idx % GEMM_BK;
      const int m = m0 + r, n = n0 + r;
      As[kk][r] = m < M ? A[(size_t)m * K + k0 + kk] : 0.0f;
      Ws[kk][r] = n < N ? w(n, k0 + kk) : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < GEMM_BK; kk++) {
      float a[4], b[4];
      for (int i = 0; i < 4; i++) {
        a[i] = As[kk][ty * 4 + i];
        b[i] = Ws[kk][tx * 4 + i];
      }
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          acc[i][j] = __dadd_rn(acc[i][j], (double)__fmul_rn(a[i], b[j]));
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; i++) {
    const int m = m0 + ty * 4 + i;
    if (m >= M)
      continue;
    for (int j = 0; j < 4; j++) {
      const int n = n0 + tx * 4 + j;
      if (n >= N)
        continue;
      float c = (float)acc[i][j];
      if (bias)
        c = __fadd_rn(c, bias[n]);
      C[(size_t)m * N + n] = c;
    }
  }
}

template <class W>
__global__ __launch_bounds__(256) void gemm_exact_kernel(
    const float *__restrict__ A, int M, W w, const float *__restrict__ bias,
    float *__restrict__ C, int N, int K) {
  gemm_tile_exact(A, M, w, bias, C, N, K, blockIdx.y * GEMM_BM,
                  blockIdx.x * GEMM_BN);
}

// Expert e multiplies rows offsets[e] .. offsets[e+1] of A by its own MXFP4
// weight and bias. Grid: (N tiles, row tiles of the largest expert, experts).
__global__ __launch_bounds__(256) void gemm_grouped_mxfp4_exact_kernel(
    const float *__restrict__ A, const int *__restrict__ offsets,
    const uint8_t *__restrict__ blocks, const uint8_t *__restrict__ scales,
    const float *__restrict__ bias, float *__restrict__ C, int N, int K) {
  const int e = blockIdx.z;
  const int r0 = offsets[e];
  const int rows = offsets[e + 1] - r0;
  const int m0 = blockIdx.y * GEMM_BM;
  if (m0 >= rows)
    return;
  const size_t groups = (size_t)N * (K / 32);
  Mxfp4Rows w{blocks + e * groups * 16, scales + e * groups, K};
  gemm_tile_exact(A + (size_t)r0 * K, rows, w, bias + (size_t)e * N,
                  C + (size_t)r0 * N, N, K, m0, blockIdx.x * GEMM_BN);
}

static void gemm_bf16_exact(hipStream_t s, const float *A, int M,
                            const uint16_t *W, const float *bias, float *C,
                            int N, int K) {
  dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  gemm_exact_kernel<<<grid, 256, 0, s>>>(A, M, Bf16Rows{W, K}, bias, C, N, K);
}

// max_rows: an upper bound on any expert's row count (the batch size).
static void gemm_grouped_mxfp4_exact(hipStream_t s, const float *A,
                                     const int *offsets, int max_rows,
                                     int n_experts, const uint8_t *blocks,
                                     const uint8_t *scales, const float *bias,
                                     float *C, int N, int K) {
  dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (max_rows + GEMM_BM - 1) / GEMM_BM,
            n_experts);
  gemm_grouped_mxfp4_exact_kernel<<<grid, 256, 0, s>>>(A, offsets, blocks,
                                                       scales, bias, C, N, K);
}
