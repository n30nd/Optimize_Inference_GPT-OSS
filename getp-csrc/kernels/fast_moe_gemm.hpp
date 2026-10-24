// Fast-path expert matrix multiply: rows offsets[e] .. offsets[e+1] of A
// (bf16) times expert e's MXFP4 weight (N, K), plus its bias, into C (fp32).
//
// This first version decodes MXFP4 to fp32 in LDS and uses plain fp32 FMAs on
// 64x64 tiles.
#pragma once

#include <hip/hip_runtime.h>

#include "fast_common.hpp"
#include "gemm.hpp"

template <class W>
__device__ void moe_tile_f32acc(const bf16_t *__restrict__ A, int M, W w,
                                 const float *__restrict__ bias,
                                 float *__restrict__ C, int N, int K, int m0,
                                 int n0) {
  __shared__ float As[GEMM_BK][GEMM_BM + 1];
  __shared__ float Ws[GEMM_BK][GEMM_BN + 1];
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
    for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += 256) {
      const int r = idx / GEMM_BK, kk = idx % GEMM_BK;
      const int m = m0 + r, n = n0 + r;
      As[kk][r] = m < M ? bf16_to_f32(A[(size_t)m * K + k0 + kk]) : 0.0f;
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
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++) {
    const int m = m0 + ty * 4 + i;
    if (m >= M)
      continue;
    for (int j = 0; j < 4; j++) {
      const int n = n0 + tx * 4 + j;
      if (n < N)
        C[(size_t)m * N + n] = acc[i][j] + (bias ? bias[n] : 0.0f);
    }
  }
}

__global__ __launch_bounds__(256) void gemm_grouped_mxfp4_f32acc_kernel(
    const bf16_t *__restrict__ A, const int *__restrict__ offsets,
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
  moe_tile_f32acc(A + (size_t)r0 * K, rows, w, bias + (size_t)e * N,
                   C + (size_t)r0 * N, N, K, m0, blockIdx.x * GEMM_BN);
}

// Rows offsets[e] .. offsets[e+1] of A times expert e's MXFP4 weight, plus
// its bias. max_rows bounds any expert's row count.
static void gemm_grouped_mxfp4(hipStream_t s, const bf16_t *A,
                               const int *offsets, int max_rows, int n_experts,
                               const uint8_t *blocks, const uint8_t *scales,
                               const float *bias, float *C, int N, int K) {
  dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (max_rows + GEMM_BM - 1) / GEMM_BM,
            n_experts);
  gemm_grouped_mxfp4_f32acc_kernel<<<grid, 256, 0, s>>>(A, offsets, blocks,
                                                        scales, bias, C, N, K);
}
