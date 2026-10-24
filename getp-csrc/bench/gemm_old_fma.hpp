// The first fast-path GEMM (plain fp32 FMAs on 64x64 LDS tiles), kept only so
// bench/gemm_bench.cpp can time it as the "before" number.
#pragma once

#include <hip/hip_runtime.h>

#include "../kernels/fast_common.hpp"
#include "../kernels/gemm.hpp"

template <class W>
__device__ void old_gemm_tile_f32acc(const bf16_t *__restrict__ A, int M, W w,
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

template <class W>
__global__ __launch_bounds__(256) void old_gemm_f32acc_kernel(
    const bf16_t *__restrict__ A, int M, W w, const float *__restrict__ bias,
    float *__restrict__ C, int N, int K) {
  old_gemm_tile_f32acc(A, M, w, bias, C, N, K, blockIdx.y * GEMM_BM,
                   blockIdx.x * GEMM_BN);
}

// C = A · W^T (+ bias). A (M, K) bf16, W (N, K) bf16, C (M, N) fp32.
static void gemm_bf16_old_fma(hipStream_t s, const bf16_t *A, int M, const bf16_t *W,
                      const float *bias, float *C, int N, int K) {
  dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  old_gemm_f32acc_kernel<<<grid, 256, 0, s>>>(A, M, Bf16Rows{W, K}, bias, C, N, K);
}
