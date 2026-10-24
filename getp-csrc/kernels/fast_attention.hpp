// Fast-path decode attention: fp32 query, bf16 KV cache, fp32 softmax with the
// head's sink, bf16 output for the next matrix multiply.
//
// This first version keeps the exact kernel's shape (one 64-thread block per
// head and sequence, scores staged in LDS) with fp32 math and parallel sums.
#pragma once

#include <hip/hip_runtime.h>

#include "fast_common.hpp"

// Grid (n_heads, batch), 64 threads. KV cache (batch, slots, kv_dim); a
// sliding layer (window > 0) keeps position t in slot t % window.
// Dynamic shared memory: (positions + 1) floats.
__global__ __launch_bounds__(64) void attention_bf16_kernel(
    const float *__restrict__ qkv, int row_stride, const bf16_t *__restrict__ kc,
    const bf16_t *__restrict__ vc, int slots, int kv_dim, int kv_mul,
    const float *__restrict__ sinks, bf16_t *__restrict__ out, int out_stride,
    int pos, int window) {
  constexpr int D = 64;
  extern __shared__ float att[];
  __shared__ float q[D];
  __shared__ float red[D];
  const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  q[tid] = qkv[(size_t)b * row_stride + h * D + tid] * 0.125f; // 1/sqrt(64)
  const int t0 = window > 0 ? max(0, pos - window + 1) : 0;
  const int n = pos - t0 + 1;
  const size_t kv_base = (size_t)b * slots * kv_dim + (h / kv_mul) * D;
  __syncthreads();

  for (int j = tid; j < n; j += D) {
    const int t = t0 + j;
    const bf16_t *k = kc + kv_base + (size_t)(window > 0 ? t % window : t) * kv_dim;
    float score = 0.0f;
    for (int i = 0; i < D; i++)
      score = fmaf(q[i], bf16_to_f32(k[i]), score);
    att[j] = score;
  }
  if (tid == 0)
    att[n] = sinks[h];
  __syncthreads();

  float m = -INFINITY;
  for (int j = tid; j <= n; j += D)
    m = fmaxf(m, att[j]);
  red[tid] = m;
  __syncthreads();
  for (int s = D / 2; s > 0; s >>= 1) {
    if (tid < s)
      red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float max_val = red[0];
  __syncthreads();
  float sum = 0.0f;
  for (int j = tid; j <= n; j += D) {
    const float e = __expf(att[j] - max_val);
    att[j] = e;
    sum += e;
  }
  red[tid] = sum;
  __syncthreads();
  for (int s = D / 2; s > 0; s >>= 1) {
    if (tid < s)
      red[tid] += red[tid + s];
    __syncthreads();
  }
  const float inv_sum = 1.0f / red[0];

  float acc = 0.0f;
  for (int j = 0; j < n; j++) {
    const int t = t0 + j;
    acc = fmaf(att[j], bf16_to_f32(vc[kv_base + (size_t)(window > 0 ? t % window : t) * kv_dim + tid]), acc);
  }
  out[(size_t)b * out_stride + h * D + tid] = f32_to_bf16(acc * inv_sum);
}

static void attention_bf16(hipStream_t s, const float *qkv, int row_stride,
                           const bf16_t *kc, const bf16_t *vc, int slots,
                           int kv_dim, int n_heads, int kv_mul,
                           const float *sinks, bf16_t *out, int out_stride,
                           int B, int pos, int window) {
  const int n_pos = window > 0 ? min(pos + 1, window) : pos + 1;
  attention_bf16_kernel<<<dim3(n_heads, B), 64, (n_pos + 1) * sizeof(float), s>>>(
      qkv, row_stride, kc, vc, slots, kv_dim, kv_mul, sinks, out, out_stride,
      pos, window);
}
