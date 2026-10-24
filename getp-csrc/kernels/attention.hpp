// Decode attention: one new token per sequence attends to its cached keys and
// values plus the head's attention sink.
#pragma once

#include <hip/hip_runtime.h>

#include "exact_math.hpp"

// The attention loop of forward() in run.cpp, bit for bit: scores summed in
// fp64, softmax over the scores and the sink with fp64 max/sum, values
// accumulated with FMA in time order. Sliding-window layers (window > 0) only
// visit the last `window` positions; the CPU's masked positions contribute
// exactly zero, so skipping them changes nothing.
//
// Grid (n_heads, batch), 64 threads (= head_dim). KV cache layout:
// (batch, slots, kv_dim); a sliding layer keeps position t in slot t % window.
// Dynamic shared memory: (positions + 1) floats.
__global__ __launch_bounds__(64) void attention_exact(
    const float *__restrict__ qkv, int row_stride, const float *__restrict__ kc,
    const float *__restrict__ vc, int slots, int kv_dim, int kv_mul,
    const float *__restrict__ sinks, float *__restrict__ out, int out_stride,
    int pos, int window) {
  constexpr int D = 64;
  extern __shared__ float att[];
  __shared__ float q[D];
  __shared__ float red[D];
  __shared__ double sum_sh;
  const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;

  q[tid] = qkv[(size_t)b * row_stride + h * D + tid];
  const int t0 = window > 0 ? max(0, pos - window + 1) : 0;
  const int n = pos - t0 + 1;
  const size_t kv_base = (size_t)b * slots * kv_dim + (h / kv_mul) * D;
  const double scale = (double)sqrt_cpu((float)D);
  __syncthreads();

  for (int j = tid; j < n; j += D) {
    const int t = t0 + j;
    const int slot = window > 0 ? t % window : t;
    const float *k = kc + kv_base + (size_t)slot * kv_dim;
    double score = 0.0;
    for (int i = 0; i < D; i++)
      score = __dadd_rn(score, (double)__fmul_rn(q[i], k[i]));
    att[j] = (float)__ddiv_rn(score, scale);
  }
  if (tid == 0)
    att[n] = sinks[h];
  __syncthreads();

  // softmax() from run.cpp over att[0..n]
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
  const double max_val = (double)red[0];
  for (int j = tid; j <= n; j += D)
    att[j] = expf_cpu((float)__dsub_rn((double)att[j], max_val));
  __syncthreads();
  if (tid == 0) {
    double sum = 0.0;
    for (int j = 0; j <= n; j++)
      sum = __dadd_rn(sum, (double)att[j]);
    sum_sh = sum;
  }
  __syncthreads();
  for (int j = tid; j <= n; j += D)
    att[j] = (float)__ddiv_rn((double)att[j], sum_sh);
  __syncthreads();

  float acc = 0.0f;
  for (int j = 0; j < n; j++) {
    const int t = t0 + j;
    const int slot = window > 0 ? t % window : t;
    acc = __fmaf_rn(att[j], vc[kv_base + (size_t)slot * kv_dim + tid], acc);
  }
  out[(size_t)b * out_stride + h * D + tid] = acc;
}
