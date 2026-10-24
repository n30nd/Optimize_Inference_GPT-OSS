// Small per-row kernels: embedding lookup, RMSNorm, RoPE, KV-cache write,
// residual add, argmax. The *_exact kernels round exactly like run.cpp's CPU
// code (see exact_math.hpp).
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "exact_math.hpp"

__device__ inline float bf16_to_float(uint16_t h) {
  return __uint_as_float((uint32_t)h << 16);
}

// x[b, :] = embedding[tokens[b], :]. One block per row.
__global__ void embed_kernel(const uint16_t *__restrict__ emb,
                             const int *__restrict__ tokens,
                             float *__restrict__ x, int H) {
  const int b = blockIdx.x;
  const uint16_t *row = emb + (size_t)tokens[b] * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    x[(size_t)b * H + i] = bf16_to_float(row[i]);
}

// rmsnorm() from run.cpp. One block per row; the squares are computed in
// parallel, then summed in order in fp64 by one thread, as the CPU does.
// Dynamic shared memory: H floats.
__global__ void rmsnorm_exact(const float *__restrict__ x,
                              const float *__restrict__ weight,
                              float *__restrict__ out, int H) {
  extern __shared__ float sq[];
  __shared__ double inv;
  const int b = blockIdx.x;
  const float *xr = x + (size_t)b * H;
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    sq[j] = __fmul_rn(xr[j], xr[j]);
  __syncthreads();
  if (threadIdx.x == 0) {
    double ss = 0.0;
    for (int j = 0; j < H; j++)
      ss = __dadd_rn(ss, (double)sq[j]);
    ss = __ddiv_rn(ss, (double)H);
    ss = __dadd_rn(ss, (double)1e-5f);
    inv = (double)__fdiv_rn(1.0f, sqrt_cpu((float)ss));
  }
  __syncthreads();
  float *o = out + (size_t)b * H;
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    o[j] = (float)__dmul_rn((double)weight[j], __dmul_rn(inv, (double)xr[j]));
}

// apply_rotary_emb() from run.cpp on the first n_heads heads of each row
// (the query heads followed by the key heads), with the CPU build's FMAs:
// x1*c - x2*s and x2*c + x1*s each fuse their first product.
__global__ void rope_exact(float *__restrict__ qkv, int row_stride,
                           const float *__restrict__ cosv,
                           const float *__restrict__ sinv, int n_heads,
                           int head_dim) {
  const int b = blockIdx.x;
  const int half = head_dim / 2;
  float *row = qkv + (size_t)b * row_stride;
  for (int idx = threadIdx.x; idx < n_heads * half; idx += blockDim.x) {
    const int h = idx / half, i = idx % half;
    float *x = row + h * head_dim;
    const float x1 = x[i], x2 = x[half + i];
    const float c = cosv[i], s = sinv[i];
    x[i] = __fmaf_rn(x1, c, -__fmul_rn(x2, s));
    x[half + i] = __fmaf_rn(x2, c, __fmul_rn(x1, s));
  }
}

// Copies each row's key and value into the cache at `slot`.
// Cache layout: (batch, slots, kv_dim).
__global__ void kv_write_kernel(const float *__restrict__ qkv, int row_stride,
                                int k_off, int v_off, float *__restrict__ kc,
                                float *__restrict__ vc, int slots, int slot,
                                int kv_dim) {
  const int b = blockIdx.x;
  const float *row = qkv + (size_t)b * row_stride;
  const size_t dst = ((size_t)b * slots + slot) * kv_dim;
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    kc[dst + i] = row[k_off + i];
    vc[dst + i] = row[v_off + i];
  }
}

// x += y, elementwise.
__global__ void add_inplace_kernel(float *__restrict__ x,
                                   const float *__restrict__ y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = __fadd_rn(x[i], y[i]);
}

// out[b] = index of the largest logit of row b; the first one on ties, like
// sample_argmax() in run.cpp. One block of 256 threads per row.
__global__ void argmax_kernel(const float *__restrict__ logits, int V,
                              int *__restrict__ out) {
  __shared__ float best_v[256];
  __shared__ int best_i[256];
  const int b = blockIdx.x;
  const float *row = logits + (size_t)b * V;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float v = row[i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  best_v[threadIdx.x] = bv;
  best_i[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = best_v[threadIdx.x + s];
      const int oi = best_i[threadIdx.x + s];
      if (ov > best_v[threadIdx.x] ||
          (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
        best_v[threadIdx.x] = ov;
        best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[b] = best_i[0] == 0x7fffffff ? 0 : best_i[0];
}
