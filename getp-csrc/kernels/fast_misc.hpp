// Fast-path elementwise and per-row kernels: fp32 math, bf16 where the result
// feeds a matrix multiply or the KV cache.
#pragma once

#include <hip/hip_runtime.h>

#include "fast_attention.hpp"
#include "fast_common.hpp"
#include "misc.hpp"

// RMSNorm of fp32 rows into bf16 rows. One block (256 threads) per row.
__global__ void rmsnorm_bf16_kernel(const float *__restrict__ x,
                                    const float *__restrict__ weight,
                                    bf16_t *__restrict__ out, int H) {
  __shared__ float part[256];
  const int b = blockIdx.x;
  const float *xr = x + (size_t)b * H;
  float ss = 0.0f;
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    ss = fmaf(xr[j], xr[j], ss);
  part[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(part[0] / H + 1e-5f);
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    out[(size_t)b * H + j] = f32_to_bf16(weight[j] * (inv * xr[j]));
}

// RoPE on the query and key heads of each fp32 qkv row, then the rotated key
// is stored as bf16 in the cache at `slot`, in the chunk64 layout of
// fast_attention.hpp (slots a multiple of 64, head_dim 64), and the value as
// int8 codes vq plus one scale per head vs (vq_index_chunk64,
// vs_index_chunk64): scale = max|v| / 127, code = round(v * 127 / max|v|).
// blockDim must be a multiple of 64, so that each wave holds one head's value.
__global__ void rope_kv_bf16_kernel(float *__restrict__ qkv, int row_stride,
                                    const float *__restrict__ cosv,
                                    const float *__restrict__ sinv,
                                    int n_q_heads, int n_kv_heads, int head_dim,
                                    bf16_t *__restrict__ kc,
                                    int8_t *__restrict__ vq, float *__restrict__ vs,
                                    int slots, int slot) {
  const int b = blockIdx.x;
  const int half = head_dim / 2;
  const int kv_dim = n_kv_heads * head_dim;
  float *row = qkv + (size_t)b * row_stride;
  float *k = row + n_q_heads * head_dim;
  const float *v = k + kv_dim;
  for (int idx = threadIdx.x; idx < (n_q_heads + n_kv_heads) * half; idx += blockDim.x) {
    const int h = idx / half, i = idx % half;
    float *xh = row + h * head_dim;
    const float x1 = xh[i], x2 = xh[half + i];
    const float c = cosv[i], s = sinv[i];
    const float o1 = x1 * c - x2 * s, o2 = x2 * c + x1 * s;
    xh[i] = o1;
    xh[half + i] = o2;
    if (h >= n_q_heads) {
      const int kh = h - n_q_heads;
      kc[kv_index_chunk64(b, slot, kh, i, slots, kv_dim)] = f32_to_bf16(o1);
      kc[kv_index_chunk64(b, slot, kh, half + i, slots, kv_dim)] = f32_to_bf16(o2);
    }
  }
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    const float x = v[i];
    float amax = fabsf(x);
#pragma unroll
    for (int o = 32; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor(amax, o));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const float c = fminf(fmaxf(rintf(x * inv), -127.0f), 127.0f);
    const int g = i / head_dim, d = i % head_dim;
    vq[vq_index_chunk64(b, slot, g, d, slots, kv_dim)] = (int8_t)c;
    if (d == 0)
      vs[vs_index_chunk64(b, slot, g, slots, kv_dim)] = amax / 127.0f;
  }
}

// SwiGLU of gpt-oss on rows of mlp1 output (gate/up interleaved) into bf16.
__global__ void swiglu_bf16_kernel(const float *__restrict__ Y,
                                   bf16_t *__restrict__ G, size_t R, int I,
                                   float limit) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= R * I)
    return;
  const size_t r = idx / I, j = idx % I;
  float val = fminf(Y[r * 2 * I + 2 * j], limit);
  float up = fminf(fmaxf(Y[r * 2 * I + 2 * j + 1], -limit), limit);
  val = val / (1.0f + __expf(-1.702f * val));
  G[idx] = f32_to_bf16(val * (up + 1.0f));
}

// Xg[row, :] = t[row_pair[row] / k, :] for bf16 rows.
__global__ void gather_bf16_kernel(const bf16_t *__restrict__ t,
                                   const int *__restrict__ row_pair, int k,
                                   bf16_t *__restrict__ Xg, int H) {
  const int row = blockIdx.x;
  const bf16_t *src = t + (size_t)(row_pair[row] / k) * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    Xg[(size_t)row * H + i] = src[i];
}

// x[b, :] += sum_j topk_w[b, j] * Z[pair_row[b, j], :].
__global__ void moe_combine_kernel(float *__restrict__ x,
                                   const float *__restrict__ Z,
                                   const int *__restrict__ pair_row,
                                   const float *__restrict__ topk_w, int k,
                                   int H) {
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < k; j++)
      acc = fmaf(Z[(size_t)pair_row[b * k + j] * H + i], topk_w[b * k + j], acc);
    x[(size_t)b * H + i] += acc;
  }
}

// RMSNorm of x's rows into bf16, the arithmetic of rmsnorm_bf16_kernel, for a
// row whose new values each thread holds in xs (elements t, t + 256, ...).
// Called by a whole 256-thread block.
__device__ inline void rmsnorm_row_bf16(const float *xs, int count,
                                        const float *__restrict__ weight,
                                        bf16_t *__restrict__ out, int H) {
  __shared__ float part[256];
  float ss = 0.0f;
  for (int c = 0; c < count; c++)
    ss = fmaf(xs[c], xs[c], ss);
  part[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(part[0] / H + 1e-5f);
  for (int c = 0, j = threadIdx.x; c < count; c++, j += blockDim.x)
    out[j] = f32_to_bf16(weight[j] * (inv * xs[c]));
}

#define FUSED_ROW_MAX 16 // H <= 256 * 16

// x += y, then RMSNorm(x) into out: add_inplace_kernel followed by
// rmsnorm_bf16_kernel, bit for bit. One block of 256 threads per row.
__global__ __launch_bounds__(256) void add_rmsnorm_bf16_kernel(
    float *__restrict__ x, const float *__restrict__ y,
    const float *__restrict__ weight, bf16_t *__restrict__ out, int H) {
  const int b = blockIdx.x;
  float *xr = x + (size_t)b * H;
  const float *yr = y + (size_t)b * H;
  float xs[FUSED_ROW_MAX];
  int count = 0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const float v = __fadd_rn(xr[j], yr[j]);
    xr[j] = v;
    xs[count++] = v;
  }
  rmsnorm_row_bf16(xs, count, weight, out + (size_t)b * H, H);
}
