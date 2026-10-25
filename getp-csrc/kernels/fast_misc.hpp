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

// RoPE on the key heads (and the query heads, with rotate_q) of each fp32
// qkv row: a rotated query goes back to qkv, the rotated key only to the
// cache at `slot`, as bf16 in the chunk64 layout of fast_attention.hpp (slots
// a multiple of 64), and the value to the cache as int8 codes vq plus one
// scale per head vs (vq_index_chunk64, vs_index_chunk64):
// scale = max|v| / 127, code = round(v * 127 / max|v|). head_dim 64.
// Without rotate_q the attention kernel rotates the query as it loads it
// (attention_q8v_chunk64 with cosv, sinv), with the same bits.
//
// One block of 256 threads per row: thread t rotates pair (i, i + 32),
// i = t % 32, of heads h0 + t / 32, h0 + t / 32 + 8, ... (NS of them; h0 = 0
// or the first key head) and quantizes values t, t + 256, ... (NV of them).
// Every load is issued first, then the results are computed, then stored:
// no load waits behind another load's use or behind a store (a wait for a
// load also waits for the stores issued before it).
template <int NS, int NV>
__global__ __launch_bounds__(256) void rope_kv_bf16_kernel(
    float *__restrict__ qkv, int row_stride, const float *__restrict__ cosv,
    const float *__restrict__ sinv, int n_q_heads, int n_kv_heads, int h0,
    bf16_t *__restrict__ kc, int8_t *__restrict__ vq, float *__restrict__ vs, int slots,
    int slot) {
  constexpr int D = 64, HALF = 32;
  const int b = blockIdx.x, t = threadIdx.x, i = t & 31;
  const int kv_dim = n_kv_heads * D, n_heads = n_q_heads + n_kv_heads;
  float *row = qkv + (size_t)b * row_stride;
  const float *v = row + n_heads * D;
  float x1[NS], x2[NS], xv[NV];
#pragma unroll
  for (int c = 0; c < NS; c++) { // past the last head: load it again, unused
    const int h = min(h0 + (t >> 5) + 8 * c, n_heads - 1);
    x1[c] = row[h * D + i];
    x2[c] = row[h * D + HALF + i];
  }
#pragma unroll
  for (int c = 0; c < NV; c++)
    xv[c] = v[min(t + 256 * c, kv_dim - 1)];
  const float cs = cosv[i], sn = sinv[i];

#pragma unroll
  for (int c = 0; c < NS; c++)
    rope_pair(x1[c], x2[c], cs, sn, x1[c], x2[c]);
  float sc[NV];
#pragma unroll
  for (int c = 0; c < NV; c++) { // whole waves: kv_dim is a multiple of 64
    float amax = fabsf(xv[c]);
#pragma unroll
    for (int o = 32; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor(amax, o));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    xv[c] = fminf(fmaxf(rintf(xv[c] * inv), -127.0f), 127.0f);
    sc[c] = amax / 127.0f;
  }

#pragma unroll
  for (int c = 0; c < NS; c++) {
    const int h = h0 + (t >> 5) + 8 * c;
    if (h < n_q_heads) {
      row[h * D + i] = x1[c];
      row[h * D + HALF + i] = x2[c];
    } else if (h < n_heads) {
      const int kh = h - n_q_heads;
      kc[kv_index_chunk64(b, slot, kh, i, slots, kv_dim)] = f32_to_bf16(x1[c]);
      kc[kv_index_chunk64(b, slot, kh, HALF + i, slots, kv_dim)] = f32_to_bf16(x2[c]);
    }
  }
#pragma unroll
  for (int c = 0; c < NV; c++) {
    const int j = t + 256 * c;
    if (j < kv_dim) {
      const int g = j / D, d = j % D;
      vq[vq_index_chunk64(b, slot, g, d, slots, kv_dim)] = (int8_t)xv[c];
      if (d == 0)
        vs[vs_index_chunk64(b, slot, g, slots, kv_dim)] = sc[c];
    }
  }
}

// rope_kv_bf16_kernel for B rows; head_dim must be 64, and n_q_heads +
// n_kv_heads at most 128, n_kv_heads at most 16.
static void rope_kv_bf16(hipStream_t s, int B, float *qkv, int row_stride, const float *cosv,
                         const float *sinv, int n_q_heads, int n_kv_heads, bf16_t *kc,
                         int8_t *vq, float *vs, int slots, int slot, bool rotate_q = true) {
  const int h0 = rotate_q ? 0 : n_q_heads;
  const int ns = (n_q_heads + n_kv_heads - h0 + 7) / 8, nv = (n_kv_heads * 64 + 255) / 256;
#define ROPE_KV_CASE(NS, NV)                                                                 \
  rope_kv_bf16_kernel<NS, NV><<<B, 256, 0, s>>>(qkv, row_stride, cosv, sinv, n_q_heads,       \
                                                n_kv_heads, h0, kc, vq, vs, slots, slot)
  if (ns == 1 && nv == 2) // gpt-oss, 8 key/value heads (query rotated by the attention)
    ROPE_KV_CASE(1, 2);
  else if (ns == 9 && nv == 2) // gpt-oss, 64 query heads and 8 key/value heads
    ROPE_KV_CASE(9, 2);
  else
    ROPE_KV_CASE(16, 4);
#undef ROPE_KV_CASE
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

#define FUSED_ROW_MAX 16 // H <= 256 * 16

// Sum of v over the 256 threads of a block, added as the halving tree of
// rmsnorm_bf16_kernel does it (part[t] += part[t + s] for s = 128, 64, ...,
// 1), so the result has the same bits; the last six levels run inside each
// wavefront with shuffles, and every thread gets the sum. part: 256 floats of
// LDS, free again only after the next __syncthreads.
__device__ inline float block_sum256(float v, float *part) {
  part[threadIdx.x] = v;
  __syncthreads();
  const int l = threadIdx.x & 63;
  float a = __fadd_rn(__fadd_rn(part[l], part[l + 128]), __fadd_rn(part[l + 64], part[l + 192]));
#pragma unroll
  for (int s = 32; s > 0; s >>= 1)
    a = __fadd_rn(a, __shfl_down(a, s));
  return __shfl(a, 0);
}

// Rows of H values handled by a 256-thread block: thread t owns elements
// t, t + 256, ..., NC of them. NC = ceil(H / 256) (only the last element may
// lie past the end of the row), or NC = FUSED_ROW_MAX for any H up to
// 256 * FUSED_ROW_MAX.
template <int NC> __device__ inline bool row_has(int c, int H) {
  return (NC < FUSED_ROW_MAX && c < NC - 1) || threadIdx.x + 256 * c < H;
}

// v[c] = p[t + 256 c] for this thread's elements. Loads past the end read
// the last element again (the value is not used), so that every load is
// issued unconditionally, before any of them is waited for.
template <int NC>
__device__ inline void load_row256(const float *__restrict__ p, int H, float (&v)[NC]) {
#pragma unroll
  for (int c = 0; c < NC; c++)
    v[c] = p[min(threadIdx.x + 256 * c, (unsigned)H - 1)];
}

// The norm of a row whose new values thread t holds in xs: 1 / RMS as
// rmsnorm_bf16_kernel computes it (same bits). Called by a whole 256-thread
// block.
template <int NC>
__device__ inline float rms_inv_row(const float (&xs)[NC], int H) {
  __shared__ float part[256];
  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < NC; c++)
    ss = row_has<NC>(c, H) ? fmaf(xs[c], xs[c], ss) : ss;
  return 1.0f / sqrtf(block_sum256(ss, part) / H + 1e-5f);
}

// out = RMSNorm(row) in bf16 from xs, the norm weights wv (load_row256) and
// inv = rms_inv_row(xs); the same values also to out2 if it is not null.
template <int NC>
__device__ inline void rmsnorm_store_bf16(const float (&xs)[NC], const float (&wv)[NC],
                                          float inv, bf16_t *__restrict__ out, int H,
                                          bf16_t *__restrict__ out2 = nullptr) {
#pragma unroll
  for (int c = 0; c < NC; c++)
    if (row_has<NC>(c, H)) {
      const bf16_t v = f32_to_bf16(wv[c] * (inv * xs[c]));
      out[threadIdx.x + 256 * c] = v;
      if (out2)
        out2[threadIdx.x + 256 * c] = v;
    }
}

// x += y, then RMSNorm(x) into out: add_inplace_kernel followed by
// rmsnorm_bf16_kernel, bit for bit. One block of 256 threads per row; all
// loads first, all stores last. out2 (may be null): a second copy of out,
// e.g. in the partner GCD's memory.
template <int NC>
__global__ __launch_bounds__(256) void add_rmsnorm_bf16_kernel(
    float *__restrict__ x, const float *__restrict__ y,
    const float *__restrict__ weight, bf16_t *__restrict__ out, int H,
    bf16_t *__restrict__ out2) {
  const int b = blockIdx.x;
  float *xr = x + (size_t)b * H;
  float xs[NC], ys[NC], wv[NC];
  load_row256(xr, H, xs);
  load_row256(y + (size_t)b * H, H, ys);
  load_row256(weight, H, wv);
#pragma unroll
  for (int c = 0; c < NC; c++)
    xs[c] = __fadd_rn(xs[c], ys[c]);
  const float inv = rms_inv_row(xs, H);
#pragma unroll
  for (int c = 0; c < NC; c++)
    if (row_has<NC>(c, H))
      xr[threadIdx.x + 256 * c] = xs[c];
  rmsnorm_store_bf16(xs, wv, inv, out + (size_t)b * H, H, out2 ? out2 + (size_t)b * H : nullptr);
}

// add_rmsnorm_bf16_kernel for B rows of H <= 256 * FUSED_ROW_MAX.
static void add_rmsnorm_bf16(hipStream_t s, int B, float *x, const float *y,
                             const float *weight, bf16_t *out, int H,
                             bf16_t *out2 = nullptr) {
  if ((H + 255) / 256 == 12) // gpt-oss: H = 2880
    add_rmsnorm_bf16_kernel<12><<<B, 256, 0, s>>>(x, y, weight, out, H, out2);
  else
    add_rmsnorm_bf16_kernel<FUSED_ROW_MAX><<<B, 256, 0, s>>>(x, y, weight, out, H, out2);
}
