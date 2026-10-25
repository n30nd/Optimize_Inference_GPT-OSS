// The small fast-path kernels as they were before they were rewritten to
// issue all their loads first (and with shorter code): the references that
// tests/small_kernels_test.cpp compares the engine's kernels with, bit for
// bit, and that bench/small_kernels_bench.cpp times them against.
#pragma once

#include <hip/hip_runtime.h>

#include "../kernels/fast_misc.hpp"
#include "../kernels/moe_ep.hpp"

__global__ void rope_kv_bf16_kernel_ref(float *__restrict__ qkv, int row_stride,
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

__device__ inline void rmsnorm_row_bf16_ref(const float *xs, int count,
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

__global__ __launch_bounds__(256) void add_rmsnorm_bf16_kernel_ref(
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
  rmsnorm_row_bf16_ref(xs, count, weight, out + (size_t)b * H, H);
}

__global__ __launch_bounds__(MOE_GROUP_THREADS) void moe_count_ep_kernel_ref(
    const int *__restrict__ topk_own, int n_own, const int *__restrict__ topk_peer,
    int n_peer, int e_begin, int *__restrict__ counts) {
  __shared__ int part[MOE_GROUP_THREADS / 64];
  const int ge = e_begin + blockIdx.x;
  int c = 0;
  for (int p = threadIdx.x; p < n_own; p += MOE_GROUP_THREADS)
    c += topk_own[p] == ge;
  for (int p = threadIdx.x; p < n_peer; p += MOE_GROUP_THREADS)
    c += topk_peer[p] == ge;
  for (int off = 32; off > 0; off >>= 1)
    c += __shfl_down(c, off);
  if (threadIdx.x % 64 == 0)
    part[threadIdx.x / 64] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int w = 0; w < MOE_GROUP_THREADS / 64; w++)
      total += part[w];
    counts[blockIdx.x] = total;
  }
}

__device__ inline void moe_place_pairs_ref(const int *__restrict__ topk, int n, int ge,
                                       bool partner, int *__restrict__ row_pair,
                                       int *__restrict__ rows, int *base) {
  __shared__ int wave_count[MOE_GROUP_THREADS / 64];
  const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
  for (int c0 = 0; c0 < n; c0 += MOE_GROUP_THREADS) {
    const int p = c0 + threadIdx.x;
    const bool hit = p < n && topk[p] == ge;
    const unsigned long long mask = __ballot(hit);
    if (lane == 0)
      wave_count[wave] = __popcll(mask);
    __syncthreads();
    int before = *base;
    for (int w = 0; w < wave; w++)
      before += wave_count[w];
    if (hit) {
      const int row = before + __popcll(mask & ((1ull << lane) - 1));
      row_pair[row] = partner ? -(p + 1) : p;
      rows[p] = row;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      for (int w = 0; w < MOE_GROUP_THREADS / 64; w++)
        *base += wave_count[w];
    __syncthreads();
  }
}

__global__ __launch_bounds__(MOE_GROUP_THREADS) void moe_place_ep_kernel_ref(
    const int *__restrict__ topk_own, int n_own, const int *__restrict__ topk_peer,
    int n_peer, int e_begin, int e_count, const int *__restrict__ counts,
    int *__restrict__ offsets, int *__restrict__ row_pair, int *__restrict__ own_row,
    int *__restrict__ peer_row) {
  __shared__ int base;
  const int e = blockIdx.x, ge = e_begin + e;
  if (threadIdx.x == 0) {
    int start = 0;
    for (int i = 0; i < e; i++)
      start += counts[i];
    base = start;
    offsets[e] = start;
    if (e == e_count - 1)
      offsets[e_count] = start + counts[e];
  }
  for (int p = e * MOE_GROUP_THREADS + threadIdx.x; p < n_own; p += e_count * MOE_GROUP_THREADS) {
    const int x = topk_own[p] - e_begin;
    if (x < 0 || x >= e_count)
      own_row[p] = -1;
  }
  for (int p = e * MOE_GROUP_THREADS + threadIdx.x; p < n_peer; p += e_count * MOE_GROUP_THREADS) {
    const int x = topk_peer[p] - e_begin;
    if (x < 0 || x >= e_count)
      peer_row[p] = -1;
  }
  __syncthreads();
  moe_place_pairs_ref(topk_own, n_own, ge, false, row_pair, own_row, &base);
  moe_place_pairs_ref(topk_peer, n_peer, ge, true, row_pair, peer_row, &base);
}

__global__ __launch_bounds__(256) void moe_combine_rmsnorm_kernel_ref(
    float *__restrict__ x, const float *__restrict__ Z, const int *__restrict__ own_row,
    const float *__restrict__ topk_w, const int *__restrict__ topk_e, int k, int H,
    const float *__restrict__ norm_w, bf16_t *__restrict__ out) {
  const int b = blockIdx.x;
  const float *rows[MOE_MAX_TOPK];
  const bool live = topk_e[(size_t)b * k] >= 0;
  for (int j = 0; j < k; j++)
    rows[j] = live ? Z + (size_t)own_row[b * k + j] * H : nullptr;
  float *xr = x + (size_t)b * H;
  float xs[FUSED_ROW_MAX];
  int count = 0;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = xr[i];
    if (live) {
      float acc = 0.0f;
      for (int j = 0; j < k; j++)
        acc = fmaf(rows[j][i], topk_w[(size_t)b * k + j], acc);
      v += acc;
      xr[i] = v;
    }
    xs[count++] = v;
  }
  rmsnorm_row_bf16_ref(xs, count, norm_w, out + (size_t)b * H, H);
}

__global__ __launch_bounds__(256) void moe_combine_partial_rmsnorm_kernel_ref(
    float *__restrict__ x, const float *__restrict__ Z_own, const int *__restrict__ own_row,
    const float *__restrict__ S_peer, const float *__restrict__ topk_w,
    const int *__restrict__ topk_e, int e_begin, int e_count, int k, int H,
    const float *__restrict__ norm_w, bf16_t *__restrict__ out) {
  const int b = blockIdx.x;
  const float *rows[MOE_MAX_TOPK];
  float wgt[MOE_MAX_TOPK];
  int n = 0;
  const bool live = topk_e[(size_t)b * k] >= 0;
  for (int j = 0; live && j < k; j++) {
    const int p = b * k + j;
    const int e = topk_e[p] - e_begin;
    if (e >= 0 && e < e_count) {
      rows[n] = Z_own + (size_t)own_row[p] * H;
      wgt[n++] = topk_w[p];
    }
  }
  float *xr = x + (size_t)b * H;
  const float *sp = S_peer + (size_t)b * H;
  float xs[FUSED_ROW_MAX];
  int count = 0;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = xr[i];
    if (live) {
      float acc = 0.0f;
      for (int j = 0; j < n; j++)
        acc = fmaf(rows[j][i], wgt[j], acc);
      v += acc + sp[i];
      xr[i] = v;
    }
    xs[count++] = v;
  }
  rmsnorm_row_bf16_ref(xs, count, norm_w, out + (size_t)b * H, H);
}

__global__ __launch_bounds__(64 * TOPK_WAVES) void router_topk_wave_kernel_ref(
    const float *__restrict__ logits, int E, int k, float *__restrict__ topk_w,
    int *__restrict__ topk_e, int B, const int *__restrict__ done,
    const int *__restrict__ slot_of) {
  const int lane = threadIdx.x % 64;
  const int b = blockIdx.x * TOPK_WAVES + threadIdx.x / 64;
  if (b >= B)
    return;
  if (done && done[b]) {
    if (lane < k) {
      topk_w[(size_t)b * k + lane] = 0.0f;
      topk_e[(size_t)b * k + lane] = -1;
    }
    return;
  }
  const float *r = logits + (size_t)b * E;
  float v[4];
  for (int j = 0; j < 4; j++)
    v[j] = lane + 64 * j < E ? r[lane + 64 * j] : 0.0f;
  unsigned taken = 0;
  int chosen[MOE_MAX_TOPK];
  float val[MOE_MAX_TOPK];
  for (int round = 0; round < k; round++) {
    float bv = 0.0f;
    int be = 0x7fffffff;
    for (int j = 0; j < 4; j++) {
      const int e = lane + 64 * j;
      if (e < E && !(taken & (1u << j)) && (be == 0x7fffffff || v[j] > bv)) {
        bv = v[j];
        be = e;
      }
    }
    for (int off = 32; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off);
      const int oe = __shfl_xor(be, off);
      if (oe != 0x7fffffff &&
          (be == 0x7fffffff || ov > bv || (ov == bv && oe < be))) {
        bv = ov;
        be = oe;
      }
    }
    if (be % 64 == lane)
      taken |= 1u << (be / 64);
    chosen[round] = be;
    val[round] = bv;
  }
  if (lane != 0)
    return;
  double max_val = val[0];
  for (int j = 1; j < k; j++)
    if (val[j] > max_val)
      max_val = val[j];
  double sum = 0.0;
  for (int j = 0; j < k; j++) {
    val[j] = expf_cpu((float)__dsub_rn((double)val[j], max_val));
    sum = __dadd_rn(sum, (double)val[j]);
  }
  for (int j = 0; j < k; j++) {
    topk_w[(size_t)b * k + j] = (float)__ddiv_rn((double)val[j], sum);
    topk_e[(size_t)b * k + j] = slot_of ? slot_of[chosen[j]] : chosen[j];
  }
}

__global__ void moe_partial_ep_kernel_ref(float *__restrict__ S,
                                      const float *__restrict__ Z_own,
                                      const int *__restrict__ peer_row,
                                      const float *__restrict__ topk_w_peer,
                                      const int *__restrict__ topk_e_peer,
                                      int e_begin, int e_count, int k, int H) {
  const int b = blockIdx.x;
  const float4 *rows[MOE_MAX_TOPK];
  float wgt[MOE_MAX_TOPK];
  int n = 0;
  for (int j = 0; j < k; j++) {
    const int p = b * k + j;
    const int e = topk_e_peer[p] - e_begin;
    if (e >= 0 && e < e_count) {
      rows[n] = (const float4 *)(Z_own + (size_t)peer_row[p] * H);
      wgt[n++] = topk_w_peer[p];
    }
  }
  float4 *out = (float4 *)(S + (size_t)b * H);
  for (int i = threadIdx.x; i < H / 4; i += blockDim.x) {
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; j++) {
      const float4 z = rows[j][i];
      acc.x = fmaf(z.x, wgt[j], acc.x);
      acc.y = fmaf(z.y, wgt[j], acc.y);
      acc.z = fmaf(z.z, wgt[j], acc.z);
      acc.w = fmaf(z.w, wgt[j], acc.w);
    }
    out[i] = acc;
  }
}
