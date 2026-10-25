// Mixture-of-experts steps when a layer's experts are split between two GCDs
// (expert parallelism). Each GCD routes its own tokens; each GCD runs only its
// own experts, on the rows from both GCDs' tokens that chose them; each GCD
// then adds up its own tokens' 4 expert outputs, reading the rows computed by
// its partner straight from the partner's memory.
//
// Pair indices: p = token * k + choice. A row of the grouped input refers to
// an own pair as p and to a partner pair as -(p + 1).
// With no partner (n_peer = 0, e_begin = 0, e_count = all experts) these do
// the same as moe_group_kernel / moe_gather_kernel / moe_combine_exact.
#pragma once

#include <hip/hip_runtime.h>

#include "fast_common.hpp"
#include "moe.hpp"

// Grouping the rows for experts e_begin .. e_begin+e_count-1: own pairs first,
// then the partner's, each in pair order. offsets (e_count + 1) are local.
// own_row[p] / peer_row[p]: row of that pair in this GCD's grouped buffers,
// or -1 if another GCD owns the expert. Two kernels, one 256-thread block per
// local expert: moe_count_ep_kernel, then moe_place_ep_kernel.

#define MOE_GROUP_THREADS 256

// counts[e] = number of own and partner pairs that chose local expert e.
__global__ __launch_bounds__(MOE_GROUP_THREADS) void moe_count_ep_kernel(
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

// Writes the rows of the pairs in topk[0..n) that chose expert `ge`, in pair
// order, from row *base on (partner pairs are stored in row_pair as -(p + 1)).
__device__ inline void moe_place_pairs(const int *__restrict__ topk, int n, int ge,
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

// Local expert blockIdx.x starts after the rows of all lower experts. Each
// block also marks its share of the pairs whose expert another GCD holds.
__global__ __launch_bounds__(MOE_GROUP_THREADS) void moe_place_ep_kernel(
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
  moe_place_pairs(topk_own, n_own, ge, false, row_pair, own_row, &base);
  moe_place_pairs(topk_peer, n_peer, ge, true, row_pair, peer_row, &base);
}

// Xg[row, :] = the normed activations of the row's token, from this GCD's
// x_own or the partner's x_peer. One block per row; n_rows is read from
// offsets[e_count] on the device.
template <class T>
__global__ void moe_gather_ep_kernel(const T *__restrict__ x_own,
                                     const T *__restrict__ x_peer,
                                     const int *__restrict__ row_pair,
                                     const int *__restrict__ offsets,
                                     int e_count, int k, T *__restrict__ Xg,
                                     int H) {
  const int row = blockIdx.x;
  if (row >= offsets[e_count])
    return;
  const int p = row_pair[row];
  const T *src = p >= 0 ? x_own + (size_t)(p / k) * H
                        : x_peer + (size_t)((-p - 1) / k) * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    Xg[(size_t)row * H + i] = src[i];
}

// x[b, :] += sum of b's expert outputs times their weights. A row of an expert
// this GCD owns is in Z_own at own_row[p]; otherwise it is in the partner's
// Z_peer at the partner's peer_row[p]. kExact: add the experts in increasing
// expert order with FMA, as forward() in run.cpp builds e_agg (same as
// moe_combine_exact); otherwise in choice order (same as moe_combine_kernel).
// One block per token.
template <bool kExact>
__global__ void moe_combine_ep_kernel(float *__restrict__ x,
                                      const float *__restrict__ Z_own,
                                      const int *__restrict__ own_row,
                                      const float *__restrict__ Z_peer,
                                      const int *__restrict__ peer_rows_of_mine,
                                      const float *__restrict__ topk_w,
                                      const int *__restrict__ topk_e,
                                      int e_begin, int e_count, int k, int H) {
  const int b = blockIdx.x;
  int order[MOE_MAX_TOPK];
  for (int j = 0; j < k; j++)
    order[j] = j;
  if (kExact)
    for (int i = 1; i < k; i++)
      for (int j = i; j > 0 && topk_e[(size_t)b * k + order[j]] <
                                   topk_e[(size_t)b * k + order[j - 1]];
           j--) {
        int t = order[j];
        order[j] = order[j - 1];
        order[j - 1] = t;
      }
  const float *rows[MOE_MAX_TOPK];
  for (int j = 0; j < k; j++) {
    const int p = b * k + order[j];
    const int e = topk_e[p] - e_begin;
    rows[j] = topk_e[p] < 0 ? nullptr // finished request: no expert
              : (e >= 0 && e < e_count) ? Z_own + (size_t)own_row[p] * H
                                         : Z_peer + (size_t)peer_rows_of_mine[p] * H;
  }
  if (rows[0] == nullptr)
    return;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.0f;
    if (kExact) {
      for (int j = 0; j < k; j++)
        acc = __fmaf_rn(rows[j][i], topk_w[(size_t)b * k + order[j]], acc);
      x[(size_t)b * H + i] = __fadd_rn(x[(size_t)b * H + i], acc);
    } else {
      for (int j = 0; j < k; j++)
        acc = fmaf(rows[j][i], topk_w[(size_t)b * k + j], acc);
      x[(size_t)b * H + i] += acc;
    }
  }
}

// Fast path with a partner: for each of the partner's tokens, the weighted
// sum of the expert outputs this GCD computed for it (its choices of this
// GCD's experts, in choice order), so the partner reads one row per token
// instead of each expert row. S (B_peer, H) fp32; H a multiple of 4.
__global__ void moe_partial_ep_kernel(float *__restrict__ S,
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

// Fast path with a partner: x[b, :] += the weighted sum of b's outputs from
// this GCD's experts (choice order) + the partner's partial sum S_peer[b, :].
__global__ void moe_combine_partial_kernel(float *__restrict__ x,
                                           const float *__restrict__ Z_own,
                                           const int *__restrict__ own_row,
                                           const float *__restrict__ S_peer,
                                           const float *__restrict__ topk_w,
                                           const int *__restrict__ topk_e,
                                           int e_begin, int e_count, int k, int H) {
  const int b = blockIdx.x;
  const float4 *rows[MOE_MAX_TOPK];
  float wgt[MOE_MAX_TOPK];
  int n = 0;
  for (int j = 0; j < k; j++) {
    const int p = b * k + j;
    const int e = topk_e[p] - e_begin;
    if (e >= 0 && e < e_count) {
      rows[n] = (const float4 *)(Z_own + (size_t)own_row[p] * H);
      wgt[n++] = topk_w[p];
    }
  }
  if (topk_e[b * k] < 0) // finished request: no expert
    return;
  float4 *xr = (float4 *)(x + (size_t)b * H);
  const float4 *sp = (const float4 *)(S_peer + (size_t)b * H);
  for (int i = threadIdx.x; i < H / 4; i += blockDim.x) {
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; j++) {
      const float4 z = rows[j][i];
      acc.x = fmaf(z.x, wgt[j], acc.x);
      acc.y = fmaf(z.y, wgt[j], acc.y);
      acc.z = fmaf(z.z, wgt[j], acc.z);
      acc.w = fmaf(z.w, wgt[j], acc.w);
    }
    const float4 s = sp[i];
    float4 v = xr[i];
    v.x += acc.x + s.x;
    v.y += acc.y + s.y;
    v.z += acc.z + s.z;
    v.w += acc.w + s.w;
    xr[i] = v;
  }
}

// Fast path, experts not split: moe_combine_ep_kernel<false> followed by
// rmsnorm_bf16_kernel of the updated row with norm_w (the next layer's
// attention norm, or the final norm), bit for bit. One 256-thread block per
// token; a finished request (expert -1) keeps x and still gets its norm.
__global__ __launch_bounds__(256) void moe_combine_rmsnorm_kernel(
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
  rmsnorm_row_bf16(xs, count, norm_w, out + (size_t)b * H, H);
}

// Fast path, experts split: moe_combine_partial_kernel followed by
// rmsnorm_bf16_kernel of the updated row with norm_w. One 256-thread block
// per token.
__global__ __launch_bounds__(256) void moe_combine_partial_rmsnorm_kernel(
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
  rmsnorm_row_bf16(xs, count, norm_w, out + (size_t)b * H, H);
}
