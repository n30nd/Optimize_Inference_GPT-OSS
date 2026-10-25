// Mixture-of-experts steps when a layer's experts are split between two GCDs
// (expert parallelism). Each GCD routes its own tokens; each GCD runs only its
// own experts, on the rows from both GCDs' tokens that chose them; each GCD
// then adds up its own tokens' 4 expert outputs. The exact path reads the
// partner's routing, rows and expert outputs straight from the partner's
// memory; on the fast path the partner writes its normed rows into this
// GCD's memory, and its expert outputs arrive as one partial sum per token.
//
// Pair indices: p = token * k + choice. A row of the grouped input refers to
// an own pair as p and to a partner pair as cap * k + p, cap = the GCD's
// batch capacity: then row_pair[row] / k is the token's row in one table
// that holds this GCD's tokens in rows 0 .. cap-1 and the partner's from row
// cap on (the fast path keeps such a table of normed rows, t16).
// With no partner (n_peer = 0, e_begin = 0, e_count = all experts) these do
// the same as moe_group_kernel / moe_gather_kernel / moe_combine_exact.
#pragma once

#include <hip/hip_runtime.h>

#include "fast_common.hpp"
#include "fast_misc.hpp"
#include "moe.hpp"

// Signals between the two GCDs of a pair. A GCD tells its partner "the data
// I wrote before this point may be read" by raising a counter in its own
// memory; the partner's stream runs a one-thread kernel that waits until the
// counter reaches the expected value. The counters live in fine-grained
// memory (kept coherent between the GCDs), and the fences make
// the data around them visible: the release writes this GCD's L2 cache back
// to memory before the counter goes up, the acquire drops the partner's data
// that this GCD's L2 may still hold from earlier reads. Same meaning as
// hipEventRecord / hipStreamWaitEvent across two GCDs, but cheaper on MI250:
// about 13 microseconds per exchange for both kernels together, against
// about 18 for the event wait.
__global__ void ep_signal_kernel(int *flag, int value) {
  __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void ep_wait_kernel(const int *flag, int value) {
  while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value)
    __builtin_amdgcn_s_sleep(2);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

// Grouping the rows for experts e_begin .. e_begin+e_count-1: own pairs first,
// then the partner's, each in pair order. offsets (e_count + 1) are local.
// own_row[p] / peer_row[p]: row of that pair in this GCD's grouped buffers,
// or -1 if another GCD owns the expert. One 256-thread block per local
// expert: moe_count_ep_kernel, then moe_place_ep_kernel, or both in one
// (moe_group_ep_kernel).

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
// order, from row *base on (row_pair gets pair0 + p).
__device__ inline void moe_place_pairs(const int *__restrict__ topk, int n, int ge,
                                       int pair0, int *__restrict__ row_pair,
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
      row_pair[row] = pair0 + p;
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
    int n_peer, int peer_pair0, int e_begin, int e_count, const int *__restrict__ counts,
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
  moe_place_pairs(topk_own, n_own, ge, 0, row_pair, own_row, &base);
  moe_place_pairs(topk_peer, n_peer, ge, peer_pair0, row_pair, peer_row, &base);
}

// moe_count_ep_kernel and moe_place_ep_kernel in one kernel, with the same
// results. Block e (one per local expert) reads every pair once, the own
// pairs then the partner's as one list: thread t takes the S consecutive
// pairs from S t on (16-byte loads, all issued together). It counts the pairs
// of the local experts before e (their sum is the expert's first row) and
// its pairs of e; a scan over the threads then gives every pair of e its row,
// in pair order. The code is kept short on purpose: a small kernel starts
// with its code out of the instruction cache, and each instruction fetch
// then waits on memory. Needs n_own and n_peer multiples of 4.
template <int S>
__global__ __launch_bounds__(MOE_GROUP_THREADS) void moe_group_ep_kernel(
    const int *__restrict__ topk_own, int n_own, const int *__restrict__ topk_peer,
    int n_peer, int peer_pair0, int e_begin, int e_count, int *__restrict__ counts,
    int *__restrict__ offsets, int *__restrict__ row_pair, int *__restrict__ own_row,
    int *__restrict__ peer_row) {
  constexpr int NW = MOE_GROUP_THREADS / 64;
  __shared__ int s_below[NW], s_hits[NW];
  const int e = blockIdx.x, ge = e_begin + e;
  const int t = threadIdx.x, lane = t % 64, wave = t / 64;
  const int n = n_own + n_peer, i0 = t * S;
  int x[S];
#pragma unroll
  for (int q = 0; q < S; q += 4) { // past the end: read pairs 0..3 again, unused
    const int i = i0 + q;
    const bool peer = i >= n_own && i < n;
    const int4 v = *(const int4 *)((peer ? topk_peer : topk_own) +
                                   (peer ? i - n_own : i < n ? i : 0));
    x[q] = v.x, x[q + 1] = v.y, x[q + 2] = v.z, x[q + 3] = v.w;
  }
  int below = 0;
  unsigned mine = 0; // bit q: pair i0 + q chose e
#pragma unroll
  for (int q = 0; q < S; q++) {
    const bool in = i0 + q < n;
    below += in && x[q] >= e_begin && x[q] < ge;
    mine |= (unsigned)(in && x[q] == ge) << q;
  }
  const int hits = __popc(mine);
  // exclusive scan of hits over the block, sum of below
  int scan = hits;
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) {
    const int y = __shfl_up(scan, off);
    scan += lane >= off ? y : 0;
  }
#pragma unroll
  for (int off = 32; off > 0; off >>= 1)
    below += __shfl_xor(below, off);
  if (lane == 63)
    s_hits[wave] = scan;
  if (lane == 0)
    s_below[wave] = below;
  __syncthreads();
  int start = 0, row = scan - hits, count = 0;
  for (int w = 0; w < NW; w++) {
    start += s_below[w];
    row += w < wave ? s_hits[w] : 0;
    count += s_hits[w];
  }
  row += start;
  for (; mine; mine &= mine - 1, row++) {
    const int i = i0 + __ffs(mine) - 1;
    if (i < n_own) {
      row_pair[row] = i;
      own_row[i] = row;
    } else {
      row_pair[row] = peer_pair0 + (i - n_own);
      peer_row[i - n_own] = row;
    }
  }
  if (t == 0) {
    offsets[e] = start;
    counts[e] = count;
    if (e == e_count - 1)
      offsets[e_count] = start + count;
  }
  // the pairs of experts on the other GCD get row -1, shared out over the
  // blocks as in moe_place_ep_kernel (these reads were just made: cache hits)
  for (int p = e * MOE_GROUP_THREADS + t; p < n_own; p += e_count * MOE_GROUP_THREADS) {
    const int y = topk_own[p] - e_begin;
    if (y < 0 || y >= e_count)
      own_row[p] = -1;
  }
  for (int p = e * MOE_GROUP_THREADS + t; p < n_peer; p += e_count * MOE_GROUP_THREADS) {
    const int y = topk_peer[p] - e_begin;
    if (y < 0 || y >= e_count)
      peer_row[p] = -1;
  }
}

// Groups the pairs as moe_count_ep_kernel + moe_place_ep_kernel do, with
// moe_group_ep_kernel when the pairs fit (at most 32 per thread).
// peer_pair0 = cap * k (see the pair indices above).
static void moe_group_ep(hipStream_t s, const int *topk_own, int n_own, const int *topk_peer,
                         int n_peer, int peer_pair0, int e_begin, int e_count, int *counts,
                         int *offsets, int *row_pair, int *own_row, int *peer_row) {
  const int per = (n_own + n_peer + MOE_GROUP_THREADS - 1) / MOE_GROUP_THREADS;
  const bool by4 = n_own % 4 == 0 && n_peer % 4 == 0;
#define MOE_GROUP_CASE(S)                                                                    \
  moe_group_ep_kernel<S><<<e_count, MOE_GROUP_THREADS, 0, s>>>(                              \
      topk_own, n_own, topk_peer, n_peer, peer_pair0, e_begin, e_count, counts, offsets,      \
      row_pair, own_row, peer_row)
  if (by4 && per <= 8)
    MOE_GROUP_CASE(8); // gpt-oss-20b: 512 tokens x 4 choices
  else if (by4 && per <= 16)
    MOE_GROUP_CASE(16); // gpt-oss-120b: own and partner pairs
  else if (by4 && per <= 32)
    MOE_GROUP_CASE(32);
  else {
    moe_count_ep_kernel<<<e_count, MOE_GROUP_THREADS, 0, s>>>(topk_own, n_own, topk_peer,
                                                               n_peer, e_begin, counts);
    moe_place_ep_kernel<<<e_count, MOE_GROUP_THREADS, 0, s>>>(
        topk_own, n_own, topk_peer, n_peer, peer_pair0, e_begin, e_count, counts, offsets,
        row_pair, own_row, peer_row);
  }
#undef MOE_GROUP_CASE
}

// Xg[row, :] = the normed activations of the row's token, from this GCD's
// x_own or the partner's x_peer. One block per row; n_rows is read from
// offsets[e_count] on the device.
template <class T>
__global__ void moe_gather_ep_kernel(const T *__restrict__ x_own,
                                     const T *__restrict__ x_peer,
                                     const int *__restrict__ row_pair,
                                     const int *__restrict__ offsets,
                                     int e_count, int k, int cap, T *__restrict__ Xg,
                                     int H) {
  const int row = blockIdx.x;
  if (row >= offsets[e_count])
    return;
  const int tok = row_pair[row] / k;
  const T *src = tok < cap ? x_own + (size_t)tok * H : x_peer + (size_t)(tok - cap) * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    Xg[(size_t)row * H + i] = src[i];
}

// x[b, :] += sum of b's expert outputs times their weights. A row of an expert
// this GCD owns is in Z_own at own_row[p]; otherwise it is in the partner's
// Z_peer at the partner's peer_row[p]. kExact: add the experts in increasing
// expert order with FMA, as forward() in run.cpp builds e_agg (same as
// moe_combine_exact); otherwise in choice order (same as moe_combine_kernel).
// When topk_e holds renumbered experts (router_topk_wave_kernel's slot_of),
// phys_of maps them back to the checkpoint's numbers for that order; null
// otherwise. One block per token.
template <bool kExact>
__global__ void moe_combine_ep_kernel(float *__restrict__ x,
                                      const float *__restrict__ Z_own,
                                      const int *__restrict__ own_row,
                                      const float *__restrict__ Z_peer,
                                      const int *__restrict__ peer_rows_of_mine,
                                      const float *__restrict__ topk_w,
                                      const int *__restrict__ topk_e,
                                      int e_begin, int e_count, int k, int H,
                                      const int *__restrict__ phys_of) {
  const int b = blockIdx.x;
  int order[MOE_MAX_TOPK], key[MOE_MAX_TOPK];
  for (int j = 0; j < k; j++) {
    order[j] = j;
    const int e = topk_e[(size_t)b * k + j];
    key[j] = phys_of && e >= 0 ? phys_of[e] : e;
  }
  if (kExact)
    for (int i = 1; i < k; i++)
      for (int j = i; j > 0 && key[order[j]] < key[order[j - 1]]; j--) {
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

// moe_gather_ep_kernel<bf16_t> with 16-byte copies, all loads of a thread
// issued together (vector t, t + 256, ... of the row, NV of them).
// H a multiple of 8.
template <int NV>
__global__ __launch_bounds__(256) void moe_gather_ep_bf16_kernel(
    const bf16_t *__restrict__ x_own, const bf16_t *__restrict__ x_peer,
    const int *__restrict__ row_pair, const int *__restrict__ offsets, int e_count, int k,
    int cap, bf16_t *__restrict__ Xg, int H) {
  const int row = blockIdx.x, nv = H / 8;
  const int n_rows = offsets[e_count], tok = row_pair[row] / k; // loaded together
  if (row >= n_rows)
    return;
  const uint4 *src = (const uint4 *)(tok < cap ? x_own + (size_t)tok * H
                                               : x_peer + (size_t)(tok - cap) * H);
  uint4 v[NV];
#pragma unroll
  for (int c = 0; c < NV; c++)
    v[c] = src[min((int)threadIdx.x + 256 * c, nv - 1)];
  uint4 *dst = (uint4 *)(Xg + (size_t)row * H);
#pragma unroll
  for (int c = 0; c < NV; c++)
    if ((int)threadIdx.x + 256 * c < nv)
      dst[threadIdx.x + 256 * c] = v[c];
}

static void moe_gather_ep_bf16(hipStream_t s, int R, const bf16_t *x_own,
                               const bf16_t *x_peer, const int *row_pair, const int *offsets,
                               int e_count, int k, int cap, bf16_t *Xg, int H) {
  if (H % 8 == 0 && H / 8 <= 2 * 256) // gpt-oss: H = 2880, 360 vectors
    moe_gather_ep_bf16_kernel<2><<<R, 256, 0, s>>>(x_own, x_peer, row_pair, offsets, e_count,
                                                   k, cap, Xg, H);
  else
    moe_gather_ep_kernel<bf16_t><<<R, 256, 0, s>>>(x_own, x_peer, row_pair, offsets, e_count,
                                                   k, cap, Xg, H);
}

// Fast path with a partner: for each of the partner's tokens, the weighted
// sum of the expert outputs this GCD computed for it (its choices of this
// GCD's experts, in choice order), so the partner reads one row per token
// instead of each expert row. S (B_peer, H) fp32; H a multiple of 4.
// KM >= k; NC >= H / 4 / 256: thread t handles float4 t, t + 256, ... of the
// row, and all its loads are issued before the first sum.
template <int KM, int NC>
__global__ __launch_bounds__(256) void moe_partial_ep_kernel(
    float *__restrict__ S, const float *__restrict__ Z_own, const int *__restrict__ peer_row,
    const float *__restrict__ topk_w_peer, const int *__restrict__ topk_e_peer, int e_begin,
    int e_count, int k, int H) {
  const int b = blockIdx.x, n4 = H / 4;
  bool mine[KM];
  float wgt[KM];
  float4 z[KM][NC];
#pragma unroll
  for (int j = 0; j < KM; j++) {
    mine[j] = false;
    if (j < k) {
      const int p = b * k + j;
      const int e = topk_e_peer[p] - e_begin;
      mine[j] = e >= 0 && e < e_count;
      if (mine[j]) {
        wgt[j] = topk_w_peer[p];
        const float4 *row = (const float4 *)(Z_own + (size_t)peer_row[p] * H);
#pragma unroll
        for (int c = 0; c < NC; c++)
          z[j][c] = row[min((int)threadIdx.x + 256 * c, n4 - 1)];
      }
    }
  }
  float4 *out = (float4 *)(S + (size_t)b * H);
#pragma unroll
  for (int c = 0; c < NC; c++) {
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < KM; j++)
      if (mine[j]) {
        acc.x = fmaf(z[j][c].x, wgt[j], acc.x);
        acc.y = fmaf(z[j][c].y, wgt[j], acc.y);
        acc.z = fmaf(z[j][c].z, wgt[j], acc.z);
        acc.w = fmaf(z[j][c].w, wgt[j], acc.w);
      }
    if ((int)threadIdx.x + 256 * c < n4)
      out[threadIdx.x + 256 * c] = acc;
  }
}

static void moe_partial_ep(hipStream_t s, int B_peer, float *S, const float *Z_own,
                           const int *peer_row, const float *topk_w_peer,
                           const int *topk_e_peer, int e_begin, int e_count, int k, int H) {
  if (k <= 4 && (H / 4 + 255) / 256 <= 3) // gpt-oss: H = 2880
    moe_partial_ep_kernel<4, 3><<<B_peer, 256, 0, s>>>(S, Z_own, peer_row, topk_w_peer,
                                                       topk_e_peer, e_begin, e_count, k, H);
  else
    moe_partial_ep_kernel<MOE_MAX_TOPK, FUSED_ROW_MAX / 4><<<B_peer, 256, 0, s>>>(
        S, Z_own, peer_row, topk_w_peer, topk_e_peer, e_begin, e_count, k, H);
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
// KM >= k: the loads of all k expert rows are issued together; NC as in
// load_row256.
template <int KM, int NC>
__global__ __launch_bounds__(256) void moe_combine_rmsnorm_kernel(
    float *__restrict__ x, const float *__restrict__ Z, const int *__restrict__ own_row,
    const float *__restrict__ topk_w, const int *__restrict__ topk_e, int k, int H,
    const float *__restrict__ norm_w, bf16_t *__restrict__ out) {
  const int b = blockIdx.x;
  const bool live = topk_e[(size_t)b * k] >= 0;
  float *xr = x + (size_t)b * H;
  float xs[NC], wv[NC];
  load_row256(xr, H, xs);
  load_row256(norm_w, H, wv);
  if (live) {
    float zs[KM][NC], wgt[KM];
#pragma unroll
    for (int j = 0; j < KM; j++)
      if (j < k) {
        wgt[j] = topk_w[(size_t)b * k + j];
        load_row256(Z + (size_t)own_row[b * k + j] * H, H, zs[j]);
      }
#pragma unroll
    for (int c = 0; c < NC; c++) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < KM; j++)
        if (j < k)
          acc = fmaf(zs[j][c], wgt[j], acc);
      xs[c] += acc;
    }
  }
  const float inv = rms_inv_row(xs, H);
  if (live)
#pragma unroll
    for (int c = 0; c < NC; c++)
      if (row_has<NC>(c, H))
        xr[threadIdx.x + 256 * c] = xs[c];
  rmsnorm_store_bf16(xs, wv, inv, out + (size_t)b * H, H);
}

// Fast path, experts split: moe_combine_partial_kernel followed by
// rmsnorm_bf16_kernel of the updated row with norm_w. One 256-thread block
// per token; KM, NC as above.
template <int KM, int NC>
__global__ __launch_bounds__(256) void moe_combine_partial_rmsnorm_kernel(
    float *__restrict__ x, const float *__restrict__ Z_own, const int *__restrict__ own_row,
    const float *__restrict__ S_peer, const float *__restrict__ topk_w,
    const int *__restrict__ topk_e, int e_begin, int e_count, int k, int H,
    const float *__restrict__ norm_w, bf16_t *__restrict__ out) {
  const int b = blockIdx.x;
  const bool live = topk_e[(size_t)b * k] >= 0;
  float *xr = x + (size_t)b * H;
  float xs[NC], wv[NC];
  load_row256(xr, H, xs);
  load_row256(norm_w, H, wv);
  if (live) {
    // this GCD's experts of the token, in choice order
    float zs[KM][NC], sp[NC], wgt[KM];
    bool mine[KM];
#pragma unroll
    for (int j = 0; j < KM; j++) {
      mine[j] = false;
      if (j < k) {
        const int p = b * k + j;
        const int e = topk_e[p] - e_begin;
        mine[j] = e >= 0 && e < e_count;
        if (mine[j]) {
          wgt[j] = topk_w[p];
          load_row256(Z_own + (size_t)own_row[p] * H, H, zs[j]);
        }
      }
    }
    load_row256(S_peer + (size_t)b * H, H, sp);
#pragma unroll
    for (int c = 0; c < NC; c++) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < KM; j++)
        if (mine[j])
          acc = fmaf(zs[j][c], wgt[j], acc);
      xs[c] += acc + sp[c];
    }
  }
  const float inv = rms_inv_row(xs, H);
  if (live)
#pragma unroll
    for (int c = 0; c < NC; c++)
      if (row_has<NC>(c, H))
        xr[threadIdx.x + 256 * c] = xs[c];
  rmsnorm_store_bf16(xs, wv, inv, out + (size_t)b * H, H);
}

// Launchers: the kernels above for gpt-oss (k <= 4, H = 2880), or for any
// k <= MOE_MAX_TOPK and H <= 256 * FUSED_ROW_MAX.
static void moe_combine_rmsnorm(hipStream_t s, int B, float *x, const float *Z,
                                const int *own_row, const float *topk_w, const int *topk_e,
                                int k, int H, const float *norm_w, bf16_t *out) {
  if (k <= 4 && (H + 255) / 256 == 12)
    moe_combine_rmsnorm_kernel<4, 12><<<B, 256, 0, s>>>(x, Z, own_row, topk_w, topk_e, k, H,
                                                        norm_w, out);
  else
    moe_combine_rmsnorm_kernel<MOE_MAX_TOPK, FUSED_ROW_MAX><<<B, 256, 0, s>>>(
        x, Z, own_row, topk_w, topk_e, k, H, norm_w, out);
}

static void moe_combine_partial_rmsnorm(hipStream_t s, int B, float *x, const float *Z_own,
                                        const int *own_row, const float *S_peer,
                                        const float *topk_w, const int *topk_e, int e_begin,
                                        int e_count, int k, int H, const float *norm_w,
                                        bf16_t *out) {
  if (k <= 4 && (H + 255) / 256 == 12)
    moe_combine_partial_rmsnorm_kernel<4, 12><<<B, 256, 0, s>>>(
        x, Z_own, own_row, S_peer, topk_w, topk_e, e_begin, e_count, k, H, norm_w, out);
  else
    moe_combine_partial_rmsnorm_kernel<MOE_MAX_TOPK, FUSED_ROW_MAX><<<B, 256, 0, s>>>(
        x, Z_own, own_row, S_peer, topk_w, topk_e, e_begin, e_count, k, H, norm_w, out);
}
