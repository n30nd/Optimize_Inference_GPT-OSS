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

// Groups the rows for experts e_begin .. e_begin+e_count-1: own pairs first,
// then the partner's, each in pair order. offsets (e_count + 1) are local.
// own_row[p] / peer_row[p]: row of that pair in this GCD's grouped buffers,
// or -1 if another GCD owns the expert. One block of at least e_count threads.
__global__ void moe_group_ep_kernel(const int *__restrict__ topk_own, int n_own,
                                    const int *__restrict__ topk_peer, int n_peer,
                                    int e_begin, int e_count,
                                    int *__restrict__ offsets,
                                    int *__restrict__ row_pair,
                                    int *__restrict__ own_row,
                                    int *__restrict__ peer_row) {
  __shared__ int start[1025];
  const int e = threadIdx.x;
  const int ge = e_begin + e;
  if (e < e_count) {
    int count = 0;
    for (int p = 0; p < n_own; p++)
      count += topk_own[p] == ge;
    for (int p = 0; p < n_peer; p++)
      count += topk_peer[p] == ge;
    start[e + 1] = count;
  }
  for (int p = threadIdx.x; p < n_own; p += blockDim.x) {
    const int x = topk_own[p] - e_begin;
    if (x < 0 || x >= e_count)
      own_row[p] = -1;
  }
  for (int p = threadIdx.x; p < n_peer; p += blockDim.x) {
    const int x = topk_peer[p] - e_begin;
    if (x < 0 || x >= e_count)
      peer_row[p] = -1;
  }
  __syncthreads();
  if (e == 0) {
    start[0] = 0;
    for (int i = 1; i <= e_count; i++)
      start[i] += start[i - 1];
    for (int i = 0; i <= e_count; i++)
      offsets[i] = start[i];
  }
  __syncthreads();
  if (e < e_count) {
    int row = start[e];
    for (int p = 0; p < n_own; p++)
      if (topk_own[p] == ge) {
        row_pair[row] = p;
        own_row[p] = row++;
      }
    for (int p = 0; p < n_peer; p++)
      if (topk_peer[p] == ge) {
        row_pair[row] = -(p + 1);
        peer_row[p] = row++;
      }
  }
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
    rows[j] = (e >= 0 && e < e_count) ? Z_own + (size_t)own_row[p] * H
                                      : Z_peer + (size_t)peer_rows_of_mine[p] * H;
  }
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
