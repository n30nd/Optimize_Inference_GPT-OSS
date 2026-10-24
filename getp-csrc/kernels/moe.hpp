// Mixture-of-experts steps: pick each token's experts, group the tokens by
// expert so each expert multiplies one contiguous block of rows, apply
// SwiGLU, and add the weighted expert outputs back.
#pragma once

#include <hip/hip_runtime.h>

#include "exact_math.hpp"

#define MOE_MAX_TOPK 8

// topk() then softmax() from run.cpp for each row of router logits: the k
// largest scores in descending order (ties: lower expert first), turned into
// weights. One thread per row.
__global__ void router_topk_exact(const float *__restrict__ logits, int E,
                                  int k, float *__restrict__ topk_w,
                                  int *__restrict__ topk_e, int B) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B)
    return;
  const float *r = logits + (size_t)b * E;
  int chosen[MOE_MAX_TOPK];
  float v[MOE_MAX_TOPK];
  for (int j = 0; j < k; j++) {
    int bi = -1;
    float bv = 0.0f;
    for (int e = 0; e < E; e++) {
      bool taken = false;
      for (int p = 0; p < j; p++)
        taken |= chosen[p] == e;
      if (!taken && (bi < 0 || r[e] > bv)) {
        bi = e;
        bv = r[e];
      }
    }
    chosen[j] = bi;
    v[j] = bv;
  }
  double max_val = v[0];
  for (int j = 1; j < k; j++)
    if (v[j] > max_val)
      max_val = v[j];
  double sum = 0.0;
  for (int j = 0; j < k; j++) {
    v[j] = expf_cpu((float)__dsub_rn((double)v[j], max_val));
    sum = __dadd_rn(sum, (double)v[j]);
  }
  for (int j = 0; j < k; j++) {
    topk_w[(size_t)b * k + j] = (float)__ddiv_rn((double)v[j], sum);
    topk_e[(size_t)b * k + j] = chosen[j];
  }
}

// Groups the n_pairs (token, choice) pairs by expert, in pair order within an
// expert: rows offsets[e] .. offsets[e+1] belong to expert e, row_pair[row]
// is the pair index (token * k + choice), pair_row is its inverse.
// One block of at least E threads.
__global__ void moe_group_kernel(const int *__restrict__ topk_e, int n_pairs,
                                 int E, int *__restrict__ offsets,
                                 int *__restrict__ row_pair,
                                 int *__restrict__ pair_row) {
  __shared__ int start[1025];
  const int e = threadIdx.x;
  int count = 0;
  if (e < E)
    for (int p = 0; p < n_pairs; p++)
      count += topk_e[p] == e;
  if (e < E)
    start[e + 1] = count;
  __syncthreads();
  if (e == 0) {
    start[0] = 0;
    for (int i = 1; i <= E; i++)
      start[i] += start[i - 1];
    for (int i = 0; i <= E; i++)
      offsets[i] = start[i];
  }
  __syncthreads();
  if (e < E) {
    int row = start[e];
    for (int p = 0; p < n_pairs; p++)
      if (topk_e[p] == e) {
        row_pair[row] = p;
        pair_row[p] = row;
        row++;
      }
  }
}

// Xg[row, :] = x[row_pair[row] / k, :]. One block per row.
__global__ void moe_gather_kernel(const float *__restrict__ x,
                                  const int *__restrict__ row_pair, int k,
                                  float *__restrict__ Xg, int H) {
  const int row = blockIdx.x;
  const float *src = x + (size_t)(row_pair[row] / k) * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    Xg[(size_t)row * H + i] = src[i];
}

// The SwiGLU of forward() in run.cpp on mlp1 output rows: Y (R, 2I) holds
// gate and up interleaved; G (R, I) receives clamp/silu(gate) * (up + 1).
__global__ void swiglu_exact(const float *__restrict__ Y,
                             float *__restrict__ G, size_t R, int I,
                             float limit) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= R * I)
    return;
  const size_t r = idx / I, j = idx % I;
  float val = Y[r * 2 * I + 2 * j];
  float up = Y[r * 2 * I + 2 * j + 1];
  if (val > limit)
    val = limit;
  if (up > limit)
    up = limit;
  if (up < -limit)
    up = -limit;
  const float alpha = 1.702f;
  val = __fmul_rn(val, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf_cpu(__fmul_rn(-alpha, val)))));
  val = __fmul_rn(val, __fadd_rn(up, 1.0f));
  G[idx] = val;
}

// x[b, :] += sum over b's experts of weight * Z[row, :], adding the experts in
// increasing expert order with FMA, as forward() builds e_agg. One block per
// token.
__global__ void moe_combine_exact(float *__restrict__ x,
                                  const float *__restrict__ Z,
                                  const int *__restrict__ pair_row,
                                  const float *__restrict__ topk_w,
                                  const int *__restrict__ topk_e, int k,
                                  int H) {
  const int b = blockIdx.x;
  int order[MOE_MAX_TOPK];
  for (int j = 0; j < k; j++)
    order[j] = j;
  for (int i = 1; i < k; i++)
    for (int j = i; j > 0 && topk_e[(size_t)b * k + order[j]] <
                                 topk_e[(size_t)b * k + order[j - 1]];
         j--) {
      int t = order[j];
      order[j] = order[j - 1];
      order[j - 1] = t;
    }
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
      const int p = b * k + order[j];
      acc = __fmaf_rn(Z[(size_t)pair_row[p] * H + i], topk_w[p], acc);
    }
    x[(size_t)b * H + i] = __fadd_rn(x[(size_t)b * H + i], acc);
  }
}
