// Fast-path decode attention: fp32 query, bf16 KV cache, fp32 softmax with the
// head's sink, bf16 output for the next matrix multiply.
//
// Main kernel (attention_gqa8_kernel), used when 8 query heads share each
// key/value head and head_dim is 64 (gpt-oss-20b and -120b):
//
// Work split
//   Each wave works on one (sequence b, key/value head g) and computes all 8
//   query heads of that group at once, so every key and value read from
//   memory is used by 8 heads (grouped-query attention): the cache is read
//   once instead of 8 times.
//   The history (slots 0 .. n-1) is cut into steps of STEP = 16 * TPS
//   positions, dealt round-robin to NS waves (flash-decoding inside the
//   workgroup). Each wave keeps its own running max, running sum and output
//   (online softmax); the NS waves of a head are merged through LDS at the
//   end, together with the sink.
//   A workgroup holds NG neighbouring key/value heads x NS splits. The
//   engine's layout runs NG = 2, NS = 4, TPS = 2 (8 waves): the two heads'
//   128-byte slices sit side by side in each cache row, and this split was
//   the fastest measured (docs/09).
//
// Matrix cores (MFMA v_mfma_f32_16x16x16bf16_1k, 16 x 16 output, K = 16)
//   Scores:  S^T (16 positions x 16 columns) = K (16 x 64) * Q^T (64 x 16),
//            4 MFMAs over the 64 head dims.
//   Output:  O (16 rows x 64 dims) += P (16 x 16 positions) * V (16 x 64),
//            4 MFMAs, one per 16-dim slice.
//   Only 8 heads exist but the MFMA has 16 columns (rows for P). The spare 8
//   carry a second bf16 piece of the same number: column c < 8 holds the
//   high part of q (or p) for head c, column c + 8 the remainder
//   (x = hi + lo, |x - hi - lo| <= 2^-18 |x|). The two halves are added after
//   the MFMA, so q and p keep ~17 bits of precision instead of bf16's 8 at no
//   extra MFMA cost.
//
// Memory access (cache layout kc/vc[(b * slots + slot) * kv_dim + g * 64 + i])
//   K: lane l reads row (l % 16) of a 16-position tile, bytes
//      [16 (l/16), +16) and [64 + 16 (l/16), +16) of that row's 128-byte
//      head slice: one 16-byte load each, every 64-byte line fully used.
//   V: lane l reads 8 bytes (dims 4 (l%16) .. +3) of rows 4 (l/16) + r,
//      r = 0..3; each load instruction covers 4 whole 128-byte rows. The four
//      rows are then transposed in registers (v_perm) into the MFMA layout.
//   A sliding-window layer keeps position t in slot t % window; the slots in
//   use are always 0 .. n-1 with n = min(pos + 1, window), and softmax does
//   not care about order, so both layer kinds simply walk slots 0 .. n-1.
//
// int8 V (Q8 = true, what the engine runs: attention_q8v_chunk64)
//   V holds int8 codes c and one fp32 scale s per (sequence, slot, head),
//   v ~= s * c. The scale of each position is multiplied into its weight
//   (p * s, still split into hi + lo), so the second MFMA multiplies by the
//   codes themselves, which bf16 holds exactly. Each lane loads its 16 codes
//   (4 positions x 4 dims, stored in that order) with one 16-byte load, and
//   its 4 scales with another.
#pragma once

#include <hip/hip_runtime.h>

#include "fast_common.hpp"

namespace fattn {

typedef short bf16x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

constexpr float kLog2e = 1.44269504088896341f;

__device__ inline bf16x4 as_bf16x4(uint32_t a, uint32_t b) {
  return __builtin_bit_cast(bf16x4, u32x2{a, b});
}

// Round to nearest even, keeping the result in the upper 16 bits (lower bits
// are garbage below the cut; no NaN/Inf ever reaches it).
__device__ inline uint32_t round_bf16_hi(uint32_t u) {
  return u + 0x7fffu + ((u >> 16) & 1u);
}

// The bf16 piece of x this column carries, in the upper 16 bits: the high
// part hi = bf16(x) or, for lo columns, bf16(x - hi). Both rounded to
// nearest, so |x - hi - lo| <= 2^-18 |x|.
__device__ inline uint32_t bf16_piece(float x, bool lo) {
  const uint32_t hi = round_bf16_hi(__float_as_uint(x)) & 0xffff0000u;
  const float rest = x - __uint_as_float(hi);
  return lo ? round_bf16_hi(__float_as_uint(rest)) : hi;
}

// Two upper halves into one dword: a in bits 0-15, b in bits 16-31.
__device__ inline uint32_t pack_hi(uint32_t a, uint32_t b) {
  return (a >> 16) | (b & 0xffff0000u);
}
// Two lower halves into one dword.
__device__ inline uint32_t pack_lo(uint32_t a, uint32_t b) {
  return (a & 0xffffu) | (b << 16);
}

// Buffer descriptor for raw loads from base: `bytes` long, reads past the end
// return 0. Word 3 = 0x00020000 is the gfx9 default (32-bit data format).
__device__ inline __amdgpu_buffer_rsrc_t make_rsrc(const void *base, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short)0, (int)bytes,
                                           0x00020000);
}

// fp32 bits of signed byte R of w (one instruction: SDWA byte select with
// sign extension; the compiler would emit shift + extract + convert).
template <int R> __device__ inline uint32_t sbyte_to_f32(uint32_t w) {
  float f;
  if constexpr (R == 0)
    asm("v_cvt_f32_i32_sdwa %0, sext(%1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0"
        : "=v"(f) : "v"(w));
  else if constexpr (R == 1)
    asm("v_cvt_f32_i32_sdwa %0, sext(%1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1"
        : "=v"(f) : "v"(w));
  else if constexpr (R == 2)
    asm("v_cvt_f32_i32_sdwa %0, sext(%1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2"
        : "=v"(f) : "v"(w));
  else
    asm("v_cvt_f32_i32_sdwa %0, sext(%1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3"
        : "=v"(f) : "v"(w));
  return __float_as_uint(f);
}

// The 4 signed bytes of w as bf16x4, exactly (|code| <= 127 fits bf16's 8
// bits): convert each to fp32, keep the upper halves (one v_perm per pair).
__device__ inline bf16x4 sbytes_to_bf16x4(uint32_t w) {
  return as_bf16x4(__builtin_amdgcn_perm(sbyte_to_f32<1>(w), sbyte_to_f32<0>(w), 0x07060302u),
                   __builtin_amdgcn_perm(sbyte_to_f32<3>(w), sbyte_to_f32<2>(w), 0x07060302u));
}

__device__ inline float row_swap8(float x) { // value of lane (l ^ 8) within its row of 16
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x128, 0xf, 0xf, false));
}

__device__ inline void store_out(float *p, float a, float b) {
  *(float2 *)p = make_float2(a, b);
}
__device__ inline void store_out(bf16_t *p, float a, float b) {
  *(uint32_t *)p = (uint32_t)f32_to_bf16(a) | ((uint32_t)f32_to_bf16(b) << 16);
}

// Where element i of key/value head g, slot `slot`, sequence b sits in the
// cache, in elements:
//   b * sb + (slot >> cshift) * cs + g * sg + (slot & (2^cshift - 1)) * st + i
// The engine's layout is {slots * kv_dim, kv_dim, 64, 30, 0} (one chunk per
// sequence). Chunks let the benchmark try layouts that keep T = 2^cshift
// consecutive slots of one head together; T must be a multiple of the step.
struct KVLayout {
  long long sb; // per sequence
  int st, sg;   // per slot, per key/value head
  int cshift;   // log2 of slots per chunk
  long long cs; // per chunk
};

// Grid (n_kv_heads / NG, B), 64 * NG * NS threads: wave w works on key/value
// head blockIdx.x * NG + w % NG and on history split w / NG. n = slots to
// attend (>= 1).
//
// Q8 = true: V is stored as int8 codes with one fp32 scale per (b, slot, g)
// (v = scale * code, see vq_index_chunk64 / vs_index_chunk64 at the end of
// this file); vc points at the codes, vs at the scales. The layout must keep
// the 16 slots of a tile of one head together (L.st == 64).
//
// ROT = true: the query in qkv is not rotated yet; the kernel applies RoPE
// with the angles cosv, sinv (head_dim / 2 of each) as it loads it, with the
// arithmetic of rope_kv_bf16_kernel (rope_pair), so q gets the same bits.
template <int NG, int NS, int TPS, typename OutT, bool Q8 = false, bool ROT = false>
__global__ __launch_bounds__(64 * NG * NS) void attention_gqa8_kernel(
    const float *__restrict__ qkv, int row_stride, const bf16_t *__restrict__ kc,
    const void *__restrict__ vc, const float *__restrict__ vs, KVLayout L,
    const float *__restrict__ sinks, OutT *__restrict__ out, int out_stride, int n,
    const float *__restrict__ cosv, const float *__restrict__ sinv) {
  constexpr int D = 64, G = 8, NW = NG * NS;
  constexpr int STEP = 16 * TPS;
  __shared__ float s_o[NW][G][D];
  __shared__ float s_m[NW][G], s_l[NW][G];

  const int b = blockIdx.y;
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int g = blockIdx.x * NG + w % NG; // key/value head of this wave
  const int split = w / NG;               // which share of the history
  const int col = lane & 15; // MFMA column / P row: head (col & 7), lo piece if col >= 8
  const int grp = lane >> 4; // MFMA k-group (0..3)
  const bool lo = col >= 8;
  const int head = col & 7;

  // Every K/V load goes through a buffer descriptor (base address + size)
  // that ends right after slot n - 1 of this (b, g): offsets are 32-bit, and
  // a row past the end reads as zeros, so the last, partial step needs no
  // special loads.
  const size_t head_off = b * L.sb + (size_t)g * L.sg; // elements, from kc
  const bf16_t *kb = kc + head_off;
  const uint32_t st2 = (uint32_t)L.st * 2, cs2 = (uint32_t)(L.cs * 2); // bytes
  const int cmask = (1 << L.cshift) - 1;
  auto row_bytes = [&](int row) { // byte offset of a slot from kb (and vb for bf16 V)
    return (uint32_t)(row >> L.cshift) * cs2 + (uint32_t)(row & cmask) * st2;
  };
  const uint32_t range = row_bytes(n - 1) + D * 2;
  const __amdgpu_buffer_rsrc_t krs = make_rsrc(kb, range);
  // bf16 V: same offsets as K. int8 V: one byte per element, so a tile's
  // codes start at half its K byte offset; the range covers the whole tile
  // of slot n - 1 (codes are always finite, and their weight is 0 past n).
  // Scales: one float per slot, at 1/32 of the K byte offset; the range ends
  // exactly after slot n - 1, so later slots read scale 0.
  const __amdgpu_buffer_rsrc_t vrs =
      Q8 ? make_rsrc((const int8_t *)vc + head_off, (row_bytes((n - 1) & ~15) >> 1) + 1024)
         : make_rsrc((const bf16_t *)vc + head_off, range);
  const __amdgpu_buffer_rsrc_t srs =
      make_rsrc(Q8 ? vs + head_off / D : nullptr, Q8 ? (row_bytes(n - 1) >> 5) + 4 : 0);

  // The loads of a step: K into kr, V into vq and sq (int8) or vr (bf16).
  // Software pipeline: the first step's loads go out before the query is
  // read, and each step loads the next one's K as soon as its scores are
  // done and the next one's V at its end, so the waits overlap the work. A
  // step past the end reads zeros (outside the descriptors' range).
  u32x4 kr[TPS][2];
  u32x2 vr[TPS][4];
  u32x4 vq[TPS], sq[TPS]; // int8 V: codes, scales
  auto load_k = [&](int step) {
    const uint32_t step_off = row_bytes(step * STEP); // a step never crosses a chunk
    const uint32_t koff = step_off + (uint32_t)col * st2 + grp * 16;
#pragma unroll
    for (int t = 0; t < TPS; t++) {
      kr[t][0] = __builtin_amdgcn_raw_buffer_load_b128(krs, koff + 16 * t * st2, 0, 0);
      kr[t][1] = __builtin_amdgcn_raw_buffer_load_b128(krs, koff + 16 * t * st2 + 64, 0, 0);
    }
  };
  auto load_v = [&](int step) {
    const uint32_t step_off = row_bytes(step * STEP);
    if constexpr (Q8) {
      // Lane (grp, col) reads the 16 codes of positions 4 grp + 0..3, dims
      // 4 col + 0..3 (one 1 KB tile per load instruction), and the 4 scales
      // of the same positions.
#pragma unroll
      for (int t = 0; t < TPS; t++) {
        vq[t] = __builtin_amdgcn_raw_buffer_load_b128(
            vrs, (step_off >> 1) + t * 1024 + grp * 256 + col * 16, 0, 0);
        sq[t] = __builtin_amdgcn_raw_buffer_load_b128(srs, (step_off >> 5) + t * 64 + grp * 16,
                                                      0, 0);
      }
    } else {
      const uint32_t voff = step_off + (uint32_t)(4 * grp) * st2 + col * 8;
#pragma unroll
      for (int t = 0; t < TPS; t++)
#pragma unroll
        for (int r = 0; r < 4; r++)
          vr[t][r] = __builtin_amdgcn_raw_buffer_load_b64(vrs, voff + (16 * t + r) * st2, 0, 0);
    }
  };
  load_k(split);
  load_v(split);

  // Q^T as the MFMA B operand: lane holds Q[head][dims of k-step s], with
  // k-step s covering dims 32 (s/2) + 8 grp + 4 (s%2) + 0..3, the same dims
  // the K loads put in each lane. Scaled by 1/sqrt(64) and log2(e) so
  // that exp2 gives the softmax weights.
  // Dims d and d + 32 (s and s + 2) form RoPE pairs.
  // The sink and the query (and the angles) are loaded together, so their
  // latencies overlap.
  const float sink = sinks[g * G + head];
  const float *q = qkv + (size_t)b * row_stride + (g * G + head) * D;
  float4 qv[4], cv[2], sv[2];
#pragma unroll
  for (int s = 0; s < 4; s++)
    qv[s] = *(const float4 *)(q + (s >> 1) * 32 + grp * 8 + (s & 1) * 4);
  if (ROT) {
#pragma unroll
    for (int s = 0; s < 2; s++) {
      cv[s] = *(const float4 *)(cosv + grp * 8 + s * 4);
      sv[s] = *(const float4 *)(sinv + grp * 8 + s * 4);
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
      rope_pair(qv[s].x, qv[s + 2].x, cv[s].x, sv[s].x, qv[s].x, qv[s + 2].x);
      rope_pair(qv[s].y, qv[s + 2].y, cv[s].y, sv[s].y, qv[s].y, qv[s + 2].y);
      rope_pair(qv[s].z, qv[s + 2].z, cv[s].z, sv[s].z, qv[s].z, qv[s + 2].z);
      rope_pair(qv[s].w, qv[s + 2].w, cv[s].w, sv[s].w, qv[s].w, qv[s + 2].w);
    }
  }
  bf16x4 qop[4];
#pragma unroll
  for (int s = 0; s < 4; s++) {
    const float4 v = qv[s];
    const float sc = 0.125f * kLog2e;
    qop[s] = as_bf16x4(pack_hi(bf16_piece(v.x * sc, lo), bf16_piece(v.y * sc, lo)),
                       pack_hi(bf16_piece(v.z * sc, lo), bf16_piece(v.w * sc, lo)));
  }

  const float sink2 = sink * kLog2e;
  float m = sink2;   // running max (log2 units) of this head over this wave's positions
  float lsum = 0.0f; // running sum of exp2(score - m) over this lane's positions
  f32x4 o[4];        // O rows 4 grp + r, dims 4 col + nn
#pragma unroll
  for (int nn = 0; nn < 4; nn++)
    o[nn] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

  const int n_steps = (n + STEP - 1) / STEP;
  for (int step = split; step < n_steps; step += NS) {
    const int base = step * STEP;
    // Scores: lane holds S^T[position base + 16 t + 4 grp + r][column col].
    float sc[TPS][4];
#pragma unroll
    for (int t = 0; t < TPS; t++) {
      f32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(as_bf16x4(kr[t][0][0], kr[t][0][1]), qop[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(as_bf16x4(kr[t][0][2], kr[t][0][3]), qop[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(as_bf16x4(kr[t][1][0], kr[t][1][1]), qop[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(as_bf16x4(kr[t][1][2], kr[t][1][3]), qop[3], acc, 0, 0, 0);
#pragma unroll
      for (int r = 0; r < 4; r++)
        sc[t][r] = acc[r] + row_swap8(acc[r]); // hi + lo piece of q
    }
    load_k(step + NS);
    if (base + STEP > n) { // last step: positions past n get weight 0
#pragma unroll
      for (int t = 0; t < TPS; t++)
#pragma unroll
        for (int r = 0; r < 4; r++)
          if (base + 16 * t + 4 * grp + r >= n)
            sc[t][r] = -INFINITY;
    }

    // Online softmax. The max of each head is raised only when a score
    // exceeds it by more than 8 (a factor 2^8): until then the weights
    // exp2(s - m) stay below 256, harmless in fp32, and the final division
    // uses the same m, so the result is unchanged. This skips rescaling the
    // output in almost every step.
    float mx = -INFINITY;
#pragma unroll
    for (int t = 0; t < TPS; t++)
#pragma unroll
      for (int r = 0; r < 4; r++)
        mx = fmaxf(mx, sc[t][r]);
    mx = fmaxf(mx, __shfl_xor(mx, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 32));
    if (__any(mx > m + 8.0f)) { // same decision for the whole wave
      const float m_new = fmaxf(m, mx);
      const float alpha = __builtin_amdgcn_exp2f(m - m_new);
      m = m_new;
      lsum *= alpha;
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const float a = __shfl(alpha, 4 * (grp & 1) + r); // O row 4 grp + r is head 4 (grp & 1) + r
#pragma unroll
        for (int nn = 0; nn < 4; nn++)
          o[nn][r] *= a;
      }
    }

#pragma unroll
    for (int t = 0; t < TPS; t++) {
      uint32_t pp[4];
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const float p = __builtin_amdgcn_exp2f(sc[t][r] - m);
        lsum += p;
        // int8 V: the position's scale goes into the weight, so the matrix
        // cores see the codes themselves (exact in bf16).
        pp[r] = bf16_piece(Q8 ? p * __uint_as_float(sq[t][r]) : p, lo);
      }
      // P as the MFMA A operand: row col, positions 4 grp + 0..3.
      const bf16x4 pa = as_bf16x4(pack_hi(pp[0], pp[1]), pack_hi(pp[2], pp[3]));
      if constexpr (Q8) {
        // Dword nn of vq holds the codes of positions 4 grp + 0..3 for dim
        // 4 col + nn: exactly the B operand of output slice nn.
#pragma unroll
        for (int nn = 0; nn < 4; nn++)
          o[nn] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(pa, sbytes_to_bf16x4(vq[t][nn]),
                                                            o[nn], 0, 0, 0);
        continue;
      }
      // V as the B operand: positions 4 grp + 0..3, dim 4 col + nn.
      const u32x2 *v = vr[t];
      o[0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          pa, as_bf16x4(pack_lo(v[0][0], v[1][0]), pack_lo(v[2][0], v[3][0])), o[0], 0, 0, 0);
      o[1] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          pa, as_bf16x4(pack_hi(v[0][0], v[1][0]), pack_hi(v[2][0], v[3][0])), o[1], 0, 0, 0);
      o[2] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          pa, as_bf16x4(pack_lo(v[0][1], v[1][1]), pack_lo(v[2][1], v[3][1])), o[2], 0, 0, 0);
      o[3] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          pa, as_bf16x4(pack_hi(v[0][1], v[1][1]), pack_hi(v[2][1], v[3][1])), o[3], 0, 0, 0);
    }
    load_v(step + NS);
  }

  // Per wave: sum over the 4 k-groups, add the lo rows (8..15, lanes 32..63)
  // to the hi rows (0..7, lanes 0..31), park in LDS.
  lsum += __shfl_xor(lsum, 16);
  lsum += __shfl_xor(lsum, 32);
#pragma unroll
  for (int nn = 0; nn < 4; nn++)
#pragma unroll
    for (int r = 0; r < 4; r++)
      o[nn][r] += __shfl_xor(o[nn][r], 32);
  if (lane < 32) {
#pragma unroll
    for (int r = 0; r < 4; r++)
      *(float4 *)&s_o[w][4 * grp + r][4 * col] = make_float4(o[0][r], o[1][r], o[2][r], o[3][r]);
  }
  if (lane < G) {
    s_m[w][lane] = m;
    s_l[w][lane] = lsum;
  }
  __syncthreads();

  // Merge the NS splits of each key/value head and the sink:
  // M = max, L = exp2(sink - M) + sum l_w 2^(m_w - M), out = sum o_w 2^(m_w - M) / L.
  for (int idx = threadIdx.x; idx < NG * G * D / 2; idx += 64 * NW) {
    const int gi = idx / (G * D / 2), hh = (idx / (D / 2)) % G, d = 2 * (idx % (D / 2));
    const int gq = (blockIdx.x * NG + gi) * G + hh; // query head
    const float sk = sinks[gq] * kLog2e;
    float M = sk;
#pragma unroll
    for (int v = 0; v < NS; v++)
      M = fmaxf(M, s_m[v * NG + gi][hh]);
    float L = __builtin_amdgcn_exp2f(sk - M), o0 = 0.0f, o1 = 0.0f;
#pragma unroll
    for (int v = 0; v < NS; v++) {
      const int wv = v * NG + gi;
      const float c = __builtin_amdgcn_exp2f(s_m[wv][hh] - M);
      L = fmaf(s_l[wv][hh], c, L);
      o0 = fmaf(s_o[wv][hh][d], c, o0);
      o1 = fmaf(s_o[wv][hh][d + 1], c, o1);
    }
    const float inv = 1.0f / L;
    store_out(out + (size_t)b * out_stride + gq * D + d, o0 * inv, o1 * inv);
  }
}

// Launch helper; the benchmark and the unit test call it directly to try
// other work splits, step sizes, cache layouts and fp32 output.
// n_kv_heads must be a multiple of NG.
template <int NG, int NS, int TPS, typename OutT>
static void attention_gqa8(hipStream_t s, const float *qkv, int row_stride,
                           const bf16_t *kc, const bf16_t *vc, KVLayout layout,
                           int n_kv_heads, const float *sinks, OutT *out,
                           int out_stride, int B, int n) {
  attention_gqa8_kernel<NG, NS, TPS, OutT><<<dim3(n_kv_heads / NG, B), 64 * NG * NS, 0, s>>>(
      qkv, row_stride, kc, vc, nullptr, layout, sinks, out, out_stride, n, nullptr, nullptr);
}

// The same with int8 V codes vq and their scales vs (layout.st must be 64);
// cosv, sinv (may be null): rotate the query as it is loaded (ROT above).
template <int NG, int NS, int TPS, typename OutT>
static void attention_gqa8_q8v(hipStream_t s, const float *qkv, int row_stride,
                               const bf16_t *kc, const int8_t *vq, const float *vs,
                               KVLayout layout, int n_kv_heads, const float *sinks, OutT *out,
                               int out_stride, int B, int n, const float *cosv = nullptr,
                               const float *sinv = nullptr) {
  if (cosv)
    attention_gqa8_kernel<NG, NS, TPS, OutT, true, true>
        <<<dim3(n_kv_heads / NG, B), 64 * NG * NS, 0, s>>>(
            qkv, row_stride, kc, vq, vs, layout, sinks, out, out_stride, n, cosv, sinv);
  else
    attention_gqa8_kernel<NG, NS, TPS, OutT, true>
        <<<dim3(n_kv_heads / NG, B), 64 * NG * NS, 0, s>>>(
            qkv, row_stride, kc, vq, vs, layout, sinks, out, out_stride, n, nullptr, nullptr);
}

} // namespace fattn

// General fallback (any kv_mul): one 64-thread block per (head, sequence),
// scores staged in LDS. Reads each key/value once per query head.
// Dynamic shared memory: (positions + 1) floats.
__global__ __launch_bounds__(64) void attention_bf16_simple_kernel(
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
  // Slots in use: 0 .. n-1 for both layer kinds (see the top of this file).
  const int n = window > 0 ? min(pos + 1, window) : pos + 1;
#ifndef GETP_ATTN_SIMPLE // -DGETP_ATTN_SIMPLE builds the previous kernel, for comparisons
  if (kv_mul == 8 && kv_dim % 128 == 0 && n_heads == 8 * (kv_dim / 64)) {
    // 2 key/value heads x 4 history splits per workgroup (8 waves), 32
    // positions per step: the fastest split measured on this layout
    // (docs/09). The split per head is the same as with 1 head per workgroup,
    // so the results are identical bit for bit.
    const fattn::KVLayout layout{(long long)slots * kv_dim, kv_dim, 64, 30, 0};
    fattn::attention_gqa8<2, 4, 2>(s, qkv, row_stride, kc, vc, layout, kv_dim / 64, sinks, out,
                                   out_stride, B, n);
    return;
  }
#endif
  attention_bf16_simple_kernel<<<dim3(n_heads, B), 64, (n + 1) * sizeof(float), s>>>(
      qkv, row_stride, kc, vc, slots, kv_dim, kv_mul, sinks, out, out_stride, pos,
      window);
}

// ---- The cache layout of the fast path (docs/09) -------------------------
// "chunk64": each sequence's cache is cut into chunks of 64 slots, and inside
// a chunk every key/value head keeps its 64 slots together (8 KB):
//   element (b, slot, g, i) at  b * slots64 * kv_dim + (slot / 64) * 64 * kv_dim
//                               + g * 64 * 64 + (slot % 64) * 64 + i
// slots64 = slots rounded up to a multiple of 64. A workgroup then reads whole
// 8 KB blocks instead of 128 bytes out of every 1 KB row; attn_bench measures
// 1,100-1,170 GB/s against ~870 for the row-major layout that attention_bf16
// reads. rope_kv_bf16_kernel writes this layout.
__host__ __device__ inline size_t kv_index_chunk64(int b, int slot, int g, int i, int slots64,
                                                   int kv_dim) {
  return ((size_t)b * slots64 + (size_t)(slot & ~63)) * kv_dim + (size_t)g * 64 * 64 +
         (size_t)(slot & 63) * 64 + i;
}

// Same arguments as attention_bf16; kc / vc in the chunk64 layout, slots a
// multiple of 64, 8 query heads per key/value head, head_dim 64.
static void attention_bf16_chunk64(hipStream_t s, const float *qkv, int row_stride,
                                   const bf16_t *kc, const bf16_t *vc, int slots, int kv_dim,
                                   int n_heads, int kv_mul, const float *sinks, bf16_t *out,
                                   int out_stride, int B, int pos, int window) {
  (void)n_heads;
  (void)kv_mul;
  const int n = window > 0 ? min(pos + 1, window) : pos + 1;
  const fattn::KVLayout layout{(long long)slots * kv_dim, 64, 64 * 64, 6, 64LL * kv_dim};
  if (n <= 128) // window layers and short histories: 4 splits of 32-position steps
    fattn::attention_gqa8<1, 4, 2>(s, qkv, row_stride, kc, vc, layout, kv_dim / 64, sinks, out,
                                   out_stride, B, n);
  else          // long histories: 8 splits of 16-position steps
    fattn::attention_gqa8<1, 8, 1>(s, qkv, row_stride, kc, vc, layout, kv_dim / 64, sinks, out,
                                   out_stride, B, n);
}

// ---- int8 V cache (docs/09) ----------------------------------------------
// K stays bf16 in the chunk64 layout above. Each value vector (64 numbers of
// one key/value head at one slot) is stored as 64 int8 codes c and one fp32
// scale s = max|v| / 127, with v ~= s * c and c = round(v / s).
// Codes keep the chunk64 blocks (one 4 KB block per head and 64 slots, at half
// the byte offset of the same K block), and inside each 16-slot tile (1 KB)
// they are in the order the attention lanes read them: the 4 codes of slots
// 4q + 0..3 for dim i sit together, and a lane's 16 bytes are 4 dims x 4 slots.
//   tile byte (slot % 16 = 4q + r, dim i = 4d + nn):  q * 256 + d * 16 + nn * 4 + r
__host__ __device__ inline size_t vq_index_chunk64(int b, int slot, int g, int i, int slots64,
                                                   int kv_dim) {
  return kv_index_chunk64(b, slot & ~15, g, 0, slots64, kv_dim) + ((slot >> 2) & 3) * 256 +
         (i >> 2) * 16 + (i & 3) * 4 + (slot & 3);
}
// Scales: per sequence and 64-slot chunk, each head's 64 scales together (256 B).
__host__ __device__ inline size_t vs_index_chunk64(int b, int slot, int g, int slots64,
                                                   int kv_dim) {
  return kv_index_chunk64(b, slot, g, 0, slots64, kv_dim) / 64;
}

// attention_bf16_chunk64 with V as int8 codes vq and scales vs (above).
// cosv, sinv (may be null): rotate the query as it is loaded (attention_gqa8_kernel, ROT).
static void attention_q8v_chunk64(hipStream_t s, const float *qkv, int row_stride,
                                  const bf16_t *kc, const int8_t *vq, const float *vs,
                                  int slots, int kv_dim, const float *sinks, bf16_t *out,
                                  int out_stride, int B, int pos, int window,
                                  const float *cosv = nullptr, const float *sinv = nullptr) {
  const int n = window > 0 ? min(pos + 1, window) : pos + 1;
  const fattn::KVLayout layout{(long long)slots * kv_dim, 64, 64 * 64, 6, 64LL * kv_dim};
  // Fastest measured split per case (docs/09): window layers 2 waves per
  // head, full layers 4 (up to 128 positions) or 8 waves.
  if (window > 0)
    fattn::attention_gqa8_q8v<1, 2, 2>(s, qkv, row_stride, kc, vq, vs, layout, kv_dim / 64,
                                       sinks, out, out_stride, B, n, cosv, sinv);
  else if (n <= 128)
    fattn::attention_gqa8_q8v<1, 4, 2>(s, qkv, row_stride, kc, vq, vs, layout, kv_dim / 64,
                                       sinks, out, out_stride, B, n, cosv, sinv);
  else
    fattn::attention_gqa8_q8v<1, 8, 1>(s, qkv, row_stride, kc, vq, vs, layout, kv_dim / 64,
                                       sinks, out, out_stride, B, n, cosv, sinv);
}
