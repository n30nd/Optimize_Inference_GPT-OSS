// Fast-path expert matrix multiply ("grouped GEMM"): rows offsets[e] ..
// offsets[e+1] of A (bf16) times expert e's MXFP4 weight W_e (N, K), plus its
// bias, into the same rows of C (fp32):
//
//   C[r, n] = sum_k A[r, k] * W_e[n, k] + bias[e, n]   for offsets[e] <= r < offsets[e+1]
//
// The work runs on the matrix cores (MFMA) with v_mfma_f32_16x16x16bf16_1k:
// one instruction multiplies a 16x16 bf16 tile of A by a 16x16 bf16 tile of W
// and adds the 16x16 result into fp32 accumulators.
//
// What limits this kernel is the number of requests to the L2 cache (one per
// 64-byte line; the L2 answers about one per channel per clock) and the rate
// at which the weights stream in from memory, more than the MFMA rate. The
// tile shape and the weight layout are chosen for both.
//
// Tile layout
// -----------
// One block of 512 threads (8 wavefronts) computes a work item: up to 64 rows
// of one expert by 256 columns. All 8 wavefronts share the same 64 rows and
// each owns 32 columns (2 MFMA tiles of 16):
//
//              columns n0 ..                                    n0 + 256
//            +--------+--------+--------+--  ...  --+--------+
//   rows     | wave 0 | wave 1 | wave 2 |           | wave 7 |
//   m0 ..    | 64x32  |        |        |           |        |
//   m0 + 64  |        |        |        |           |        |
//            +--------+--------+--------+--  ...  --+--------+
//
// K is walked in steps of 128 (one MXFP4 group of 32 per lane). Per step the
// block copies the 64 x 128 slice of A into LDS (shared by the 8 wavefronts),
// and each lane reads 16 bytes (32 codes, one group) per MFMA column tile
// straight from global memory into registers, decodes them to bf16 and feeds
// them to the MFMA: 8 MFMA k-steps of 16 per step. The A slice of the next
// step goes through registers into the other LDS buffer while the current
// one is multiplied, and the weight bytes of the next step are loaded into
// registers meanwhile. Each lane loads the scale bytes it needs for the whole
// item (one per step and column tile) into registers once, at the item's start.
//
// Packed weights (moe_pack_mxfp4)
// -------------------------------
// The kernel does not read the checkpoint layout (one row of K/2 code bytes
// per column) but a copy repacked once at load time, in which every (256-
// column tile, step of 128 k) is one contiguous run of 16 KB, in the order
// the lanes read it. Per expert, with nc = 16-column pieces in the tile (16,
// fewer only in the last tile; columns are padded with zeros to a multiple
// of 16):
//
//   codes:  tile t, step p, lane group kq, piece c, column j of the piece:
//           offset 256t * K/2 + p * nc * 1024 + (kq * nc + c) * 256 + 16 * j
//           = the 16 code bytes of column 256t + 16c + j, group 4p + kq
//   scales: piece P = n / 16 (1536 bytes each); lane (j, kq) owns 24 bytes,
//           byte i = scale of column 16P + j, group 4i + kq (0 past K):
//           bytes 0..15 at P * 1536 + 16 * lane, 16..23 at P * 1536 + 1024 + 8 * lane
//
// So the codes of one load instruction of a wavefront are 4 runs of 256
// bytes, and each lane gets its 6 scale registers with 2 loads. With kq
// outermost, a last half step (K % 128 == 64) is the first half of a full
// one, so a tile still takes 8 * K bytes per 16 columns, as in the
// checkpoint: lanes kq = 2, 3 read past the end of the tile's buffer
// descriptor there and get zero codes without memory traffic. The bytes
// that reach each register are the same as with the checkpoint layout, so
// the results are bit for bit the same.
//
// MFMA operand layout (16x16x16): lane l holds 4 consecutive k values of row
// (l % 16) of the A tile and of column (l % 16) of the W tile; lanes 0-15 hold
// k 0-3, lanes 16-31 k 4-7, and so on. The sum over k does not care in which
// order the k values come, as long as A and W use the same order. We use that
// freedom twice:
//   - lane group kq = l / 16 owns the 32 k values 32*kq .. 32*kq+31 of each
//     step of 128 for all 8 k-steps, so each lane reads 16 contiguous MXFP4
//     bytes and one scale;
//   - inside each group of 8 k the decoder produces the even k first
//     (k 0 2 4 6, then 1 3 5 7), and A is stored in LDS in that same order.
// K = 2880 is 22.5 steps of 128: in the last, half step the lanes whose 32 k
// lie past K get zero codes (the A values they meet are finite, so they add 0).
//
// MXFP4 decoding (fp4_tables, fp4x8_to_bf16)
// ------------------------------------------
// A code is 4 bits: sign, 2 exponent bits, 1 mantissa bit, so the magnitude
// is one of 8 values {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Times the group's scale
// 2^(s-127) every value is still an exact bf16. For each (column, group of 32
// k) we build two 8-entry byte tables in registers: the high byte and the low
// byte of the bf16 of each scaled magnitude. v_perm_b32 then looks up 4
// magnitudes at once (it picks bytes of a 64-bit table by 3-bit index), the
// sign bit is OR-ed into the high byte, and two more v_perm_b32 interleave
// low and high bytes into bf16 pairs: about 2 instructions per weight.
//   Exact for scale bytes 2 .. 252 (the checkpoint uses 115 .. 136); below
//   that 0.5 * 2^(s-127) is not a normal bf16.
//
// Uneven experts (scheduling)
// ---------------------------
// The host does not know how many rows each expert got (offsets live on the
// GPU), so the kernel is persistent: a fixed grid of (CUs x blocks per CU)
// blocks reads offsets, counts each expert's 64-row tiles, and takes work
// items from a queue (one atomic counter per device). Items are ordered
// expert, then column tile, then row tile, so the row tiles that need the
// same weight slice start together and share it through the L2 cache.
// Experts are taken largest first, so the last items of a call, which decide
// when it ends, are the cheap one-tile experts. Experts with 0 rows have no
// items. A row tile with fewer than 64 valid rows skips the 16-row MFMA tiles
// that are entirely empty, so an expert with 20 rows pays for 32 rows.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>

#include "fast_common.hpp"
#include "gemm.hpp"

#define MOE_MT 64                     // rows per work item
#define MOE_KS 128                    // k per pipeline step
#define MOE_WAVES 8                   // wavefronts per block
#define MOE_NS 2                      // 16-column MFMA tiles per wavefront
#define MOE_NT (MOE_WAVES * 16 * MOE_NS) // columns per work item (256)
#define MOE_THREADS (MOE_WAVES * 64)
#define MOE_SREGS 6                   // scale registers per column tile: K <= 4 * 6 * 128
#define MOE_MAX_E 256                 // largest expert count supported
#define MOE_SPIECE (64 * 4 * MOE_SREGS) // packed scale bytes per 16 columns (1536)

// Bytes of one expert's packed codes and scales (moe_pack_mxfp4).
__host__ __device__ inline size_t moe_packed_blocks_bytes(int N, int K) {
  return (size_t)((N + 15) / 16) * 8 * K;
}
__host__ __device__ inline size_t moe_packed_scales_bytes(int N, int K) {
  return (size_t)((N + 15) / 16) * MOE_SPIECE;
}

typedef short moe_bf16x4 __attribute__((ext_vector_type(4)));
typedef float moe_f32x4 __attribute__((ext_vector_type(4)));

// High-byte and low-byte tables of bf16(m * 2^(s-127)) for the 8 magnitudes
// m = {0, 0.5, 1, 1.5, 2, 3, 4, 6}; byte i of (lo0, lo1) = entry i (0-3, 4-7).
struct Fp4Tables {
  uint32_t h0, h1, l0, l1;
};

__device__ inline Fp4Tables fp4_tables(uint32_t s) {
  // The bf16 bit patterns, two per dword: magnitude 2i in the low half,
  // 2i+1 in the high half. 0.5 * 2^(s-127) has exponent field s-1; 1 has s;
  // 2 has s+1; 4 has s+2; the 1.5 / 3 / 6 entries add mantissa bit 0x40.
  const uint32_t base = __umul24(s, 0x00800080u); // (s << 7) in both halves
  const uint32_t d0 = (s << 23) - 0x00800000u;   // 0,   0.5
  const uint32_t d1 = base + 0x00400000u;        // 1,   1.5
  const uint32_t d2 = base + 0x00C00080u;        // 2,   3
  const uint32_t d3 = base + 0x01400100u;        // 4,   6
  Fp4Tables t;
  t.h0 = __builtin_amdgcn_perm(d1, d0, 0x07050301u);
  t.l0 = __builtin_amdgcn_perm(d1, d0, 0x06040200u);
  t.h1 = __builtin_amdgcn_perm(d3, d2, 0x07050301u);
  t.l1 = __builtin_amdgcn_perm(d3, d2, 0x06040200u);
  return t;
}

// The 8 codes of one dword (byte j: code 2j in the low nibble, 2j+1 in the
// high nibble) to 4 dwords of bf16 pairs, low half first:
//   o[0] = (c0, c2), o[1] = (c4, c6), o[2] = (c1, c3), o[3] = (c5, c7).
__device__ inline void fp4x8_to_bf16(uint32_t q, const Fp4Tables &t,
                                     uint32_t o[4]) {
  const uint32_t ilo = q & 0x07070707u;        // magnitudes of the even codes
  const uint32_t ihi = (q >> 4) & 0x07070707u; // magnitudes of the odd codes
  const uint32_t hlo = __builtin_amdgcn_perm(t.h1, t.h0, ilo) | ((q << 4) & 0x80808080u);
  const uint32_t llo = __builtin_amdgcn_perm(t.l1, t.l0, ilo);
  const uint32_t hhi = __builtin_amdgcn_perm(t.h1, t.h0, ihi) | (q & 0x80808080u);
  const uint32_t lhi = __builtin_amdgcn_perm(t.l1, t.l0, ihi);
  o[0] = __builtin_amdgcn_perm(hlo, llo, 0x05010400u);
  o[1] = __builtin_amdgcn_perm(hlo, llo, 0x07030602u);
  o[2] = __builtin_amdgcn_perm(hhi, lhi, 0x05010400u);
  o[3] = __builtin_amdgcn_perm(hhi, lhi, 0x07030602u);
}

__device__ inline moe_bf16x4 as_bf16x4(uint32_t lo, uint32_t hi) {
  union {
    uint2 u;
    moe_bf16x4 v;
  } x;
  x.u = make_uint2(lo, hi);
  return x.v;
}

// LDS of one block, 32 KB, so 2 blocks (16 wavefronts) fit in a CU's 64 KB.
// a[buffer][t][kq][row] holds the 4 bf16 that lane (row % 16, kq) feeds to
// the MFMA at k-step t of a step of 128.
struct MoeSmem {
  uint2 a[2][8][4][MOE_MT];
};

// Work queue of the persistent kernel: [0] next item, [1] blocks finished.
// The last block to finish resets both, so every launch starts from 0. One
// counter per device: calls on two streams of the same device must not
// overlap.
__device__ unsigned int g_moe_queue[2];

// The work list, kept in registers by every wavefront. Experts are taken in
// order of decreasing row count (ties: lower expert first), so the last items
// of a call belong to the smallest experts and the cheapest tiles fill the
// tail. Lane l of group g describes rank p = 64*g + l: expert[g] is the
// expert of that rank and pre[g] the number of row tiles (of MOE_MT rows)
// of all experts ranked before it.
struct MoeTiles {
  int expert[MOE_MAX_E / 64];
  int pre[MOE_MAX_E / 64];
  int total;
};

// `scratch` (E ints of LDS) is free before the first item; the caller puts a
// barrier between this and the first use of LDS.
__device__ inline MoeTiles moe_tile_prefix(const int *__restrict__ offsets,
                                           int E, int *scratch) {
  const int lane = threadIdx.x & 63;
  constexpr int G = MOE_MAX_E / 64;
  int rows[G], rank[G];
#pragma unroll
  for (int g = 0; g < G; g++) {
    const int e = g * 64 + lane;
    rows[g] = e < E ? offsets[e + 1] - offsets[e] : -1;
    rank[g] = 0;
  }
  // rank = experts with more rows, or as many rows and a lower number.
  for (int e2 = 0; e2 < E; e2++) {
    int r2 = 0;
#pragma unroll
    for (int g = 0; g < G; g++)
      if (e2 / 64 == g)
        r2 = __builtin_amdgcn_readlane(rows[g], e2 % 64);
#pragma unroll
    for (int g = 0; g < G; g++)
      rank[g] += (r2 > rows[g]) || (r2 == rows[g] && e2 < g * 64 + lane);
  }
#pragma unroll
  for (int g = 0; g < G; g++)
    if (g * 64 + lane < E)
      scratch[rank[g]] = g * 64 + lane; // every wavefront writes the same
  __syncthreads();
  MoeTiles t;
  int base = 0;
#pragma unroll
  for (int g = 0; g < G; g++) {
    const int p = g * 64 + lane;
    t.expert[g] = p < E ? scratch[p] : 0;
    const int e = t.expert[g];
    const int cnt = p < E ? (offsets[e + 1] - offsets[e] + MOE_MT - 1) / MOE_MT : 0;
    int incl = cnt; // inclusive scan over the wavefront
#pragma unroll
    for (int d = 1; d < 64; d <<= 1) {
      const int v = __shfl_up(incl, d, 64);
      if (lane >= d)
        incl += v;
    }
    t.pre[g] = base + incl - cnt;
    base += __shfl(incl, 63, 64);
  }
  t.total = base;
  return t;
}

// The rank that owns `item`: the last rank p with pre[p] * n_tiles <= item
// (empty experts have pre[p] == pre[p+1], so they are never picked).
__device__ inline int moe_find_rank(const MoeTiles &t, int E, int n_tiles,
                                    int item) {
  const int lane = threadIdx.x & 63;
  int count = 0;
#pragma unroll
  for (int g = 0; g < MOE_MAX_E / 64; g++)
    count += __popcll(__ballot(g * 64 + lane < E && t.pre[g] * n_tiles <= item));
  return count - 1;
}

// Thread (row, part) loads 16 k of one A row per step (k 16*part ..
// 16*part+15 of the 128, two 16-byte loads) and stores them in the MFMA
// order: they belong to lane group kq = part / 2 and k-steps t0 .. t0+3 with
// t0 = 4 * (part % 2); each 8 k give one even-k and one odd-k step.
__device__ inline void moe_store_a(MoeSmem &sm, int buf, int row, int part,
                                   const uint4 &x0, const uint4 &x1) {
  const uint32_t EVEN = 0x05040100u, ODD = 0x07060302u;
  const int kq = part >> 1, t0 = 4 * (part & 1);
  sm.a[buf][t0 + 0][kq][row] = make_uint2(__builtin_amdgcn_perm(x0.y, x0.x, EVEN),
                                          __builtin_amdgcn_perm(x0.w, x0.z, EVEN));
  sm.a[buf][t0 + 1][kq][row] = make_uint2(__builtin_amdgcn_perm(x0.y, x0.x, ODD),
                                          __builtin_amdgcn_perm(x0.w, x0.z, ODD));
  sm.a[buf][t0 + 2][kq][row] = make_uint2(__builtin_amdgcn_perm(x1.y, x1.x, EVEN),
                                          __builtin_amdgcn_perm(x1.w, x1.z, EVEN));
  sm.a[buf][t0 + 3][kq][row] = make_uint2(__builtin_amdgcn_perm(x1.y, x1.x, ODD),
                                          __builtin_amdgcn_perm(x1.w, x1.z, ODD));
}

// Buffer loads: a 128-bit descriptor in scalar registers holds the base
// address and size; each lane adds a 32-bit offset, and the per-step offset
// goes in a scalar register, so the loop does no 64-bit address arithmetic.
// Reads past the size return 0. Word 3 = 0x00020000 is the gfx9 default
// (32-bit data format).
__device__ inline __amdgpu_buffer_rsrc_t moe_rsrc(const void *base, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short)0,
                                           (int)bytes, 0x00020000);
}

__device__ inline uint4 moe_load16(__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
  const auto x = __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0);
  return make_uint4(x[0], x[1], x[2], x[3]);
}

__device__ inline uint2 moe_load8(__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
  const auto x = __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0);
  return make_uint2(x[0], x[1]);
}

// Per-lane load state of one work item.
struct MoeLanes {
  __amdgpu_buffer_rsrc_t a_rsrc, w_rsrc;
  uint32_t a_off;         // byte offset of this thread's 32 A bytes in step 0
  uint32_t w_off[MOE_NS]; // byte offset of this lane's 16 code bytes in step 0
  uint32_t w_step;        // bytes of one step of the item's packed tile
};

// Wave priorities (s_setprio; 0 is the default) for items of at most 32
// rows, in the three phases of a step: 2 while the loads of the next step
// are issued, 1 in the MFMA loop, 0 for the A store and the barrier. The
// SIMD issues from the ready wavefront of highest priority, so a wave that
// has just passed the barrier gets its loads out before the waves in their
// MFMA loop go on, and those go before a wave that only stores A and waits.
// These items are bound by the weight stream; earlier loads help it. Items
// of 48 and 64 rows keep the default: with priorities they were up to 12 %
// (48 rows, mlp2) and 1 % (64 rows) slower.
template <int MSA> struct MoePrio {
  static constexpr int load = MSA <= 2 ? 2 : 0, mfma = MSA <= 2 ? 1 : 0;
};

// One step of 128 k. Decodes the weights of step p from wq with the scale
// in the low byte of sc (then shifts sc down by one byte), refills wq with
// step p+1, loads the A slice of step p+1 and, after the 8 MFMA k-steps,
// stores it into the other LDS buffer. Past the last step the loads re-read
// the last one: with no branch in the loop body the compiler waits for
// exactly the loads each use needs (behind a branch it waited for all).
template <int MSA>
__device__ __forceinline__ void moe_step(MoeSmem &sm, int p, int np,
                                         uint4 (&wq)[MOE_NS],
                                         uint32_t (&sc)[MOE_NS][MOE_SREGS],
                                         const MoeLanes &ln, int arow, int apart,
                                         int j, int kq,
                                         moe_f32x4 (&acc)[MSA][MOE_NS]) {
  const int buf = p & 1;
  uint32_t q[MOE_NS][4];
  Fp4Tables tb[MOE_NS];
#pragma unroll
  for (int jn = 0; jn < MOE_NS; jn++) {
    q[jn][0] = wq[jn].x;
    q[jn][1] = wq[jn].y;
    q[jn][2] = wq[jn].z;
    q[jn][3] = wq[jn].w;
    tb[jn] = fp4_tables(sc[jn][0] & 0xffu);
#pragma unroll
    for (int r = 0; r + 1 < MOE_SREGS; r++)
      sc[jn][r] = __builtin_amdgcn_alignbit(sc[jn][r + 1], sc[jn][r], 8);
    sc[jn][MOE_SREGS - 1] >>= 8;
  }

  const int pn = min(p + 1, np - 1);
  if (MoePrio<MSA>::load)
    __builtin_amdgcn_s_setprio(MoePrio<MSA>::load);
  const uint4 x0 = moe_load16(ln.a_rsrc, ln.a_off, pn * MOE_KS * 2);
  const uint4 x1 = moe_load16(ln.a_rsrc, ln.a_off + 16, pn * MOE_KS * 2);
#pragma unroll
  for (int jn = 0; jn < MOE_NS; jn++)
    wq[jn] = moe_load16(ln.w_rsrc, ln.w_off[jn], pn * ln.w_step);
  // Keep the loads here, at the start of the step: left alone, the compiler
  // moves them to the end (to free registers), where the A loads' latency
  // is no longer hidden behind the MFMAs.
  __builtin_amdgcn_sched_barrier(0);
  if (MoePrio<MSA>::load != MoePrio<MSA>::mfma)
    __builtin_amdgcn_s_setprio(MoePrio<MSA>::mfma);

#pragma unroll
  for (int d = 0; d < 4; d++) {
    // dword d of the 16 code bytes: k 8d .. 8d+7 of this lane's 32;
    // k-step 2d takes the even k, 2d+1 the odd k
    uint32_t o[MOE_NS][4];
#pragma unroll
    for (int jn = 0; jn < MOE_NS; jn++)
      fp4x8_to_bf16(q[jn][d], tb[jn], o[jn]);
#pragma unroll
    for (int par = 0; par < 2; par++) {
      const int t = 2 * d + par;
      moe_bf16x4 af[MSA];
#pragma unroll
      for (int i = 0; i < MSA; i++) {
        const uint2 v = sm.a[buf][t][kq][i * 16 + j];
        af[i] = as_bf16x4(v.x, v.y);
      }
#pragma unroll
      for (int i = 0; i < MSA; i++)
#pragma unroll
        for (int jn = 0; jn < MOE_NS; jn++)
          acc[i][jn] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              af[i], as_bf16x4(o[jn][2 * par], o[jn][2 * par + 1]), acc[i][jn], 0, 0, 0);
    }
  }

  // And keep the reordering of A for the store here, after the MFMAs: done
  // early it would wait for the loads just issued.
  __builtin_amdgcn_sched_barrier(0);
  if (MoePrio<MSA>::mfma)
    __builtin_amdgcn_s_setprio(0);
  moe_store_a(sm, buf ^ 1, arow, apart, x0, x1);
  __syncthreads();
}

// One work item: rows m0 .. m0+mv-1 (mv <= 64) of expert e (C, G and gather
// already offset to the expert's first row), columns n0 .. n0+255. A holds
// the expert's rows, or, when gather is set, is the token matrix and row r of
// the expert is token gather[r] / topk. wblocks and wscales point to the
// expert's packed weights. MSA = number of 16-row tiles that hold valid rows
// (1 .. 4).
template <int MSA>
__device__ void moe_item(MoeSmem &sm, const bf16_t *__restrict__ A,
                         const uint8_t *__restrict__ wblocks,
                         const uint8_t *__restrict__ wscales,
                         const float *__restrict__ bias, float *__restrict__ C,
                         bf16_t *__restrict__ G, float limit,
                         const int *__restrict__ gather, int topk, int m0,
                         int mv, int n0, int N, int K) {
  const int tid = threadIdx.x;
  const int lane = tid & 63, wave = tid >> 6;
  const int j = lane & 15, kq = lane >> 4;
  const int np = (K + MOE_KS - 1) / MOE_KS;
  const int nc = min(16, (N + 15) / 16 - n0 / 16); // 16-column pieces in this tile

  // 1. Thread (arow, apart) copies 16 A values of row m0 + arow per step
  //    (rows past mv repeat the last one; their results are never stored).
  //    In a wavefront, lanes 2i and 2i+1 read the two halves of one 64-byte
  //    line. Lane (j, kq) of column tile jn reads piece c = 2 * wave + jn of
  //    the packed tile; pieces past nc read zeros (an offset past the end of
  //    the buffer: no memory traffic).
  const int arow = ((lane >> 1) & 31) + 32 * (wave & 1);
  const int apart = 2 * (wave >> 1) + (lane & 1);
  MoeLanes ln;
  const int r = m0 + min(arow, mv - 1);
  ln.a_rsrc = moe_rsrc(A, gather ? 0xffffffffu : (uint32_t)(m0 + mv) * K * 2);
  ln.a_off = (uint32_t)(gather ? gather[r] / topk : r) * K * 2 + 32 * apart;
  ln.w_rsrc = moe_rsrc(wblocks + (size_t)n0 * (K / 2), (uint32_t)nc * 8 * K);
  ln.w_step = (uint32_t)nc * 1024;
  const __amdgpu_buffer_rsrc_t s_rsrc =
      moe_rsrc(wscales + (size_t)(n0 / 16) * MOE_SPIECE, (uint32_t)nc * MOE_SPIECE);
  int ncol[MOE_NS];
  uint32_t s_off[MOE_NS];
#pragma unroll
  for (int jn = 0; jn < MOE_NS; jn++) {
    const int c = wave * MOE_NS + jn;
    ncol[jn] = n0 + 16 * c + j;
    ln.w_off[jn] = c < nc ? (uint32_t)(kq * nc + c) * 256 + 16 * j : 0x80000000u;
    s_off[jn] = (uint32_t)c * MOE_SPIECE;
  }

  // 2. Lane (j, kq) keeps the scales of groups 4p + kq, p = 0, 1, ..., of
  //    its column of each tile, 4 per register, step 0 in the low byte:
  //    24 contiguous bytes of the packed scales.
  static_assert(MOE_SREGS == 6, "the scale loads below fill 6 registers");
  uint32_t sc[MOE_NS][MOE_SREGS];
#pragma unroll
  for (int jn = 0; jn < MOE_NS; jn++) {
    const uint4 s0 = moe_load16(s_rsrc, s_off[jn] + 16 * lane, 0);
    const uint2 s1 = moe_load8(s_rsrc, s_off[jn] + 1024 + 8 * lane, 0);
    sc[jn][0] = s0.x;
    sc[jn][1] = s0.y;
    sc[jn][2] = s0.z;
    sc[jn][3] = s0.w;
    sc[jn][4] = s1.x;
    sc[jn][5] = s1.y;
  }
  // Keep the scale loads before the loads of step 0: the wait for the A
  // loads (before their LDS store) then also covers the scales. Issued after
  // them, a scale load can still be pending when the loop starts, and the
  // compiler then waits at the top of every loop turn for more loads than
  // the step needs.
  __builtin_amdgcn_sched_barrier(0);

  moe_f32x4 acc[MSA][MOE_NS];
#pragma unroll
  for (int i = 0; i < MSA; i++)
#pragma unroll
    for (int jn = 0; jn < MOE_NS; jn++)
      acc[i][jn] = moe_f32x4{0.0f, 0.0f, 0.0f, 0.0f};

  // The weights of step 0; then each step loads those of the next one.
  uint4 wq[MOE_NS];
  {
    const uint4 x0 = moe_load16(ln.a_rsrc, ln.a_off, 0);
    const uint4 x1 = moe_load16(ln.a_rsrc, ln.a_off + 16, 0);
#pragma unroll
    for (int jn = 0; jn < MOE_NS; jn++)
      wq[jn] = moe_load16(ln.w_rsrc, ln.w_off[jn], 0);
    moe_store_a(sm, 0, arow, apart, x0, x1);
  }
  __syncthreads();

  // Two steps per turn, so that the LDS buffer of each step is a constant;
  // the compiler then also alternates two register sets for wq. (Check the
  // machine code: a loop turn must not end with moves into wq after an
  // s_waitcnt vmcnt(0), which would wait for the loads of the next step.)
  int p = 0;
  for (; p + 1 < np; p += 2) {
    moe_step<MSA>(sm, p, np, wq, sc, ln, arow, apart, j, kq, acc);
    moe_step<MSA>(sm, p + 1, np, wq, sc, ln, arow, apart, j, kq, acc);
  }
  if (p < np)
    moe_step<MSA>(sm, p, np, wq, sc, ln, arow, apart, j, kq, acc);

  // Lane (j, kq) holds rows 4*kq + 0..3 of column j of each 16x16 tile.
  if (G == nullptr) {
#pragma unroll
    for (int jn = 0; jn < MOE_NS; jn++) {
      const int n = ncol[jn];
      if (n >= N)
        continue;
      const float bn = bias ? bias[n] : 0.0f;
#pragma unroll
      for (int i = 0; i < MSA; i++)
#pragma unroll
        for (int rr = 0; rr < 4; rr++) {
          const int m = i * 16 + kq * 4 + rr;
          if (m < mv)
            C[(size_t)(m0 + m) * N + n] = acc[i][jn][rr] + bn;
        }
    }
    return;
  }
  // SwiGLU: columns 2q (gate) and 2q+1 (up) sit in neighbouring lanes; the
  // even lane writes G[m, q] with the same arithmetic as swiglu_bf16_kernel.
#pragma unroll
  for (int jn = 0; jn < MOE_NS; jn++) {
    const int n = ncol[jn];
    const float bn = bias && n < N ? bias[n] : 0.0f;
#pragma unroll
    for (int i = 0; i < MSA; i++)
#pragma unroll
      for (int rr = 0; rr < 4; rr++) {
        const float y = acc[i][jn][rr] + bn;
        const float up_raw = __shfl_xor(y, 1, 64);
        const int m = i * 16 + kq * 4 + rr;
        if ((j & 1) == 0 && n < N && m < mv) {
          float val = fminf(y, limit);
          const float up = fminf(fmaxf(up_raw, -limit), limit);
          val = val / (1.0f + __expf(-1.702f * val));
          G[(size_t)(m0 + m) * (N / 2) + n / 2] = f32_to_bf16(val * (up + 1.0f));
        }
      }
  }
}

__global__ __launch_bounds__(MOE_THREADS) __attribute__((amdgpu_waves_per_eu(4)))
void gemm_grouped_mxfp4_mfma_kernel(
    const bf16_t *__restrict__ A, const int *__restrict__ offsets, int n_experts,
    const uint8_t *__restrict__ blocks, const uint8_t *__restrict__ scales,
    const float *__restrict__ bias, float *__restrict__ C, bf16_t *__restrict__ G,
    float limit, const int *__restrict__ row_pair, int topk, int N, int K) {
  __shared__ MoeSmem sm;
  const int tid = threadIdx.x;
  const int n_tiles = (N + MOE_NT - 1) / MOE_NT;
  const MoeTiles tiles = moe_tile_prefix(offsets, n_experts, (int *)&sm);
  const int total = tiles.total * n_tiles;
  const size_t wsize = moe_packed_blocks_bytes(N, K), ssize = moe_packed_scales_bytes(N, K);
  // The item number goes from thread 0 to the block through the last LDS
  // word, which is free between items (it is first written again at the end
  // of the next item's first step, after everyone has read it).
  volatile unsigned int *slot = (volatile unsigned int *)&sm.a[1][7][3][MOE_MT - 1];

  for (;;) {
    if (tid == 0)
      *slot = atomicAdd(&g_moe_queue[0], 1u);
    __syncthreads();
    // readfirstlane: the compiler cannot know that every lane read the same
    // word; without it, everything derived from the item (the A load
    // descriptor among them) would sit in vector registers, and each A load
    // would become a loop over the distinct descriptor values.
    const int item = __builtin_amdgcn_readfirstlane((int)*slot);
    if (item >= total)
      break;
    // Items: expert (largest first), then column tile, then row tile.
    const int p = moe_find_rank(tiles, n_experts, n_tiles, item);
    int e = 0, e_first = 0; // held by lane p % 64 of group p / 64
#pragma unroll
    for (int g = 0; g < MOE_MAX_E / 64; g++)
      if (p / 64 == g) {
        e = __builtin_amdgcn_readlane(tiles.expert[g], p % 64);
        e_first = __builtin_amdgcn_readlane(tiles.pre[g], p % 64);
      }
    const int r0 = offsets[e];
    const int rows = offsets[e + 1] - r0;
    const int e_tiles = (rows + MOE_MT - 1) / MOE_MT;
    const int rest = item - e_first * n_tiles;
    const int nt = rest / e_tiles, mt = rest % e_tiles;
    const int m0 = mt * MOE_MT;
    const int mv = min(MOE_MT, rows - m0);
    const bf16_t *Ae = row_pair ? A : A + (size_t)r0 * K;
    const int *gather = row_pair ? row_pair + r0 : nullptr;
    float *Ce = C ? C + (size_t)r0 * N : nullptr;
    bf16_t *Ge = G ? G + (size_t)r0 * (N / 2) : nullptr;
    const uint8_t *wb = blocks + e * wsize, *wsc = scales + e * ssize;
    const float *be = bias ? bias + (size_t)e * N : nullptr;
    switch ((mv + 15) >> 4) {
    case 1:
      moe_item<1>(sm, Ae, wb, wsc, be, Ce, Ge, limit, gather, topk, m0, mv, nt * MOE_NT, N, K);
      break;
    case 2:
      moe_item<2>(sm, Ae, wb, wsc, be, Ce, Ge, limit, gather, topk, m0, mv, nt * MOE_NT, N, K);
      break;
    case 3:
      moe_item<3>(sm, Ae, wb, wsc, be, Ce, Ge, limit, gather, topk, m0, mv, nt * MOE_NT, N, K);
      break;
    default:
      moe_item<4>(sm, Ae, wb, wsc, be, Ce, Ge, limit, gather, topk, m0, mv, nt * MOE_NT, N, K);
      break;
    }
  }

  // The last block out resets the queue for the next launch.
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(&g_moe_queue[1], 1u) == gridDim.x - 1) {
      atomicExch(&g_moe_queue[0], 0u);
      atomicExch(&g_moe_queue[1], 0u);
    }
  }
}

// ---------------------------------------------------------------------------
// Repacking the checkpoint layout (blocks (E, N, K/32, 16), scales (E, N, K/32))
// into the packed layout described at the top. One thread per group of 32
// codes (16 bytes), over the columns padded to a multiple of 16.
__global__ void moe_pack_blocks_kernel(const uint8_t *__restrict__ src,
                                       uint8_t *__restrict__ dst, int E, int N, int K) {
  const int kg = K / 32, n16 = (N + 15) / 16;
  const size_t per_e = (size_t)n16 * 16 * kg;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)E * per_e)
    return;
  const int e = (int)(idx / per_e);
  const size_t rem = idx % per_e;
  const int n = (int)(rem / kg), g = (int)(rem % kg);
  const int piece = n / 16, j = n % 16, t = piece / 16, c = piece % 16;
  const int nc = min(16, n16 - 16 * t);
  const int p = g / 4, kq = g % 4;
  const size_t off = (size_t)e * moe_packed_blocks_bytes(N, K) + (size_t)t * 128 * K +
                     (size_t)p * nc * 1024 + (size_t)(kq * nc + c) * 256 + 16 * j;
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (n < N)
    v = *(const uint4 *)(src + (((size_t)e * N + n) * kg + g) * 16);
  *(uint4 *)(dst + off) = v;
}

// One thread per (expert, 16-column piece, lane): that lane's 24 scale bytes.
__global__ void moe_pack_scales_kernel(const uint8_t *__restrict__ src,
                                       uint8_t *__restrict__ dst, int E, int N, int K) {
  const int kg = K / 32, n16 = (N + 15) / 16;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)E * n16 * 64)
    return;
  const int lane = (int)(idx % 64);
  const size_t piece = idx / 64; // e * n16 + P
  const int e = (int)(piece / n16), n = (int)(piece % n16) * 16 + lane % 16;
  const int kq = lane / 16;
  uint8_t *d = dst + piece * MOE_SPIECE;
  for (int i = 0; i < 4 * MOE_SREGS; i++) {
    const int g = 4 * i + kq;
    const uint8_t v = n < N && g < kg ? src[((size_t)e * N + n) * kg + g] : 0;
    if (i < 16)
      d[16 * lane + i] = v;
    else
      d[1024 + 8 * lane + (i - 16)] = v;
  }
}

// Writes the packed copy of n_experts experts' MXFP4 weights (blocks and
// scales in the checkpoint layout) to pblocks (n_experts *
// moe_packed_blocks_bytes(N, K) bytes) and pscales (n_experts *
// moe_packed_scales_bytes(N, K) bytes). Needs moe_mfma_ok(n_experts, N, K).
static void moe_pack_mxfp4(hipStream_t s, const uint8_t *blocks, const uint8_t *scales,
                           int n_experts, int N, int K, uint8_t *pblocks,
                           uint8_t *pscales) {
  const size_t nb = (size_t)n_experts * ((N + 15) / 16) * 16 * (K / 32);
  const size_t ns = (size_t)n_experts * ((N + 15) / 16) * 64;
  moe_pack_blocks_kernel<<<(unsigned)((nb + 255) / 256), 256, 0, s>>>(blocks, pblocks,
                                                                     n_experts, N, K);
  moe_pack_scales_kernel<<<(unsigned)((ns + 255) / 256), 256, 0, s>>>(scales, pscales,
                                                                     n_experts, N, K);
}

// ---------------------------------------------------------------------------
// Simple version (kept as the fallback for shapes the MFMA kernel does not
// take): MXFP4 decoded to fp32 in LDS, plain fp32 FMAs on 64x64 tiles.
template <class W>
__device__ void moe_tile_f32acc(const bf16_t *__restrict__ A, int M, W w,
                                const float *__restrict__ bias,
                                float *__restrict__ C, int N, int K, int m0,
                                int n0) {
  __shared__ float As[GEMM_BK][GEMM_BM + 1];
  __shared__ float Ws[GEMM_BK][GEMM_BN + 1];
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
    for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += 256) {
      const int r = idx / GEMM_BK, kk = idx % GEMM_BK;
      const int m = m0 + r, n = n0 + r;
      As[kk][r] = m < M ? bf16_to_f32(A[(size_t)m * K + k0 + kk]) : 0.0f;
      Ws[kk][r] = n < N ? w(n, k0 + kk) : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < GEMM_BK; kk++) {
      float a[4], b[4];
      for (int i = 0; i < 4; i++) {
        a[i] = As[kk][ty * 4 + i];
        b[i] = Ws[kk][tx * 4 + i];
      }
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++) {
    const int m = m0 + ty * 4 + i;
    if (m >= M)
      continue;
    for (int j = 0; j < 4; j++) {
      const int n = n0 + tx * 4 + j;
      if (n < N)
        C[(size_t)m * N + n] = acc[i][j] + (bias ? bias[n] : 0.0f);
    }
  }
}

__global__ __launch_bounds__(256) void gemm_grouped_mxfp4_f32acc_kernel(
    const bf16_t *__restrict__ A, const int *__restrict__ offsets,
    const uint8_t *__restrict__ blocks, const uint8_t *__restrict__ scales,
    const float *__restrict__ bias, float *__restrict__ C, int N, int K) {
  const int e = blockIdx.z;
  const int r0 = offsets[e];
  const int rows = offsets[e + 1] - r0;
  const int m0 = blockIdx.y * GEMM_BM;
  if (m0 >= rows)
    return;
  const size_t groups = (size_t)N * (K / 32);
  Mxfp4Rows w{blocks + e * groups * 16, scales + e * groups, K};
  moe_tile_f32acc(A + (size_t)r0 * K, rows, w, bias + (size_t)e * N,
                  C + (size_t)r0 * N, N, K, m0, blockIdx.x * GEMM_BN);
}

static void gemm_grouped_mxfp4_simple(hipStream_t s, const bf16_t *A,
                                      const int *offsets, int max_rows,
                                      int n_experts, const uint8_t *blocks,
                                      const uint8_t *scales, const float *bias,
                                      float *C, int N, int K) {
  dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (max_rows + GEMM_BM - 1) / GEMM_BM,
            n_experts);
  gemm_grouped_mxfp4_f32acc_kernel<<<grid, 256, 0, s>>>(A, offsets, blocks,
                                                        scales, bias, C, N, K);
}

// Blocks of the persistent MFMA kernel that fit on the device at once.
static int moe_mfma_grid(int max_items) {
  static int cached[64] = {0};
  int dev = 0;
  hipGetDevice(&dev);
  if (dev < 0 || dev >= 64)
    dev = 0;
  if (cached[dev] == 0) {
    int cus = 0, per_cu = 0;
    hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, gemm_grouped_mxfp4_mfma_kernel,
                                                 MOE_THREADS, 0);
    cached[dev] = std::max(1, cus * std::max(1, per_cu));
  }
  return std::max(1, std::min(cached[dev], max_items));
}

// The MFMA kernel needs K a multiple of 64 and at most 4 * MOE_SREGS * 128
// (the scale registers), at most MOE_MAX_E experts, and one expert's packed
// weight bytes below 2^31 (offsets from 2^31 up serve as "past the end").
// The fast entry points take the packed weights (moe_pack_mxfp4) exactly
// when this holds, and the checkpoint layout otherwise (simple version).
static bool moe_mfma_ok(int n_experts, int N, int K) {
  return K % 64 == 0 && K <= 4 * MOE_SREGS * MOE_KS && n_experts <= MOE_MAX_E &&
         moe_packed_blocks_bytes(N, K) < 0x80000000ull;
}

static void gemm_grouped_mxfp4_launch(hipStream_t s, const bf16_t *A,
                                      const int *row_pair, int topk,
                                      const int *offsets, int max_rows,
                                      int n_experts, const uint8_t *blocks,
                                      const uint8_t *scales, const float *bias,
                                      float *C, bf16_t *G, float limit, int N,
                                      int K) {
  const long long max_items = (long long)((N + MOE_NT - 1) / MOE_NT) * n_experts *
                              ((max_rows + MOE_MT - 1) / MOE_MT);
  const int grid = moe_mfma_grid((int)std::min<long long>(max_items, 1 << 30));
  gemm_grouped_mxfp4_mfma_kernel<<<grid, MOE_THREADS, 0, s>>>(
      A, offsets, n_experts, blocks, scales, bias, C, G, limit, row_pair, topk, N, K);
}

// Rows offsets[e] .. offsets[e+1] of A times expert e's MXFP4 weight, plus its
// bias, into the same rows of C. offsets: device array of n_experts+1 ints.
// A (R, K) bf16 row-major; C (R, N) fp32; bias (n_experts, N) fp32.
// max_rows bounds any expert's row count. blocks / scales: packed by
// moe_pack_mxfp4 when moe_mfma_ok(n_experts, N, K), else the checkpoint layout.
static void gemm_grouped_mxfp4(hipStream_t s, const bf16_t *A,
                               const int *offsets, int max_rows, int n_experts,
                               const uint8_t *blocks, const uint8_t *scales,
                               const float *bias, float *C, int N, int K) {
  if (!moe_mfma_ok(n_experts, N, K)) {
    gemm_grouped_mxfp4_simple(s, A, offsets, max_rows, n_experts, blocks,
                              scales, bias, C, N, K);
    return;
  }
  gemm_grouped_mxfp4_launch(s, A, nullptr, 1, offsets, max_rows, n_experts,
                            blocks, scales, bias, C, nullptr, 0.0f, N, K);
}

// Fusion for mlp1: the same multiply,
// then gpt-oss's SwiGLU on each (gate, up) column pair, written as bf16 to
// G (R, N/2) — what gemm_grouped_mxfp4 followed by swiglu_bf16_kernel
// produce, bit for bit, without writing and re-reading the fp32 (R, N) C.
// Needs moe_mfma_ok(n_experts, N, K), N even and packed weights.
static void gemm_grouped_mxfp4_swiglu(hipStream_t s, const bf16_t *A,
                                      const int *offsets, int max_rows,
                                      int n_experts, const uint8_t *blocks,
                                      const uint8_t *scales, const float *bias,
                                      bf16_t *G, float limit, int N, int K) {
  gemm_grouped_mxfp4_launch(s, A, nullptr, 1, offsets, max_rows, n_experts,
                            blocks, scales, bias, nullptr, G, limit, N, K);
}

// Fusion for mlp1 without a partner GCD: gather, multiply and
// SwiGLU in one kernel. X (B, K) bf16 holds one row per token; row r of the
// grouped order is token row_pair[r] / topk (moe_group_kernel's row_pair).
// Equal, bit for bit, to gather_bf16_kernel + gemm_grouped_mxfp4 +
// swiglu_bf16_kernel. Needs moe_mfma_ok(n_experts, N, K), N even and packed
// weights.
static void gemm_grouped_mxfp4_gather_swiglu(hipStream_t s, const bf16_t *X,
                                             const int *row_pair, int topk,
                                             const int *offsets, int max_rows,
                                             int n_experts, const uint8_t *blocks,
                                             const uint8_t *scales,
                                             const float *bias, bf16_t *G,
                                             float limit, int N, int K) {
  gemm_grouped_mxfp4_launch(s, X, row_pair, topk, offsets, max_rows, n_experts,
                            blocks, scales, bias, nullptr, G, limit, N, K);
}
