// Fast-path dense matrix multiply on the matrix cores:
//   C (M, N) fp32 = A (M, K) bf16 · W (N, K)^T  (+ bias (N) fp32)
// Used for qkv, attention out, router and LM head.
//
// Both inputs are stored k-contiguous (A row-major, W row-major (N, K)), which
// is exactly what the MFMA instruction wants: each lane feeds it a few
// consecutive k values of one row of A and one row of W.
//
// MFMA used: v_mfma_f32_32x32x8bf16_1k. One wavefront (64 lanes) multiplies a
// 32x8 slice of A by an 8x32 slice of W^T and adds the 32x32 product to fp32
// accumulators that live in registers:
//   A operand: lane l gives row (l % 32), k = 4 * (l / 32) + 0..3 (4 bf16)
//   B operand: lane l gives col (l % 32), k = 4 * (l / 32) + 0..3 (4 bf16)
//   C/D:       lane l holds col (l % 32), rows 8 * (v / 4) + 4 * (l / 32) + v % 4
//              for v = 0..15 (16 fp32)
// Which physical k a "k slot" stands for does not matter as long as A and W
// use the same mapping, so every lane reads 8 consecutive k values (16 bytes)
// with one load and spends them on two MFMAs: k 0..3 then k 4..7 of its 8.
// Products of two bf16 numbers are exact in fp32 and the sums are fp32, so
// the result matches an fp32 reference to fp32 rounding (~1e-6 relative).
//
// Kernels:
// - gemm_mfma_kernel: one block per BM x BN output tile, optionally split
//   along K into slices that add up in a fixed order. Tile sizes are picked
//   per shape (gemm_pick_config at the end of this file).
// - gemm_splitk_kernel: for outputs too narrow to fill the GPU (the router,
//   N = 32 or 128). One 32x32 output tile per block; the block's waves split
//   K between them, read straight from global memory, and add their partial
//   sums through LDS in a fixed order (deterministic).
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>

#include "fast_common.hpp"

namespace fgemm {

typedef short bf16x4 __attribute__((ext_vector_type(4)));
typedef short bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x16 __attribute__((ext_vector_type(16)));

__device__ inline bf16x4 lo4(bf16x8 v) { return __builtin_shufflevector(v, v, 0, 1, 2, 3); }
__device__ inline bf16x4 hi4(bf16x8 v) { return __builtin_shufflevector(v, v, 4, 5, 6, 7); }

// Two MFMA k-steps: 8 consecutive k values of a row of A times the same 8 of
// a row of W, added into the 32x32 accumulator c.
__device__ inline f32x16 mfma_k8x2(bf16x8 a, bf16x8 w, f32x16 c) {
  c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(lo4(a), lo4(w), c, 0, 0, 0);
  c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(hi4(a), hi4(w), c, 0, 0, 0);
  return c;
}

// LDS tile layout: row r of the tile is VPR 16-byte vectors (VPR = BK / 8).
// Vector c of row r is stored at slot c ^ swz(r) of that row. LDS has 32
// banks of 4 bytes; a 16-byte access by 8 neighbouring lanes is conflict-free
// only if the 8 lanes hit 8 different 16-byte bank groups. The lanes of one
// MFMA operand read 8 consecutive rows at the same c, so the XOR spreads them
// over all 8 groups. (Without it, with 64-byte rows, rows r and r+2 collide.)
// The copy writes 8 consecutive vectors of one or two rows: also conflict-free.
template <int VPR>
__device__ inline int lds_slot(int r, int c) {
  constexpr int rows_per_128B = VPR >= 8 ? 1 : 8 / VPR;
  return r * VPR + (c ^ ((r / rows_per_128B) % VPR));
}

// Spin until *flag == value (one thread), then make the other blocks' writes
// visible to the whole block.
__device__ inline void wait_flag(const unsigned *flag, unsigned value) {
  if (threadIdx.x == 0)
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != value)
      __builtin_amdgcn_s_sleep(1);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// Publish this block's global writes, then set *flag = value.
__device__ inline void set_flag(unsigned *flag, unsigned value) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

//                  K -->
//        A tile  +--------+        W tile (BN x BK, rows = output columns)
//   BM   rows of |  sA    |        +--------+
//   ...          +--------+   x    |  sW    |^T   =  C tile (BM x BN)
//                                  +--------+
//   waves:  WAVES_M x WAVES_N, wave (wm, wn) owns rows wm*WM.., cols wn*WN..
//
// Software pipeline (mainloop). K step kt uses LDS buffer kt & 1 and is split
// in NC = BK / 16 chunks of 16 k values:
//   chunk 0      : read operands of chunk 1 from LDS; copy K step kt+1 (in
//                  registers since the previous step) into the other LDS
//                  buffer and start loading K step kt+2 from global memory
//                  into the freed registers | MFMAs of chunk 0
//   chunk c      : read operands of chunk c+1 | MFMAs of chunk c
//   last chunk   : barrier, read chunk 0 of K step kt+1 | MFMAs of last chunk
// So LDS reads overlap MFMAs, global loads have a whole K step to arrive,
// the LDS writes have finished long before the barrier, and there is one
// barrier per K step. The other buffer is free at chunk 0 because every wave
// read it last before the previous barrier. Loads past the end of K are
// clamped to the last K step (harmless repeats), so the loop has no branches.
template <int BM, int BN, int BK, int WAVES_M, int WAVES_N>
struct TileGemm {
  static constexpr int NT = 64 * WAVES_M * WAVES_N;
  static constexpr int WM = BM / WAVES_M, WN = BN / WAVES_N; // one wave's tile
  static constexpr int TM = WM / 32, TN = WN / 32;           // its MFMA accumulators
  static constexpr int VPR = BK / 8;                         // 16-byte vectors per row
  static constexpr int NC = BK / 16;                         // 16-wide k chunks per step
  static constexpr int A_VECS = BM * VPR, W_VECS = BN * VPR;
  static constexpr int A_LOADS = (A_VECS + NT - 1) / NT, W_LOADS = (W_VECS + NT - 1) / NT;
  static_assert(WM * WAVES_M == BM && WN * WAVES_N == BN, "tile shape");
  static_assert(WM % 32 == 0 && WN % 32 == 0 && BK % 32 == 0, "tile shape");
  static_assert(2 * (BM + BN) * BK * 2 <= 65536, "LDS holds 64 KB");

  struct Smem {
    bf16x8 a[2][A_VECS];
    bf16x8 w[2][W_VECS];
  };
  // Waves per SIMD that LDS allows (4 SIMDs per compute unit); the register
  // allocator is told to stay within that occupancy's register budget.
  static constexpr int WAVES_PER_SIMD =
      std::max(1, std::min(8, (int)(65536 / sizeof(Smem)) * (NT / 64) / 4));
  typedef f32x16 Acc[TM][TN];

  // acc = sum over K steps [k_begin, k_begin + nk) of tile (m0, n0); nk >= 1.
  __device__ static void mainloop(Smem &sm, const bf16_t *__restrict__ A, int M,
                                  const bf16_t *__restrict__ W, int N, int K, int m0,
                                  int n0, int k_begin, int nk, Acc &acc) {
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;

    // Copy slots: thread tid copies vectors tid, tid + NT, ... of each tile.
    // When the tile is not a multiple of NT vectors, the spare threads repeat
    // the last vector (same value to the same LDS slot). Rows past M or N are
    // clamped to the last row: they are computed but never stored, which
    // avoids branches in the main loop.
    const bf16x8 *ga[A_LOADS], *gw[W_LOADS];
    int la[A_LOADS], lw[W_LOADS];
#pragma unroll
    for (int i = 0; i < A_LOADS; i++) {
      const int v = min(tid + i * NT, A_VECS - 1), r = v / VPR, c = v % VPR;
      ga[i] = (const bf16x8 *)(A + (size_t)min(m0 + r, M - 1) * K) + k_begin * VPR + c;
      la[i] = lds_slot<VPR>(r, c);
    }
#pragma unroll
    for (int i = 0; i < W_LOADS; i++) {
      const int v = min(tid + i * NT, W_VECS - 1), r = v / VPR, c = v % VPR;
      gw[i] = (const bf16x8 *)(W + (size_t)min(n0 + r, N - 1) * K) + k_begin * VPR + c;
      lw[i] = lds_slot<VPR>(r, c);
    }

    bf16x8 ra[A_LOADS], rw[W_LOADS];         // next K step, on its way to LDS
    bf16x8 fa[2][TM], fw[2][TN];             // MFMA operands, double buffered
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++)
        acc[i][j] = f32x16{};

    // All indices below are compile-time constants after unrolling, so every
    // array stays in registers.
    auto load = [&](int kt) { // global -> registers
      kt = min(kt, nk - 1);
#pragma unroll
      for (int i = 0; i < A_LOADS; i++)
        ra[i] = ga[i][kt * VPR];
#pragma unroll
      for (int i = 0; i < W_LOADS; i++)
        rw[i] = gw[i][kt * VPR];
    };
    auto store = [&](int buf) { // registers -> LDS
#pragma unroll
      for (int i = 0; i < A_LOADS; i++)
        sm.a[buf][la[i]] = ra[i];
#pragma unroll
      for (int i = 0; i < W_LOADS; i++)
        sm.w[buf][lw[i]] = rw[i];
    };
    auto read_frags = [&](int buf, int kc, int f) { // LDS -> MFMA operands
      // Lanes 0-31 take k 0..7 of the 16-wide chunk, lanes 32-63 k 8..15.
      const int c = kc * 2 + (lane >> 5);
#pragma unroll
      for (int i = 0; i < TM; i++)
        fa[f][i] = sm.a[buf][lds_slot<VPR>(wm * WM + i * 32 + (lane & 31), c)];
#pragma unroll
      for (int j = 0; j < TN; j++)
        fw[f][j] = sm.w[buf][lds_slot<VPR>(wn * WN + j * 32 + (lane & 31), c)];
    };
    auto mfmas = [&](int f) { // 2 MFMA k-steps on every accumulator
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++)
            acc[i][j] = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
                h ? hi4(fa[f][i]) : lo4(fa[f][i]), h ? hi4(fw[f][j]) : lo4(fw[f][j]),
                acc[i][j], 0, 0, 0);
    };

    // Instruction order inside one region between barriers: the LLVM
    // scheduler is told to alternate MFMAs with the memory instructions of the
    // region (sched_group_barrier: mask 0x008 MFMA, 0x100 LDS read, 0x200 LDS
    // write, 0x020 global read), so the matrix unit never waits behind a burst
    // of loads, and LDS reads are issued early enough to arrive in time.
    constexpr int N_MFMA = 2 * TM * TN, N_DSR = TM + TN, N_COPY = A_LOADS + W_LOADS;
    // Chunks 0..NC-2 (one region): every chunk alternates its MFMAs with the
    // next chunk's LDS reads, then with its share of the copy (LDS write of
    // K step kt+1 + global load of K step kt+2).
    auto sched_before_barrier = [&]() {
      constexpr int PER_CHUNK = (N_COPY + NC - 2) / (NC - 1);
#pragma unroll
      for (int c = 0; c < NC - 1; c++)
#pragma unroll
        for (int i = 0; i < N_MFMA; i++) {
          __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
          const int copy = c * PER_CHUNK + i - N_DSR;
          if (i < N_DSR)
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
          else if (i - N_DSR < PER_CHUNK && copy < N_COPY) {
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          }
        }
    };
    // Last chunk (after the barrier): the LDS reads of chunk 0 of the next K
    // step go out first, the MFMAs cover their latency.
    auto sched_after_barrier = [&]() {
#pragma unroll
      for (int i = 0; i < N_MFMA; i++) {
        if (i < N_DSR)
          __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
      }
    };

    load(0);
    store(0);
    load(1);
    __syncthreads();
    read_frags(0, 0, 0);

    for (int kt = 0; kt < nk; kt++) {
      const int b = kt & 1;
      // chunk 0: also copy K step kt+1 to the other buffer, load kt+2
      read_frags(b, 1, 1);
      store(b ^ 1);
      load(kt + 2);
      mfmas(0);
#pragma unroll
      for (int c = 1; c < NC - 1; c++) {
        read_frags(b, c + 1, (c + 1) & 1);
        mfmas(c & 1);
      }
      sched_before_barrier();
      __syncthreads();
      // last chunk: its MFMAs cover the LDS reads of chunk 0 of K step kt+1
      read_frags(b ^ 1, 0, 0);
      mfmas(1);
      sched_after_barrier();
    }
  }

  // This lane's bias values (one per output column it writes), loaded before
  // the main loop so they have long arrived when the results are written.
  __device__ static void load_bias(const float *__restrict__ bias, int N, int n0,
                                   float (&bv)[TN]) {
    const int lane = threadIdx.x & 63, wn = (threadIdx.x >> 6) / WAVES_M;
#pragma unroll
    for (int j = 0; j < TN; j++)
      bv[j] = bias ? bias[min(n0 + wn * WN + j * 32 + (lane & 31), N - 1)] : 0.0f;
  }

  // Writes acc + bv (or + the value already in C) to the tile (m0, n0).
  // Each lane writes column n of 16 rows per accumulator; for a fixed v the
  // 64 lanes write two runs of 32 consecutive floats.
  __device__ static void store_c(float *__restrict__ C, int M, int N, int m0, int n0,
                                 const float (&bv)[TN], bool add_to_c, Acc &acc) {
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    const bool full = m0 + BM <= M && n0 + BN <= N;
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const int n = n0 + wn * WN + j * 32 + (lane & 31);
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int v = 0; v < 16; v++) {
          const int m = m0 + wm * WM + i * 32 + 8 * (v / 4) + 4 * (lane >> 5) + (v % 4);
          float *c = &C[(size_t)m * N + n];
          if (full || (m < M && n < N))
            *c = acc[i][j][v] + (add_to_c ? *c : bv[j]);
        }
    }
  }
};

// One flag per output tile for split K (one copy per GCD). Values only ever
// grow: every call gets a fresh range of 16 values (epoch), so the flags never
// need clearing. Calls that use split K must not run concurrently on the same
// GCD (one stream per GCD, as in the engine).
constexpr int kMaxSplitTiles = 16384;
__device__ unsigned g_split_flags[kMaxSplitTiles];

// Data parallel: block = one tile (m fastest: consecutive blocks share the
// same W columns, so a W tile is read from memory once and found in the L2
// cache by the other row blocks).
// Split K (k_slices > 1): the grid holds k_slices copies of the tile grid;
// copy s computes K steps [s * nk / k_slices, (s + 1) * nk / k_slices) of its
// tile. Slice 0 stores its sum plus the bias, slice s > 0 waits until
// g_split_flags[tile] says slice s - 1 has stored, then adds its sum to C. The
// order of the additions is fixed, so the result is the same on every run.
// Slice s - 1 always has a lower block index, so it is already running or
// done when slice s starts waiting (blocks are dispatched in index order).
template <int BM, int BN, int BK, int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(64 * WAVES_M *WAVES_N) __attribute__((amdgpu_waves_per_eu(
    TileGemm<BM, BN, BK, WAVES_M, WAVES_N>::WAVES_PER_SIMD))) void gemm_mfma_kernel(
    const bf16_t *__restrict__ A, int M, const bf16_t *__restrict__ W,
    const float *__restrict__ bias, float *__restrict__ C, int N, int K,
    int m_blocks, int k_slices, unsigned epoch) {
  typedef TileGemm<BM, BN, BK, WAVES_M, WAVES_N> T;
  __shared__ typename T::Smem sm;
  const int tiles = gridDim.x / k_slices;
  const int tile = blockIdx.x % tiles, ks = blockIdx.x / tiles;
  const int m0 = (tile % m_blocks) * BM, n0 = (tile / m_blocks) * BN;
  const int nk_all = K / BK;
  const int k_begin = nk_all * ks / k_slices;
  const int nk = nk_all * (ks + 1) / k_slices - k_begin;

  float bv[T::TN];
  T::load_bias(ks == 0 ? bias : nullptr, N, n0, bv);
  typename T::Acc acc;
  T::mainloop(sm, A, M, W, N, K, m0, n0, k_begin, nk, acc);
  if (ks > 0)
    wait_flag(&g_split_flags[tile], epoch + ks);
  T::store_c(C, M, N, m0, n0, bv, ks > 0, acc);
  if (ks + 1 < k_slices)
    set_flag(&g_split_flags[tile], epoch + ks + 1);
}

// One 32x32 output tile per block, KW waves splitting K: wave w takes the
// 32-wide k chunks w, w + KW, w + 2 KW, ... Within a chunk lane l reads 16
// consecutive k values (two 16-byte loads) of its row of A and of W.
template <int KW>
__global__ __launch_bounds__(64 * KW) void gemm_splitk_kernel(
    const bf16_t *__restrict__ A, int M, const bf16_t *__restrict__ W,
    const float *__restrict__ bias, float *__restrict__ C, int N, int K,
    int m_blocks) {
  __shared__ float part[KW - 1][16][64];
  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  const int m0 = (blockIdx.x % m_blocks) * 32;
  const int n0 = (blockIdx.x / m_blocks) * 32;
  const int koff = (lane >> 5) * 16;
  const bf16x8 *pa =
      (const bf16x8 *)(A + (size_t)min(m0 + (lane & 31), M - 1) * K + koff);
  const bf16x8 *pw =
      (const bf16x8 *)(W + (size_t)min(n0 + (lane & 31), N - 1) * K + koff);

  const int n = n0 + (lane & 31);
  const float b = bias ? bias[min(n, N - 1)] : 0.0f;

  f32x16 acc = {};
  const int nchunks = K / 32;
  int ch = wave;
  for (; ch + KW < nchunks; ch += 2 * KW) { // two chunks in flight
    const bf16x8 a0 = pa[ch * 4], a1 = pa[ch * 4 + 1];
    const bf16x8 w0 = pw[ch * 4], w1 = pw[ch * 4 + 1];
    const bf16x8 a2 = pa[(ch + KW) * 4], a3 = pa[(ch + KW) * 4 + 1];
    const bf16x8 w2 = pw[(ch + KW) * 4], w3 = pw[(ch + KW) * 4 + 1];
    acc = mfma_k8x2(a0, w0, acc);
    acc = mfma_k8x2(a1, w1, acc);
    acc = mfma_k8x2(a2, w2, acc);
    acc = mfma_k8x2(a3, w3, acc);
  }
  if (ch < nchunks) {
    acc = mfma_k8x2(pa[ch * 4], pw[ch * 4], acc);
    acc = mfma_k8x2(pa[ch * 4 + 1], pw[ch * 4 + 1], acc);
  }

  if (wave > 0)
#pragma unroll
    for (int v = 0; v < 16; v++)
      part[wave - 1][v][lane] = acc[v];
  __syncthreads();
  if (wave > 0)
    return;
#pragma unroll
  for (int w = 0; w < KW - 1; w++)
#pragma unroll
    for (int v = 0; v < 16; v++)
      acc[v] += part[w][v][lane];
  if (n >= N)
    return;
#pragma unroll
  for (int v = 0; v < 16; v++) {
    const int m = m0 + 8 * (v / 4) + 4 * (lane >> 5) + (v % 4);
    if (m < M)
      C[(size_t)m * N + n] = acc[v] + b;
  }
}

static unsigned next_epoch() {
  static std::atomic<unsigned> epoch{0};
  return (epoch.fetch_add(1) + 1) * 16;
}

// k_slices > 1: split K into that many slices (at most 15 and K / BK).
template <int BM, int BN, int BK, int WAVES_M, int WAVES_N>
static void launch_tiled(hipStream_t s, const bf16_t *A, int M, const bf16_t *W,
                         const float *bias, float *C, int N, int K, int k_slices) {
  const int mb = (M + BM - 1) / BM, nb = (N + BN - 1) / BN;
  const int threads = 64 * WAVES_M * WAVES_N;
  k_slices = std::max(1, std::min({k_slices, 15, K / BK}));
  if (mb * nb > kMaxSplitTiles)
    k_slices = 1;
  const unsigned epoch = k_slices > 1 ? next_epoch() : 0;
  gemm_mfma_kernel<BM, BN, BK, WAVES_M, WAVES_N>
      <<<mb * nb * k_slices, threads, 0, s>>>(A, M, W, bias, C, N, K, mb, k_slices, epoch);
}

template <int KW>
static void launch_splitk(hipStream_t s, const bf16_t *A, int M, const bf16_t *W,
                          const float *bias, float *C, int N, int K, int) {
  const int mb = (M + 31) / 32, nb = (N + 31) / 32;
  gemm_splitk_kernel<KW><<<mb * nb, 64 * KW, 0, s>>>(A, M, W, bias, C, N, K, mb);
}

typedef void (*GemmLaunch)(hipStream_t, const bf16_t *, int, const bf16_t *,
                           const float *, float *, int, int, int k_slices);

struct GemmConfig {
  const char *name;
  GemmLaunch launch;
  int bk; // K must be a multiple of this
};

// The kernel shapes gemm_bf16() chooses from: block tile BM x BN x BK with
// WAVES_M x WAVES_N waves. "LDS" is the double-buffered tile copy; 64 KB of
// LDS per compute unit (CU) decides how many blocks share a CU.
// The benchmark (bench/gemm_bench.cpp --sweep) times all of them.
enum {                // wave tile, LDS, blocks per CU
  CFG_128x128x32_2x4, // 64x32,  32 KB, 2 (4 waves per SIMD)
  CFG_128x256x32_2x4, // 64x64,  48 KB, 1
  CFG_256x256x32_2x4, // 128x64, 64 KB, 1
  CFG_128x128x64_2x4, // 64x32,  64 KB, 1 (half the barriers per K)
  CFG_64x128x64_2x4,  // 32x32,  48 KB, 1 (small M: more blocks)
  CFG_64x64x64_2x2,   // 32x32,  32 KB, 2 (small M or N)
  CFG_128x128x32_2x2, // 64x64,  32 KB, 2 (4 waves: the first version)
  CFG_ROUTER_4W,      // gemm_splitk_kernel, 32x32 tiles, 4 waves split K
  CFG_ROUTER_8W,      // same with 8 waves
};
static const GemmConfig kGemmConfigs[] = {
    {"128x128x32 2x4", launch_tiled<128, 128, 32, 2, 4>, 32},
    {"128x256x32 2x4", launch_tiled<128, 256, 32, 2, 4>, 32},
    {"256x256x32 2x4", launch_tiled<256, 256, 32, 2, 4>, 32},
    {"128x128x64 2x4", launch_tiled<128, 128, 64, 2, 4>, 64},
    {"64x128x64 2x4", launch_tiled<64, 128, 64, 2, 4>, 64},
    {"64x64x64 2x2", launch_tiled<64, 64, 64, 2, 2>, 64},
    {"128x128x32 2x2", launch_tiled<128, 128, 32, 2, 2>, 32},
    {"splitk32x32 4w", launch_splitk<4>, 32},
    {"splitk32x32 8w", launch_splitk<8>, 32},
};
static const int kNumGemmConfigs = sizeof(kGemmConfigs) / sizeof(kGemmConfigs[0]);

// Index into kGemmConfigs for this shape, and how many K slices to use.
// Picked from bench/gemm_bench.cpp --sweep on one MI250 GCD (fastest at each
// M of 64..2048 for the gpt-oss shapes; a batch between two measured sizes
// uses the choice of the larger one). Other shapes get the qkv choices.
static int gemm_pick_config(int M, int N, int K, int *k_slices) {
  *k_slices = 1;
  const bool k64 = K % 64 == 0;
  if (N <= 256) { // router: 32 or 128 outputs
    if (M <= 512)
      return CFG_ROUTER_4W;
    *k_slices = 2;
    return k64 ? CFG_64x64x64_2x2 : CFG_ROUTER_8W;
  }
  if (N == 2880 && K == 4096) { // attention out
    if (M <= 64) {
      *k_slices = 2;
      return CFG_64x64x64_2x2;
    }
    if (M <= 128) {
      *k_slices = 2;
      return CFG_64x128x64_2x4;
    }
    if (M <= 256)
      return CFG_64x128x64_2x4;
    if (M <= 512)
      return CFG_128x128x64_2x4;
    if (M <= 1024)
      return CFG_128x256x32_2x4;
    return CFG_256x256x32_2x4;
  }
  if (N > 65536) { // LM head
    if (M <= 64)
      return k64 ? CFG_64x128x64_2x4 : CFG_128x128x32_2x4;
    if (M <= 128)
      return k64 ? CFG_128x128x64_2x4 : CFG_128x128x32_2x4;
    if (M <= 256 || M > 512)
      return CFG_256x256x32_2x4;
    return CFG_128x128x32_2x4;
  }
  // qkv and anything else
  if (M <= 128 && k64) {
    *k_slices = M <= 64 ? 2 : 1;
    return CFG_64x128x64_2x4;
  }
  if (M <= 512 || M > 1024)
    return CFG_128x128x32_2x4;
  return CFG_256x256x32_2x4;
}

} // namespace fgemm

// C = A · W^T (+ bias). A (M, K) bf16 row-major, W (N, K) bf16 row-major,
// C (M, N) fp32 row-major, bias (N) fp32 or nullptr. K must be a multiple of
// 32 and A, W 16-byte aligned. Any M, N >= 1.
static void gemm_bf16(hipStream_t s, const bf16_t *A, int M, const bf16_t *W,
                      const float *bias, float *C, int N, int K) {
  if (M <= 0 || N <= 0)
    return;
  int k_slices = 1;
  const fgemm::GemmConfig &cfg = fgemm::kGemmConfigs[fgemm::gemm_pick_config(M, N, K, &k_slices)];
  cfg.launch(s, A, M, W, bias, C, N, K, k_slices);
}
