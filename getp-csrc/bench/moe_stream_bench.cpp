// Memory ceiling of the expert multiply's weight stream: the reads that
// gemm_grouped_mxfp4_mfma_kernel (kernels/fast_moe_gemm.hpp) makes for the
// packed codes, without A, decoding or MFMAs.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a \
//     getp-csrc/bench/moe_stream_bench.cpp -o moe_stream_bench
// Run:
//   tools/gpulock.sh <hip id> -- ./moe_stream_bench
//
// 32 experts x 23 tiles of 256 columns (gpt-oss mlp1, K = 2880): 736 work
// items of 23 steps of 16 KB, taken from a queue by a persistent grid of
// 512-thread blocks, as in the kernel. Per step each thread loads 2 x 16
// bytes (lane (j, kq) of wavefront w, piece c = 2w + jn: offset
// (16 kq + c) * 256 + 16 j of the step's 16 KB). It reports GB/s for:
//   - grid: 104 or 208 blocks (1 or 2 per CU; the kernel runs 208),
//   - depth: steps in flight per thread (1 = the kernel; 2, 4 = deeper),
//   - a barrier per step or not,
//   - layout: tile-major (each item's 23 steps contiguous, the kernel's
//     packed layout) or step-major (step p of all 23 tiles of an expert
//     side by side, so the items of one expert read one region per step),
//   - what one load instruction of a wavefront reads: 4 runs of 256 bytes
//     (the kernel) or 1 KB in one piece,
//   - one block of 16 wavefronts per CU with 512-column items (32 KB per
//     step) instead of 8 wavefronts and 256 columns.
#include <hip/hip_runtime.h>
#include <stdio.h>

#include <algorithm>

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int kTiles = 23, kExperts = 32, kK = 2880, kStep = 16384;

__device__ unsigned g_stream_queue;

__device__ inline uint4 ld16(__amdgpu_buffer_rsrc_t r, uint32_t v, uint32_t s) {
  const auto x = __builtin_amdgcn_raw_buffer_load_b128(r, v, s, 0);
  return make_uint4(x[0], x[1], x[2], x[3]);
}

template <int D, bool BAR, bool STEP_MAJOR, int WAVES = 8, bool CONTIG = false>
__global__ __launch_bounds__(WAVES * 64) __attribute__((amdgpu_waves_per_eu(4)))
void stream_kernel(const uint8_t *W, int items, int K, unsigned *sink) {
  __shared__ unsigned slot;
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int j = lane & 15, kq = lane >> 4;
  // K is a kernel argument, as in the real kernel: with a compile-time trip
  // count the compiler builds a different loop, and the times change.
  const int np = (K + 127) / 128;
  const uint32_t tile_bytes = 128 * K; // 256 columns x K/2
  uint32_t acc = 0;
  for (;;) {
    if (threadIdx.x == 0)
      slot = atomicAdd(&g_stream_queue, 1u);
    __syncthreads();
    const int it = __builtin_amdgcn_readfirstlane((int)slot);
    __syncthreads();
    if (it >= items)
      break;
    // WAVES = 16: one item is 512 columns (pieces 0..31), its steps of 32 KB
    // contiguous (tile-major); the caller passes the number of such items.
    const int tiles = kTiles * 8 / WAVES;
    const int e = it / tiles, t = it % tiles;
    const uint32_t ibytes = tile_bytes * WAVES / 8, istep = kStep * WAVES / 8;
    const uint8_t *base = STEP_MAJOR ? W + (size_t)e * tiles * ibytes
                                     : W + (size_t)it * ibytes;
    const uint32_t step = STEP_MAJOR ? tiles * istep : istep;
    const auto r = __builtin_amdgcn_make_buffer_rsrc(
        (void *)base, (short)0, STEP_MAJOR ? tiles * ibytes : ibytes, 0x00020000);
    uint32_t off[2];
    for (int jn = 0; jn < 2; jn++)
      off[jn] = (STEP_MAJOR ? t * istep : 0) +
                (CONTIG ? (uint32_t)(2 * wave + jn) * 1024 + 16 * lane
                        : (uint32_t)(2 * WAVES * kq + 2 * wave + jn) * 256 + 16 * j);
    uint4 w[D][2];
#pragma unroll
    for (int d = 0; d < D; d++)
      for (int jn = 0; jn < 2; jn++)
        w[d][jn] = ld16(r, off[jn], min(d, np - 1) * step);
    for (int p = 0; p < np; p += D) {
#pragma unroll
      for (int d = 0; d < D; d++) {
#pragma unroll
        for (int jn = 0; jn < 2; jn++)
          acc ^= w[d][jn].x + w[d][jn].y + w[d][jn].z + w[d][jn].w;
#pragma unroll
        for (int jn = 0; jn < 2; jn++)
          w[d][jn] = ld16(r, off[jn], min(p + d + D, np - 1) * step);
        if (BAR)
          __syncthreads();
      }
    }
  }
  if (acc == 0x12345678u)
    sink[0] = acc;
}

template <int D, bool BAR, bool STEP_MAJOR, int WAVES = 8, bool CONTIG = false>
static void run(const uint8_t *W, int items, unsigned *sink, double bytes, int grid) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  const unsigned zero = 0;
  const int n = 10;
  float sum = 0;
  for (int i = 0; i < n + 2; i++) {
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_stream_queue), &zero, 4));
    CHECK(hipEventRecord(a));
    stream_kernel<D, BAR, STEP_MAJOR, WAVES, CONTIG><<<grid, WAVES * 64>>>(W, items, kK, sink);
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms;
    CHECK(hipEventElapsedTime(&ms, a, b));
    if (i >= 2)
      sum += ms;
  }
  printf("grid %3d x %2d waves  depth %d  barrier %d  %-11s  %-9s  %6.1f us  %5.0f GB/s\n",
         grid, WAVES, D, (int)BAR, STEP_MAJOR ? "step-major" : "tile-major",
         CONTIG ? "1 KB" : "4 x 256 B", sum / n * 1e3, bytes / (sum / n) * 1e-6);
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
}

int main() {
  const int items = kExperts * kTiles;
  const size_t bytes = (size_t)items * 128 * kK;
  uint8_t *W;
  unsigned *sink;
  CHECK(hipMalloc(&W, bytes));
  CHECK(hipMalloc(&sink, 4));
  CHECK(hipMemset(W, 1, bytes));
  printf("%d items x %d KB = %.0f MB of codes\n", items, 128 * kK / 1024, bytes / 1e6);
  for (int grid : {104, 208}) {
    run<1, true, false>(W, items, sink, bytes, grid);
    run<2, true, false>(W, items, sink, bytes, grid);
    run<4, true, false>(W, items, sink, bytes, grid);
    run<1, false, false>(W, items, sink, bytes, grid);
    run<1, true, true>(W, items, sink, bytes, grid);
    run<2, true, true>(W, items, sink, bytes, grid);
    run<1, true, false, 8, true>(W, items, sink, bytes, grid);
  }
  // One block of 16 wavefronts per CU, 512-column items (32 KB per step):
  // 11 such tiles per expert (5,632 of the 5,888 columns).
  const int items16 = kExperts * (kTiles * 8 / 16);
  const double bytes16 = (double)items16 * 256 * kK;
  run<1, true, false, 16>(W, items16, sink, bytes16, 104);
  run<2, true, false, 16>(W, items16, sink, bytes16, 104);
  run<1, false, false, 16>(W, items16, sink, bytes16, 104);
  run<1, true, true, 16>(W, items16, sink, bytes16, 104);
  run<1, true, false, 16, true>(W, items16, sink, bytes16, 104);
  CHECK(hipFree(W));
  CHECK(hipFree(sink));
  return 0;
}
