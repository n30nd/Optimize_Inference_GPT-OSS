// Micro-benchmark of decode attention (kernels/fast_attention.hpp) on the
// shapes of gpt-oss: 64 query heads, 8 key/value heads, head_dim 64,
// window layers (128 slots) and full layers (1024 slots).
//
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/attn_bench.cpp -o attn_bench
//   ./attn_bench            # before (previous kernel) vs after (attention_bf16)
//   ./attn_bench variants   # also every wave-count / step-size / layout variant
//
// Each line: mean time per call over 30 calls (after 5 warm-up calls),
// measured with hipEvents around the 30 calls, and the achieved bandwidth on
// the bytes of cache the call must read: B * n * 8 heads * 64 * 2 (K, V) * 2 B.
#include <hip/hip_runtime.h>

#include <stdio.h>
#include <string.h>

#include <math.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "../kernels/fast_attention.hpp"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int D = 64, NH = 64, KVM = 8, NKV = 8, KVD = NKV * D;
constexpr int ROW = NH * D + 2 * KVD, OUT = NH * D;
constexpr int WARMUP = 5, CALLS = 30;

__global__ void fill_bf16(bf16_t *p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = f32_to_bf16(((h & 0xffff) / 32768.0f - 1.0f) * 2.0f);
  }
}
__global__ void fill_f32(float *p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = ((h & 0xffff) / 32768.0f - 1.0f) * scale;
  }
}

// Reads a and b once (16-byte loads); writes only if the xor of everything
// is a magic value, so the loads cannot be removed.
__global__ void read_stream(const uint4 *__restrict__ a, const uint4 *__restrict__ b, size_t n,
                            uint32_t *out) {
  uint32_t x = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 u = a[i], v = b[i];
    x ^= u.x ^ u.y ^ u.z ^ u.w ^ v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x12345678u)
    *out = x;
}

// Comparison point, not used by ./run: grouped-query sharing with plain
// vector math (no matrix cores). One 256-thread workgroup per (b, g) computes
// all 8 heads: scores of every position in LDS (8 x n floats), exact max and
// sum, then thread (head, 2 dims) sums p * v over the positions. Each key and
// value comes from memory once; the 8 heads re-read V from the L1 cache.
__global__ __launch_bounds__(256) void gqa_vector_kernel(
    const float *__restrict__ qkv, int row_stride, const bf16_t *__restrict__ kc,
    const bf16_t *__restrict__ vc, int slots, const float *__restrict__ sinks,
    bf16_t *__restrict__ out, int out_stride, int n) {
  constexpr int G = 8;
  extern __shared__ float sc[]; // [G][n]
  __shared__ float q[G][D];
  __shared__ float red[G][2];
  const int g = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  for (int i = tid; i < G * D; i += 256)
    q[i / D][i % D] = qkv[(size_t)b * row_stride + g * G * D + i] * 0.125f;
  __syncthreads();
  const bf16_t *kb = kc + (size_t)b * slots * KVD + g * D;
  const bf16_t *vb = vc + (size_t)b * slots * KVD + g * D;
  for (int j = tid; j < n; j += 256) {
    const uint4 *kr = (const uint4 *)(kb + (size_t)j * KVD);
    float acc[G] = {};
#pragma unroll 1
    for (int c = 0; c < 8; c++) {
      const uint4 u = kr[c];
      const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
      for (int e = 0; e < 8; e++) {
        const float kv = __uint_as_float(e & 1 ? (w4[e / 2] & 0xffff0000u) : (w4[e / 2] << 16));
        for (int h = 0; h < G; h++)
          acc[h] = fmaf(q[h][c * 8 + e], kv, acc[h]);
      }
    }
    for (int h = 0; h < G; h++)
      sc[h * n + j] = acc[h];
  }
  __syncthreads();
  // Wave h/2 (32 lanes per head) finds max and sum of head h.
  const int h = tid / 32, l = tid % 32;
  float mx = sinks[g * G + h];
  for (int j = l; j < n; j += 32)
    mx = fmaxf(mx, sc[h * n + j]);
  for (int o = 16; o > 0; o >>= 1)
    mx = fmaxf(mx, __shfl_xor(mx, o, 32));
  float sum = l == 0 ? __expf(sinks[g * G + h] - mx) : 0.0f;
  for (int j = l; j < n; j += 32) {
    const float e = __expf(sc[h * n + j] - mx);
    sc[h * n + j] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1)
    sum += __shfl_xor(sum, o, 32);
  __syncthreads();
  // Thread (h, l) owns dims 2l, 2l + 1 of head h.
  float o0 = 0.0f, o1 = 0.0f;
  for (int j = 0; j < n; j++) {
    const uint32_t v2 = *(const uint32_t *)(vb + (size_t)j * KVD + 2 * l);
    const float p = sc[h * n + j];
    o0 = fmaf(p, __uint_as_float(v2 << 16), o0);
    o1 = fmaf(p, __uint_as_float(v2 & 0xffff0000u), o1);
  }
  const float inv = 1.0f / sum;
  *(uint32_t *)(out + (size_t)b * out_stride + (g * G + h) * D + 2 * l) =
      (uint32_t)f32_to_bf16(o0 * inv) | ((uint32_t)f32_to_bf16(o1 * inv) << 16);
}

static float max_bf16_diff(const bf16_t *a, const bf16_t *b, size_t n) {
  std::vector<bf16_t> x(n), y(n);
  CHECK(hipMemcpy(x.data(), a, n * 2, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(y.data(), b, n * 2, hipMemcpyDeviceToHost));
  float m = 0;
  for (size_t i = 0; i < n; i++) {
    uint32_t u = (uint32_t)x[i] << 16, v = (uint32_t)y[i] << 16;
    float f, g;
    memcpy(&f, &u, 4);
    memcpy(&g, &v, 4);
    m = std::max(m, fabsf(f - g));
  }
  return m;
}

// attention_gqa8_kernel's loads (same work split, same addresses) with the
// arithmetic removed.
template <int NS, int TPS>
__global__ __launch_bounds__(64 * NS) void loads_only_kernel(const bf16_t *kc, const bf16_t *vc,
                                                             fattn::KVLayout L, int n,
                                                             uint32_t *out) {
  constexpr int STEP = 16 * TPS;
  const int g = blockIdx.x, b = blockIdx.y;
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6, col = lane & 15, grp = lane >> 4;
  const bf16_t *kb = kc + b * L.sb + (size_t)g * L.sg, *vb = vc + b * L.sb + (size_t)g * L.sg;
  const uint32_t st2 = L.st * 2, cs2 = (uint32_t)(L.cs * 2);
  const int cmask = (1 << L.cshift) - 1;
  auto row_bytes = [&](int row) {
    return (uint32_t)(row >> L.cshift) * cs2 + (uint32_t)(row & cmask) * st2;
  };
  const uint32_t range = row_bytes(n - 1) + 128;
  const auto krs = fattn::make_rsrc(kb, range), vrs = fattn::make_rsrc(vb, range);
  uint32_t x = 0;
  for (int step = w; step < (n + STEP - 1) / STEP; step += NS) {
    const uint32_t so = row_bytes(step * STEP);
    const uint32_t koff = so + col * st2 + grp * 16, voff = so + 4 * grp * st2 + col * 8;
    for (int t = 0; t < TPS; t++) {
      const auto a = __builtin_amdgcn_raw_buffer_load_b128(krs, koff + 16 * t * st2, 0, 0);
      const auto c = __builtin_amdgcn_raw_buffer_load_b128(krs, koff + 16 * t * st2 + 64, 0, 0);
      x ^= a[0] ^ a[1] ^ a[2] ^ a[3] ^ c[0] ^ c[1] ^ c[2] ^ c[3];
      for (int r = 0; r < 4; r++) {
        const auto v = __builtin_amdgcn_raw_buffer_load_b64(vrs, voff + (16 * t + r) * st2, 0, 0);
        x ^= v[0] ^ v[1];
      }
    }
  }
  if (x == 0x12345678u)
    *out = x;
}

static float time_us(const std::function<void()> &f) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  for (int i = 0; i < WARMUP; i++)
    f();
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventRecord(a, 0));
  for (int i = 0; i < CALLS; i++)
    f();
  CHECK(hipEventRecord(b, 0));
  CHECK(hipEventSynchronize(b));
  CHECK(hipGetLastError());
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, a, b));
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return ms * 1000.0f / CALLS;
}

int main(int argc, char **argv) {
  const bool variants = argc > 1 && !strcmp(argv[1], "variants");
  const int Bmax = 1024;
  // Room for the padded layout below (slot stride KVD + 64 elements).
  const size_t full = (size_t)Bmax * 1024 * (KVD + 64), win = (size_t)Bmax * 128 * (KVD + 64);
  bf16_t *kf, *vf, *kw, *vw, *out;
  float *qkv, *sinks;
  CHECK(hipMalloc(&kf, full * 2));
  CHECK(hipMalloc(&vf, full * 2));
  CHECK(hipMalloc(&kw, win * 2));
  CHECK(hipMalloc(&vw, win * 2));
  CHECK(hipMalloc(&qkv, (size_t)Bmax * ROW * 4));
  CHECK(hipMalloc(&sinks, NH * 4));
  CHECK(hipMalloc(&out, (size_t)Bmax * OUT * 2));
  bf16_t *out2;
  CHECK(hipMalloc(&out2, (size_t)Bmax * OUT * 2));
  fill_bf16<<<4096, 256>>>(kf, full, 1);
  fill_bf16<<<4096, 256>>>(vf, full, 2);
  fill_bf16<<<4096, 256>>>(kw, win, 3);
  fill_bf16<<<4096, 256>>>(vw, win, 4);
  fill_f32<<<1024, 256>>>(qkv, (size_t)Bmax * ROW, 5, 1.5f);
  fill_f32<<<1, 64>>>(sinks, NH, 6, 3.0f);
  CHECK(hipDeviceSynchronize());

  int dev = 0;
  hipDeviceProp_t prop;
  CHECK(hipGetDevice(&dev));
  CHECK(hipGetDeviceProperties(&prop, dev));
  printf("device %d: %s, %d CUs; %d calls per measurement after %d warm-up\n", dev,
         prop.gcnArchName, prop.multiProcessorCount, CALLS, WARMUP);
  printf("%-30s %-6s %5s %5s %5s %10s %8s\n", "kernel", "layer", "B", "pos", "n", "us/call",
         "GB/s");

  for (int B : {512, 1024})
    for (int window : {128, 0})
      for (int pos : {64, 256, 512, 1023}) {
        const int slots = window ? 128 : 1024;
        const int n = window ? std::min(pos + 1, window) : pos + 1;
        const bf16_t *kc = window ? kw : kf, *vc = window ? vw : vf;
        const double bytes = (double)B * n * NKV * D * 2 * 2;
        auto report = [&](const char *name, float us) {
          printf("%-30s %-6s %5d %5d %5d %10.1f %8.0f\n", name, window ? "window" : "full", B,
                 pos, n, us, bytes / (us * 1e-6) / 1e9);
        };
        const long long sb = (long long)slots * KVD;
        const fattn::KVLayout rowL{sb, KVD, D, 30, 0};                 // engine's layout
        const fattn::KVLayout headL{sb, D, slots * D, 30, 0};          // (b, g, slot, 64)
        const fattn::KVLayout padL{(long long)slots * (KVD + 64), KVD + 64, D, 30, 0};
        auto chunkL = [&](int sh) { // (b, slot / T, g, slot % T, 64), T = 2^sh
          return fattn::KVLayout{sb, D, (1 << sh) * D, sh, (long long)(1 << sh) * KVD};
        };
        report("before (simple)", time_us([&] {
                 attention_bf16_simple_kernel<<<dim3(NH, B), 64, (n + 1) * sizeof(float)>>>(
                     qkv, ROW, kc, vc, slots, KVD, KVM, sinks, out, OUT, pos, window);
               }));
        report("after (attention_bf16)", time_us([&] {
                 attention_bf16(0, qkv, ROW, kc, vc, slots, KVD, NH, KVM, sinks, out, OUT, B,
                                pos, window);
               }));
        if (!variants)
          continue;
        report("proposed: chunk64 layout", time_us([&] {
                 attention_bf16_chunk64(0, qkv, ROW, kc, vc, slots, KVD, NH, KVM, sinks, out, OUT,
                                        B, pos, window);
               }));
        report("gqa vector (no MFMA)", time_us([&] {
                 gqa_vector_kernel<<<dim3(NKV, B), 256, NKV * n * sizeof(float)>>>(
                     qkv, ROW, kc, vc, slots, sinks, out2, OUT, n);
               }));
        attention_bf16(0, qkv, ROW, kc, vc, slots, KVD, NH, KVM, sinks, out, OUT, B, pos, window);
        CHECK(hipDeviceSynchronize());
        printf("  (gqa vector vs attention_bf16: max |diff| %.2e)\n",
               max_bf16_diff(out, out2, (size_t)B * OUT));
#define VAR(NG, NS, TPS)                                                      \
  report("gqa8 NG" #NG " NS" #NS " TPS" #TPS, time_us([&] {                    \
           fattn::attention_gqa8<NG, NS, TPS>(0, qkv, ROW, kc, vc, rowL, NKV, sinks, out, OUT, B, \
                                              n);                              \
         }))
        // NG key/value heads of one sequence per workgroup, NS history splits.
        VAR(1, 1, 2);
        VAR(1, 2, 2);
        VAR(1, 4, 1);
        VAR(1, 4, 2);
        VAR(1, 4, 4);
        VAR(1, 8, 1);
        VAR(1, 8, 2);
        VAR(1, 16, 1);
        VAR(2, 4, 2);
        VAR(2, 8, 1);
        VAR(4, 2, 2);
        VAR(4, 4, 1);
        VAR(4, 4, 2);
        VAR(8, 1, 2);
        VAR(8, 1, 4);
        VAR(8, 2, 1);
        VAR(8, 2, 2);
#undef VAR
        // Head-major cache (b, g, slot, 64): each (b, g) history is one
        // contiguous block. Also with 256 bytes of padding between blocks.
#define LAY(NAME, NG, NS, TPS, LAYOUT)                                         \
  report(NAME, time_us([&] {                                                   \
           fattn::attention_gqa8<NG, NS, TPS>(0, qkv, ROW, kc, vc, LAYOUT, NKV, sinks, out, OUT, \
                                              B, n);                           \
         }))
        LAY("NG1 NS4 TPS2 head-major", 1, 4, 2, headL);
        LAY("NG2 NS4 TPS2 head-major", 2, 4, 2, headL);
        LAY("NG1 NS8 TPS1 head-major", 1, 8, 1, headL);
        LAY("NG1 NS4 TPS2 padded rows", 1, 4, 2, padL);
        LAY("NG2 NS4 TPS2 padded rows", 2, 4, 2, padL);
        LAY("NG1 NS4 TPS2 chunk32", 1, 4, 2, chunkL(5));
        LAY("NG2 NS4 TPS2 chunk32", 2, 4, 2, chunkL(5));
        LAY("NG1 NS4 TPS2 chunk64", 1, 4, 2, chunkL(6));
        LAY("NG2 NS4 TPS2 chunk64", 2, 4, 2, chunkL(6));
        LAY("NG1 NS8 TPS1 chunk64", 1, 8, 1, chunkL(6));
        LAY("NG1 NS4 TPS2 chunk128", 1, 4, 2, chunkL(7));
        LAY("NG2 NS4 TPS2 chunk128", 2, 4, 2, chunkL(7));
#undef LAY
        // The same loads without any arithmetic: how fast the access pattern
        // alone can go.
        report("loads only, engine layout", time_us([&] {
                 loads_only_kernel<4, 2><<<dim3(NKV, B), 256>>>(kc, vc, rowL, n, (uint32_t *)out);
               }));
        report("loads only, head-major", time_us([&] {
                 loads_only_kernel<4, 2><<<dim3(NKV, B), 256>>>(kc, vc, headL, n, (uint32_t *)out);
               }));
        report("loads only, chunk64", time_us([&] {
                 loads_only_kernel<4, 2><<<dim3(NKV, B), 256>>>(kc, vc, chunkL(6), n, (uint32_t *)out);
               }));
      }
  // Plain streaming read of the same bytes (one full layer at pos 1023, K and
  // V): the most a kernel of ours gets from memory on this GCD.
  for (int B : {512, 1024}) {
    const size_t n16 = (size_t)B * 1024 * KVD * 2 / 16; // uint4 per buffer
    uint32_t *sink_out;
    CHECK(hipMalloc(&sink_out, 4));
    const float us = time_us([&] {
      read_stream<<<104 * 16, 256>>>((const uint4 *)kf, (const uint4 *)vf, n16, sink_out);
    });
    printf("%-30s %-6s %5d %5d %5d %10.1f %8.0f\n", "plain read (yardstick)", "full", B, 1023,
           1024, us, 2.0 * n16 * 16 / (us * 1e-6) / 1e9);
    CHECK(hipFree(sink_out));
  }
  return 0;
}
