// Unit test for kernels/fast_attention.hpp: decode attention with sinks,
// sliding window and grouped-query attention, against an fp64 CPU reference
// on the same inputs (fp32 q, bf16 K/V cache).
//
//   hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/attn_test.cpp -o attn_test
//   ./attn_test
//
// Cache slots that hold no position are filled with NaN, so any read of them
// that leaks into the result shows up. Prints the largest absolute error of
// the fp32 output (before bf16 rounding) and of the bf16 output, for the new
// kernel and for the previous one (attention_bf16_simple_kernel).
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include <random>
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

constexpr int D = 64, NH = 64, KVM = 8, NKV = NH / KVM, KVD = NKV * D;
constexpr int ROW = NH * D + 2 * KVD, OUT = NH * D;

static float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

struct Case {
  int B, slots, window, pos;
  float q_scale;
  int layout; // 0: engine's (b, slot, g, 64); 1: head-major (b, g, slot, 64);
              // 2: chunks of 64 slots (b, slot / 64, g, slot % 64, 64)
};

static fattn::KVLayout layout_of(const Case &c) {
  if (c.layout == 1)
    return {(long long)c.slots * KVD, D, c.slots * D, 30, 0};
  if (c.layout == 2)
    return {(long long)c.slots * KVD, D, 64 * D, 6, 64LL * KVD};
  return {(long long)c.slots * KVD, KVD, D, 30, 0};
}

// Index of element i of head g, slot `slot`, sequence b.
static size_t kv_index(const Case &c, int b, int slot, int g, int i) {
  const fattn::KVLayout L = layout_of(c);
  return b * L.sb + (slot >> L.cshift) * L.cs + (size_t)g * L.sg +
         (size_t)(slot & ((1 << L.cshift) - 1)) * L.st + i;
}

template <int NG, int NS, int TPS>
static void run_new(const Case &c, const float *q, const bf16_t *k, const bf16_t *v,
                    const float *s, float *o32, bf16_t *o16) {
  const int n = c.window > 0 ? std::min(c.pos + 1, c.window) : c.pos + 1;
  fattn::attention_gqa8<NG, NS, TPS, float>(0, q, ROW, k, v, layout_of(c), NKV, s, o32, OUT, c.B, n);
  fattn::attention_gqa8<NG, NS, TPS, bf16_t>(0, q, ROW, k, v, layout_of(c), NKV, s, o16, OUT, c.B, n);
}

struct Err {
  double f32 = 0, bf16 = 0, old_bf16 = 0;
};

static Err run_case(const Case &c, int variant, std::mt19937 &rng) {
  const int B = c.B;
  const int n = c.window > 0 ? std::min(c.pos + 1, c.window) : c.pos + 1;
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<float> q((size_t)B * ROW), sinks(NH);
  for (auto &x : q)
    x = u(rng) * c.q_scale;
  for (auto &x : sinks)
    x = 3.0f * u(rng) + 1.0f;
  const size_t cache = (size_t)B * c.slots * KVD;
  std::vector<uint16_t> kc(cache, 0x7fc0), vc(cache, 0x7fc0); // NaN = slot not in use
  const int t0 = c.window > 0 ? std::max(0, c.pos - c.window + 1) : 0;
  for (int b = 0; b < B; b++)
    for (int t = t0; t <= c.pos; t++) {
      const int slot = c.window > 0 ? t % c.window : t;
      for (int g = 0; g < NKV; g++)
        for (int i = 0; i < D; i++) {
          kc[kv_index(c, b, slot, g, i)] = f2bf(2.0f * u(rng));
          vc[kv_index(c, b, slot, g, i)] = f2bf(2.0f * u(rng));
        }
    }

  // fp64 reference, written from the interface description.
  std::vector<double> ref((size_t)B * OUT);
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++)
    for (int h = 0; h < NH; h++) {
      std::vector<double> sc(n + 1);
      const int g = h / KVM;
      double mx = sinks[h];
      for (int t = t0; t <= c.pos; t++) {
        const int slot = c.window > 0 ? t % c.window : t;
        double d = 0;
        for (int i = 0; i < D; i++)
          d += (double)q[(size_t)b * ROW + h * D + i] * bf2f(kc[kv_index(c, b, slot, g, i)]);
        sc[t - t0] = d / 8.0;
        mx = std::max(mx, sc[t - t0]);
      }
      double sum = exp(sinks[h] - mx);
      for (int j = 0; j < n; j++)
        sum += (sc[j] = exp(sc[j] - mx));
      for (int i = 0; i < D; i++) {
        double acc = 0;
        for (int t = t0; t <= c.pos; t++) {
          const int slot = c.window > 0 ? t % c.window : t;
          acc += sc[t - t0] * bf2f(vc[kv_index(c, b, slot, g, i)]);
        }
        ref[(size_t)b * OUT + h * D + i] = acc / sum;
      }
    }

  float *dq, *ds, *do32;
  bf16_t *dk, *dv, *do16, *dold;
  CHECK(hipMalloc(&dq, q.size() * 4));
  CHECK(hipMalloc(&ds, NH * 4));
  CHECK(hipMalloc(&dk, cache * 2));
  CHECK(hipMalloc(&dv, cache * 2));
  CHECK(hipMalloc(&do32, (size_t)B * OUT * 4));
  CHECK(hipMalloc(&do16, (size_t)B * OUT * 2));
  CHECK(hipMalloc(&dold, (size_t)B * OUT * 2));
  CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ds, sinks.data(), NH * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dk, kc.data(), cache * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dv, vc.data(), cache * 2, hipMemcpyHostToDevice));

  switch (variant) {
  case 0: run_new<1, 4, 2>(c, dq, dk, dv, ds, do32, do16); break; // keep in sync with attention_bf16
  case 1: run_new<1, 1, 1>(c, dq, dk, dv, ds, do32, do16); break;
  case 2: run_new<8, 1, 2>(c, dq, dk, dv, ds, do32, do16); break;
  case 3: run_new<8, 2, 1>(c, dq, dk, dv, ds, do32, do16); break;
  case 4: run_new<4, 2, 2>(c, dq, dk, dv, ds, do32, do16); break;
  case 5: run_new<2, 4, 2>(c, dq, dk, dv, ds, do32, do16); break;
  case 6: run_new<1, 8, 1>(c, dq, dk, dv, ds, do32, do16); break;
  case 7: run_new<8, 2, 2>(c, dq, dk, dv, ds, do32, do16); break;
  }
  if (c.layout == 0) {
    // The public entry point must agree with variant 0 on the default layout.
    if (variant == 0) {
      bf16_t *dpub;
      CHECK(hipMalloc(&dpub, (size_t)B * OUT * 2));
      attention_bf16(0, dq, ROW, dk, dv, c.slots, KVD, NH, KVM, ds, dpub, OUT, B, c.pos, c.window);
      std::vector<uint16_t> a((size_t)B * OUT), p((size_t)B * OUT);
      CHECK(hipMemcpy(a.data(), do16, a.size() * 2, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(p.data(), dpub, p.size() * 2, hipMemcpyDeviceToHost));
      if (a != p) {
        fprintf(stderr, "attention_bf16 differs from variant 0\n");
        exit(1);
      }
      CHECK(hipFree(dpub));
    }
    attention_bf16_simple_kernel<<<dim3(NH, B), 64, (n + 1) * sizeof(float)>>>(
        dq, ROW, dk, dv, c.slots, KVD, KVM, ds, dold, OUT, c.pos, c.window);
  }
  std::vector<uint16_t> o16c;
  if (c.layout == 2 && variant == 0) { // the proposed-layout entry point
    attention_bf16_chunk64(0, dq, ROW, dk, dv, c.slots, KVD, NH, KVM, ds, dold, OUT, B, c.pos,
                           c.window);
    o16c.resize((size_t)B * OUT);
    CHECK(hipMemcpy(o16c.data(), dold, o16c.size() * 2, hipMemcpyDeviceToHost));
  }
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  std::vector<float> o32((size_t)B * OUT);
  std::vector<uint16_t> o16((size_t)B * OUT), old((size_t)B * OUT);
  CHECK(hipMemcpy(o32.data(), do32, o32.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(o16.data(), do16, o16.size() * 2, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(old.data(), dold, old.size() * 2, hipMemcpyDeviceToHost));
  Err e;
  for (size_t i = 0; i < ref.size(); i++) {
    auto upd = [](double &m, double d) { m = std::isfinite(d) ? std::max(m, d) : INFINITY; };
    upd(e.f32, fabs(o32[i] - ref[i]));
    upd(e.bf16, fabs(bf2f(o16[i]) - ref[i]));
    if (c.layout == 0)
      upd(e.old_bf16, fabs(bf2f(old[i]) - ref[i]));
    if (!o16c.empty())
      upd(e.bf16, fabs(bf2f(o16c[i]) - ref[i]));
  }
  for (void *p : {(void *)dq, (void *)ds, (void *)dk, (void *)dv, (void *)do32, (void *)do16, (void *)dold})
    CHECK(hipFree(p));
  return e;
}

// Outputs are at most 2 in size (|v| <= 2). bf16 rounding of the output alone
// is up to 2^-8 * 2 = 7.8e-3; the fp32 output must be far below that. The
// kernel carries q and the weights as two bf16 pieces (error <= 2^-18
// relative), which with the large scores of the q_scale = 6 cases gives ~3e-5.
constexpr double kTolF32 = 5e-5, kTolBf16 = 8e-3;

int main() {
  std::mt19937 rng(1234);
  const int positions[] = {0, 1, 127, 128, 129, 1023};
  const char *names[] = {"NG1 NS4 TPS2", "NG1 NS1 TPS1", "NG8 NS1 TPS2", "NG8 NS2 TPS1",
                         "NG4 NS2 TPS2", "NG2 NS4 TPS2", "NG1 NS8 TPS1", "NG8 NS2 TPS2"};
  bool ok = true;
  double worst = 0;
  printf("%-20s %-6s %5s %4s %6s %5s  %-10s %-10s %-10s\n", "kernel", "layer", "pos", "B", "qscale",
         "lay", "f32 err", "bf16 err", "old bf16");
  for (int variant = 0; variant < 8; variant++)
    for (int window : {128, 0})
      for (int pos : positions)
        for (int lay = 0; lay < 3; lay++) {
          if (variant > 0 && lay == 0 && pos != 1023 && pos != 129)
            continue; // the other variants on fewer cases
          if (lay > 0 && variant != 0 && variant != 3 && variant != 6)
            continue;
          const int slots = window > 0 ? 128 : 1024;
          const float qs = (pos == 129) ? 6.0f : 1.5f; // pos 129: large, peaky scores
          Case c{37, slots, window, pos, qs, lay};
          const Err e = run_case(c, variant, rng);
          const bool pass = e.f32 < kTolF32 && e.bf16 < kTolBf16;
          ok &= pass;
          worst = std::max(worst, e.f32);
          char old[16] = "-";
          if (c.layout == 0)
            snprintf(old, sizeof old, "%.2e", e.old_bf16);
          printf("%-20s %-6s %5d %4d %6.1f %5s  %-10.2e %-10.2e %-10s %s\n", names[variant],
                 window ? "window" : "full", pos, c.B, qs, lay == 1 ? "head" : lay == 2 ? "chunk" : "row", e.f32, e.bf16,
                 old, pass ? "ok" : "FAIL");
        }
  // One full-size batch for the default kernel.
  for (int window : {128, 0}) {
    Case c{512, window ? 128 : 1024, window, 1023, 1.5f, 0};
    const Err e = run_case(c, 0, rng);
    const bool pass = e.f32 < kTolF32 && e.bf16 < kTolBf16;
    ok &= pass;
    worst = std::max(worst, e.f32);
    printf("%-20s %-6s %5d %4d %6.1f %5s  %-10.2e %-10.2e %-10.2e %s\n", names[0],
           window ? "window" : "full", 1023, c.B, 1.5f, "row", e.f32, e.bf16, e.old_bf16,
           pass ? "ok" : "FAIL");
  }
  printf("largest fp32-output error: %.2e\n%s\n", worst, ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
