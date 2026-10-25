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
// Second part: the int8 V cache (attention_q8v_chunk64 and the cache writer
// rope_kv_bf16_kernel), see run_case_q8 below.
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include <random>
#include <vector>

#include "../kernels/fast_misc.hpp"

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

// ---- int8 V (attention_q8v_chunk64) ----------------------------------------
// V is drawn in fp32 and quantized on the CPU with the documented rule
// (scale = max|v| / 127, code = round(v * 127 / max|v|)); K is bf16, both in
// the chunk64 layout. Three fp64 references:
//   deq: the dequantized values scale * code -> error of the kernel's arithmetic
//   f32: the original fp32 values            -> error incl. int8 rounding
// and the bf16 kernel (attention_bf16_chunk64) on bf16(v) against f32, the
// error the int8 cache replaces. Unused slots: K NaN, scales NaN, codes random.
static void quantize_v(const float *v, int8_t *codes, float *scale) { // 64 values
  float amax = 0.0f;
  for (int i = 0; i < D; i++)
    amax = std::max(amax, fabsf(v[i]));
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  for (int i = 0; i < D; i++)
    codes[i] = (int8_t)std::min(std::max(rintf(v[i] * inv), -127.0f), 127.0f);
  *scale = amax / 127.0f;
}

struct ErrQ8 {
  double deq_f32 = 0, deq_bf16 = 0, f32_bf16 = 0, base_bf16 = 0; // see above
  double rms_q8 = 0, rms_base = 0; // root-mean-square error vs f32, int8 and bf16 kernels
};

template <int NS, int TPS>
static void run_q8(int B, int n, const float *q, const bf16_t *k, const int8_t *vq, const float *vs,
                   const float *sk, float *o32, bf16_t *o16, int slots) {
  const fattn::KVLayout L{(long long)slots * KVD, D, 64 * D, 6, 64LL * KVD};
  fattn::attention_gqa8_q8v<1, NS, TPS, float>(0, q, ROW, k, vq, vs, L, NKV, sk, o32, OUT, B, n);
  fattn::attention_gqa8_q8v<1, NS, TPS, bf16_t>(0, q, ROW, k, vq, vs, L, NKV, sk, o16, OUT, B, n);
}

// outlier: one dim of each value vector 8x larger (real value vectors have a
// few large channels; the other 63 then get coarser codes).
static ErrQ8 run_case_q8(const Case &c, int variant, bool outlier, std::mt19937 &rng) {
  const int B = c.B, slots = c.slots;
  const int n = c.window > 0 ? std::min(c.pos + 1, c.window) : c.pos + 1;
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::uniform_int_distribution<int> ub(-128, 127);
  std::vector<float> q((size_t)B * ROW), sinks(NH);
  for (auto &x : q)
    x = u(rng) * c.q_scale;
  for (auto &x : sinks)
    x = 3.0f * u(rng) + 1.0f;
  const size_t cache = (size_t)B * slots * KVD, nsc = (size_t)B * slots * NKV;
  std::vector<uint16_t> kc(cache, 0x7fc0), vb(cache, 0x7fc0);
  std::vector<int8_t> vq(cache);
  for (auto &x : vq)
    x = (int8_t)ub(rng); // slots not in use: any code
  std::vector<float> vs(nsc, NAN), vf(cache, NAN), vdq(cache, NAN);
  const int t0 = c.window > 0 ? std::max(0, c.pos - c.window + 1) : 0;
  for (int b = 0; b < B; b++)
    for (int t = t0; t <= c.pos; t++) {
      const int slot = c.window > 0 ? t % c.window : t;
      for (int g = 0; g < NKV; g++) {
        float v[D];
        int8_t cd[D];
        float sc;
        const int big = (int)(rng() % D);
        for (int i = 0; i < D; i++) {
          kc[kv_index_chunk64(b, slot, g, i, slots, KVD)] = f2bf(2.0f * u(rng));
          v[i] = 2.0f * u(rng) * (outlier && i == big ? 8.0f : 1.0f);
        }
        quantize_v(v, cd, &sc);
        vs[vs_index_chunk64(b, slot, g, slots, KVD)] = sc;
        for (int i = 0; i < D; i++) {
          const size_t e = kv_index_chunk64(b, slot, g, i, slots, KVD);
          vq[vq_index_chunk64(b, slot, g, i, slots, KVD)] = cd[i];
          vf[e] = v[i];
          vb[e] = f2bf(v[i]);
          vdq[e] = sc * (float)cd[i];
        }
      }
    }

  // fp64 references for value tables vf (fp32) and vdq (dequantized).
  std::vector<double> ref_f((size_t)B * OUT), ref_d((size_t)B * OUT);
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++)
    for (int h = 0; h < NH; h++) {
      std::vector<double> sc(n);
      const int g = h / KVM;
      double mx = sinks[h];
      for (int t = t0; t <= c.pos; t++) {
        const int slot = c.window > 0 ? t % c.window : t;
        double d = 0;
        for (int i = 0; i < D; i++)
          d += (double)q[(size_t)b * ROW + h * D + i] * bf2f(kc[kv_index_chunk64(b, slot, g, i, slots, KVD)]);
        sc[t - t0] = d / 8.0;
        mx = std::max(mx, sc[t - t0]);
      }
      double sum = exp(sinks[h] - mx);
      for (int j = 0; j < n; j++)
        sum += (sc[j] = exp(sc[j] - mx));
      for (int i = 0; i < D; i++) {
        double af = 0, ad = 0;
        for (int t = t0; t <= c.pos; t++) {
          const int slot = c.window > 0 ? t % c.window : t;
          const size_t e = kv_index_chunk64(b, slot, g, i, slots, KVD);
          af += sc[t - t0] * vf[e];
          ad += sc[t - t0] * vdq[e];
        }
        ref_f[(size_t)b * OUT + h * D + i] = af / sum;
        ref_d[(size_t)b * OUT + h * D + i] = ad / sum;
      }
    }

  float *dq, *ds, *dvs, *do32;
  bf16_t *dk, *dvb, *do16, *dpub, *dbase;
  int8_t *dvq;
  CHECK(hipMalloc(&dq, q.size() * 4));
  CHECK(hipMalloc(&ds, NH * 4));
  CHECK(hipMalloc(&dk, cache * 2));
  CHECK(hipMalloc(&dvb, cache * 2));
  CHECK(hipMalloc(&dvq, cache));
  CHECK(hipMalloc(&dvs, nsc * 4));
  CHECK(hipMalloc(&do32, (size_t)B * OUT * 4));
  CHECK(hipMalloc(&do16, (size_t)B * OUT * 2));
  CHECK(hipMalloc(&dpub, (size_t)B * OUT * 2));
  CHECK(hipMalloc(&dbase, (size_t)B * OUT * 2));
  CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ds, sinks.data(), NH * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dk, kc.data(), cache * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dvb, vb.data(), cache * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dvq, vq.data(), cache, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dvs, vs.data(), nsc * 4, hipMemcpyHostToDevice));
  switch (variant) {
  case 0: run_q8<4, 2>(B, n, dq, dk, dvq, dvs, ds, do32, do16, slots); break;
  case 1: run_q8<8, 1>(B, n, dq, dk, dvq, dvs, ds, do32, do16, slots); break;
  case 2: run_q8<1, 1>(B, n, dq, dk, dvq, dvs, ds, do32, do16, slots); break;
  case 3: run_q8<4, 4>(B, n, dq, dk, dvq, dvs, ds, do32, do16, slots); break;
  case 4: run_q8<8, 2>(B, n, dq, dk, dvq, dvs, ds, do32, do16, slots); break;
  case 5: run_q8<2, 2>(B, n, dq, dk, dvq, dvs, ds, do32, do16, slots); break;
  }
  attention_q8v_chunk64(0, dq, ROW, dk, dvq, dvs, slots, KVD, ds, dpub, OUT, B, c.pos, c.window);
  attention_bf16_chunk64(0, dq, ROW, dk, dvb, slots, KVD, NH, KVM, ds, dbase, OUT, B, c.pos,
                         c.window);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  std::vector<float> o32((size_t)B * OUT);
  std::vector<uint16_t> o16((size_t)B * OUT), pub((size_t)B * OUT), base((size_t)B * OUT);
  CHECK(hipMemcpy(o32.data(), do32, o32.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(o16.data(), do16, o16.size() * 2, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(pub.data(), dpub, pub.size() * 2, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(base.data(), dbase, base.size() * 2, hipMemcpyDeviceToHost));
  // The entry point picks NS2 TPS2 on window layers, NS4 TPS2 on full layers
  // up to 128 positions, NS8 TPS1 above.
  if ((variant == 5 && c.window > 0) || (variant == 0 && c.window == 0 && n <= 128) ||
      (variant == 1 && c.window == 0 && n > 128))
    if (pub != o16) {
      fprintf(stderr, "attention_q8v_chunk64 differs from its variant\n");
      exit(1);
    }
  ErrQ8 e;
  auto upd = [](double &m, double d) { m = std::isfinite(d) ? std::max(m, d) : INFINITY; };
  double s2q = 0, s2b = 0;
  for (size_t i = 0; i < ref_f.size(); i++) {
    upd(e.deq_f32, fabs(o32[i] - ref_d[i]));
    upd(e.deq_bf16, fabs(bf2f(o16[i]) - ref_d[i]));
    upd(e.f32_bf16, fabs(bf2f(pub[i]) - ref_f[i]));
    upd(e.base_bf16, fabs(bf2f(base[i]) - ref_f[i]));
    s2q += (bf2f(pub[i]) - ref_f[i]) * (bf2f(pub[i]) - ref_f[i]);
    s2b += (bf2f(base[i]) - ref_f[i]) * (bf2f(base[i]) - ref_f[i]);
  }
  e.rms_q8 = sqrt(s2q / ref_f.size());
  e.rms_base = sqrt(s2b / ref_f.size());
  for (void *p : {(void *)dq, (void *)ds, (void *)dk, (void *)dvb, (void *)dvq, (void *)dvs,
                  (void *)do32, (void *)do16, (void *)dpub, (void *)dbase})
    CHECK(hipFree(p));
  return e;
}

// The cache writer (rope_kv_bf16_kernel, kernels/fast_misc.hpp) against the
// CPU rule: rotated K in bf16, V codes and scales. Writes 3 slots of a cache
// of 128 slots for 37 rows; codes must match exactly.
static bool check_writer(std::mt19937 &rng) {
  const int B = 37, slots = 128, half = D / 2;
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<float> qkv((size_t)B * ROW), cosv(half), sinv(half);
  for (auto &x : qkv)
    x = 3.0f * u(rng);
  for (int b = 0; b < B; b++) // some all-zero and some outlier value vectors
    for (int i = 0; i < D; i++) {
      qkv[(size_t)b * ROW + NH * D + KVD + 0 * D + i] = 0.0f;
      if (i == 5)
        qkv[(size_t)b * ROW + NH * D + KVD + 1 * D + i] = 40.0f;
    }
  for (int i = 0; i < half; i++) {
    cosv[i] = cosf(0.1f * i);
    sinv[i] = sinf(0.1f * i);
  }
  float *dqkv, *dc, *dsn, *dvs;
  bf16_t *dk;
  int8_t *dvq;
  const size_t cache = (size_t)B * slots * KVD;
  CHECK(hipMalloc(&dqkv, qkv.size() * 4));
  CHECK(hipMalloc(&dc, half * 4));
  CHECK(hipMalloc(&dsn, half * 4));
  CHECK(hipMalloc(&dk, cache * 2));
  CHECK(hipMalloc(&dvq, cache));
  CHECK(hipMalloc(&dvs, (size_t)B * slots * NKV * 4));
  CHECK(hipMemcpy(dc, cosv.data(), half * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dsn, sinv.data(), half * 4, hipMemcpyHostToDevice));
  bool ok = true;
  long long codes = 0;
  for (int slot : {0, 77, 127}) {
    CHECK(hipMemcpy(dqkv, qkv.data(), qkv.size() * 4, hipMemcpyHostToDevice));
    rope_kv_bf16_kernel<<<B, 256>>>(dqkv, ROW, dc, dsn, NH, NKV, D, dk, dvq, dvs, slots, slot);
    CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> k(cache);
    std::vector<int8_t> vq(cache);
    std::vector<float> vs((size_t)B * slots * NKV);
    CHECK(hipMemcpy(k.data(), dk, cache * 2, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(vq.data(), dvq, cache, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(vs.data(), dvs, vs.size() * 4, hipMemcpyDeviceToHost));
    for (int b = 0; b < B; b++)
      for (int g = 0; g < NKV; g++) {
        const float *kr = &qkv[(size_t)b * ROW + NH * D + g * D];
        const float *vr = &qkv[(size_t)b * ROW + NH * D + KVD + g * D];
        int8_t cd[D];
        float sc;
        quantize_v(vr, cd, &sc);
        if (vs[vs_index_chunk64(b, slot, g, slots, KVD)] != sc)
          ok = false;
        for (int i = 0; i < D; i++) {
          if (vq[vq_index_chunk64(b, slot, g, i, slots, KVD)] != cd[i])
            ok = false;
          codes++;
          const int j = i % half;
          const float x1 = kr[j], x2 = kr[half + j];
          const float rk = i < half ? x1 * cosv[j] - x2 * sinv[j] : x2 * cosv[j] + x1 * sinv[j];
          if (fabsf(bf2f(k[kv_index_chunk64(b, slot, g, i, slots, KVD)]) - rk) >
              fabsf(rk) * 8e-3f + 1e-6f)
            ok = false;
        }
      }
  }
  printf("writer (rope_kv_bf16_kernel): %lld codes and their scales vs the CPU rule: %s\n", codes,
         ok ? "identical" : "DIFFERENT");
  for (void *p : {(void *)dqkv, (void *)dc, (void *)dsn, (void *)dk, (void *)dvq, (void *)dvs})
    CHECK(hipFree(p));
  return ok;
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
  printf("largest fp32-output error: %.2e\n", worst);

  // int8 V. deq: kernel vs fp64 on the dequantized values (its own
  // arithmetic, same tolerances as above). vs f32: against the original
  // values, for the int8 kernel and the bf16 kernel.
  printf("\nint8 V (chunk64)\n%-14s %-6s %5s %4s %6s %3s  %-9s %-9s | %-9s %-9s %-9s %-9s\n",
         "kernel", "layer", "pos", "B", "qscale", "out", "deq f32", "deq bf16", "q8 vs f32",
         "bf16 vs f32", "q8 rms", "bf16 rms");
  const char *qnames[] = {"q8 NS4 TPS2", "q8 NS8 TPS1", "q8 NS1 TPS1", "q8 NS4 TPS4", "q8 NS8 TPS2",
                          "q8 NS2 TPS2"};
  double worst_q8 = 0, worst_base = 0;
  auto report_q8 = [&](int variant, const Case &c, bool outl) {
    const ErrQ8 e = run_case_q8(c, variant, outl, rng);
    // Against the original values the int8 rounding adds up to half a code
    // step, max|v| / 254 (2 / 254, or 16 / 254 with an outlier), on top of
    // the bf16 rounding of the output.
    const double tol_f = (outl ? 16.0 : 2.0) / 254.0 + kTolBf16;
    const bool pass = e.deq_f32 < kTolF32 * (outl ? 8 : 1) && e.deq_bf16 < kTolBf16 * (outl ? 8 : 1) &&
                      e.f32_bf16 < tol_f;
    ok &= pass;
    if (!outl) {
      worst_q8 = std::max(worst_q8, e.f32_bf16);
      worst_base = std::max(worst_base, e.base_bf16);
    }
    printf("%-14s %-6s %5d %4d %6.1f %3s  %-9.2e %-9.2e | %-9.2e %-11.2e %-9.2e %-9.2e %s\n",
           qnames[variant], c.window ? "window" : "full", c.pos, c.B, c.q_scale, outl ? "yes" : "no",
           e.deq_f32, e.deq_bf16, e.f32_bf16, e.base_bf16, e.rms_q8, e.rms_base,
           pass ? "ok" : "FAIL");
  };
  for (int variant = 0; variant < 6; variant++)
    for (int window : {128, 0})
      for (int pos : positions) {
        if (variant == 5 && window == 0 && pos != 1023 && pos != 129)
          continue;
        if (variant > 1 && variant < 5 && pos != 1023 && pos != 129)
          continue;
        const float qs = (pos == 129) ? 6.0f : 1.5f;
        report_q8(variant, Case{37, window > 0 ? 128 : 1024, window, pos, qs, 2}, false);
        if (variant == 0 && (pos == 129 || pos == 1023))
          report_q8(variant, Case{37, window > 0 ? 128 : 1024, window, pos, qs, 2}, true);
      }
  for (int window : {128, 0})
    report_q8(window ? 5 : 1, Case{512, window ? 128 : 1024, window, 1023, 1.5f, 2}, false);
  printf("largest bf16-output error vs the original values (no outliers): int8 V %.2e, bf16 V "
         "%.2e\n",
         worst_q8, worst_base);
  ok &= check_writer(rng);
  printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
