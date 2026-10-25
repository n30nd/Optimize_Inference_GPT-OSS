// Unit test for the small fast-path kernels: each one against the kernels it
// replaced (tests/small_kernels_ref.hpp, or the unfused pair still in the
// tree), bit for bit, on the engine's shapes and on odd ones.
//
//   add_rmsnorm_bf16           = add_inplace_kernel + rmsnorm_bf16_kernel (also
//                                the second copy of the output)
//   rope_kv_bf16 (+ attention) = rope_kv_bf16_kernel_ref, with the query
//                                rotated by rope_kv or by the attention kernel
//   moe_combine_(partial_)rmsnorm = the kernels before
//   moe_group_ep               = moe_count_ep_kernel + moe_place_ep_kernel
//   moe_gather_ep_bf16         = moe_gather_ep_kernel<bf16_t>
//   moe_partial_ep             = the kernel before
//   router_topk_wave_kernel    = router_topk_wave_kernel_ref (planted ties,
//                                +-0, finished rows, renumbered experts)
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/tests/small_kernels_test.cpp -o small_kernels_test
// Run: tools/gpulock.sh $(tools/gpus.py --free 1) -- ./small_kernels_test
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <random>
#include <vector>

#include "../kernels/fast_misc.hpp"
#include "../kernels/moe.hpp"
#include "../kernels/moe_ep.hpp"
#include "small_kernels_ref.hpp"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static std::mt19937 g_rng(12345);
static int g_failures = 0;

// Device buffer with a host copy.
template <class T> struct Buf {
  T *d = nullptr;
  size_t n = 0;
  explicit Buf(size_t n_) : n(n_) { CHECK(hipMalloc(&d, std::max<size_t>(n, 1) * sizeof(T))); }
  Buf(const Buf &) = delete;
  ~Buf() { (void)hipFree(d); }
  void put(const std::vector<T> &h) {
    CHECK(hipMemcpy(d, h.data(), n * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> get() const {
    std::vector<T> h(n);
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
  }
  void fill(int byte) { CHECK(hipMemset(d, byte, n * sizeof(T))); }
  void copy_from(const Buf &o) {
    CHECK(hipMemcpy(d, o.d, n * sizeof(T), hipMemcpyDeviceToDevice));
  }
};

template <class T> static bool same(const Buf<T> &a, const Buf<T> &b) {
  CHECK(hipDeviceSynchronize());
  const auto x = a.get(), y = b.get();
  return memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

static void report(const char *what, bool ok) {
  printf("%-64s %s\n", what, ok ? "identical" : "DIFFERENT");
  g_failures += !ok;
}

static std::vector<float> normal(size_t n, float scale) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto &x : v)
    x = d(g_rng);
  return v;
}

// ---- add + RMSNorm ----------------------------------------------------------

static void test_add_rmsnorm(int B, int H) {
  Buf<float> x0((size_t)B * H), x1((size_t)B * H), y((size_t)B * H), w(H);
  Buf<bf16_t> o0((size_t)B * H), o1((size_t)B * H);
  x0.put(normal(x0.n, 1.0f));
  x1.copy_from(x0);
  y.put(normal(y.n, 0.5f));
  w.put(normal(H, 1.0f));
  Buf<float> x2((size_t)B * H);
  Buf<bf16_t> o2((size_t)B * H), o3((size_t)B * H);
  x2.copy_from(x0);
  add_inplace_kernel<<<((size_t)B * H + 255) / 256, 256>>>(x0.d, y.d, (size_t)B * H);
  rmsnorm_bf16_kernel<<<B, 256>>>(x0.d, w.d, o0.d, H);
  add_rmsnorm_bf16(0, B, x1.d, y.d, w.d, o1.d, H);
  add_rmsnorm_bf16(0, B, x2.d, y.d, w.d, o2.d, H, o3.d); // with the second copy
  char name[128];
  snprintf(name, sizeof name, "add_rmsnorm_bf16 B=%d H=%d (x, out, second copy)", B, H);
  report(name, same(x0, x1) && same(o0, o1) && same(x0, x2) && same(o0, o2) && same(o0, o3));
}

// ---- RoPE and the KV cache --------------------------------------------------

static constexpr int D = 64;

static void test_rope(int B, int NQ, int NKV, int slots, int slot) {
  const int KVD = NKV * D, ROW = (NQ + 2 * NKV) * D;
  Buf<float> q0((size_t)B * ROW), q1((size_t)B * ROW), q2((size_t)B * ROW), c(D / 2), s(D / 2);
  auto hq = normal(q0.n, 2.0f);
  for (int b = 0; b < B; b += 3) // all-zero value vectors (scale 0)
    for (int i = 0; i < D; i++)
      hq[(size_t)b * ROW + (NQ + NKV) * D + i] = 0.0f;
  q0.put(hq);
  q1.copy_from(q0);
  q2.copy_from(q0);
  std::vector<float> hc(D / 2), hs(D / 2);
  for (int i = 0; i < D / 2; i++)
    hc[i] = cosf(0.77f * i + 0.3f), hs[i] = sinf(0.77f * i + 0.3f);
  c.put(hc);
  s.put(hs);
  const size_t cache = (size_t)B * slots * KVD;
  Buf<bf16_t> k0(cache), k1(cache), k2(cache);
  Buf<int8_t> v0(cache), v1(cache), v2(cache);
  Buf<float> s0((size_t)B * slots * NKV), s1(s0.n), s2(s0.n);
  for (auto *p : {&k0, &k1, &k2})
    p->fill(0x55);
  for (auto *p : {&v0, &v1, &v2})
    p->fill(0x55);
  for (auto *p : {&s0, &s1, &s2})
    p->fill(0x55);
  rope_kv_bf16_kernel_ref<<<B, 256>>>(q0.d, ROW, c.d, s.d, NQ, NKV, D, k0.d, v0.d, s0.d, slots,
                                      slot);
  rope_kv_bf16(0, B, q1.d, ROW, c.d, s.d, NQ, NKV, k1.d, v1.d, s1.d, slots, slot, true);
  rope_kv_bf16(0, B, q2.d, ROW, c.d, s.d, NQ, NKV, k2.d, v2.d, s2.d, slots, slot, false);
  CHECK(hipDeviceSynchronize());
  // the query part of qkv: rotated (rotate_q) or untouched
  const auto a = q0.get(), b1 = q1.get(), b2 = q2.get();
  bool q_rot = true, q_kept = true;
  for (int b = 0; b < B; b++) {
    q_rot &= memcmp(&a[(size_t)b * ROW], &b1[(size_t)b * ROW], NQ * D * 4) == 0;
    q_kept &= memcmp(&hq[(size_t)b * ROW], &b2[(size_t)b * ROW], NQ * D * 4) == 0;
  }
  char name[128];
  snprintf(name, sizeof name, "rope_kv_bf16 B=%d heads %d+%d slot %d/%d: rotated q", B, NQ, NKV,
           slot, slots);
  report(name, q_rot && q_kept);
  snprintf(name, sizeof name, "rope_kv_bf16 B=%d heads %d+%d slot %d/%d: K, V codes, scales", B,
           NQ, NKV, slot, slots);
  report(name, same(k0, k1) && same(v0, v1) && same(s0, s1) && same(k0, k2) && same(v0, v2) &&
                   same(s0, s2));
}

static __device__ uint32_t hash32(uint32_t h) {
  h *= 2654435761u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  return h;
}

// Keys: bf16 of (-1, 1); value codes: -127..127; scales: (0, 0.05).
__global__ void fill_cache(bf16_t *k, int8_t *v, float *sc, size_t n, size_t ns, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = hash32((uint32_t)i ^ seed);
    k[i] = f32_to_bf16((h >> 8) * (2.0f / 16777216.0f) - 1.0f);
    v[i] = (int8_t)((int)(h % 255) - 127);
    if (i < ns)
      sc[i] = (hash32(h) >> 8) * (0.05f / 16777216.0f);
  }
}

// rope_kv_bf16_kernel_ref + attention against rope_kv_bf16 (keys only) +
// attention rotating the query, after a history of `pos` positions.
static void test_rope_attention(int B, int pos, int window) {
  const int NQ = 64, NKV = 8, KVD = NKV * D, ROW = (NQ + 2 * NKV) * D;
  const int slots = window ? window : 1024;
  Buf<float> q((size_t)B * ROW), qa((size_t)B * ROW), c(D / 2), s(D / 2), sk(NQ);
  (void)D;
  q.put(normal(q.n, 2.0f));
  sk.put(normal(NQ, 1.0f));
  std::vector<float> hc(D / 2), hs(D / 2);
  for (int i = 0; i < D / 2; i++)
    hc[i] = cosf(0.11f * i + 1.0f), hs[i] = sinf(0.11f * i + 1.0f);
  c.put(hc);
  s.put(hs);
  const size_t cache = (size_t)B * slots * KVD;
  Buf<bf16_t> k0(cache), k1(cache), o0((size_t)B * NQ * D), o1((size_t)B * NQ * D);
  Buf<int8_t> v0(cache), v1(cache);
  Buf<float> s0((size_t)B * slots * NKV), s1(s0.n);
  // the history: random keys, codes and scales
  fill_cache<<<1024, 256>>>(k0.d, v0.d, s0.d, cache, s0.n, (uint32_t)(pos * 31 + window));
  k1.copy_from(k0), v1.copy_from(v0), s1.copy_from(s0);
  const int slot = window ? pos % window : pos;
  qa.copy_from(q);
  rope_kv_bf16_kernel_ref<<<B, 256>>>(qa.d, ROW, c.d, s.d, NQ, NKV, D, k0.d, v0.d, s0.d, slots,
                                      slot);
  attention_q8v_chunk64(0, qa.d, ROW, k0.d, v0.d, s0.d, slots, KVD, sk.d, o0.d, NQ * D, B, pos,
                        window);
  rope_kv_bf16(0, B, q.d, ROW, c.d, s.d, NQ, NKV, k1.d, v1.d, s1.d, slots, slot, false);
  attention_q8v_chunk64(0, q.d, ROW, k1.d, v1.d, s1.d, slots, KVD, sk.d, o1.d, NQ * D, B, pos,
                        window, c.d, s.d);
  char name[128];
  snprintf(name, sizeof name, "rope + attention, query rotated inside, B=%d pos %d window %d", B,
           pos, window);
  report(name, same(o0, o1) && same(k0, k1) && same(v0, v1) && same(s0, s1));
}

// ---- expert grouping --------------------------------------------------------

// n pairs of tokens with k distinct experts each out of E; finished tokens
// (expert -1) now and then.
static std::vector<int> routing(int tokens, int k, int E) {
  std::vector<int> t((size_t)tokens * k);
  for (int b = 0; b < tokens; b++) {
    const bool done = g_rng() % 29 == 0;
    for (int j = 0; j < k; j++) {
      int e;
      bool again;
      do {
        e = g_rng() % E;
        again = false;
        for (int p = 0; p < j; p++)
          again |= t[(size_t)b * k + p] == e;
      } while (again);
      t[(size_t)b * k + j] = done ? -1 : e;
    }
  }
  return t;
}

static void test_group(int tok_own, int tok_peer, int k, int E, int e_begin, int e_count) {
  const int n_own = tok_own * k, n_peer = tok_peer * k, R = n_own + n_peer;
  Buf<int> own(n_own), peer(n_peer);
  own.put(routing(tok_own, k, E));
  if (n_peer)
    peer.put(routing(tok_peer, k, E));
  Buf<int> c0(e_count), c1(e_count), f0(e_count + 1), f1(e_count + 1), rp0(R), rp1(R);
  Buf<int> or0(n_own), or1(n_own), pr0(n_peer), pr1(n_peer);
  for (auto *p : {&c0, &c1, &f0, &f1, &rp0, &rp1, &or0, &or1, &pr0, &pr1})
    p->fill(0x7f);
  const int *pp = n_peer ? peer.d : nullptr;
  moe_count_ep_kernel_ref<<<e_count, MOE_GROUP_THREADS>>>(own.d, n_own, pp, n_peer, e_begin,
                                                          c0.d);
  // partner pairs numbered from n_own on: the own tokens fill the table's first rows
  moe_place_ep_kernel_ref<<<e_count, MOE_GROUP_THREADS>>>(own.d, n_own, pp, n_peer, n_own,
                                                          e_begin, e_count, c0.d, f0.d, rp0.d,
                                                          or0.d, pr0.d);
  moe_group_ep(0, own.d, n_own, pp, n_peer, n_own, e_begin, e_count, c1.d, f1.d, rp1.d, or1.d,
               pr1.d);
  char name[128];
  snprintf(name, sizeof name, "moe_group_ep %d+%d tokens x %d, experts %d..%d of %d", tok_own,
           tok_peer, k, e_begin, e_begin + e_count - 1, E);
  report(name, same(c0, c1) && same(f0, f1) && same(rp0, rp1) && same(or0, or1) &&
                   same(pr0, pr1));
}

// ---- combine + RMSNorm ------------------------------------------------------

static void test_combine(int B, int k, int H, bool partner) {
  const int E = partner ? 128 : 32, e_begin = partner ? 64 : 0, e_count = partner ? 64 : 32;
  const int R = B * k;
  auto topk = routing(B, k, E);
  std::vector<int> rows(R);
  for (int i = 0; i < R; i++)
    rows[i] = i;
  std::shuffle(rows.begin(), rows.end(), g_rng);
  Buf<float> x0((size_t)B * H), x1((size_t)B * H), Z((size_t)R * H), S((size_t)B * H), w(H),
      tw(R);
  Buf<int> te(R), orow(R);
  Buf<bf16_t> o0((size_t)B * H), o1((size_t)B * H);
  x0.put(normal(x0.n, 1.0f));
  x1.copy_from(x0);
  Z.put(normal(Z.n, 0.3f));
  S.put(normal(S.n, 0.3f));
  w.put(normal(H, 1.0f));
  tw.put(normal(R, 0.25f));
  te.put(topk);
  orow.put(rows);
  if (partner) {
    moe_combine_partial_rmsnorm_kernel_ref<<<B, 256>>>(x0.d, Z.d, orow.d, S.d, tw.d, te.d,
                                                       e_begin, e_count, k, H, w.d, o0.d);
    moe_combine_partial_rmsnorm(0, B, x1.d, Z.d, orow.d, S.d, tw.d, te.d, e_begin, e_count, k,
                                H, w.d, o1.d);
  } else {
    moe_combine_rmsnorm_kernel_ref<<<B, 256>>>(x0.d, Z.d, orow.d, tw.d, te.d, k, H, w.d, o0.d);
    moe_combine_rmsnorm(0, B, x1.d, Z.d, orow.d, tw.d, te.d, k, H, w.d, o1.d);
  }
  char name[128];
  snprintf(name, sizeof name, "moe_combine%s_rmsnorm B=%d k=%d H=%d (x and out)",
           partner ? "_partial" : "", B, k, H);
  report(name, same(x0, x1) && same(o0, o1));
}


// ---- the partner's rows: gather and partial sums ----------------------------

static void test_gather_partial(int B, int B_peer, int k, int H) {
  const int E = 128, e_begin = 64, e_count = 64;
  const int n_own = B * k, n_peer = B_peer * k, R = n_own + n_peer;
  Buf<int> own(n_own), peer(n_peer);
  const auto ho = routing(B, k, E), hp = routing(B_peer, k, E);
  own.put(ho);
  peer.put(hp);
  Buf<int> cnt(e_count), off(e_count + 1), rp(R), orow(n_own), prow(n_peer);
  moe_count_ep_kernel_ref<<<e_count, MOE_GROUP_THREADS>>>(own.d, n_own, peer.d, n_peer, e_begin,
                                                          cnt.d);
  moe_place_ep_kernel_ref<<<e_count, MOE_GROUP_THREADS>>>(own.d, n_own, peer.d, n_peer, n_own,
                                                          e_begin, e_count, cnt.d, off.d, rp.d,
                                                          orow.d, prow.d);
  // gather of bf16 rows from both GCDs' activations
  std::vector<bf16_t> xo((size_t)B * H), xp((size_t)B_peer * H);
  for (auto &v : xo)
    v = (bf16_t)g_rng();
  for (auto &v : xp)
    v = (bf16_t)g_rng();
  Buf<bf16_t> dxo(xo.size()), dxp(xp.size()), g0((size_t)R * H), g1((size_t)R * H);
  dxo.put(xo);
  dxp.put(xp);
  g0.fill(0x11);
  g1.fill(0x11);
  moe_gather_ep_kernel<bf16_t><<<R, 256>>>(dxo.d, dxp.d, rp.d, off.d, e_count, k, B, g0.d, H);
  moe_gather_ep_bf16(0, R, dxo.d, dxp.d, rp.d, off.d, e_count, k, B, g1.d, H);
  char name[128];
  snprintf(name, sizeof name, "moe_gather_ep_bf16 %d+%d tokens x %d, H=%d", B, B_peer, k, H);
  report(name, same(g0, g1));
  // partial sums for the partner's tokens
  Buf<float> Z((size_t)R * H), tw(n_peer), S0((size_t)B_peer * H), S1((size_t)B_peer * H);
  Z.put(normal(Z.n, 0.3f));
  tw.put(normal(n_peer, 0.25f));
  S0.fill(0x22);
  S1.fill(0x22);
  moe_partial_ep_kernel_ref<<<B_peer, 256>>>(S0.d, Z.d, prow.d, tw.d, peer.d, e_begin, e_count,
                                             k, H);
  moe_partial_ep(0, B_peer, S1.d, Z.d, prow.d, tw.d, peer.d, e_begin, e_count, k, H);
  snprintf(name, sizeof name, "moe_partial_ep %d partner tokens x %d, H=%d", B_peer, k, H);
  report(name, same(S0, S1));
}

// ---- router top-k -----------------------------------------------------------

static void test_topk(int B, int E, int k, bool renumber) {
  std::vector<float> lg = normal((size_t)B * E, 1.0f);
  // ties and signed zeros: copies of another expert's score, +0 and -0
  for (int b = 0; b < B; b++) {
    float *r = &lg[(size_t)b * E];
    if (b % 3 == 0)
      r[E - 1] = r[g_rng() % (E - 1)];
    if (b % 5 == 0) {
      r[1] = 0.0f;
      r[E / 2] = -0.0f;
      r[0] = r[E - 2] = 50.0f;
    }
    if (b % 7 == 0)
      for (int e = 0; e < E; e++)
        r[e] = (float)(int)(r[e] * 2); // many equal scores
  }
  std::vector<int> done(B, 0), slot(E);
  for (int b = 0; b < B; b += 11)
    done[b] = 1;
  for (int e = 0; e < E; e++)
    slot[e] = (e * 37 + 5) % E;
  Buf<float> L((size_t)B * E), w0((size_t)B * k), w1((size_t)B * k);
  Buf<int> e0((size_t)B * k), e1((size_t)B * k), dn(B), sl(E);
  L.put(lg);
  dn.put(done);
  sl.put(slot);
  const int *so = renumber ? sl.d : nullptr;
  const int grid = (B + TOPK_WAVES - 1) / TOPK_WAVES;
  for (const int *dd : {(const int *)nullptr, (const int *)dn.d}) {
    router_topk_wave_kernel_ref<<<grid, 64 * TOPK_WAVES>>>(L.d, E, k, w0.d, e0.d, B, dd, so);
    router_topk_wave_kernel<<<grid, 64 * TOPK_WAVES>>>(L.d, E, k, w1.d, e1.d, B, dd, so);
    char name[128];
    snprintf(name, sizeof name, "router_topk_wave_kernel B=%d E=%d k=%d%s%s", B, E, k,
             dd ? " done rows" : "", so ? " renumbered" : "");
    report(name, same(w0, w1) && same(e0, e1));
  }
}

int main() {
  for (int B : {1, 7, 512})
    for (int H : {2880, 64, 1000, 4096})
      test_add_rmsnorm(B, H);
  for (int B : {1, 37, 512}) {
    test_rope(B, 64, 8, 1024, 517);
    test_rope(B, 64, 8, 128, 127);
    test_rope(B, 8, 1, 64, 0); // other head counts: the generic version
    test_rope(B, 32, 4, 64, 63);
  }
  for (int pos : {0, 1, 17, 100, 300})
    test_rope_attention(13, pos, 0);
  for (int pos : {5, 127, 128, 300})
    test_rope_attention(13, pos, 128);
  test_rope_attention(512, 512, 0);
  test_rope_attention(512, 512, 128);
  test_group(512, 0, 4, 32, 0, 32);      // 20b
  test_group(512, 512, 4, 128, 0, 64);   // 120b, both halves
  test_group(512, 512, 4, 128, 64, 64);
  test_group(512, 511, 4, 128, 64, 64);
  test_group(1024, 1024, 4, 128, 0, 64); // 32 pairs per thread
  test_group(3, 0, 4, 32, 0, 32);
  test_group(5, 9, 4, 128, 64, 64);
  test_group(1, 0, 4, 32, 0, 32);
  test_group(511, 0, 3, 32, 0, 32);      // n_own not a multiple of 4: the old kernels
  test_group(4096, 0, 4, 32, 0, 32);     // too many pairs: the old kernels
  for (int B : {1, 9, 512}) {
    test_combine(B, 4, 2880, false);
    test_combine(B, 4, 2880, true);
  }
  test_gather_partial(512, 512, 4, 2880);
  test_gather_partial(7, 300, 4, 2880);
  test_gather_partial(33, 17, 2, 1000);
  test_gather_partial(33, 17, 6, 4096);
  test_combine(33, 2, 1000, false);
  test_combine(33, 6, 1000, true);
  test_combine(33, 8, 4096, false);
  for (int E : {32, 128, 200})
    for (bool rn : {false, true}) {
      test_topk(512, E, 4, rn);
      test_topk(7, E, 2, rn);
    }
  test_topk(33, 16, 8, false);
  printf("%s\n", g_failures ? "FAILED" : "all identical");
  return g_failures != 0;
}
