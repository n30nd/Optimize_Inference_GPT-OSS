// Times the small fast-path kernels between the matrix multiplies, old
// (tests/small_kernels_ref.hpp) against new, on the shapes of one GCD with
// 512 requests, and checks that both give the same bits.
//
// Each kernel is timed the way the engine runs it: right after the kernel
// that precedes it there (by default the attention-out multiply, 512 x 2880
// x 4096; the qkv multiply for RoPE), which leaves the L2 cache full of other
// data. Time of the pair minus the preceding kernel alone, mean of --iters
// pairs, in alternating rounds; "alone" = back to back.
// --evict also runs four other multiplies before each timed call, so that the
// small kernel's code is no longer in the instruction cache, as in the engine,
// where a whole layer of other kernels runs between two calls. Short kernels
// with much code are much slower that way.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/small_kernels_bench.cpp -o small_bench
// Run on one GCD (two for the partner reads of "group"):
//   tools/gpulock.sh $(tools/gpus.py --free 2) -- ./small_bench [--iters N] [--rounds R] [--evict]
//       [--only add_rmsnorm,rope,rope_attn,router,group,combine]
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "../kernels/fast_attention.hpp"
#include "../kernels/fast_gemm.hpp"
#include "../kernels/fast_misc.hpp"
#include "../kernels/moe.hpp"
#include "../kernels/moe_ep.hpp"
#include "../tests/small_kernels_ref.hpp"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int g_iters = 50, g_rounds = 3;
static bool g_evict = false;
static std::vector<std::string> g_only;

static bool wanted(const char *name) {
  if (g_only.empty())
    return true;
  for (auto &s : g_only)
    if (s == name)
      return true;
  return false;
}

template <class T> static T *dalloc(size_t n) {
  T *p;
  CHECK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

template <class T> static void upload(T *d, const std::vector<T> &h) {
  CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
}

template <class T> static std::vector<T> download(const T *d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

template <class T> static bool same(const T *a, const T *b, size_t n, const char *what) {
  auto x = download(a, n), y = download(b, n);
  size_t diff = 0;
  for (size_t i = 0; i < n; i++)
    diff += memcmp(&x[i], &y[i], sizeof(T)) != 0;
  if (diff)
    printf("  %s: %zu of %zu differ\n", what, diff, n);
  return diff == 0;
}

static std::vector<float> randf(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto &x : v)
    x = d(rng);
  return v;
}

// The big kernel every timed call follows.
struct Big {
  bf16_t *A, *W;
  float *C;
  Big() {
    A = dalloc<bf16_t>(512 * 4096);
    W = dalloc<bf16_t>(2880 * 4096);
    C = dalloc<float>(512 * 2880);
    CHECK(hipMemset(A, 0x3c, 512 * 4096 * 2));
    CHECK(hipMemset(W, 0x3c, 2880 * 4096 * 2));
  }
  void run() { gemm_bf16(0, A, 512, W, nullptr, C, 2880, 4096); }
};
static Big *g_big;
// The kernel each timed call follows; the attention-out multiply by default.
static std::function<void()> g_pre;

static float time_loop(const std::function<void()> &f, bool with_big) {
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0, 0));
  for (int i = 0; i < g_iters; i++) {
    if (with_big)
      g_pre();
    if (f)
      f();
  }
  CHECK(hipEventRecord(e1, 0));
  CHECK(hipEventSynchronize(e1));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  CHECK(hipEventDestroy(e0));
  CHECK(hipEventDestroy(e1));
  return ms * 1000.0f / g_iters;
}

// Prints old and new time (us per call after the big kernel), and alone.
static void compare(const char *name, const std::function<void()> &old_f,
                    const std::function<void()> &new_f) {
  for (int w = 0; w < 3; w++) {
    old_f();
    new_f();
    g_pre();
  }
  CHECK(hipDeviceSynchronize());
  double t_big = 0, t_old = 0, t_new = 0, a_old = 0, a_new = 0;
  for (int r = 0; r < g_rounds; r++) {
    t_big += time_loop(nullptr, true);
    t_old += time_loop(old_f, true);
    t_new += time_loop(new_f, true);
    a_old += time_loop(old_f, false);
    a_new += time_loop(new_f, false);
  }
  t_big /= g_rounds, t_old /= g_rounds, t_new /= g_rounds, a_old /= g_rounds, a_new /= g_rounds;
  printf("%-28s after pre: old %7.2f  new %7.2f us (%+5.1f%%) | alone: old %7.2f  new %7.2f us\n",
         name, t_old - t_big, t_new - t_big, 100.0 * ((t_new - t_big) / (t_old - t_big) - 1),
         a_old, a_new);
}

static constexpr int B = 512, H = 2880, D = 64, NQ = 64, NKV = 8, KVD = NKV * D;
static constexpr int QKV = NQ * D + 2 * KVD, K = 4;

static void bench_add_rmsnorm() {
  auto hx = randf((size_t)B * H, 1.0f, 1), hy = randf((size_t)B * H, 0.5f, 2);
  auto hw = randf(H, 1.0f, 3);
  float *x0 = dalloc<float>((size_t)B * H), *x1 = dalloc<float>((size_t)B * H);
  float *y = dalloc<float>((size_t)B * H), *w = dalloc<float>(H);
  bf16_t *o0 = dalloc<bf16_t>((size_t)B * H), *o1 = dalloc<bf16_t>((size_t)B * H);
  upload(x0, hx), upload(x1, hx), upload(y, hy), upload(w, hw);
  add_rmsnorm_bf16_kernel_ref<<<B, 256>>>(x0, y, w, o0, H);
  add_rmsnorm_bf16(0, B, x1, y, w, o1, H);
  CHECK(hipDeviceSynchronize());
  const bool ok = same(x0, x1, (size_t)B * H, "x") & same(o0, o1, (size_t)B * H, "out");
  printf("add_rmsnorm: %s\n", ok ? "bit-identical" : "DIFFERENT");
  compare(
      "add_rmsnorm", [&] { add_rmsnorm_bf16_kernel_ref<<<B, 256>>>(x0, y, w, o0, H); },
      [&] { add_rmsnorm_bf16(0, B, x1, y, w, o1, H); });
}


static void bench_rope() {
  const int slots = 1024, slot = 517, half = D / 2;
  auto hq = randf((size_t)B * QKV, 2.0f, 4);
  std::vector<float> hc(half), hs(half);
  for (int i = 0; i < half; i++)
    hc[i] = cosf(0.37f * i + 0.1f), hs[i] = sinf(0.37f * i + 0.1f);
  float *q0 = dalloc<float>((size_t)B * QKV), *q1 = dalloc<float>((size_t)B * QKV);
  float *dc = dalloc<float>(half), *ds = dalloc<float>(half);
  const size_t cache = (size_t)B * slots * KVD;
  bf16_t *k0 = dalloc<bf16_t>(cache), *k1 = dalloc<bf16_t>(cache);
  int8_t *v0 = dalloc<int8_t>(cache), *v1 = dalloc<int8_t>(cache);
  float *s0 = dalloc<float>((size_t)B * slots * NKV), *s1 = dalloc<float>((size_t)B * slots * NKV);
  upload(q0, hq), upload(q1, hq), upload(dc, hc), upload(ds, hs);
  CHECK(hipMemset(k0, 0, cache * 2));
  CHECK(hipMemset(k1, 0, cache * 2));
  CHECK(hipMemset(v0, 0, cache));
  CHECK(hipMemset(v1, 0, cache));
  CHECK(hipMemset(s0, 0, (size_t)B * slots * NKV * 4));
  CHECK(hipMemset(s1, 0, (size_t)B * slots * NKV * 4));
  rope_kv_bf16_kernel_ref<<<B, 256>>>(q0, QKV, dc, ds, NQ, NKV, D, k0, v0, s0, slots, slot);
  rope_kv_bf16(0, B, q1, QKV, dc, ds, NQ, NKV, k1, v1, s1, slots, slot);
  CHECK(hipDeviceSynchronize());
  bf16_t *wq = dalloc<bf16_t>((size_t)QKV * H);
  CHECK(hipMemset(wq, 0x3c, (size_t)QKV * H * 2));
  // the query part of qkv, and the caches
  bool ok = true;
  {
    auto a = download(q0, (size_t)B * QKV), b = download(q1, (size_t)B * QKV);
    size_t diff = 0;
    for (int r = 0; r < B; r++)
      diff += memcmp(&a[(size_t)r * QKV], &b[(size_t)r * QKV], NQ * D * 4) != 0;
    if (diff)
      printf("  q: %zu rows differ\n", diff), ok = false;
  }
  ok &= same(k0, k1, cache, "K cache") & same(v0, v1, cache, "V codes") &
        same(s0, s1, (size_t)B * slots * NKV, "V scales");
  printf("rope_kv: %s\n", ok ? "bit-identical" : "DIFFERENT");
  auto saved = g_pre;
  g_pre = [&] { gemm_bf16(0, g_big->A, B, wq, nullptr, q1, QKV, H); };
  compare(
      "rope_kv",
      [&] {
        rope_kv_bf16_kernel_ref<<<B, 256>>>(q1, QKV, dc, ds, NQ, NKV, D, k0, v0, s0, slots, slot);
      },
      [&] {
        rope_kv_bf16(0, B, q1, QKV, dc, ds, NQ, NKV, k1, v1, s1, slots, slot);
      });
  g_pre = saved;
}

// Routing of B tokens to E experts (k distinct each), 3 % finished (-1).
static std::vector<int> random_topk(int E, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<int> t((size_t)B * K);
  for (int b = 0; b < B; b++) {
    const bool done = rng() % 32 == 0;
    for (int j = 0; j < K; j++) {
      int e;
      bool again;
      do {
        e = rng() % E;
        again = false;
        for (int p = 0; p < j; p++)
          again |= t[(size_t)b * K + p] == e;
      } while (again);
      t[(size_t)b * K + j] = done ? -1 : e;
    }
  }
  return t;
}


// rope_kv + attention: the query rotated by rope_kv (before) or by the
// attention kernel as it loads it (after). Full layers at n = 513 and window
// layers at n = 128, the shapes of position 512.
static void bench_rope_attention() {
  const int slots = 1024, half = D / 2;
  float *qkv = dalloc<float>((size_t)B * QKV), *q0 = dalloc<float>((size_t)B * QKV);
  auto hq = randf((size_t)B * QKV, 2.0f, 21);
  upload(q0, hq);
  std::vector<float> hc(half), hs(half), hsink = randf(NQ, 1.0f, 22);
  for (int i = 0; i < half; i++)
    hc[i] = cosf(0.37f * i + 0.1f), hs[i] = sinf(0.37f * i + 0.1f);
  float *dc = dalloc<float>(half), *ds = dalloc<float>(half), *sk = dalloc<float>(NQ);
  upload(dc, hc), upload(ds, hs), upload(sk, hsink);
  const size_t cache = (size_t)B * slots * KVD;
  bf16_t *kc[2] = {dalloc<bf16_t>(cache), dalloc<bf16_t>(cache)};
  int8_t *vq[2] = {dalloc<int8_t>(cache), dalloc<int8_t>(cache)};
  float *vs[2] = {dalloc<float>((size_t)B * slots * NKV), dalloc<float>((size_t)B * slots * NKV)};
  bf16_t *out[2] = {dalloc<bf16_t>((size_t)B * NQ * D), dalloc<bf16_t>((size_t)B * NQ * D)};
  // history: fill both caches with the same keys and values (positions 0..511)
  for (int slot = 0; slot < 512; slot++)
    for (int v = 0; v < 2; v++) {
      CHECK(hipMemcpy(qkv, q0, (size_t)B * QKV * 4, hipMemcpyDeviceToDevice));
      rope_kv_bf16_kernel_ref<<<B, 256>>>(qkv, QKV, dc, ds, NQ, NKV, D, kc[v], vq[v], vs[v],
                                          slots, slot);
    }
  for (int timing = 0; timing < 2; timing++)
    for (int window : {0, 128}) {
      const int pos = 512, slot = window ? pos % window : pos;
      auto run_old = [&] {
        rope_kv_bf16_kernel_ref<<<B, 256>>>(qkv, QKV, dc, ds, NQ, NKV, D, kc[0], vq[0], vs[0],
                                            slots, slot);
        attention_q8v_chunk64(0, qkv, QKV, kc[0], vq[0], vs[0], slots, KVD, sk, out[0], NQ * D,
                              B, pos, window);
      };
      auto run_new = [&] {
        rope_kv_bf16(0, B, qkv, QKV, dc, ds, NQ, NKV, kc[1], vq[1], vs[1], slots, slot, false);
        attention_q8v_chunk64(0, qkv, QKV, kc[1], vq[1], vs[1], slots, KVD, sk, out[1], NQ * D,
                              B, pos, window, dc, ds);
      };
      if (!timing) {
        CHECK(hipMemcpy(qkv, q0, (size_t)B * QKV * 4, hipMemcpyDeviceToDevice));
        run_old();
        CHECK(hipMemcpy(qkv, q0, (size_t)B * QKV * 4, hipMemcpyDeviceToDevice));
        run_new();
        CHECK(hipDeviceSynchronize());
        bool ok = same(kc[0], kc[1], cache, "K cache") & same(vq[0], vq[1], cache, "V codes") &
                  same(vs[0], vs[1], (size_t)B * slots * NKV, "V scales") &
                  same(out[0], out[1], (size_t)B * NQ * D, "attention out");
        printf("rope + attention (window %d): %s\n", window, ok ? "bit-identical" : "DIFFERENT");
        continue;
      }
      // the qkv multiply first, then rope + attention (q rotated in place
      // again and again: values drift but stay finite for the timing)
      bf16_t *wq = dalloc<bf16_t>((size_t)QKV * H);
      CHECK(hipMemset(wq, 0x3c, (size_t)QKV * H * 2));
      auto saved = g_pre;
      g_pre = [&] { gemm_bf16(0, g_big->A, B, wq, nullptr, qkv, QKV, H); };
      compare(window ? "rope+attention window n=128" : "rope+attention full n=513", run_old,
              run_new);
      auto att_only = [&](const float *c, const float *sn) {
        return [&, c, sn] {
          attention_q8v_chunk64(0, qkv, QKV, kc[1], vq[1], vs[1], slots, KVD, sk, out[1],
                                NQ * D, B, pos, window, c, sn);
        };
      };
      compare("  attention alone: no rot | rot", att_only(nullptr, nullptr), att_only(dc, ds));
      g_pre = saved;
      CHECK(hipFree(wq));
    }
}


// Grouping by expert: moe_count_ep_kernel + moe_place_ep_kernel against
// moe_group_ep. 20b: 2,048 own pairs, 32 experts; 120b: own and partner
// pairs, 64 of 128 experts (both halves). With a second visible GPU the
// partner's pairs live in its memory, as in the engine.
static void bench_group() {
  int n_dev = 0;
  CHECK(hipGetDeviceCount(&n_dev));
  int *peer_mem = nullptr;
  const int NP = B * K;
  if (n_dev > 1) {
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&peer_mem, NP * sizeof(int)));
    CHECK(hipSetDevice(0));
    hipError_t err = hipDeviceEnablePeerAccess(1, 0);
    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
      CHECK(err);
  }
  struct Case {
    const char *name;
    int E, e_begin, e_count;
    bool peer;
  } cases[] = {{"group 20b (32 experts)", 32, 0, 32, false},
               {"group 120b, experts 0-63", 128, 0, 64, true},
               {"group 120b, experts 64-127", 128, 64, 64, true}};
  for (const Case &cs : cases) {
    auto own = random_topk(cs.E, 31), peer = random_topk(cs.E, 32);
    int *d_own = dalloc<int>(NP), *d_peer = peer_mem ? peer_mem : dalloc<int>(NP);
    upload(d_own, own);
    upload(d_peer, peer);
    const int n_peer = cs.peer ? NP : 0, R = NP + n_peer;
    int *o[2][5]; // counts, offsets, row_pair, own_row, peer_row
    const int sz[5] = {cs.e_count, cs.e_count + 1, R, NP, NP};
    for (int v = 0; v < 2; v++)
      for (int a = 0; a < 5; a++) {
        o[v][a] = dalloc<int>(sz[a]);
        CHECK(hipMemset(o[v][a], 0x7f, sz[a] * sizeof(int)));
      }
    auto run_old = [&] {
      moe_count_ep_kernel_ref<<<cs.e_count, MOE_GROUP_THREADS>>>(
          d_own, NP, cs.peer ? d_peer : nullptr, n_peer, cs.e_begin, o[0][0]);
      moe_place_ep_kernel_ref<<<cs.e_count, MOE_GROUP_THREADS>>>(
          d_own, NP, cs.peer ? d_peer : nullptr, n_peer, NP, cs.e_begin, cs.e_count, o[0][0],
          o[0][1], o[0][2], o[0][3], o[0][4]);
    };
    auto run_new = [&] {
      moe_group_ep(0, d_own, NP, cs.peer ? d_peer : nullptr, n_peer, NP, cs.e_begin,
                   cs.e_count, o[1][0], o[1][1], o[1][2], o[1][3], o[1][4]);
    };
    run_old();
    run_new();
    CHECK(hipDeviceSynchronize());
    bool ok = true;
    const char *what[5] = {"counts", "offsets", "row_pair", "own_row", "peer_row"};
    for (int a = 0; a < 5; a++)
      ok &= same(o[0][a], o[1][a], sz[a], what[a]);
    printf("%s: %s%s\n", cs.name, ok ? "bit-identical" : "DIFFERENT",
           cs.peer && peer_mem ? " (partner pairs on GPU 1)" : "");
    compare(cs.name, run_old, run_new);
  }
}


// bf16 copies of normal(0, scale) numbers.
static std::vector<bf16_t> randbf(size_t n, float scale, uint32_t seed) {
  auto f = randf(n, scale, seed);
  std::vector<bf16_t> v(n);
  for (size_t i = 0; i < n; i++) {
    uint32_t u;
    memcpy(&u, &f[i], 4);
    v[i] = (bf16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
  }
  return v;
}

// Router top-k: router_topk_wave_kernel before (tests/small_kernels_ref.hpp)
// and now, after the router multiply.
static void bench_router() {
  for (int E : {32, 128}) {
    auto ha = randbf((size_t)B * H, 1.0f, 41), hw = randbf((size_t)E * H, 0.03f, 42);
    auto hb = randf(E, 0.1f, 43);
    std::vector<int> hdone(B, 0), hslot(E);
    for (int b = 0; b < B; b += 37)
      hdone[b] = 1;
    for (int e = 0; e < E; e++)
      hslot[e] = (e * 37 + 11) % E; // a permutation (37 odd)
    bf16_t *a = dalloc<bf16_t>((size_t)B * H), *w = dalloc<bf16_t>((size_t)E * H);
    float *bias = dalloc<float>(E), *C = dalloc<float>((size_t)B * E);
    float *tw[2] = {dalloc<float>(B * K), dalloc<float>(B * K)};
    int *te[2] = {dalloc<int>(B * K), dalloc<int>(B * K)};
    int *done = dalloc<int>(B), *slot = dalloc<int>(E);
    upload(a, ha), upload(w, hw), upload(bias, hb), upload(done, hdone), upload(slot, hslot);
    const int *so = E == 128 ? slot : nullptr;
    const int grid = (B + TOPK_WAVES - 1) / TOPK_WAVES;
    auto run_old = [&] {
      gemm_bf16(0, a, B, w, bias, C, E, H);
      router_topk_wave_kernel_ref<<<grid, 64 * TOPK_WAVES>>>(C, E, K, tw[0], te[0], B, done, so);
    };
    auto run_new = [&] {
      gemm_bf16(0, a, B, w, bias, C, E, H);
      router_topk_wave_kernel<<<grid, 64 * TOPK_WAVES>>>(C, E, K, tw[1], te[1], B, done, so);
    };
    run_old();
    run_new();
    CHECK(hipDeviceSynchronize());
    bool ok = same(tw[0], tw[1], B * K, "topk_w") & same(te[0], te[1], B * K, "topk_e");
    printf("router top-k E=%d: %s\n", E, ok ? "bit-identical" : "DIFFERENT");
    compare(E == 32 ? "router + top-k (E=32)" : "router + top-k (E=128)", run_old, run_new);
    compare(
        "  top-k alone",
        [&] {
          router_topk_wave_kernel_ref<<<grid, 64 * TOPK_WAVES>>>(C, E, K, tw[0], te[0], B, done,
                                                                 so);
        },
        [&] {
          router_topk_wave_kernel<<<grid, 64 * TOPK_WAVES>>>(C, E, K, tw[1], te[1], B, done, so);
        });
  }
}

static void bench_combine() {
  // no partner: 32 experts, own_row = a permutation of the pairs
  auto topk = random_topk(32, 5);
  std::vector<int> rows((size_t)B * K);
  for (size_t i = 0; i < rows.size(); i++)
    rows[i] = (int)i;
  std::shuffle(rows.begin(), rows.end(), std::mt19937(6));
  auto hx = randf((size_t)B * H, 1.0f, 7), hz = randf((size_t)B * K * H, 0.3f, 8);
  auto hw = randf(H, 1.0f, 9), htw = randf((size_t)B * K, 0.25f, 10);
  float *x0 = dalloc<float>((size_t)B * H), *x1 = dalloc<float>((size_t)B * H);
  float *z = dalloc<float>((size_t)B * K * H), *w = dalloc<float>(H), *tw = dalloc<float>(B * K);
  int *te = dalloc<int>(B * K), *orow = dalloc<int>(B * K);
  bf16_t *o0 = dalloc<bf16_t>((size_t)B * H), *o1 = dalloc<bf16_t>((size_t)B * H);
  upload(x0, hx), upload(x1, hx), upload(z, hz), upload(w, hw), upload(tw, htw);
  upload(te, topk), upload(orow, rows);
  moe_combine_rmsnorm_kernel_ref<<<B, 256>>>(x0, z, orow, tw, te, K, H, w, o0);
  moe_combine_rmsnorm(0, B, x1, z, orow, tw, te, K, H, w, o1);
  CHECK(hipDeviceSynchronize());
  bool ok = same(x0, x1, (size_t)B * H, "x") & same(o0, o1, (size_t)B * H, "out");
  printf("combine_rmsnorm: %s\n", ok ? "bit-identical" : "DIFFERENT");
  compare(
      "combine_rmsnorm",
      [&] { moe_combine_rmsnorm_kernel_ref<<<B, 256>>>(x0, z, orow, tw, te, K, H, w, o0); },
      [&] { moe_combine_rmsnorm(0, B, x1, z, orow, tw, te, K, H, w, o1); });

  // partner: 128 experts, this GCD holds 0..63; S of the partner in local
  // memory here (the engine reads it from the partner)
  auto topk2 = random_topk(128, 11);
  float *S = dalloc<float>((size_t)B * H);
  upload(S, randf((size_t)B * H, 0.3f, 12));
  upload(te, topk2);
  upload(x0, hx), upload(x1, hx);
  moe_combine_partial_rmsnorm_kernel_ref<<<B, 256>>>(x0, z, orow, S, tw, te, 0, 64, K, H, w, o0);
  moe_combine_partial_rmsnorm(0, B, x1, z, orow, S, tw, te, 0, 64, K, H, w, o1);
  CHECK(hipDeviceSynchronize());
  ok = same(x0, x1, (size_t)B * H, "x") & same(o0, o1, (size_t)B * H, "out");
  printf("combine_partial_rmsnorm: %s\n", ok ? "bit-identical" : "DIFFERENT");
  compare(
      "combine_partial_rmsnorm (S local)",
      [&] {
        moe_combine_partial_rmsnorm_kernel_ref<<<B, 256>>>(x0, z, orow, S, tw, te, 0, 64, K, H,
                                                           w, o0);
      },
      [&] { moe_combine_partial_rmsnorm(0, B, x1, z, orow, S, tw, te, 0, 64, K, H, w, o1); });
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      g_iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--evict"))
      g_evict = true;
    else if (!strcmp(argv[i], "--rounds") && i + 1 < argc)
      g_rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
      std::string s = argv[++i];
      size_t p = 0;
      while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos)
          q = s.size();
        g_only.push_back(s.substr(p, q - p));
        p = q + 1;
      }
    }
  }
  g_big = new Big();
  g_pre = [] { g_big->run(); };
  if (g_evict) {
    // other kernels' code between the timed calls, as in the engine (the
    // instruction cache does not keep a small kernel's code from one layer
    // to the next)
    static float *c2 = dalloc<float>((size_t)512 * 5120);
    g_pre = [] {
      g_big->run();
      gemm_bf16(0, g_big->A, 512, g_big->W, nullptr, c2, 4096, 2880);  // stream kernel
      gemm_bf16(0, g_big->A, 512, g_big->W, nullptr, c2, 32, 2880);    // router kernel
      gemm_bf16(0, g_big->A, 2048, g_big->W, nullptr, c2, 640, 1024);  // 256x256 tiles
      gemm_bf16(0, g_big->A, 64, g_big->W, nullptr, c2, 2880, 4096);   // 64x64 split K
    };
  }
  if (wanted("add_rmsnorm"))
    bench_add_rmsnorm();
  if (wanted("rope"))
    bench_rope();
  if (wanted("rope_attn"))
    bench_rope_attention();
  if (wanted("router"))
    bench_router();
  if (wanted("group"))
    bench_group();
  if (wanted("combine"))
    bench_combine();
  return 0;
}
