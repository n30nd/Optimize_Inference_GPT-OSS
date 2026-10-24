// Checks gemm_bf16() (kernels/fast_gemm.hpp) against an fp64 CPU reference on
// the same bf16 inputs: A ~ N(0, 1), W ~ N(0, 0.05), bias ~ N(0, 0.1).
// Every gpt-oss shape at every batch size, plus odd M and N (not multiples of
// any tile), with and without bias.
//
// Error measure: max |gpu - ref| over the checked outputs, divided by the RMS
// of the reference outputs ("rel"). fp32 accumulation gives ~1e-6; a bf16
// rounding anywhere would give ~4e-3.
// Small outputs are checked in full; large ones on every row and column of
// the last (partial) tiles plus 200,000 random positions.
// With --all-configs every kernel configuration is checked, the tiled ones
// also with split K (3 slices). Split-K runs (also when the picked
// configuration uses split K) are repeated and must be bit-identical: the
// slices add up in a fixed order.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/gemm_test.cpp -o gemm_test
// Run: tools/gpulock.sh $(tools/gpus.py --free 1) -- ./gemm_test [--all-configs] [--shapes a,b]
//   --shapes a,b   only these shapes (names as printed)
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <vector>

#include "../kernels/fast_gemm.hpp"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static uint16_t to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static double from_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// Normal samples from a counter-based hash (fast, reproducible, thread-safe).
static float normal(uint64_t i, uint64_t seed, float sd) {
  auto h = [](uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  };
  const uint64_t a = h(i * 2 + seed * 0x100000001ull), b = h(i * 2 + 1 + seed * 0x100000001ull);
  const double u1 = ((a >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  const double u2 = (b >> 11) * (1.0 / 9007199254740992.0);
  return (float)(sd * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static void fill(std::vector<uint16_t> &v, uint64_t seed, float sd) {
#pragma omp parallel for
  for (size_t i = 0; i < v.size(); i++)
    v[i] = to_bf16(normal(i, seed, sd));
}

// The positions checked for an M x N output, and the fp64 reference there.
struct Reference {
  std::vector<std::pair<int, int>> pts;
  std::vector<double> ref;
  double rms = 0;
};

static Reference make_reference(const std::vector<uint16_t> &A, const std::vector<uint16_t> &W,
                                const float *bias, int M, int N, int K) {
  Reference R;
  if ((size_t)M * N <= 600000) {
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
        R.pts.push_back({m, n});
  } else {
    // Last 33 rows over a stripe of columns, the last 300 columns over all
    // rows (the partial tiles), plus random points.
    const int m_edge = std::max(0, M - 33), n_edge = std::max(0, N - 300);
    for (int m = m_edge; m < M; m++)
      for (int n = 0; n < N; n += std::max(1, N / 4096))
        R.pts.push_back({m, n});
    for (int m = 0; m < M; m++)
      for (int n = n_edge; n < N; n++)
        R.pts.push_back({m, n});
    std::mt19937_64 rng(M * 31 + N);
    for (int i = 0; i < 200000; i++)
      R.pts.push_back({(int)(rng() % M), (int)(rng() % N)});
  }
  R.ref.resize(R.pts.size());
  double ss = 0;
#pragma omp parallel for reduction(+ : ss) schedule(static, 256)
  for (size_t i = 0; i < R.pts.size(); i++) {
    const int m = R.pts[i].first, n = R.pts[i].second;
    double s = bias ? bias[n] : 0.0;
    const uint16_t *a = &A[(size_t)m * K], *w = &W[(size_t)n * K];
    for (int k = 0; k < K; k++)
      s += from_bf16(a[k]) * from_bf16(w[k]);
    R.ref[i] = s;
    ss += s * s;
  }
  R.rms = sqrt(ss / R.pts.size());
  return R;
}

static double rel_error(const Reference &R, const std::vector<float> &C, int N) {
  double worst = 0;
  for (size_t i = 0; i < R.pts.size(); i++) {
    const double d = fabs((double)C[(size_t)R.pts[i].first * N + R.pts[i].second] - R.ref[i]);
    worst = std::max(worst, std::isfinite(d) ? d : INFINITY);
  }
  return worst / R.rms;
}

int main(int argc, char **argv) {
  bool all = false;
  const char *only = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--all-configs"))
      all = true;
    else if (!strcmp(argv[i], "--shapes") && i + 1 < argc)
      only = argv[++i];
  }
  struct NK {
    const char *name;
    int N, K;
  };
  const NK shapes[] = {{"qkv", 5120, 2880},       {"attn_out", 2880, 4096},
                       {"router_20b", 32, 2880},  {"router_120b", 128, 2880},
                       {"lm_head", 201088, 2880}, {"odd_n", 1000, 2880},
                       {"odd_n_small", 77, 64},   {"k32", 300, 32}};
  const int ms[] = {64, 128, 256, 512, 1024, 2048, 1, 7, 100, 333, 513};
  double worst = 0;
  int cases = 0, fails = 0;
  for (const NK &s : shapes) {
    if (only) { // comma-separated list of names
      bool want = false;
      for (const char *p = only; *p;) {
        const char *e = strchr(p, ',');
        const size_t len = e ? (size_t)(e - p) : strlen(p);
        want |= len == strlen(s.name) && !strncmp(p, s.name, len);
        p += len + (e ? 1 : 0);
      }
      if (!want)
        continue;
    }
    const int N = s.N, K = s.K;
    std::vector<uint16_t> W((size_t)N * K);
    std::vector<float> bias(N);
    fill(W, 2000 + N * 3 + K, 0.05f);
    for (int n = 0; n < N; n++)
      bias[n] = normal(n, 3, 0.1f);
    const bool with_bias = strcmp(s.name, "lm_head") != 0;
    bf16_t *dW;
    float *dB;
    CHECK(hipMalloc(&dW, W.size() * 2));
    CHECK(hipMalloc(&dB, N * 4));
    CHECK(hipMemcpy(dW, W.data(), W.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, bias.data(), N * 4, hipMemcpyHostToDevice));
    const float *b = with_bias ? dB : nullptr;

    for (int M : ms) {
      if (!strcmp(s.name, "lm_head") && M != 512 && M != 2048 && M != 333)
        continue; // the LM head reference is slow; three sizes are enough
      std::vector<uint16_t> A((size_t)M * K);
      fill(A, 1000 + M * 7 + K, 1.0f);
      const Reference R = make_reference(A, W, with_bias ? bias.data() : nullptr, M, N, K);
      bf16_t *dA;
      float *dC;
      CHECK(hipMalloc(&dA, A.size() * 2));
      CHECK(hipMalloc(&dC, (size_t)M * N * 4));
      CHECK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
      std::vector<float> C((size_t)M * N), C2;

      // One run into a NaN-filled output (unwritten outputs fail).
      auto run = [&](int config, int ks, std::vector<float> &out) {
        CHECK(hipMemset(dC, 0xff, (size_t)M * N * 4));
        if (config < 0)
          gemm_bf16(0, dA, M, dW, b, dC, N, K);
        else
          fgemm::kGemmConfigs[config].launch(0, dA, M, dW, b, dC, N, K, ks);
        CHECK(hipGetLastError());
        out.resize((size_t)M * N);
        CHECK(hipMemcpy(out.data(), dC, out.size() * 4, hipMemcpyDeviceToHost));
      };

      for (int c = all ? 0 : -1; c < (all ? fgemm::kNumGemmConfigs : 0); c++)
        for (int ks : {1, 3}) {
          if (c >= 0 && K % fgemm::kGemmConfigs[c].bk)
            continue;
          if (c < 0 && ks != 1)
            continue; // the picked configuration chooses its own K slices
          if (c >= 0 && ks != 1 && !strncmp(fgemm::kGemmConfigs[c].name, "splitk", 6))
            continue; // the router kernel has no K-slice option
          int used_ks = ks;
          const char *cfg;
          if (c < 0)
            cfg = fgemm::kGemmConfigs[fgemm::gemm_pick_config(M, N, K, &used_ks)].name;
          else
            cfg = fgemm::kGemmConfigs[c].name;
          run(c, ks, C);
          const double rel = rel_error(R, C, N);
          bool repeat_ok = true;
          if (used_ks != 1) {
            run(c, ks, C2);
            repeat_ok = memcmp(C.data(), C2.data(), C.size() * 4) == 0;
          }
          const bool ok = rel < 1e-4 && repeat_ok;
          worst = std::max(worst, std::isnan(rel) ? INFINITY : rel);
          cases++;
          fails += !ok;
          printf("%-12s M=%5d N=%6d K=%5d bias=%d  %-22s ks%-2d rel err %.2e %s%s\n", s.name,
                 M, N, K, with_bias, cfg, used_ks, rel, ok ? "ok" : "FAIL",
                 repeat_ok ? "" : " (not repeatable)");
          fflush(stdout);
        }
      CHECK(hipFree(dA));
      CHECK(hipFree(dC));
    }
    CHECK(hipFree(dW));
    CHECK(hipFree(dB));
  }
  printf("%d cases, %d failed, worst rel err %.2e -> %s\n", cases, fails, worst,
         fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
