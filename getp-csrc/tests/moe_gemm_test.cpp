// Unit test of gemm_grouped_mxfp4 (kernels/fast_moe_gemm.hpp) against a CPU
// reference in double precision computed from the same bf16 A and the decoded
// MXFP4 weights.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a \
//     getp-csrc/tests/moe_gemm_test.cpp -o moe_gemm_test
// Run:
//   tools/gpulock.sh <hip id> -- ./moe_gemm_test [routing dump file]
//
// The dump file holds lines "B pos layer offsets[0] .. offsets[E]" (real
// routing of gpt-oss-20b); without it only synthetic routings are tested.
//
// Checks:
//  1. decode: one-hot A rows make C equal the decoded weights, which must be
//     bit-identical to LUT[code] * 2^(scale - 127), for all 16 codes and all
//     scale bytes 2 .. 252 (scale 1 makes 0.5 * 2^-126, below bf16's normal range);
//  2. every shape (mlp1 5760x2880, mlp2 2880x2880) with 32, 64 (one GCD of
//     gpt-oss-120b) and 128 experts, uneven, real and edge-case routings
//     (empty experts, one expert with all rows, row counts around multiples
//     of 16 and 64). The error of each
//     output is divided by sum_k |A||W| + |bias|, the size fp32 rounding
//     errors scale with; it must stay below 1e-5;
//  3. the optional fused mlp1 entry points (gemm_grouped_mxfp4_swiglu and
//     gemm_grouped_mxfp4_gather_swiglu) equal the separate kernels
//     (gather_bf16_kernel, gemm_grouped_mxfp4, swiglu_bf16_kernel) bit for bit.
#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "../kernels/fast_misc.hpp"
#include "../kernels/fast_moe_gemm.hpp"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const float kLut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                               4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                               -2.0f, -3.0f, -4.0f, -6.0f};

static float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t f2bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

// Weight (n, k) of one expert.
static float wval(const uint8_t *blk, const uint8_t *scl, int K, int n, int k) {
  const size_t g = (size_t)n * (K / 32) + k / 32;
  const uint8_t byte = blk[g * 16 + (k % 32) / 2];
  const int code = (k & 1) ? byte >> 4 : byte & 15;
  return kLut[code] * ldexpf(1.0f, (int)scl[g] - 127);
}

struct Case {
  std::string name;
  int E, N, K;
  std::vector<int> counts; // rows per expert
  int smin = 115, smax = 136;
};

static std::vector<int> counts_from_offsets(const std::vector<int> &off) {
  std::vector<int> c(off.size() - 1);
  for (size_t e = 0; e + 1 < off.size(); e++)
    c[e] = off[e + 1] - off[e];
  return c;
}

// B tokens each picking 4 distinct of E_all experts, with a skewed
// popularity; returns the row counts of the first E experts.
static std::vector<int> random_routing(int E, int B, unsigned seed, double skew,
                                       int E_all = 0) {
  if (E_all == 0)
    E_all = E;
  std::mt19937 rng(seed);
  std::vector<double> p(E_all);
  std::gamma_distribution<double> gam(1.0 / skew, 1.0);
  for (auto &x : p)
    x = gam(rng);
  std::vector<int> c(E_all, 0);
  for (int b = 0; b < B; b++) {
    std::vector<double> q = p;
    for (int j = 0; j < 4; j++) {
      std::discrete_distribution<int> d(q.begin(), q.end());
      const int e = d(rng);
      c[e]++;
      q[e] = 0.0;
    }
  }
  c.resize(E);
  return c;
}

// Returns the number of failures.
static int run_case(const Case &cs, bool verbose = true) {
  const int E = cs.E, N = cs.N, K = cs.K;
  std::vector<int> off(E + 1, 0);
  int max_rows = 0;
  for (int e = 0; e < E; e++) {
    off[e + 1] = off[e] + cs.counts[e];
    max_rows = std::max(max_rows, cs.counts[e]);
  }
  const int R = off[E];
  std::mt19937 rng(1234 + N + E + R);
  const size_t wbytes = (size_t)N * K / 2, sbytes = (size_t)N * K / 32;
  std::vector<uint8_t> blk((size_t)E * wbytes), scl((size_t)E * sbytes);
  for (auto &b : blk)
    b = rng() & 0xff;
  std::uniform_int_distribution<int> sd(cs.smin, cs.smax);
  for (size_t i = 0; i < scl.size(); i++)
    scl[i] = (uint8_t)(cs.smin == 1 ? 1 + i % 252 : sd(rng));
  std::vector<uint16_t> A((size_t)std::max(R, 1) * K);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (auto &a : A)
    a = f2bf16(nd(rng));
  std::vector<float> bias((size_t)E * N);
  for (auto &b : bias)
    b = nd(rng) * 0.1f;

  uint16_t *dA;
  int *dOff;
  uint8_t *dB, *dS;
  float *dBias, *dC;
  CHECK(hipMalloc(&dA, A.size() * 2));
  CHECK(hipMalloc(&dOff, off.size() * 4));
  CHECK(hipMalloc(&dB, blk.size()));
  CHECK(hipMalloc(&dS, scl.size()));
  CHECK(hipMalloc(&dBias, bias.size() * 4));
  CHECK(hipMalloc(&dC, (size_t)std::max(R, 1) * N * 4));
  CHECK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dOff, off.data(), off.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dB, blk.data(), blk.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dS, scl.data(), scl.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dBias, bias.data(), bias.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemset(dC, 0xff, (size_t)std::max(R, 1) * N * 4)); // NaN: unwritten rows show
  gemm_grouped_mxfp4(0, dA, dOff, max_rows, E, dB, dS, dBias, dC, N, K);
  CHECK(hipGetLastError());
  std::vector<float> C((size_t)std::max(R, 1) * N);
  CHECK(hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));

  double worst = 0.0, worst_rel = 0.0;
  long long bad = 0;
  for (int e = 0; e < E; e++) {
    if (cs.counts[e] == 0)
      continue;
    const uint8_t *eb = blk.data() + e * wbytes, *es = scl.data() + e * sbytes;
    std::vector<float> W((size_t)N * K);
#pragma omp parallel for
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++)
        W[(size_t)n * K + k] = wval(eb, es, K, n, k);
#pragma omp parallel for collapse(2) reduction(max : worst, worst_rel) reduction(+ : bad)
    for (int r = off[e]; r < off[e + 1]; r++)
      for (int n = 0; n < N; n++) {
        const uint16_t *ar = &A[(size_t)r * K];
        const float *wr = &W[(size_t)n * K];
        double s = 0.0, mag = 0.0;
        for (int k = 0; k < K; k++) {
          const double p = (double)bf16f(ar[k]) * wr[k];
          s += p;
          mag += fabs(p);
        }
        s += bias[(size_t)e * N + n];
        mag += fabs(bias[(size_t)e * N + n]);
        const double got = C[(size_t)r * N + n];
        const double err = fabs(got - s) / (mag > 0 ? mag : 1.0);
        if (!(err <= 1e-5))
          bad++;
        if (!(err <= worst))
          worst = std::isnan(err) ? INFINITY : err;
        if (fabs(s) > 1e-2 * mag)
          worst_rel = std::max(worst_rel, fabs(got - s) / fabs(s));
      }
  }
  int nz = 0;
  for (int c : cs.counts)
    nz += c == 0;
  if (verbose)
    printf("%-34s E=%3d N=%4d K=%4d rows=%5d max_rows=%4d empty=%3d  "
           "max err/sum|AW| %.2e  max rel err %.2e  %s\n",
           cs.name.c_str(), E, N, K, R, max_rows, nz, worst, worst_rel,
           bad ? "FAIL" : "ok");
  CHECK(hipFree(dA));
  CHECK(hipFree(dOff));
  CHECK(hipFree(dB));
  CHECK(hipFree(dS));
  CHECK(hipFree(dBias));
  CHECK(hipFree(dC));
  return bad ? 1 : 0;
}

// One-hot rows: row r has 1.0 at k = r, so C[r, n] = W[n, r] exactly.
static int decode_test() {
  const int N = 256, K = 2880, R = K;
  std::mt19937 rng(7);
  std::vector<uint8_t> blk((size_t)N * K / 2), scl((size_t)N * K / 32);
  for (size_t i = 0; i < blk.size(); i++)
    blk[i] = (uint8_t)((i % 16) | ((rng() & 15) << 4)); // every code, both nibbles
  for (size_t i = 0; i < scl.size(); i++)
    scl[i] = (uint8_t)(2 + i % 251); // every scale byte 2 .. 252
  std::vector<uint16_t> A((size_t)R * K, 0);
  for (int r = 0; r < R; r++)
    A[(size_t)r * K + r] = 0x3f80; // 1.0
  std::vector<int> off = {0, R};
  uint16_t *dA;
  int *dOff;
  uint8_t *dB, *dS;
  float *dC;
  CHECK(hipMalloc(&dA, A.size() * 2));
  CHECK(hipMalloc(&dOff, 8));
  CHECK(hipMalloc(&dB, blk.size()));
  CHECK(hipMalloc(&dS, scl.size()));
  CHECK(hipMalloc(&dC, (size_t)R * N * 4));
  CHECK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dOff, off.data(), 8, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dB, blk.data(), blk.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dS, scl.data(), scl.size(), hipMemcpyHostToDevice));
  gemm_grouped_mxfp4(0, dA, dOff, R, 1, dB, dS, nullptr, dC, N, K);
  CHECK(hipGetLastError());
  std::vector<float> C((size_t)R * N);
  CHECK(hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));
  long long bad = 0;
  for (int r = 0; r < R; r++)
    for (int n = 0; n < N; n++) {
      const float want = wval(blk.data(), scl.data(), K, n, r);
      const float got = C[(size_t)r * N + n];
      if (memcmp(&want, &got, 4) != 0 && !(want == 0.0f && got == 0.0f)) {
        if (bad < 5)
          printf("  decode mismatch n=%d k=%d: want %g got %g\n", n, r, want, got);
        bad++;
      }
    }
  printf("%-34s %lld of %lld weights differ  %s\n", "decode (all codes, scales 2..252)",
         bad, (long long)R * N, bad ? "FAIL" : "ok");
  CHECK(hipFree(dA));
  CHECK(hipFree(dOff));
  CHECK(hipFree(dB));
  CHECK(hipFree(dS));
  CHECK(hipFree(dC));
  return bad ? 1 : 0;
}

// The fused variants against the separate kernels, bit for bit: B tokens
// pick 4 distinct experts (skewed); row_pair is built as moe_group_kernel
// builds it (rows grouped by expert, in pair order).
static int fused_test(int E, int B) {
  const int N = 5760, K = 2880, I = N / 2, topk = 4;
  const float limit = 7.0f;
  std::mt19937 rng(99);
  std::gamma_distribution<double> gam(1.0, 1.0);
  std::vector<double> pop(E);
  for (auto &x : pop)
    x = gam(rng);
  std::vector<int> topk_e((size_t)B * topk);
  for (int b = 0; b < B; b++) {
    std::vector<double> q = pop;
    for (int j = 0; j < topk; j++) {
      std::discrete_distribution<int> d(q.begin(), q.end());
      topk_e[b * topk + j] = d(rng);
      q[topk_e[b * topk + j]] = 0.0;
    }
  }
  std::vector<int> off(E + 1, 0), row_pair;
  for (int e = 0; e < E; e++) {
    for (int p = 0; p < B * topk; p++)
      if (topk_e[p] == e)
        row_pair.push_back(p);
    off[e + 1] = (int)row_pair.size();
  }
  int max_rows = 0;
  for (int e = 0; e < E; e++)
    max_rows = std::max(max_rows, off[e + 1] - off[e]);
  const int R = off[E];
  std::vector<uint8_t> blk((size_t)E * N * K / 2), scl((size_t)E * N * K / 32);
  for (auto &b : blk)
    b = rng() & 0xff;
  for (auto &x : scl)
    x = 118 + rng() % 8; // outputs of a few units, so the clamp at 7 is hit
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> X((size_t)B * K), A((size_t)R * K);
  for (auto &a : X)
    a = f2bf16(nd(rng));
  for (int r = 0; r < R; r++)
    memcpy(&A[(size_t)r * K], &X[(size_t)(row_pair[r] / topk) * K], K * 2);
  std::vector<float> bias((size_t)E * N);
  for (auto &b : bias)
    b = nd(rng);
  uint16_t *dX, *dA, *dG[3];
  int *dOff, *dRp;
  uint8_t *dB, *dS;
  float *dBias, *dC;
  CHECK(hipMalloc(&dX, X.size() * 2));
  CHECK(hipMalloc(&dA, A.size() * 2));
  CHECK(hipMalloc(&dOff, off.size() * 4));
  CHECK(hipMalloc(&dRp, row_pair.size() * 4));
  CHECK(hipMalloc(&dB, blk.size()));
  CHECK(hipMalloc(&dS, scl.size()));
  CHECK(hipMalloc(&dBias, bias.size() * 4));
  CHECK(hipMalloc(&dC, (size_t)R * N * 4));
  for (auto &g : dG) {
    CHECK(hipMalloc(&g, (size_t)R * I * 2));
    CHECK(hipMemset(g, 0xff, (size_t)R * I * 2));
  }
  CHECK(hipMemcpy(dX, X.data(), X.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dOff, off.data(), off.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dRp, row_pair.data(), row_pair.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dB, blk.data(), blk.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dS, scl.data(), scl.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dBias, bias.data(), bias.size() * 4, hipMemcpyHostToDevice));
  gemm_grouped_mxfp4(0, dA, dOff, max_rows, E, dB, dS, dBias, dC, N, K);
  swiglu_bf16_kernel<<<((size_t)R * I + 255) / 256, 256>>>(dC, dG[0], R, I, limit);
  gemm_grouped_mxfp4_swiglu(0, dA, dOff, max_rows, E, dB, dS, dBias, dG[1], limit, N, K);
  gemm_grouped_mxfp4_gather_swiglu(0, dX, dRp, topk, dOff, max_rows, E, dB, dS, dBias,
                                   dG[2], limit, N, K);
  CHECK(hipGetLastError());
  std::vector<uint16_t> g[3];
  for (int i = 0; i < 3; i++) {
    g[i].resize((size_t)R * I);
    CHECK(hipMemcpy(g[i].data(), dG[i], g[i].size() * 2, hipMemcpyDeviceToHost));
  }
  long long d1 = 0, d2 = 0, clamped = 0;
  for (size_t i = 0; i < g[0].size(); i++) {
    d1 += g[0][i] != g[1][i];
    d2 += g[0][i] != g[2][i];
    clamped += fabsf(bf16f(g[0][i])) >= 7.0f;
  }
  printf("%-34s E=%3d rows=%5d  swiglu: %lld differ, gather+swiglu: %lld differ "
         "of %lld (%lld at |G| >= 7)  %s\n",
         "mlp1 fused vs separate kernels", E, R, d1, d2, (long long)g[0].size(), clamped,
         d1 || d2 ? "FAIL" : "ok");
  for (void *p : {(void *)dX, (void *)dA, (void *)dOff, (void *)dRp, (void *)dB,
                  (void *)dS, (void *)dBias, (void *)dC, (void *)dG[0], (void *)dG[1],
                  (void *)dG[2]})
    CHECK(hipFree(p));
  return d1 || d2 ? 1 : 0;
}

int main(int argc, char **argv) {
  int fails = decode_test();
  fails += fused_test(32, 512);
  fails += fused_test(128, 512);
  std::vector<Case> cases;
  const int shapes[2][2] = {{5760, 2880}, {2880, 2880}};
  const char *sname[2] = {"mlp1", "mlp2"};
  for (int s = 0; s < 2; s++) {
    const int N = shapes[s][0], K = shapes[s][1];
    std::string p = sname[s];
    // Real routing (gpt-oss-20b, B = 512), first and last dumped layer.
    if (argc > 1) {
      FILE *f = fopen(argv[1], "r");
      std::vector<std::vector<int>> lines;
      int B, pos, l;
      while (f && fscanf(f, "%d %d %d", &B, &pos, &l) == 3) {
        std::vector<int> off(33);
        for (int i = 0; i <= 32; i++)
          if (fscanf(f, "%d", &off[i]) != 1)
            return 1;
        lines.push_back(off);
      }
      if (f)
        fclose(f);
      if (!lines.empty()) {
        cases.push_back({p + " real routing first", 32, N, K, counts_from_offsets(lines.front())});
        cases.push_back({p + " real routing last", 32, N, K, counts_from_offsets(lines.back())});
      }
    }
    cases.push_back({p + " 20b B=512 skewed", 32, N, K, random_routing(32, 512, 1 + s, 1.0)});
    cases.push_back({p + " 120b B=512 skewed", 128, N, K, random_routing(128, 512, 3 + s, 1.0)});
    // Every token on the same 4 experts; all others empty.
    std::vector<int> c4(32, 0);
    c4[3] = c4[9] = c4[20] = c4[31] = 512;
    cases.push_back({p + " 4 experts take all B=512", 32, N, K, c4});
    // One GCD of gpt-oss-120b: experts 0..63 of 128, rows from both GCDs'
    // 512 tokens.
    cases.push_back({p + " 120b GCD half B=512", 64, N, K, random_routing(64, 1024, 5 + s, 1.0, 128)});
    // One expert has every row; all others are empty.
    std::vector<int> c1(32, 0);
    c1[17] = 2048;
    cases.push_back({p + " 1 expert takes all 2048", 32, N, K, c1});
    // Row counts around the 16- and 64-row tile edges, some experts empty.
    const int odd[] = {1, 15, 16, 17, 0, 31, 33, 47, 63, 64, 65, 0, 79, 127, 128, 129,
                       0, 2, 191, 193, 3, 0, 255, 257, 5, 7, 0, 11, 13, 48, 49, 50};
    cases.push_back({p + " tile-edge row counts", 32, N, K, std::vector<int>(odd, odd + 32)});
  }
  cases.push_back({"mlp1 20b B=2048 skewed", 32, 5760, 2880, random_routing(32, 2048, 9, 1.0)});
  // Small shapes: N not a multiple of the 128-column tile, one expert.
  cases.push_back({"N=200 K=128 small", 3, 200, 128, {5, 0, 70}});
  cases.push_back({"N=64 K=64 single row", 1, 64, 64, {1}});
  for (const Case &cs : cases)
    fails += run_case(cs);
  printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
  return fails ? 1 : 0;
}
