// Micro-benchmark of the expert matrix multiply gemm_grouped_mxfp4
// (kernels/fast_moe_gemm.hpp) on real routing, against the first version
// (gemm_grouped_mxfp4_simple) and an estimate of the vendor library.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a \
//     getp-csrc/bench/moe_gemm_bench.cpp -o moe_gemm_bench
// Run:
//   tools/gpulock.sh <hip id> -- ./moe_gemm_bench <dump B=512> <dump B=1024> <dump B=2048> [iters=40] [old_iters=6]
// MOE_ONLY="20b mlp1 512" limits the run to one configuration; MOE_UNIFORM=64
// gives every expert 64 rows; MOE_FUSED=1 also times, for mlp1, the engine's
// three kernels (gather_bf16_kernel + gemm_grouped_mxfp4 + swiglu_bf16_kernel)
// against the fused gemm_grouped_mxfp4_gather_swiglu. MOE_DUMP120=<path with
// %d for B> replaces the synthetic 120b routing by dumped real routing.
//
// A dump file holds lines "B pos layer offsets[0] .. offsets[32]": the real
// routing of gpt-oss-20b, one line per layer and step. Each timed call uses
// the next line, so the calls see the routing of different layers, as in a
// decode step. gpt-oss-120b splits its 128 experts over the two GCDs of a
// card: each GCD runs experts 0..63 (or 64..127) on the rows of both GCDs'
// B tokens (about 4 * B rows, at most 2 * B per expert). Its routing is
// synthetic (2 * B tokens picking 4 of 128 experts, the first 64 kept), with
// the same unevenness as the measured 20b one, unless MOE_DUMP120 points to
// dumps of the real one (see below; the real one is more uneven).
//
// The fast kernel reads the weights repacked by moe_pack_mxfp4 (done once,
// not timed), the simple version the checkpoint layout.
//
// Per configuration it prints the mean time over `iters` calls (after 3
// warm-up calls), the useful work in TFLOP/s (2 * rows * N * K per call), the
// bytes per second (weights of the experts that got rows + A + C), and the
// library estimate: the sum over experts of the per-expert bf16 library time
// for that expert's row count, interpolated from yardsticks/gemm_bench.json.
#include <hip/hip_runtime.h>
#include <math.h>
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

// Library (hipBLASLt via PyTorch, bf16 weights) time in us for one expert of
// M rows, from yardsticks/gemm_bench.json.
static const double kLibM[8] = {1, 16, 64, 128, 256, 512, 1024, 2048};
static const double kLibMlp1[8] = {42.2, 43.3, 48.6, 54.9, 80.2, 162.8, 302.3, 592.9};
static const double kLibMlp2[8] = {30.0, 30.4, 32.5, 39.0, 59.4, 78.5, 158.8, 308.9};

static double lib_us(const double *t, int m) {
  if (m <= 0)
    return 0.0;
  if (m >= 2048)
    return t[7] * m / 2048.0;
  int i = 0;
  while (kLibM[i + 1] < m)
    i++;
  const double f = (m - kLibM[i]) / (kLibM[i + 1] - kLibM[i]);
  return t[i] + f * (t[i + 1] - t[i]);
}

static std::vector<std::vector<int>> read_dump(const char *path) {
  std::vector<std::vector<int>> sets;
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  int B, pos, l;
  while (fscanf(f, "%d %d %d", &B, &pos, &l) == 3) {
    std::vector<int> off(33);
    for (int i = 0; i <= 32; i++)
      if (fscanf(f, "%d", &off[i]) != 1)
        exit(1);
    sets.push_back(off);
  }
  fclose(f);
  return sets;
}

// B tokens each picking 4 distinct of E_all experts; expert popularity drawn
// from a gamma distribution of shape `shape` (smaller = more uneven). Returns
// the offsets of the first E experts (E <= E_all).
static std::vector<int> synth_offsets(int E, int B, unsigned seed, double shape,
                                      int E_all = 0) {
  if (E_all == 0)
    E_all = E;
  std::mt19937 rng(seed);
  std::gamma_distribution<double> gam(shape, 1.0);
  std::vector<double> p(E_all);
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
  std::vector<int> off(E + 1, 0);
  for (int e = 0; e < E; e++)
    off[e + 1] = off[e] + c[e];
  return off;
}

static double cv_of(const std::vector<int> &off) {
  const int E = (int)off.size() - 1;
  double m = (double)off[E] / E, v = 0;
  for (int e = 0; e < E; e++)
    v += (off[e + 1] - off[e] - m) * (off[e + 1] - off[e] - m);
  return sqrt(v / E) / m;
}

struct Timing {
  double mean, mn, mx;
};

template <class F>
static Timing time_calls(F call, int nsets, int iters) {
  for (int i = 0; i < 3; i++)
    call(i % nsets);
  CHECK(hipDeviceSynchronize());
  std::vector<hipEvent_t> ev(iters + 1);
  for (auto &e : ev)
    CHECK(hipEventCreate(&e));
  CHECK(hipEventRecord(ev[0], 0));
  for (int i = 0; i < iters; i++) {
    call(i % nsets);
    CHECK(hipEventRecord(ev[i + 1], 0));
  }
  CHECK(hipEventSynchronize(ev[iters]));
  Timing t{0, 1e30, 0};
  for (int i = 0; i < iters; i++) {
    float ms;
    CHECK(hipEventElapsedTime(&ms, ev[i], ev[i + 1]));
    t.mean += ms * 1000.0 / iters;
    t.mn = std::min(t.mn, ms * 1000.0);
    t.mx = std::max(t.mx, ms * 1000.0);
  }
  for (auto &e : ev)
    CHECK(hipEventDestroy(e));
  return t;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <dump B=512> <dump B=1024> <dump B=2048> [iters] [old_iters]\n", argv[0]);
    return 1;
  }
  const int iters = argc > 4 ? atoi(argv[4]) : 40;
  const int old_iters = argc > 5 ? atoi(argv[5]) : 6;
  const int K = 2880;
  struct Shape {
    const char *name;
    int N;
    const double *lib;
  } shapes[2] = {{"mlp1", 5760, kLibMlp1}, {"mlp2", 2880, kLibMlp2}};

  // Routing sets per (model, B).
  struct Config {
    std::string model;
    int E, B, max_rows;
    std::vector<std::vector<int>> sets;
  };
  std::vector<Config> cfgs;
  const int Bs[3] = {512, 1024, 2048};
  for (int i = 0; i < 3; i++)
    cfgs.push_back({"20b", 32, Bs[i], Bs[i], read_dump(argv[1 + i])});
  // Synthetic 120b routing: same gamma shape for every B, chosen so that the
  // synthetic 20b routing at B=512 has the measured unevenness.
  const double shape = 0.45;
  {
    double cv20 = 0, cvreal = 0;
    for (int s = 0; s < 24; s++)
      cv20 += cv_of(synth_offsets(32, 512, 100 + s, shape)) / 24;
    for (auto &o : cfgs[0].sets)
      cvreal += cv_of(o) / cfgs[0].sets.size();
    printf("routing unevenness (std/mean of rows per expert): real 20b B=512 %.2f, "
           "synthetic 20b B=512 %.2f\n", cvreal, cv20);
  }
  // MOE_DUMP120=<path with %d> uses real routing of the gpt-oss-120b split
  // instead (one GCD's 64 experts per line: "rows e_begin layer offsets[0]
  // .. offsets[64]", 72 lines per step), when the file for that B exists:
  // positions from 16 on (the first ones are the prompts' shared prefix),
  // every 37th line, so the calls mix layers and both GCDs.
  for (int i = 0; i < 3; i++) {
    Config c{"120b", 64, Bs[i], 2 * Bs[i], {}};
    if (const char *pat = getenv("MOE_DUMP120")) {
      char path[1024];
      snprintf(path, sizeof path, pat, Bs[i]);
      if (FILE *f = fopen(path, "r")) {
        int rows, e0, l, n = 0;
        while (fscanf(f, "%d %d %d", &rows, &e0, &l) == 3) {
          std::vector<int> off(65);
          for (auto &o : off)
            if (fscanf(f, "%d", &o) != 1)
              exit(1);
          if (n++ >= 16 * 72 && n % 37 == 0)
            c.sets.push_back(off);
        }
        fclose(f);
      }
    }
    for (int s = 0; c.sets.empty() && s < 24; s++)
      c.sets.push_back(synth_offsets(64, 2 * Bs[i], 1000 * i + s, shape, 128));
    cfgs.push_back(c);
  }

  // MOE_UNIFORM=<rows> replaces every routing by <rows> rows per expert (a
  // diagnostic of the kernel itself, without the unevenness).
  if (const char *u = getenv("MOE_UNIFORM")) {
    const int rows = atoi(u);
    for (auto &c : cfgs) {
      std::vector<int> off(c.E + 1);
      for (int e = 0; e <= c.E; e++)
        off[e] = e * rows;
      c.sets.assign(1, off);
      c.max_rows = rows;
    }
  }

  int dev;
  CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  printf("device %s, %d CUs; work item %d rows x %d columns\n\n", prop.gcnArchName,
         prop.multiProcessorCount, MOE_MT, MOE_NT);
  printf("%-5s %-5s %5s | %9s %9s %9s | %9s %7s %7s | %9s %9s | %7s\n", "model", "shape",
         "B", "old us", "new us", "new min", "TFLOP/s", "GB/s", "speedup", "lib est",
         "lib avg", "vs lib");

  // MOE_ONLY="20b mlp1 512" runs one configuration (for profilers).
  const char *only = getenv("MOE_ONLY");
  std::mt19937 rng(42);
  for (auto &cf : cfgs) {
    for (auto &sh : shapes) {
      if (only && cf.model + " " + sh.name + " " + std::to_string(cf.B) != only)
        continue;
      const int E = cf.E, N = sh.N;
      int R = 0; // rows of the largest set (buffer size)
      double R_avg = 0;
      for (int i = 0; i < iters; i++)
        R_avg += (double)cf.sets[i % cf.sets.size()][E] / iters;
      for (auto &o : cf.sets)
        R = std::max(R, o[E]);
      const size_t wbytes = (size_t)N * K / 2, sbytes = (size_t)N * K / 32;
      std::vector<uint8_t> hb((size_t)E * wbytes), hs((size_t)E * sbytes);
      for (size_t i = 0; i < hb.size(); i += 4) {
        const uint32_t r = rng();
        memcpy(&hb[i], &r, 4);
      }
      for (auto &s : hs)
        s = 115 + rng() % 22;
      std::vector<uint16_t> hA((size_t)R * K);
      for (auto &a : hA)
        a = (uint16_t)((0x3f80 | (rng() & 0x7f)) ^ ((rng() & 1) << 15)); // +-[1, 2)
      uint8_t *dB, *dS;
      uint16_t *dA;
      float *dBias, *dC;
      int *dOff;
      const int nsets = (int)cf.sets.size();
      CHECK(hipMalloc(&dB, hb.size()));
      CHECK(hipMalloc(&dS, hs.size()));
      CHECK(hipMalloc(&dA, hA.size() * 2));
      CHECK(hipMalloc(&dBias, (size_t)E * N * 4));
      CHECK(hipMalloc(&dC, (size_t)R * N * 4));
      CHECK(hipMalloc(&dOff, (size_t)nsets * (E + 1) * 4));
      CHECK(hipMemcpy(dB, hb.data(), hb.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dS, hs.data(), hs.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
      CHECK(hipMemset(dBias, 0, (size_t)E * N * 4));
      for (int s = 0; s < nsets; s++)
        CHECK(hipMemcpy(dOff + s * (E + 1), cf.sets[s].data(), (E + 1) * 4,
                        hipMemcpyHostToDevice));
      uint8_t *dPB, *dPS; // packed copies for the fast kernel
      CHECK(hipMalloc(&dPB, (size_t)E * moe_packed_blocks_bytes(N, K)));
      CHECK(hipMalloc(&dPS, (size_t)E * moe_packed_scales_bytes(N, K)));
      moe_pack_mxfp4(0, dB, dS, E, N, K, dPB, dPS);
      CHECK(hipDeviceSynchronize());

      // Per-call averages of the work done.
      double lib = 0, bytes = 0;
      const int calls = iters;
      for (int i = 0; i < calls; i++) {
        const auto &o = cf.sets[i % nsets];
        for (int e = 0; e < E; e++) {
          const int m = o[e + 1] - o[e];
          lib += lib_us(sh.lib, m) / calls;
          if (m > 0)
            bytes += (double)(wbytes + sbytes) / calls;
        }
      }
      bytes += R_avg * K * 2 + R_avg * N * 4;
      const double flops = 2.0 * R_avg * N * K;

      auto call_new = [&](int s) {
        gemm_grouped_mxfp4(0, dA, dOff + s * (E + 1), cf.max_rows, E, dPB, dPS, dBias, dC, N, K);
      };
      auto call_old = [&](int s) {
        gemm_grouped_mxfp4_simple(0, dA, dOff + s * (E + 1), cf.max_rows, E, dB, dS, dBias, dC, N, K);
      };
      const Timing tn = time_calls(call_new, nsets, iters);
      Timing to{0, 0, 0};
      if (old_iters > 0)
        to = time_calls(call_old, nsets, old_iters);
      CHECK(hipGetLastError());
      const double lib64 = E * lib_us(sh.lib, (int)(R_avg / E + 0.5));
      printf("%-5s %-5s %5d | %9.1f %9.1f %9.1f | %9.1f %7.0f %6.1fx | %9.1f %9.1f | %6.2fx\n",
             cf.model.c_str(), sh.name, cf.B, to.mean, tn.mean, tn.mn,
             flops / tn.mean * 1e-6, bytes / tn.mean * 1e-3,
             old_iters > 0 ? to.mean / tn.mean : 0.0, lib, lib64, lib / tn.mean);
      fflush(stdout);
      if (getenv("MOE_FUSED") && N == 5760) {
        // Tokens X (B, K); row r of the grouped order is token row_pair[r] / 4.
        const int I = N / 2, topk = 4;
        std::vector<int> rp(R);
        for (int r = 0; r < R; r++)
          rp[r] = (int)(((long long)r * 7919) % (cf.B * topk));
        int *dRp;
        uint16_t *dX, *dG;
        CHECK(hipMalloc(&dRp, (size_t)R * 4));
        CHECK(hipMalloc(&dX, (size_t)cf.B * K * 2));
        CHECK(hipMalloc(&dG, (size_t)R * I * 2));
        CHECK(hipMemcpy(dRp, rp.data(), (size_t)R * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dX, hA.data(), (size_t)cf.B * K * 2, hipMemcpyHostToDevice));
        std::vector<int> rows(nsets);
        for (int s = 0; s < nsets; s++)
          rows[s] = cf.sets[s][E];
        auto call_sep = [&](int s) {
          gather_bf16_kernel<<<rows[s], 256, 0, 0>>>(dX, dRp, topk, dA, K);
          gemm_grouped_mxfp4(0, dA, dOff + s * (E + 1), cf.max_rows, E, dPB, dPS, dBias, dC, N, K);
          swiglu_bf16_kernel<<<((size_t)rows[s] * I + 255) / 256, 256, 0, 0>>>(dC, dG, rows[s], I, 7.0f);
        };
        auto call_fused = [&](int s) {
          gemm_grouped_mxfp4_gather_swiglu(0, dX, dRp, topk, dOff + s * (E + 1), cf.max_rows, E,
                                           dPB, dPS, dBias, dG, 7.0f, N, K);
        };
        auto call_swiglu = [&](int s) {
          gemm_grouped_mxfp4_swiglu(0, dA, dOff + s * (E + 1), cf.max_rows, E, dPB, dPS, dBias, dG,
                                    7.0f, N, K);
        };
        const Timing ts = time_calls(call_sep, nsets, iters);
        const Timing tw = time_calls(call_swiglu, nsets, iters);
        const Timing tf = time_calls(call_fused, nsets, iters);
        CHECK(hipGetLastError());
        printf("      mlp1 fusion: gather + gemm + swiglu %.1f us | gemm+swiglu %.1f us (+ gather) | "
               "gather+gemm+swiglu %.1f us\n", ts.mean, tw.mean, tf.mean);
        CHECK(hipFree(dRp));
        CHECK(hipFree(dX));
        CHECK(hipFree(dG));
      }
      CHECK(hipFree(dB));
      CHECK(hipFree(dS));
      CHECK(hipFree(dPB));
      CHECK(hipFree(dPS));
      CHECK(hipFree(dA));
      CHECK(hipFree(dBias));
      CHECK(hipFree(dC));
      CHECK(hipFree(dOff));
    }
  }
  printf("\nold/new: mean over calls (old %d, new %d); TFLOP/s and GB/s of the new kernel;\n"
         "lib est: sum over experts of the library time at that expert's rows; "
         "lib avg: E x library time at the average rows per expert.\n",
         old_iters, iters);
  return 0;
}
