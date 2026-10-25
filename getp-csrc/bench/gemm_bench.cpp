// Times gemm_bf16() (kernels/fast_gemm.hpp) on every dense matrix multiply
// shape of gpt-oss: warm-up calls, then the mean of --iters calls between two
// hipEvents.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/gemm_bench.cpp -o gemm_bench
// Run on one GCD:
//   tools/gpulock.sh $(tools/gpus.py --free 1) -- ./gemm_bench [options]
// Options:
//   --old            also time the first fp32-FMA kernel (bench/gemm_old_fma.hpp)
//   --sweep          time every kernel configuration in kGemmConfigs, not only the picked one
//   --ks 1,2,4       with --sweep: K-slice counts to try on the tiled kernels (default 1)
//   --cfgs 0,3,7     with --sweep: only these configuration indices
//   --iters N        timed calls per measurement (default 20)
//   --ms 64,512,...  batch sizes (default 64,128,256,512,1024,2048)
//   --shapes a,b     subset of qkv,attn_out,router_20b,router_120b,lm_head
//   --interleave     time each call between attention-out multiplies at 512 rows
//                    (time of both minus the attention-out alone), as in the
//                    engine, where a small multiply finds its inputs out of the
//                    L2 cache and follows a big kernel
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../kernels/fast_gemm.hpp"
#include "gemm_old_fma.hpp"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Small random values (|x| < 0.5) so no input is a NaN or infinity.
__global__ void fill_bf16(bf16_t *p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    const float u = (h >> 8) * (1.0f / 16777216.0f) - 0.5f;
    p[i] = (bf16_t)(__float_as_uint(u) >> 16);
  }
}

struct Shape {
  const char *name;
  int N, K;
};

static const Shape kShapes[] = {
    {"qkv", 5120, 2880},       {"attn_out", 2880, 4096},
    {"router_20b", 32, 2880},  {"router_120b", 128, 2880},
    {"lm_head", 201088, 2880},
};

template <class F>
static double time_us(F f, int iters) {
  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  for (int i = 0; i < 3; i++)
    f();
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventRecord(t0, 0));
  for (int i = 0; i < iters; i++)
    f();
  CHECK(hipEventRecord(t1, 0));
  CHECK(hipEventSynchronize(t1));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, t0, t1));
  CHECK(hipEventDestroy(t0));
  CHECK(hipEventDestroy(t1));
  return ms * 1000.0 / iters;
}

static std::vector<std::string> split(const char *s) {
  std::vector<std::string> out;
  std::string cur;
  for (; *s; s++) {
    if (*s == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += *s;
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

int main(int argc, char **argv) {
  bool old = false, sweep = false, interleave = false;
  int iters = 20;
  std::vector<int> ms = {64, 128, 256, 512, 1024, 2048};
  std::vector<std::string> shapes;
  std::vector<int> kss = {1}, cfgs;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--old"))
      old = true;
    else if (!strcmp(argv[i], "--sweep"))
      sweep = true;
    else if (!strcmp(argv[i], "--interleave"))
      interleave = true;
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ms") && i + 1 < argc) {
      ms.clear();
      for (auto &t : split(argv[++i]))
        ms.push_back(atoi(t.c_str()));
    } else if (!strcmp(argv[i], "--ks") && i + 1 < argc) {
      kss.clear();
      for (auto &t : split(argv[++i]))
        kss.push_back(atoi(t.c_str()));
    } else if (!strcmp(argv[i], "--cfgs") && i + 1 < argc) {
      for (auto &t : split(argv[++i]))
        cfgs.push_back(atoi(t.c_str()));
    } else if (!strcmp(argv[i], "--shapes") && i + 1 < argc)
      shapes = split(argv[++i]);
    else {
      fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }

  int maxM = 0;
  for (int m : ms)
    maxM = std::max(maxM, m);
  bf16_t *A, *W;
  float *bias, *C;
  const size_t maxK = 4096, maxN = 201088;
  CHECK(hipMalloc(&A, maxM * maxK * 2));
  CHECK(hipMalloc(&W, maxN * 2880 * 2));
  CHECK(hipMalloc(&bias, maxN * 4));
  CHECK(hipMalloc(&C, (size_t)maxM * maxN * 4));
  fill_bf16<<<1024, 256>>>(A, maxM * maxK, 1);
  fill_bf16<<<1024, 256>>>(W, maxN * 2880, 2);
  CHECK(hipMemset(bias, 0, maxN * 4));
  CHECK(hipDeviceSynchronize());

  // the multiply the timed calls are interleaved with (--interleave)
  bf16_t *A2, *W2;
  float *C2;
  CHECK(hipMalloc(&A2, 512 * 4096 * 2));
  CHECK(hipMalloc(&W2, 2880 * 4096 * 2));
  CHECK(hipMalloc(&C2, 512 * 2880 * 4));
  fill_bf16<<<1024, 256>>>(A2, 512 * 4096, 3);
  fill_bf16<<<1024, 256>>>(W2, 2880 * 4096, 4);
  CHECK(hipDeviceSynchronize());
  auto big = [&] { gemm_bf16(0, A2, 512, W2, bias, C2, 2880, 4096); };
  const double big_us = interleave ? time_us(big, iters) : 0.0;
  auto timed = [&](auto f) {
    if (!interleave)
      return time_us(f, iters);
    return time_us(
               [&] {
                 big();
                 f();
               },
               iters) -
           big_us;
  };

  printf("%-12s %5s %7s %5s  %-26s %10s %8s\n", "shape", "M", "N", "K", "kernel", "us",
         "TFLOP/s");
  for (const Shape &sh : kShapes) {
    if (!shapes.empty()) {
      bool want = false;
      for (auto &n : shapes)
        want |= n == sh.name;
      if (!want)
        continue;
    }
    for (int M : ms) {
      const double flop = 2.0 * M * sh.N * sh.K;
      auto row = [&](const char *kname, double us) {
        printf("%-12s %5d %7d %5d  %-26s %10.1f %8.1f\n", sh.name, M, sh.N, sh.K, kname, us,
               flop / us * 1e-6);
        fflush(stdout);
      };
      const float *b = strncmp(sh.name, "lm_head", 7) ? bias : nullptr;
      const int N = sh.N, K = sh.K;
      if (old)
        row("old fp32 FMA",
            timed([&] { gemm_bf16_old_fma(0, A, M, W, b, C, N, K); }));
      int pick_ks = 1;
      const int pick = fgemm::gemm_pick_config(M, N, K, &pick_ks);
      if (!sweep) {
        char name[64];
        snprintf(name, sizeof name, "%s ks%d", fgemm::kGemmConfigs[pick].name, pick_ks);
        row(name, timed([&] { gemm_bf16(0, A, M, W, b, C, N, K); }));
        continue;
      }
      for (int c = 0; c < fgemm::kNumGemmConfigs; c++) {
        if (!cfgs.empty() && std::find(cfgs.begin(), cfgs.end(), c) == cfgs.end())
          continue;
        const fgemm::GemmConfig &cfg = fgemm::kGemmConfigs[c];
        if (K % cfg.bk)
          continue;
        const bool splitk = !strncmp(cfg.name, "splitk", 6);
        if (splitk && N > 8192)
          continue; // far too slow on the LM head; not a candidate there
        for (int ks : kss) {
          if (splitk && ks > 1)
            continue;
          char name[64];
          snprintf(name, sizeof name, "%s ks%d%s", cfg.name, ks,
                   c == pick && ks == pick_ks ? " *" : "");
          row(name, timed([&] { cfg.launch(0, A, M, W, b, C, N, K, ks); }));
        }
      }
    }
  }
  return 0;
}
