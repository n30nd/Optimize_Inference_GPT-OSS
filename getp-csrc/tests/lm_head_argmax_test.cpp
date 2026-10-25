// Checks gemm_bf16_argmax() (kernels/fast_gemm.hpp) against gemm_bf16() +
// argmax_kernel (kernels/misc.hpp): the chosen column must be the same for
// every row. LM head shape (N = 201,088, K = 2,880) at the batch sizes the
// engine uses and at odd ones, with A ~ N(0, 1) and W ~ N(0, 0.05), plus
// planted ties:
//   - rows 1, 2, 3 of A have their largest value in two identical columns of
//     W (in different column tiles, the same tile, and the same lane quarter);
//     the lower column must win;
//   - row 0 of A is all zeros, so every value of that row is 0: column 0 wins.
// Each size runs twice (the keys must be reset by the first call).
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/lm_head_argmax_test.cpp -o lm_head_argmax_test
// Run: tools/gpulock.sh $(tools/gpus.py --free 1) -- ./lm_head_argmax_test
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "../kernels/fast_gemm.hpp"
#include "../kernels/misc.hpp"

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

int main() {
  const int N = 201088, K = 2880, M_MAX = 2048;
  const int sizes[] = {1, 33, 64, 100, 128, 200, 256, 384, 512, 513, 700, 1024, 2048};
  // planted ties: (row of A, lower column, higher column)
  const int ties[3][3] = {{1, 777, 150000}, {2, 3000, 3001}, {3, 64005, 64006}};

  std::vector<uint16_t> A((size_t)M_MAX * K), W((size_t)N * K);
#pragma omp parallel for
  for (size_t i = 0; i < A.size(); i++)
    A[i] = to_bf16(normal(i, 1, 1.0f));
#pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++)
    W[i] = to_bf16(normal(i, 2, 0.05f));
  for (int k = 0; k < K; k++)
    A[k] = 0;
  for (auto &t : ties)
    for (int k = 0; k < K; k++) {
      const uint16_t v = to_bf16(0.1f * normal((size_t)t[0] * K + k, 1, 1.0f));
      W[(size_t)t[1] * K + k] = W[(size_t)t[2] * K + k] = v;
    }

  bf16_t *dA, *dW;
  float *dC;
  int *d_ref, *d_got;
  unsigned long long *d_keys;
  CHECK(hipMalloc(&dA, A.size() * 2));
  CHECK(hipMalloc(&dW, W.size() * 2));
  CHECK(hipMalloc(&dC, (size_t)M_MAX * N * 4));
  CHECK(hipMalloc(&d_ref, M_MAX * 4));
  CHECK(hipMalloc(&d_got, M_MAX * 4));
  CHECK(hipMalloc(&d_keys, M_MAX * 8));
  CHECK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dW, W.data(), W.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_keys, 0, M_MAX * 8));

  int failures = 0;
  std::vector<int> ref(M_MAX), got(M_MAX);
  for (int M : sizes) {
    gemm_bf16(0, dA, M, dW, nullptr, dC, N, K);
    argmax_kernel<<<M, 256>>>(dC, N, d_ref);
    CHECK(hipMemcpy(ref.data(), d_ref, M * 4, hipMemcpyDeviceToHost));
    for (int rep = 0; rep < 2; rep++) {
      CHECK(hipMemset(d_got, 0xff, M * 4));
      gemm_bf16_argmax(0, dA, M, dW, nullptr, N, K, d_keys, d_got);
      CHECK(hipMemcpy(got.data(), d_got, M * 4, hipMemcpyDeviceToHost));
      int bad = 0;
      for (int m = 0; m < M; m++)
        bad += got[m] != ref[m];
      printf("M %4d run %d: %d / %d rows differ", M, rep, bad, M);
      for (auto &t : ties)
        if (t[0] < M)
          printf("  row %d -> %d", t[0], got[t[0]]);
      printf("  row 0 -> %d\n", got[0]);
      failures += bad;
      if (got[0] != 0)
        failures++;
      for (auto &t : ties)
        if (t[0] < M && (ref[t[0]] != t[1] || got[t[0]] != t[1]))
          failures++;
    }
  }
  std::vector<unsigned long long> keys(M_MAX);
  CHECK(hipMemcpy(keys.data(), d_keys, M_MAX * 8, hipMemcpyDeviceToHost));
  for (auto k : keys)
    failures += k != 0;
  printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
