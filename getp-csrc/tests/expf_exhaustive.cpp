// Checks expf_cpu() (kernels/exact_math.hpp) against the host expf for every
// one of the 2^32 float bit patterns.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a \
//     getp-csrc/tests/expf_exhaustive.cpp -o expf_exhaustive
// Run on one GPU: tools/gpulock.sh <hip id> -- ./expf_exhaustive
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "../kernels/exact_math.hpp"

__global__ void expf_all(uint32_t base, uint32_t n, float *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf_cpu(__uint_as_float(base + i));
}

int main() {
  const uint32_t chunk = 1u << 28;
  float *d_out;
  if (hipMalloc(&d_out, chunk * sizeof(float)) != hipSuccess)
    return 1;
  std::vector<float> gpu(chunk);
  unsigned long long mismatches = 0, nan_both = 0;
  for (uint64_t base = 0; base < (1ull << 32); base += chunk) {
    expf_all<<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out);
    if (hipMemcpy(gpu.data(), d_out, chunk * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
      return 1;
    unsigned long long mm = 0, nn = 0;
    long long first_bad = -1;
#pragma omp parallel for reduction(+ : mm, nn)
    for (long long i = 0; i < (long long)chunk; i++) {
      uint32_t bits = (uint32_t)(base + i);
      float x;
      memcpy(&x, &bits, 4);
      float h = expf(x);
      if (isnan(h) && isnan(gpu[i])) {
        nn++;
        continue;
      }
      if (memcmp(&h, &gpu[i], 4) != 0) {
        mm++;
#pragma omp critical
        if (first_bad < 0)
          first_bad = i;
      }
    }
    mismatches += mm;
    nan_both += nn;
    if (first_bad >= 0) {
      uint32_t bits = (uint32_t)(base + first_bad);
      float x;
      memcpy(&x, &bits, 4);
      printf("chunk %llx: %llu mismatches, e.g. x=%a host=%a gpu=%a\n",
             (unsigned long long)base, mm, x, expf(x), gpu[first_bad]);
    }
  }
  printf("%llu mismatches over 2^32 inputs (%llu NaN inputs agree as NaN)\n",
         mismatches, nan_both);
  return mismatches == 0 ? 0 : 1;
}
