// Checks GPU fp32 square root and reciprocal against the host (IEEE, correctly
// rounded) on all 2^32 inputs.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a \
//     getp-csrc/tests/fp32_ops_exhaustive.cpp -o fp32_ops_exhaustive
// Run on one GPU: tools/gpulock.sh <hip id> -- ./fp32_ops_exhaustive
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "../kernels/exact_math.hpp"

template <int OP>
__device__ float apply(float x) {
  if (OP == 0) return __fsqrt_rn(x);
  if (OP == 1) return sqrtf(x);
  if (OP == 2) return __fdiv_rn(1.0f, x);
  if (OP == 3) return 1.0f / x;
  if (OP == 4) return sqrt_cpu(x);
  return div_cpu(1.0f, x);
}

static float host_apply(int op, float x) {
  if (op == 0 || op == 1 || op == 4) return sqrtf(x);
  return 1.0f / x;
}

template <int OP>
__global__ void run(uint32_t base, uint32_t n, float *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = apply<OP>(__uint_as_float(base + i));
}

int main() {
  const char *names[] = {"__fsqrt_rn", "sqrtf", "__fdiv_rn(1,x)", "1.0f/x", "sqrt_cpu", "div_cpu(1,x)"};
  const uint32_t chunk = 1u << 28;
  float *d_out;
  if (hipMalloc(&d_out, chunk * sizeof(float)) != hipSuccess)
    return 1;
  std::vector<float> gpu(chunk);
  for (int op = 0; op < 6; op++) {
    unsigned long long mismatches = 0;
    uint32_t example = 0;
    for (uint64_t base = 0; base < (1ull << 32); base += chunk) {
      switch (op) {
      case 0: run<0><<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out); break;
      case 1: run<1><<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out); break;
      case 2: run<2><<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out); break;
      case 3: run<3><<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out); break;
      case 4: run<4><<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out); break;
      case 5: run<5><<<chunk / 256, 256>>>((uint32_t)base, chunk, d_out); break;
      }
      if (hipMemcpy(gpu.data(), d_out, chunk * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
        return 1;
      unsigned long long mm = 0;
#pragma omp parallel for reduction(+ : mm)
      for (long long i = 0; i < (long long)chunk; i++) {
        uint32_t bits = (uint32_t)(base + i);
        float x;
        memcpy(&x, &bits, 4);
        float h = host_apply(op, x);
        if (isnan(h) && isnan(gpu[i]))
          continue;
        if (memcmp(&h, &gpu[i], 4) != 0) {
          mm++;
          example = bits;
        }
      }
      mismatches += mm;
    }
    float x;
    memcpy(&x, &example, 4);
    printf("%-16s %llu mismatches over 2^32 inputs%s", names[op], mismatches,
           mismatches ? "" : "\n");
    if (mismatches)
      printf(" (e.g. x=%a host=%a)\n", x, host_apply(op, x));
  }
  return 0;
}
