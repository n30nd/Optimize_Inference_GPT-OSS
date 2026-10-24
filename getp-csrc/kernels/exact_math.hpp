// Device math that rounds exactly like the CPU build of run.cpp, so the first
// GPU kernels can reproduce the CPU forward pass bit for bit.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

// expf with the algorithm glibc 2.39 uses on x86-64 CPUs with FMA (from ARM
// optimized-routines, MIT licence): exp(x) = 2^(k/32) * 2^(r/32) with a table
// for 2^(i/32) and a cubic polynomial, all in double, rounded to float once.
// tests/expf_exhaustive.cpp checks it against the host expf on all 2^32 inputs.
__device__ static const uint64_t kExp2fTable[32] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};

__device__ inline float expf_cpu(float x) {
  const double kInvLn2N = 0x1.71547652b82fep+0 * 32;
  const double kShift = 0x1.8p+52;
  const double kC0 = 0x1.c6af84b912394p-5 / (32.0 * 32.0 * 32.0);
  const double kC1 = 0x1.ebfce50fac4f3p-3 / (32.0 * 32.0);
  const double kC2 = 0x1.62e42ff0c52d6p-1 / 32.0;

  uint32_t ux = __float_as_uint(x);
  uint32_t abstop = (ux >> 20) & 0x7ff;
  if (abstop >= (__float_as_uint(88.0f) >> 20)) {
    if (ux == __float_as_uint(-INFINITY))
      return 0.0f;
    if (abstop >= (__float_as_uint(INFINITY) >> 20))
      return x + x;
    if (x > 0x1.62e42ep6f)
      return INFINITY;
    if (x < -0x1.9fe368p6f)
      return 0.0f;
  }
  double z = __dmul_rn(kInvLn2N, (double)x);
  double kd = __dadd_rn(z, kShift);
  uint64_t ki = (uint64_t)__double_as_longlong(kd);
  kd = __dsub_rn(kd, kShift);
  double r = __dsub_rn(z, kd);
  uint64_t t = kExp2fTable[ki % 32] + (ki << (52 - 5));
  double s = __longlong_as_double((long long)t);
  double p = __fma_rn(kC0, r, kC1);
  double y = __fma_rn(kC2, r, 1.0);
  y = __fma_rn(p, __dmul_rn(r, r), y);
  return (float)__dmul_rn(y, s);
}

// Correctly rounded fp32 square root and division, like the CPU's sqrtss and
// divss: computed in fp64, then rounded once (exact for these operations,
// since fp64 has more than 2 * 24 + 2 bits).
__device__ inline float sqrt_cpu(float x) { return (float)__dsqrt_rn((double)x); }
__device__ inline float div_cpu(float a, float b) {
  return (float)__ddiv_rn((double)a, (double)b);
}
