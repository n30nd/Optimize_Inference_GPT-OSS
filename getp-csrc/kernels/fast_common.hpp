// Shared pieces of the fast path: bf16 storage helpers.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint16_t bf16_t; // raw bfloat16 bits

__device__ inline float bf16_to_f32(bf16_t h) {
  return __uint_as_float((uint32_t)h << 16);
}

// Round to nearest even; NaN stays NaN.
__device__ inline bf16_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffff) > 0x7f800000)
    return (bf16_t)((u >> 16) | 0x40);
  u += 0x7fff + ((u >> 16) & 1);
  return (bf16_t)(u >> 16);
}
