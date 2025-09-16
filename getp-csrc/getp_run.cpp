// getp-csrc/getp_run.cpp
// End-to-end "getp" path with multi-GPU: DP x (TP+EP).
// Default targets DP=2 and TP+EP per group=4 (total WORLD_GPUS=8),
// where each DP group owns 4 GPUs and runs the same per-token inputs.
// - No external GPU libs. Naive HIP kernels focused on correctness.
// - Weights:    All stored as BF16 (MLP1 now BF16, previously FP8-like)
// - Activations: FP32 (GPURunState concept)
// - All-reduce: P2P ring (hipMemcpyPeerAsync) for Attention W_o partials
// - Warm-up does sharding + chunked copy (512 MB default)
// - Inference runs one token stream per request (driver in getp_eval.cpp)
// - Router computed on host; MoE dispatch via P2P alltoallv between GPUs

#include "getp_eval.cpp"

#ifndef GETP_RUN
#define GETP_RUN

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <limits>

// Model/data parallel sizing
#ifndef DP_GROUPS
#define DP_GROUPS 1
#endif
#ifndef MP_GROUP_SIZE
#define MP_GROUP_SIZE 4
#endif
#ifndef WORLD_GPUS
#define WORLD_GPUS (DP_GROUPS * MP_GROUP_SIZE)
#endif

// Default micro-batch size per DP group
#ifndef GROUP_BATCH
#define GROUP_BATCH 4
#endif

// Runtime-configurable microbatch size (defaults to GROUP_BATCH)
static int g_group_batch = GROUP_BATCH;

#ifndef CHUNK_BYTES_MB
#define CHUNK_BYTES_MB 512  // set to 1024 for 1GB chunks if you prefer
#endif

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    exit(EXIT_FAILURE);     \
  }                         \
} while(0)

static size_t CHUNK_BYTES = (size_t)CHUNK_BYTES_MB * 1024ull * 1024ull;

// Helpers for DP/MP rank mapping
static inline int group_id_from_rank(int r) { return r / MP_GROUP_SIZE; }
static inline int mp_rank_from_rank(int r) { return r % MP_GROUP_SIZE; }
static inline int group_base_rank(int g) { return g * MP_GROUP_SIZE; }

using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;

__device__ inline hip_bfloat16 float_to_bf16_dev(float x) {
  hip_bfloat16 out;
  uint16_t* p = reinterpret_cast<uint16_t*>(&out);
  *p = (uint16_t)(__float_as_uint(x) >> 16);
  return out;
}


// ---------------------------------------------
// Small helpers: BF16 <-> FP32 and chunked copy
// ---------------------------------------------
// Host-side converters
static inline float bf16_to_fp32_host(hip_bfloat16 v) {
  uint16_t u16 = *reinterpret_cast<uint16_t*>(&v);
  uint32_t u32 = ((uint32_t)u16) << 16;
  float f; memcpy(&f, &u32, sizeof(float)); return f;
}
static inline hip_bfloat16 fp32_to_bf16_host(float f) {
  uint32_t u32; memcpy(&u32, &f, sizeof(uint32_t));
  uint16_t u16 = (uint16_t)(u32 >> 16);
  hip_bfloat16 out; memcpy(&out, &u16, sizeof(uint16_t)); return out;
}

// Device/host dual-use converters
__host__ __device__ inline float bf16_to_fp32_bi(hip_bfloat16 v) {
#if defined(__HIP_DEVICE_COMPILE__)
  // reinterpret as 16-bit then shift
  uint16_t u16 = *(const uint16_t*)&v;
  uint32_t u32 = ((uint32_t)u16) << 16;
  union { uint32_t u; float f; } u; u.u = u32; return u.f;
#else
  return bf16_to_fp32_host(v);
#endif
}

static void memcpyHtoD_chunked(void* dst, const void* src, size_t nbytes) {
  const uint8_t* p = (const uint8_t*)src;
  uint8_t* q = (uint8_t*)dst;
  size_t off = 0;
  while (off < nbytes) {
    size_t nb = std::min(CHUNK_BYTES, nbytes - off);
    HIP_CHECK(hipMemcpy(q + off, p + off, nb, hipMemcpyHostToDevice));
    off += nb;
  }
}
static void memcpyDtoH_chunked(void* dst, const void* src, size_t nbytes) {
  const uint8_t* p = (const uint8_t*)src;
  uint8_t* q = (uint8_t*)dst;
  size_t off = 0;
  while (off < nbytes) {
    size_t nb = std::min(CHUNK_BYTES, nbytes - off);
    HIP_CHECK(hipMemcpy(q + off, p + off, nb, hipMemcpyDeviceToHost));
    off += nb;
  }
}

static void safeHipFree(void* p) {
  if (p) {
    hipError_t e = hipFree(p);
    if (e != hipSuccess) {
      fprintf(stderr, "hipFree error: %s\n", hipGetErrorString(e));
    }
  }
}

static void enable_peer_access_all() {
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  for (int i = 0; i < ndev; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < ndev; ++j) {
      if (i == j) continue;
      int can = 0;
      hipError_t e = hipDeviceCanAccessPeer(&can, i, j);
      if (e == hipSuccess && can) {
        hipError_t e2 = hipDeviceEnablePeerAccess(j, 0);
        if (e2 != hipSuccess && e2 != hipErrorPeerAccessAlreadyEnabled) {
          fprintf(stderr, "hipDeviceEnablePeerAccess(%d->%d) failed: %s\n", i, j, hipGetErrorString(e2));
        }
      }
    }
  }
}

// ---------------------------------------------
// Very small, naive device kernels (correctness)
// ---------------------------------------------

__host__ __device__ inline float silu(float x);

// y[out] = W[out, in] (BF16) @ x[in] (FP32). One block per output row.
__global__ void k_matvec_bf16_fp32(const hip_bfloat16* __restrict__ W,
                                   const float*        __restrict__ x,
                                   float*                    y,
                                   int in_features, int out_features) {
  int row = blockIdx.x;
  if (row >= out_features) return;
  const hip_bfloat16* wrow = W + (size_t)row * in_features;
  float sum = 0.f;
  for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
    float w = bf16_to_fp32_bi(wrow[j]);
    sum += w * x[j];
  }
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) y[row] = sdata[0];
}

// Batched BF16xFP32 matvec: Y[b,out] = W[out,in] @ X[b,in]
__global__ void k_qkv_matvec_bf16_fp32_batched(const hip_bfloat16* __restrict__ W,
                                           const float*        __restrict__ X,
                                           float*              __restrict__ Y,
                                           int in_features, int out_features) {
  int row = blockIdx.x;
  int b   = blockIdx.y;
  if (row >= out_features) return;
  const hip_bfloat16* wrow = W + (size_t)row * in_features;
  const float* x = X + (size_t)b * in_features;
  float sum = 0.f;
  for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
    float w = bf16_to_fp32_bi(wrow[j]);
    sum += w * x[j];
  }
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) Y[(size_t)b * out_features + row] = sdata[0];
}

// Batched BF16xFP32 matvec: Y[b,out] = W[out,in] @ X[b,in]
__global__ __launch_bounds__(64)
void k_mlp1_matvec_bf16_fp32_batched(const hip_bfloat16* __restrict__ W,
                                      const float* __restrict__ X,
                                      float* __restrict__ gate_up,
                                      const float* __restrict__ bias,
                                      int in_features,
                                      int out_features,
                                      int batch,
                                      float clampU,
                                      float clampL,
                                      int Mpad,
                                      int Kpad) {
  using float4v = __attribute__((ext_vector_type(4))) float;

  const int tx = threadIdx.x;  // lane within tile (0..15)
  const int ty = threadIdx.y;  // row group within tile (0..3)
  const int lane = ty * blockDim.x + tx;

  const int tile_m = blockIdx.x;
  const int row0 = tile_m * 16;
  const int col0 = blockIdx.y * 16;
  const int I = out_features >> 1;
  const bool has_bias = (bias != nullptr);
  (void)Mpad;

  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float        sWf[16 * 16];
  __shared__ __align__(16) float        sX [16 * 16];
  __shared__ float sBias[16];

  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int Ktiles = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)((Kpad + 15) / 16);

  if (has_bias) {
    if (lane < 16) {
      const int r = row0 + lane;
      sBias[lane] = (r < out_features) ? bias[r] : 0.f;
    }
    __syncthreads();
  }

  for (int kk = 0; kk < Ktiles; ++kk) {
    if (lane < 32) {
      const size_t tile_base = ((size_t)tile_m * tiles_per_row + (size_t)kk) * (size_t)256;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(W + tile_base)[lane];
    }

    const int col = col0 + tx;
    const int x_base = kk * 16 + ty * 4;
    float xvals[4] = {0.f, 0.f, 0.f, 0.f};
    if (col < batch && x_base < in_features) {
      const float* x_col = X + (size_t)col * (size_t)in_features;
      if (x_base + 3 < in_features) {
        float4v pack = reinterpret_cast<const float4v*>(x_col + x_base)[0];
        xvals[0] = pack.x;
        xvals[1] = pack.y;
        xvals[2] = pack.z;
        xvals[3] = pack.w;
      } else {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
          int idx = x_base + t;
          if (idx < in_features) xvals[t] = x_col[idx];
        }
      }
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k_local = ty * 4 + t;
      sX[k_local * 16 + tx] = xvals[t];
    }

    __syncthreads();

    int idx4 = lane * 4;
    if (idx4 < 256) {
      sWf[idx4 + 0] = bf16_to_fp32_bi(sWb[idx4 + 0]);
      sWf[idx4 + 1] = bf16_to_fp32_bi(sWb[idx4 + 1]);
      sWf[idx4 + 2] = bf16_to_fp32_bi(sWb[idx4 + 2]);
      sWf[idx4 + 3] = bf16_to_fp32_bi(sWb[idx4 + 3]);
    }

    __syncthreads();

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k4 = ty + 4 * t;
      float a = sWf[tx * 16 + k4];
      float b = sX[k4 * 16 + tx];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }

    __syncthreads();
  }

  const int col = col0 + tx;
  if (col < batch) {
    float* out_col = gate_up + (size_t)col * (size_t)I;
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
      const int offset = ty * 4 + iter * 2;
      const int r  = row0 + offset;
      const int rp = r + 1;
      if (r < out_features && rp < out_features) {
        float gate = (iter == 0) ? acc.x : acc.z;
        float up   = (iter == 0) ? acc.y : acc.w;
        if (has_bias) {
          gate += sBias[offset];
          up   += sBias[offset + 1];
        }
        if (gate > clampU) gate = clampU;
        if (up < clampL)   up   = clampL;
        if (up > clampU)   up   = clampU;
        gate = silu(gate);
        const int idx = r >> 1;
        if (idx < I) {
          out_col[idx] = gate * (up + 1.f);
        }
      }
    }
  }
}

__global__ void pack_convert_fp32_to_bf16_tiles16_kernel(const float *src_fp32,
                                                         int row_offset,
                                                         int rows,
                                                         int M,
                                                         int K,
                                                         hip_bfloat16 *dst_packed,
                                                         int Mpad,
                                                         int Kpad) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)rows * (size_t)K;
  if (idx >= total) return;
  int r = (int)(idx / K);
  int c = (int)(idx % K);
  int m = row_offset + r;
  if (m >= M) return;
  int tile_m = m / 16;
  int tile_k = c / 16;
  int im = m % 16;
  int ik = c % 16;
  size_t tiles_per_row = (size_t)((Kpad + 15) / 16);
  size_t dst_index = ((size_t)tile_m * tiles_per_row + (size_t)tile_k) * (size_t)256 + (size_t)im * 16 + (size_t)ik;
  dst_packed[dst_index] = float_to_bf16_dev(src_fp32[idx]);
}

// Batched BF16xFP32 matvec: Y[b,out] = W[out,in] @ X[b,in]
__global__ void k_mlp2_matvec_bf16_fp32_batched(const hip_bfloat16* __restrict__ W,
                                           const float*        __restrict__ X,
                                           float*              __restrict__ Y,
                                           int in_features, int out_features) {
  int row = blockIdx.x;
  int b   = blockIdx.y;
  if (row >= out_features) return;
  const hip_bfloat16* wrow = W + (size_t)row * in_features;
  const float* x = X + (size_t)b * in_features;
  float sum = 0.f;
  for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
    float w = bf16_to_fp32_bi(wrow[j]);
    sum += w * x[j];
  }
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) Y[(size_t)b * out_features + row] = sdata[0];
}

// RMSNorm for a single vector: y = weight * x / sqrt(mean(x^2)+eps)
__global__ void k_rmsnorm_vec(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ y,
                              int size) {
  __shared__ float ssum[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = x[i];
    local += v * v;
  }
  ssum[threadIdx.x] = local; __syncthreads();
  if (blockDim.x >= 256) { if (threadIdx.x < 128) ssum[threadIdx.x] += ssum[threadIdx.x + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (threadIdx.x <  64) ssum[threadIdx.x] += ssum[threadIdx.x +  64]; __syncthreads(); }
  if (threadIdx.x < 32) {
    volatile float* v = ssum;
    v[threadIdx.x] += v[threadIdx.x + 32];
    v[threadIdx.x] += v[threadIdx.x + 16];
    v[threadIdx.x] += v[threadIdx.x +  8];
    v[threadIdx.x] += v[threadIdx.x +  4];
    v[threadIdx.x] += v[threadIdx.x +  2];
    v[threadIdx.x] += v[threadIdx.x +  1];
  }
  __syncthreads();
  float inv = rsqrtf(ssum[0] / (float)size + 1e-5f);
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    y[i] = weight[i] * (x[i] * inv);
  }
}

// Batched RMSNorm over B vectors
__global__ void k_rmsnorm_vec_batched(const float* __restrict__ X,
                                      const float* __restrict__ weight,
                                      float* __restrict__ Y,
                                      int size) {
  int b = blockIdx.y;
  const float* x = X + (size_t)b * size;
  float* y = Y + (size_t)b * size;
  __shared__ float ssum[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) local += x[i]*x[i];
  ssum[threadIdx.x] = local; __syncthreads();
  if (blockDim.x >= 256) { if (threadIdx.x < 128) ssum[threadIdx.x] += ssum[threadIdx.x + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (threadIdx.x <  64) ssum[threadIdx.x] += ssum[threadIdx.x +  64]; __syncthreads(); }
  if (threadIdx.x < 32) {
    volatile float* v = ssum;
    v[threadIdx.x] += v[threadIdx.x + 32];
    v[threadIdx.x] += v[threadIdx.x + 16];
    v[threadIdx.x] += v[threadIdx.x +  8];
    v[threadIdx.x] += v[threadIdx.x +  4];
    v[threadIdx.x] += v[threadIdx.x +  2];
    v[threadIdx.x] += v[threadIdx.x +  1];
  }
  __syncthreads();
  float inv = rsqrtf(ssum[0] / (float)size + 1e-5f);
  for (int i = threadIdx.x; i < size; i += blockDim.x) y[i] = weight[i] * (x[i] * inv);
}

// Generic FP32 matvec: y[out] = W[out,in] @ x[in]
__global__ void k_matvec_fp32(const float* __restrict__ W,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int in_n, int out_n) {
  int row = blockIdx.x;
  if (row >= out_n) return;
  const float* wrow = W + (size_t)row * in_n;
  float sum = 0.f;
  for (int j = threadIdx.x; j < in_n; j += blockDim.x) sum += wrow[j] * x[j];
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) y[row] = sdata[0];
}

// Batched FP32 matvec: Y[b,out] = W[out,in] @ X[b,in]
__global__ void k_matvec_fp32_batched(const float* __restrict__ W,
                                      const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      int in_n, int out_n) {
  int row = blockIdx.x;
  int b   = blockIdx.y;
  if (row >= out_n) return;
  const float* wrow = W + (size_t)row * in_n;
  const float* x = X + (size_t)b * in_n;
  float sum = 0.f;
  for (int j = threadIdx.x; j < in_n; j += blockDim.x) sum += wrow[j] * x[j];
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) Y[(size_t)b * out_n + row] = sdata[0];
}

// In-place softmax for a small vector (naive)
__global__ void k_softmax_inplace(float* x, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float m = x[0];
  for (int i = 1; i < n; ++i) m = fmaxf(m, x[i]);
  float s = 0.f;
  for (int i = 0; i < n; ++i) { x[i] = expf(x[i] - m); s += x[i]; }
  float inv = 1.f / s;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

__global__ void k_add_bias_batched(float* Y, const float* B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < n) Y[(size_t)b * n + i] += B[i];
}

__global__ void k_add_bias(float* y, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i];
}

// In-place add: dst[i] += src[i] for i in [0,n)
__global__ void k_add_inplace(float* dst, const float* src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// Batched in-place add over B rows of length n: X[b,:] += Y[b,:]
__global__ void k_add_inplace_batched(float* X, const float* Y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < n) X[(size_t)b * n + i] += Y[(size_t)b * n + i];
}

// Batched AXPY: X[b,:] += alpha * Y[b,:]
__global__ void k_axpy_batched(float* X, const float* Y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < n) X[(size_t)b * n + i] += alpha * Y[(size_t)b * n + i];
}

// Gather rows: dst[r,:] = src[idx[r],:], r in [0,nb)

__global__ void k_gather_rows_f32(const float* __restrict__ src,
                                  const int*   __restrict__ idx,
                                  int nb, int D,
                                  float* __restrict__ dst) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // col
  int r = blockIdx.y; // row
  if (r < nb && j < D) {
    int i_src = idx[r];
    dst[(size_t)r * D + j] = src[(size_t)i_src * D + j];
  }
}

// Scatter-add with weights: dst[idx[r],:] += w[r] * src[r,:]
__global__ void k_scatter_axpy_rows_f32(float* __restrict__ dst,
                                        const int* __restrict__ idx,
                                        const float* __restrict__ w,
                                        const float* __restrict__ src,
                                        int nb, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // col
  int r = blockIdx.y; // row
  if (r < nb && j < D) {
    int i_dst = idx[r];
    float alpha = w[r];
    atomicAdd(&dst[(size_t)i_dst * D + j], alpha * src[(size_t)r * D + j]);
  }
}

// SwiGLU piece
__host__ __device__ inline float silu(float x) { return x / (1.0f + expf(-1.702f * x)); }

// Rotary on Q or K for local heads
__global__ void k_apply_rotary(float* vec,       // shape: n_heads_local * head_dim
                               const float* cos_vals, // head_dim/2 (same for all heads)
                               const float* sin_vals,
                               int n_heads_local, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x; // 0..head_dim/2-1
  if (h >= n_heads_local || i >= head_dim/2) return;
  int half = head_dim/2;
  float* base = vec + h * head_dim;
  float x1 = base[i];
  float x2 = base[half + i];
  float c = cos_vals[i];
  float s = sin_vals[i];
  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;
  base[i] = o1;
  base[half + i] = o2;
}

// Batched rotary with per-sample positions; computes cos/sin on the fly
__global__ void k_apply_rotary_dynamic_batched_strided(
    float* vec_b,
    const int* __restrict__ pos_list,
    float base,
    int head_dim,
    float scaling_factor,
    float initial_context_length,
    float ntk_beta,
    float ntk_alpha,
    int n_heads_local,
    int sample_stride) {
  int h = blockIdx.x;
  int b = blockIdx.y;
  int i = threadIdx.x;
  if (h >= n_heads_local || i >= head_dim/2) return;
  float* vec_base = vec_b + (size_t)b * sample_stride + h * head_dim;
  int half = head_dim/2;
  float x1 = vec_base[i];
  float x2 = vec_base[half + i];
  // Compute cos/sin for this sample position and dimension i
  int pos_b = pos_list[b];
  const float PI = 3.14159265358979323846f;
  float exponent = (2.0f * (float)i) / (float)head_dim;
  float freq = powf(base, exponent);
  float concentration = (scaling_factor > 1.0f) ? (0.1f * logf(scaling_factor) + 1.0f) : 1.0f;
  float inv_freq;
  if (scaling_factor > 1.0f) {
    float low = (float)half * logf(initial_context_length / (ntk_beta * 2.0f * PI)) / logf(base);
    float high = (float)half * logf(initial_context_length / (ntk_alpha * 2.0f * PI)) / logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    if (ramp < 0.f) ramp = 0.f; if (ramp > 1.f) ramp = 1.f;
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }
  float val = (float)pos_b * inv_freq;
  float c = cosf(val) * concentration;
  float s = sinf(val) * concentration;
  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;
  vec_base[i] = o1;
  vec_base[half + i] = o2;
}

// Compute RoPE cos/sin on device for a given position and config
__global__ void k_compute_cos_sin(int pos,
                                  float base,
                                  int head_dim,
                                  float scaling_factor,
                                  float initial_context_length,
                                  float ntk_beta,
                                  float ntk_alpha,
                                  float* __restrict__ cos_out,
                                  float* __restrict__ sin_out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim / 2;
  if (j >= d_half) return;
  const float PI = 3.14159265358979323846f;

  float exponent = (2.0f * (float)j) / (float)head_dim;
  float freq = powf(base, exponent);
  float concentration = (scaling_factor > 1.0f) ? (0.1f * logf(scaling_factor) + 1.0f) : 1.0f;

  float inv_freq;
  if (scaling_factor > 1.0f) {
    float low = (float)d_half * logf(initial_context_length / (ntk_beta * 2.0f * PI)) / logf(base);
    float high = (float)d_half * logf(initial_context_length / (ntk_alpha * 2.0f * PI)) / logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)j - low) / (high - low);
    if (ramp < 0.f) ramp = 0.f; if (ramp > 1.f) ramp = 1.f;
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }

  float val = (float)pos * inv_freq;
  cos_out[j] = cosf(val) * concentration;
  sin_out[j] = sinf(val) * concentration;
}

// Attention for local heads: computes tb_local
__global__ void k_attention_step(
    const float* __restrict__ q_local,
    const float* __restrict__ kv_cache_k,
    const float* __restrict__ kv_cache_v,
    float*       __restrict__ tb_local,
    int pos_plus1, int head_dim,
    int n_qh_local, int n_kvh_local, int kv_mul,
    const float* __restrict__ mask_row_or_null, // (pos+1) or null
    const float* __restrict__ attn_sinks_l      // (n_qh_local) or null
) {
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (h >= n_qh_local || d >= head_dim) return;

  extern __shared__ float smem[];
  float* logits = smem; // length = pos_plus1 + 1  (the +1 is for sink)
  const float* q = q_local + h * head_dim;

  float inv_sqrt_d = rsqrtf((float)head_dim);

  // real tokens: t = 0..pos
  for (int t = 0; t < pos_plus1; ++t) {
    __syncthreads();
    if (d == 0) {
      const float* k = kv_cache_k + t*(n_kvh_local*head_dim) + (h/kv_mul)*head_dim;
      float score = 0.f;
      for (int i = 0; i < head_dim; ++i) score += q[i]*k[i];
      score *= inv_sqrt_d;
      if (mask_row_or_null) score += mask_row_or_null[t];
      logits[t] = score;
    }
    __syncthreads();
  }

  // extra sink logit at index pos_plus1
  if (d == 0) {
    float sink = (attn_sinks_l) ? attn_sinks_l[h] : 0.f;
    logits[pos_plus1] = sink;
  }
  __syncthreads();

  // softmax over len = pos_plus1 + 1
  int len = pos_plus1 + 1;
  if (d == 0) {
    float m = logits[0];
    for (int t = 1; t < len; ++t) m = fmaxf(m, logits[t]);
    float s = 0.f;
    for (int t = 0; t < len; ++t) { logits[t] = expf(logits[t] - m); s += logits[t]; }
    for (int t = 0; t < len; ++t) logits[t] /= s;
  }
  __syncthreads();

  // weighted sum ONLY over real tokens (0..pos), ignore sink at index pos_plus1
  float acc = 0.f;
  for (int t = 0; t < pos_plus1; ++t) {
    const float* v = kv_cache_v + t*(n_kvh_local*head_dim) + (h/kv_mul)*head_dim;
    float a = logits[t];
    acc += a * v[d];
  }
  tb_local[h*head_dim + d] = acc;
}

__global__ void k_attention_step_batched(
    const float* __restrict__ q_local_b,   // [B, n_qh_local*hd]
    const float* __restrict__ kv_cache_k_b,// [B, seq_len, n_kvh_local*hd]
    const float* __restrict__ kv_cache_v_b,
    float*       __restrict__ tb_local_b,  // [B, n_qh_local*hd]
    const int*   __restrict__ pos_list,
    int head_dim,
    int n_qh_local, int n_kvh_local, int kv_mul,
    int kv_block_stride,                   // seq_len * (n_kvh_local*hd)
    const float* __restrict__ mask_base_or_null, // [seq_len, seq_len] or null
    int seq_len,
    const float* __restrict__ attn_sinks_l // [n_qh_local] or null
) {
  int h  = blockIdx.x;
  int b  = blockIdx.y;
  int d  = threadIdx.x;
  if (h >= n_qh_local || d >= head_dim) return;

  int pos_b = pos_list[b];
  if (pos_b < 0) return;

  const float* q = q_local_b + (size_t)b * (size_t)n_qh_local * head_dim + (size_t)h * head_dim;
  const float* k_base = kv_cache_k_b + (size_t)b * (size_t)kv_block_stride;
  const float* v_base = kv_cache_v_b + (size_t)b * (size_t)kv_block_stride;
  const float* mask_row = (mask_base_or_null) ? (mask_base_or_null + (size_t)pos_b * seq_len) : nullptr;

  const int kvdim = n_kvh_local * head_dim;
  const int kvh_off = (h / kv_mul) * head_dim;
  const float inv_sqrt_d = rsqrtf((float)head_dim);

  extern __shared__ float smem[]; // size >= (max_pos+2) + head_dim
  float* logits = smem;           // [0 .. pos_b+1]
  float* redbuf = logits + (pos_b + 2); // [0 .. head_dim-1]

  // 1) Tính logits[t] = softmax(q·k_t / sqrt(d)) với dot song song theo d
  for (int t = 0; t <= pos_b; ++t) {
    const float* k = k_base + (size_t)t * kvdim + kvh_off;
    float prod = q[d] * k[d];
    redbuf[d] = prod;
    __syncthreads();

    if (blockDim.x >= 256) { if (d < 128) redbuf[d] += redbuf[d + 128]; __syncthreads(); }
    if (blockDim.x >= 128) { if (d <  64) redbuf[d] += redbuf[d +  64]; __syncthreads(); }
    if (d < 32) {
      volatile float* v = redbuf;
      v[d] += v[d + 32]; v[d] += v[d + 16]; v[d] += v[d + 8];
      v[d] += v[d +  4]; v[d] += v[d +  2]; v[d] += v[d + 1];
    }
    if (d == 0) {
      float score = redbuf[0] * inv_sqrt_d;
      if (mask_row) score += mask_row[t];
      logits[t] = score;
    }
    __syncthreads();
  }
  if (d == 0) {
    logits[pos_b + 1] = attn_sinks_l ? attn_sinks_l[h] : 0.f; // sink
    int len = pos_b + 2;
    float m = logits[0];
    for (int i = 1; i < len; ++i) m = fmaxf(m, logits[i]);
    float s = 0.f;
    for (int i = 0; i < len; ++i) { logits[i] = expf(logits[i] - m); s += logits[i]; }
    float inv = 1.f / s;
    for (int i = 0; i < len; ++i) logits[i] *= inv;
  }
  __syncthreads();

  // 2) Tính tb = Σ_t a_t * v_t (mỗi thread d tích lũy thành phần d)
  float acc = 0.f;
  for (int t = 0; t <= pos_b; ++t) {
    const float* v = v_base + (size_t)t * kvdim + kvh_off;
    acc += logits[t] * v[d];
  }
  tb_local_b[(size_t)b * (size_t)n_qh_local * head_dim + (size_t)h * head_dim + d] = acc;
}

// Batched attention with slot id indirection for KV cache addressing
__global__ void k_attention_step_batched_sid(
    const float* __restrict__ q_local_b,   // [B, n_qh_local*hd]
    const float* __restrict__ kv_cache_k_b,// [GROUP_BATCH, seq_len, n_kvh_local*hd]
    const float* __restrict__ kv_cache_v_b,
    float*       __restrict__ tb_local_b,  // [B, n_qh_local*hd]
    const int*    __restrict__ pos_list,
    const int*    __restrict__ slot_ids,   // [B]
    int head_dim,
    int n_qh_local, int n_kvh_local, int kv_mul,
    int kv_block_stride, // stride between samples in kv cache (seq_len * kvdim)
    const float* __restrict__ mask_base_or_null, // [seq_len, seq_len]
    int seq_len,
    const float* __restrict__ attn_sinks_l) {
  int h = blockIdx.x;
  int b = blockIdx.y;
  int d = threadIdx.x;
  if (h >= n_qh_local || d >= head_dim) return;

  // Shared memory layout: [logits(0..pos_b+1)] [redbuf(0..head_dim-1)]
  extern __shared__ float smem[];
  const float* q = q_local_b + (size_t)b * n_qh_local * head_dim + h * head_dim;
  const float inv_sqrt_d = rsqrtf((float)head_dim);
  const int kvdim = n_kvh_local * head_dim;
  const int kvh_off = (h / kv_mul) * head_dim;

  const int sid = slot_ids[b];
  const float* k_base = kv_cache_k_b + (size_t)sid * kv_block_stride;
  const float* v_base = kv_cache_v_b + (size_t)sid * kv_block_stride;

  const int pos_b = pos_list[b];
  const int pos_plus1 = pos_b + 1;
  float* logits = smem;                // length = pos_plus1 + 1
  float* redbuf = logits + (pos_plus1 + 1); // reduction buffer over head_dim

  const float* mask_row_or_null = (mask_base_or_null)
      ? (mask_base_or_null + (size_t)pos_b * seq_len)
      : nullptr;

  // 1) Parallel dot for logits over t=0..pos_b
  for (int t = 0; t < pos_plus1; ++t) {
    const float* k = k_base + (size_t)t * kvdim + kvh_off;
    float prod = q[d] * k[d];
    redbuf[d] = prod;
    __syncthreads();

    if (blockDim.x >= 256) { if (d < 128) redbuf[d] += redbuf[d + 128]; __syncthreads(); }
    if (blockDim.x >= 128) { if (d <  64) redbuf[d] += redbuf[d +  64]; __syncthreads(); }
    if (d < 32) {
      volatile float* v = redbuf;
      v[d] += v[d + 32]; v[d] += v[d + 16]; v[d] += v[d + 8];
      v[d] += v[d +  4]; v[d] += v[d +  2]; v[d] += v[d + 1];
    }
    if (d == 0) {
      float score = redbuf[0] * inv_sqrt_d;
      if (mask_row_or_null) score += mask_row_or_null[t];
      logits[t] = score;
    }
    __syncthreads();
  }

  // 2) Append sink logit and softmax over len=pos_plus1+1
  if (d == 0) {
    logits[pos_plus1] = attn_sinks_l ? attn_sinks_l[h] : 0.f;
    int len = pos_plus1 + 1;
    float m = logits[0];
    for (int i = 1; i < len; ++i) m = fmaxf(m, logits[i]);
    float s = 0.f;
    for (int i = 0; i < len; ++i) { logits[i] = expf(logits[i] - m); s += logits[i]; }
    float inv = 1.f / s;
    for (int i = 0; i < len; ++i) logits[i] *= inv;
  }
  __syncthreads();

  // 3) Weighted sum over real tokens (exclude sink)
  float acc = 0.f;
  for (int t = 0; t < pos_plus1; ++t) {
    const float* v = v_base + (size_t)t * kvdim + kvh_off;
    acc += logits[t] * v[d];
  }
  tb_local_b[(size_t)b * n_qh_local * head_dim + h * head_dim + d] = acc;
}


// y_partial[hidden_dim] = W_o_local(hidden_dim, cols_local BF16) @ tb_local(cols_local)
__global__ void k_matvec_colslice_bf16(const hip_bfloat16* __restrict__ Wcols,
                                       const float*        __restrict__ xcols,
                                       float*                    y,
                                       int hidden_dim, int cols_local) {
  int row = blockIdx.x; // one block per row
  if (row >= hidden_dim) return;
  const hip_bfloat16* wrow = Wcols + (size_t)row * cols_local;
  float sum = 0.f;
  for (int j = threadIdx.x; j < cols_local; j += blockDim.x) {
    float w = bf16_to_fp32_bi(wrow[j]);
    sum += w * xcols[j];
  }
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) y[row] = sdata[0];
}

// Batched colslice matvec for W_o partials
__global__ void k_matvec_colslice_bf16_batched(const hip_bfloat16* __restrict__ Wcols,
                                               const float*        __restrict__ Xcols,
                                               float*              __restrict__ Y,
                                               int hidden_dim, int cols_local) {
  int row = blockIdx.x; // hidden dim
  int b   = blockIdx.y;
  if (row >= hidden_dim) return;
  const hip_bfloat16* wrow = Wcols + (size_t)row * cols_local;
  const float* xcols = Xcols + (size_t)b * cols_local;
  float sum = 0.f;
  for (int j = threadIdx.x; j < cols_local; j += blockDim.x) {
    float w = bf16_to_fp32_bi(wrow[j]);
    sum += w * xcols[j];
  }
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  sdata[tid] = sum; __syncthreads();
  if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  if (tid == 0) Y[(size_t)b * hidden_dim + row] = sdata[0];
}

// Zero out per-sample rows if inactive (pos_list[b] < 0)
__global__ void k_zero_inactive_rows(float* buf, const int* __restrict__ pos_list,
                                     int row_size) {
  int b = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_list[b] < 0 && i < row_size) {
    buf[(size_t)b * row_size + i] = 0.0f;
  }
}

// FP8-like (uint8 + per-row scale) matvec: out_rows x in_cols
// Legacy path retained for future use; not used in current flow.
#ifndef KEEP_FP8_UTILS
#define KEEP_FP8_UTILS 1
#endif
#if KEEP_FP8_UTILS
__global__ void k_matvec_fp8_rowscale(const uint8_t* __restrict__ W_u8,
                                      const float*   __restrict__ scale_rows,
                                      const float*   __restrict__ x,
                                      float*               y,
                                      int in_cols, int out_rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= out_rows) return;
  const uint8_t* row = W_u8 + (size_t)r * in_cols;
  float s = scale_rows[r];
  float acc = 0.f;
  for (int j = 0; j < in_cols; ++j) {
    float w = (float)((int)row[j] - 128) * s;
    acc += w * x[j];
  }
  y[r] = acc;
}
#endif

// ---------------------------------------------
// Sharding helpers
// ---------------------------------------------
static inline void split_even(int total, int rank, int world, int& start, int& count) {
  int base = total / world;
  int rem  = total % world;
  count = base + (rank < rem ? 1 : 0);
  start = rank * base + std::min(rank, rem);
}

struct HeadShard {
  int q_start, q_count;
  int kv_start, kv_count;
};
static HeadShard compute_head_shard(const Config* p, int local_rank, int world_in_group) {
  HeadShard hs{};
  split_even(p->n_attn_heads, local_rank, world_in_group, hs.q_start, hs.q_count);
  split_even(p->n_kv_heads,  local_rank, world_in_group, hs.kv_start, hs.kv_count);
  return hs;
}

struct ExpertShard { int e_start, e_count; };
static ExpertShard compute_expert_shard(const Config* p, int local_rank, int world_in_group) {
  ExpertShard es{};
  split_even(p->n_experts, local_rank, world_in_group, es.e_start, es.e_count);
  return es;
}

// ---------------------------------------------
// Device-side containers per GPU
// ---------------------------------------------
struct GPUTransformerWeights {
  hip_bfloat16* d_w_qkv_rows = nullptr;
  float*        d_b_qkv_rows = nullptr;
  size_t w_qkv_rows_stride = 0;
  int rows_per_layer_local = 0;

  hip_bfloat16* d_w_o_cols = nullptr;
  int cols_per_layer_local = 0;

  hip_bfloat16* d_w_mlp1_packed = nullptr; // MLP1 packed as 16x16 BF16 tiles
  int mlp1_Mpad = 0;
  int mlp1_Kpad = 0;
#if KEEP_FP8_UTILS
  // Legacy FP8 packing kept for future experiments
  uint8_t* d_w_mlp1_u8 = nullptr;
  float*   d_w_mlp1_scale = nullptr;
#endif
  float*   d_b_mlp1 = nullptr;

  hip_bfloat16* d_w_mlp2 = nullptr;
  float*        d_b_mlp2 = nullptr;

  // Router (FP32) per layer
  float* d_w_router = nullptr; // size L * E * D (row-major: E x D per layer)
  float* d_b_router = nullptr; // size L * E

  // Final classifier (FP32) stored on GPU0 only
  float* d_w_out = nullptr; // size vocab_size * D (row-major)

  // RMSNorm weights
  float* d_rms_attn_w = nullptr; // size L*D
  float* d_rms_ffn_w  = nullptr; // size L*D
  float* d_rms_out_w  = nullptr; // size D

  float* d_attn_sinks = nullptr; // size = n_layers * q_count

  // W_o bias per layer on device (FP32)
  float* d_b_o = nullptr; // size L * D
};

struct GPURunState {
  float* d_kv_k = nullptr;
  float* d_kv_v = nullptr;

  float* d_x = nullptr;
  float* d_t = nullptr;
  float* d_qkv = nullptr;
  float* d_q_local = nullptr;
  float* d_tb_local = nullptr;
  float* d_tb2_partial = nullptr;

  float* d_gate_up  = nullptr;
  float* d_ffn_out  = nullptr;

  float* d_cos = nullptr;   // size head_dim/2
  float* d_sin = nullptr;   // size head_dim/2
  float* d_mask = nullptr;  // size seq_len * seq_len  (nullable)

  // Router temporaries
  float* d_router = nullptr; // size E
  float* d_topk   = nullptr; // size K (experts_per_token)

  // Per-batch dynamic positions
  int* d_pos_list = nullptr; // size GROUP_BATCH
  // Per-batch slot ids (maps compact batch index -> persistent slot id 0..GROUP_BATCH-1)
  int* d_slot_ids = nullptr; // size GROUP_BATCH

  // Logits (GPU0 only)
  float* d_logits = nullptr; // size vocab_size
  int* d_next_token = nullptr; // single int result for greedy argmax

  // P2P communication helpers
  float* d_comm_tmp = nullptr;   // size GROUP_BATCH * D (receive scratch)
  float* d_moe_accum = nullptr;  // size GROUP_BATCH * D (accumulator on root)
  hipStream_t comm_stream = nullptr;

  // Batched MoE staging
  float* d_in_batch = nullptr;    // size GROUP_BATCH * D
  int*   d_idx_tmp = nullptr;     // size GROUP_BATCH
  float* d_weight_tmp = nullptr;  // size GROUP_BATCH
};

static GPUTransformerWeights g_weights[WORLD_GPUS];
static GPURunState g_state[WORLD_GPUS];
static HeadShard  g_head_shard[WORLD_GPUS];
static ExpertShard g_expert_shard[WORLD_GPUS];
static int g_world = WORLD_GPUS;
static std::mutex g_reduce_mutex;

static float* h_tb2_partials[WORLD_GPUS] = {};
static float* h_tb2_sum = nullptr; // legacy; not used in group-forward path

// Simple ring-allreduce using peer copies and in-place accumulation.
// Operates over g_state[r].d_tb2_partial for r in [base, base+group_size).
// Root‑only reduction: accumulate all shard partials into root's d_tb2_partial
static void root_reduce_tb2(int base, int group_size, int B, int D, int root) {
  size_t elems = (size_t)B * (size_t)D;
  size_t bytes = elems * sizeof(float);
  // Ensure producers finished computing tb2_partial on default streams
  for (int off = 0; off < group_size; ++off) {
    int r = base + off;
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipDeviceSynchronize());
  }
  // Accumulate on root
  HIP_CHECK(hipSetDevice(root));
  int blocks = (int)((elems + 255) / 256);
  for (int off = 0; off < group_size; ++off) {
    int r = base + off;
    if (r == root) continue; // root already has its partial
    HIP_CHECK(hipMemcpyPeerAsync(g_state[root].d_comm_tmp, root,
                                 g_state[r].d_tb2_partial, r,
                                 bytes, g_state[root].comm_stream));
    k_add_inplace<<<blocks, 256, 0, g_state[root].comm_stream>>>(
        g_state[root].d_tb2_partial, g_state[root].d_comm_tmp, elems);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(g_state[root].comm_stream));
  }
}

// ---------------------------------------------
// Packing helpers
// ---------------------------------------------
static void pack_layer_w_qkv_local_bf16(const TransformerWeights* w, const Config* p,
                                        int layer, const HeadShard& hs,
                                        hip_bfloat16* dst_rows,
                                        float*        dst_bias) {
  int D = p->hidden_dim;
  int hd = p->head_dim;
  int q_rows_total = p->n_attn_heads * hd;
  int k_rows_total = p->n_kv_heads * hd;
  int v_rows_total = p->n_kv_heads * hd;

  const float* W = w->w_qkv + 1ll*layer*D*(q_rows_total + k_rows_total + v_rows_total);
  const float* B = w->b_qkv + 1ll*layer*(q_rows_total + k_rows_total + v_rows_total);

  int row_off = 0;
  // Q rows
  for (int h = 0; h < hs.q_count*hd; ++h) {
    int src_row = hs.q_start*hd + h;
    const float* src = W + 1ll*src_row*D;
    hip_bfloat16* dst = dst_rows + 1ll*(row_off)*D;
    for (int j = 0; j < D; ++j) dst[j] = fp32_to_bf16_host(src[j]);
    dst_bias[row_off] = B[src_row];
    row_off++;
  }
  // K rows
  for (int h = 0; h < hs.kv_count*hd; ++h) {
    int src_row = q_rows_total + (hs.kv_start*hd + h);
    const float* src = W + 1ll*src_row*D;
    hip_bfloat16* dst = dst_rows + 1ll*(row_off)*D;
    for (int j = 0; j < D; ++j) dst[j] = fp32_to_bf16_host(src[j]);
    dst_bias[row_off] = B[src_row];
    row_off++;
  }
  // V rows
  for (int h = 0; h < hs.kv_count*hd; ++h) {
    int src_row = q_rows_total + k_rows_total + (hs.kv_start*hd + h);
    const float* src = W + 1ll*src_row*D;
    hip_bfloat16* dst = dst_rows + 1ll*(row_off)*D;
    for (int j = 0; j < D; ++j) dst[j] = fp32_to_bf16_host(src[j]);
    dst_bias[row_off] = B[src_row];
    row_off++;
  }
}

static void pack_layer_w_o_colslice_bf16(const TransformerWeights* w, const Config* p,
                                         int layer, const HeadShard& hs,
                                         hip_bfloat16* dst_cols) {
  int D = p->hidden_dim;
  int hd = p->head_dim;
  int cols_total = p->n_attn_heads * hd;

  const float* W = w->w_o + 1ll*layer*D*cols_total; // row-major (out,in)
  for (int row = 0; row < D; ++row) {
    hip_bfloat16* dst_row = dst_cols + 1ll*row*(hs.q_count*hd);
    int col_off = 0;
    for (int h = 0; h < hs.q_count; ++h) {
      int h_global = hs.q_start + h;
      int c0 = h_global * hd;
      const float* src_block = W + 1ll*row*cols_total + c0;
      for (int c = 0; c < hd; ++c) {
        dst_row[col_off++] = fp32_to_bf16_host(src_block[c]);
      }
    }
  }
}

// Legacy: keep rowwise FP8 pack util for future use
#if KEEP_FP8_UTILS
static void pack_rowwise_fp8_u8(const float* src, uint8_t* dst_u8, float* scale_rows,
                                int out_rows, int in_cols) {
  for (int r = 0; r < out_rows; ++r) {
    const float* row = src + 1ll*r*in_cols;
    float amax = 0.f;
    for (int j = 0; j < in_cols; ++j) amax = std::max(amax, fabsf(row[j]));
    float s = (amax > 0.f) ? (amax / 127.f) : 1.f;
    scale_rows[r] = s;
    for (int j = 0; j < in_cols; ++j) {
      int v = (int)lrintf(row[j] / s);
      v = std::max(-127, std::min(127, v));
      dst_u8[1ll*r*in_cols + j] = (uint8_t)(v + 128);
    }
  }
}
#endif

// ---------------------------------------------
// Memory utility functions
// ---------------------------------------------
static void get_gpu_memory_info(int gpu_id, size_t* free_mem, size_t* total_mem) {
  HIP_CHECK(hipSetDevice(gpu_id));
  HIP_CHECK(hipMemGetInfo(free_mem, total_mem));
}

static void print_memory_usage(int gpu_id, const char* stage) {
  size_t free_mem, total_mem;
  get_gpu_memory_info(gpu_id, &free_mem, &total_mem);
  size_t used_mem = total_mem - free_mem;
  printf("[GPU%d] %s - Memory: %.2f/%.2f GB used (%.1f%%)\n", 
         gpu_id, stage, 
         used_mem / (1024.0 * 1024.0 * 1024.0),
         total_mem / (1024.0 * 1024.0 * 1024.0),
         (used_mem * 100.0) / total_mem);
  fflush(stdout);
}

// ---------------------------------------------
// Greedy argmax kernel (one block) over vocab
// Assumes logits length = vocab_size, writes index of max to out_idx
// ---------------------------------------------
__global__ void k_argmax(const float* __restrict__ logits, int vocab, int* __restrict__ out_idx) {
  extern __shared__ unsigned char karg_smem[];
  float* svals = reinterpret_cast<float*>(karg_smem);
  int* sidx = reinterpret_cast<int*>(svals + blockDim.x);
  int tid = threadIdx.x;
  float best_v = -INFINITY;
  int best_i = 0;
  for (int i = tid; i < vocab; i += blockDim.x) {
    float v = logits[i];
    if (v > best_v) { best_v = v; best_i = i; }
  }
  svals[tid] = best_v;
  sidx[tid] = best_i;
  __syncthreads();
  for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      float v_other = svals[tid + offset];
      if (v_other > svals[tid] || (v_other == svals[tid] && sidx[tid + offset] < sidx[tid])) {
        svals[tid] = v_other;
        sidx[tid] = sidx[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) *out_idx = sidx[0];
}

// Batched argmax: one block per sample (assumes blockDim.x>=1024 for V)
__global__ void k_argmax_batched(const float* __restrict__ logits, int batch, int vocab, int* __restrict__ out_idx) {
  int b = blockIdx.x;
  if (b >= batch) return;
  extern __shared__ unsigned char karg_smem[];
  float* svals = reinterpret_cast<float*>(karg_smem);
  int* sidx = reinterpret_cast<int*>(svals + blockDim.x);
  int tid = threadIdx.x;
  float best_v = -INFINITY;
  int best_i = 0;
  const float* l = logits + (size_t)b * vocab;
  for (int i = tid; i < vocab; i += blockDim.x) {
    float v = l[i];
    if (v > best_v) { best_v = v; best_i = i; }
  }
  svals[tid] = best_v;
  sidx[tid] = best_i;
  __syncthreads();
  for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      float v_other = svals[tid + offset];
      if (v_other > svals[tid] || (v_other == svals[tid] && sidx[tid + offset] < sidx[tid])) {
        svals[tid] = v_other;
        sidx[tid] = sidx[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) out_idx[b] = sidx[0];
}


// Forward declarations
void warm_up(Transformer *transformer, Tokenizer *tokenizer);
void finish(Transformer *transformer, Tokenizer *tokenizer);
static void gpu_copy_weights(Transformer* transformer, int r);
static void gpu_malloc_run_state(Transformer* transformer, int r);


// ---------------------------------------------
// One-token forward for a DP group with batch size B (lock-step)
// ---------------------------------------------
// Compact-batch forward: same as above but KV-cache addressing uses slot_ids
static void forward_one_group_batched_compact(Transformer* tr, int group_id,
                                              int B,
                                              const int* tokens,
                                              const int* pos_list,
                                              const int* slot_ids, // [B] compact index -> persistent slot id
                                              std::vector<float>& xB,
                                              std::vector<float>& tB,
                                              std::vector<int>& next_tokens_out) {
  Config* p = &tr->config;
  const int L  = p->n_layers;
  const int D  = p->hidden_dim;
  const int hd = p->head_dim;
  const int kv_mul = p->n_attn_heads / p->n_kv_heads;
  const int I  = p->intermediate_dim;
  const int E  = p->n_experts;

  const int base = group_base_rank(group_id);
  const int root = base; // mp_rank==0 device in group

  // Prepare embeddings on host xB (B*D)
  for (int b = 0; b < B; ++b) {
    const float* emb_row = tr->weights.token_embedding_table + (size_t)tokens[b] * D;
    std::copy(emb_row, emb_row + D, xB.begin() + (size_t)b * D);
  }

  // Layers
  for (int l = 0; l < L; ++l) {
    // 1) RMSNorm tB = rms(xB)
    for (int r = base; r < base + MP_GROUP_SIZE; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& devw = g_weights[r];
      auto& st   = g_state[r];
      HIP_CHECK(hipMemcpy(st.d_x, xB.data(), (size_t)B*D*sizeof(float), hipMemcpyHostToDevice));
      const float* w_attn = devw.d_rms_attn_w + (size_t)l * D;
      dim3 grid(1, B), block(256);
      k_rmsnorm_vec_batched<<<grid, block>>>(st.d_x, w_attn, st.d_t, D);
      HIP_CHECK(hipGetLastError());
    }

    // 2) QKV matvec + bias per shard (batched)
    for (int r = base; r < base + MP_GROUP_SIZE; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& devw = g_weights[r];
      auto& st   = g_state[r];
      int rows = devw.rows_per_layer_local;
      const hip_bfloat16* W = devw.d_w_qkv_rows + (size_t)l * devw.w_qkv_rows_stride;
      dim3 grid(rows, B), block(256);
      k_qkv_matvec_bf16_fp32_batched<<<grid, block>>>(W, st.d_t, st.d_qkv, D, rows);
      HIP_CHECK(hipGetLastError());
      float* b = devw.d_b_qkv_rows + (size_t)l * rows;
      dim3 gridb((rows+255)/256, B);
      k_add_bias_batched<<<gridb, 256>>>(st.d_qkv, b, rows);
      HIP_CHECK(hipGetLastError());
    }

    // 3) RoPE (dynamic pos) and KV cache update, extract local Q (batched)
    for (int r = base; r < base + MP_GROUP_SIZE; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& hs = g_head_shard[r];
      auto& st = g_state[r];
      // Upload positions and slot ids for this batch
      HIP_CHECK(hipMemcpy(st.d_pos_list, pos_list, (size_t)B * sizeof(int), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(st.d_slot_ids, slot_ids, (size_t)B * sizeof(int), hipMemcpyHostToDevice));
      int q_rows = hs.q_count * hd;
      int kv_rows = hs.kv_count * hd;

      // Apply RoPE to Q and K with dynamic per-sample positions
      float* q_ptr_b = st.d_qkv; // samples separated by sample_stride = (q_rows + 2*kv_rows)
      float* k_ptr_b = st.d_qkv + q_rows;
      int sample_stride = q_rows + 2*kv_rows;
      dim3 grid_q(hs.q_count, B), block_q(hd/2);
      k_apply_rotary_dynamic_batched_strided<<<grid_q, block_q>>>(
          q_ptr_b, st.d_pos_list, p->rope_theta, p->head_dim,
          p->rope_scaling_factor, p->initial_context_length,
          32.0f, 1.0f,
          hs.q_count, sample_stride);
      HIP_CHECK(hipGetLastError());
      dim3 grid_k(hs.kv_count, B), block_k(hd/2);
      k_apply_rotary_dynamic_batched_strided<<<grid_k, block_k>>>(
          k_ptr_b, st.d_pos_list, p->rope_theta, p->head_dim,
          p->rope_scaling_factor, p->initial_context_length,
          32.0f, 1.0f,
          hs.kv_count, sample_stride);
      HIP_CHECK(hipGetLastError());

      // Update K/V cache for each active sample (use slot_ids to select cache row)
      int kv_dim_local = hs.kv_count * hd;
      size_t kv_block = (size_t)p->seq_len * kv_dim_local;
      float* k_cache_layer_base = st.d_kv_k + (size_t)l * g_group_batch * kv_block;
      float* v_cache_layer_base = st.d_kv_v + (size_t)l * g_group_batch * kv_block;
      for (int b = 0; b < B; ++b) {
        int pos_b = pos_list[b];
        int sid = slot_ids[b];
        float* k_cache_b = k_cache_layer_base + (size_t)sid * kv_block;
        float* v_cache_b = v_cache_layer_base + (size_t)sid * kv_block;
        const float* k_ptr = st.d_qkv + (size_t)b * (q_rows + 2*kv_rows) + q_rows;
        const float* v_ptr = st.d_qkv + (size_t)b * (q_rows + 2*kv_rows) + q_rows + kv_rows;
        HIP_CHECK(hipMemcpy(k_cache_b + (size_t)pos_b * kv_dim_local, k_ptr, kv_dim_local*sizeof(float), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(v_cache_b + (size_t)pos_b * kv_dim_local, v_ptr, kv_dim_local*sizeof(float), hipMemcpyDeviceToDevice));
      }

      // Copy Q local into contiguous B buffer
      for (int b = 0; b < B; ++b) {
        const float* q_ptr = st.d_qkv + (size_t)b * (q_rows + 2*kv_rows);
        float* qdst = st.d_q_local + (size_t)b * q_rows;
        HIP_CHECK(hipMemcpy(qdst, q_ptr, q_rows*sizeof(float), hipMemcpyDeviceToDevice));
      }
    }

    // 4) Attention (batched) -> tb_local_b (KV cache addressed by slot_ids)
    for (int r = base; r < base + MP_GROUP_SIZE; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& hs = g_head_shard[r];
      auto& st = g_state[r];
      int kv_dim_local = hs.kv_count * hd;
      size_t kv_block = (size_t)p->seq_len * kv_dim_local;
      const float* k_cache_layer_base = st.d_kv_k + (size_t)l * g_group_batch * kv_block;
      const float* v_cache_layer_base = st.d_kv_v + (size_t)l * g_group_batch * kv_block;
      int max_pos = -1;
      for (int b = 0; b < B; ++b) if (pos_list[b] > max_pos) max_pos = pos_list[b];
      int smem_len = (max_pos >= 0 ? (max_pos + 2) : 1);
      const float* sinks_l = nullptr;
      if (g_weights[r].d_attn_sinks) sinks_l = g_weights[r].d_attn_sinks + (size_t)l * g_head_shard[r].q_count;

      size_t smem = (size_t)(smem_len + hd) * sizeof(float);
      dim3 grid(hs.q_count, B), block(hd);
      k_attention_step_batched_sid<<<grid, block, smem>>>(
        st.d_q_local,
        k_cache_layer_base,
        v_cache_layer_base,
        st.d_tb_local,
        st.d_pos_list,
        st.d_slot_ids,
        hd, hs.q_count, hs.kv_count, kv_mul,
        (int)((size_t)p->seq_len * kv_dim_local),
        st.d_mask, p->seq_len,
        sinks_l);
      HIP_CHECK(hipGetLastError());
    }

    // 5) W_o matvec (batched) per shard
    for (int r = base; r < base + MP_GROUP_SIZE; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& devw = g_weights[r];
      auto& st   = g_state[r];
      const hip_bfloat16* Wcols = devw.d_w_o_cols + (size_t)l * (size_t)D * devw.cols_per_layer_local;
      dim3 grid(D, B), block(256);
      k_matvec_colslice_bf16_batched<<<grid, block>>>(Wcols, st.d_tb_local, st.d_tb2_partial, D, devw.cols_per_layer_local);
      HIP_CHECK(hipGetLastError());
    }

    // 6) Reduce all shards to root via P2P, then add bias and residual on root device
    int root = base; // group root device
    root_reduce_tb2(base, MP_GROUP_SIZE, B, D, root);
    HIP_CHECK(hipSetDevice(root));
    // Add W_o bias to tb2 partials on device
    const float* b_o_dev = g_weights[root].d_b_o + (size_t)l * D;
    dim3 gridb_b((D+255)/256, B);
    k_add_bias_batched<<<gridb_b, 256>>>(g_state[root].d_tb2_partial, b_o_dev, D);
    HIP_CHECK(hipGetLastError());
    // Residual add: xB += tb2_sum (on device root)
    // Root already has xB on device from step 1
    dim3 grid_add((D+255)/256, B);
    k_add_inplace_batched<<<grid_add, 256>>>(g_state[root].d_x, g_state[root].d_tb2_partial, D);
    HIP_CHECK(hipGetLastError());

    // 7) FFN RMSNorm on root (batched); keep tB on device for MoE dispatch
    {
      HIP_CHECK(hipSetDevice(root));
      auto& devwr = g_weights[root];
      auto& str   = g_state[root];
      const float* w_ffn = devwr.d_rms_ffn_w + (size_t)l * D;
      dim3 grid(1, B), block(256);
      k_rmsnorm_vec_batched<<<grid, block>>>(str.d_x, w_ffn, str.d_t, D);
      HIP_CHECK(hipGetLastError());
    }

    // 8) Router matvec on root (batched) + bias; copy to host
    std::vector<float> router((size_t)B * E);
    {
      HIP_CHECK(hipSetDevice(root));
      auto& devwr = g_weights[root];
      auto& str   = g_state[root];
      const float* Wrouter_l = devwr.d_w_router + (size_t)l * (size_t)E * D;
      dim3 grid(E, B), block(256);
      k_matvec_fp32_batched<<<grid, block>>>(Wrouter_l, str.d_t, str.d_router, D, E);
      HIP_CHECK(hipGetLastError());
      dim3 gridb((E+255)/256, B);
      const float* b_router_l = devwr.d_b_router + (size_t)l * E;
      k_add_bias_batched<<<gridb, 256>>>(str.d_router, b_router_l, E);
      HIP_CHECK(hipGetLastError());
      memcpyDtoH_chunked(router.data(), str.d_router, (size_t)B * E * sizeof(float));
    }

    // 9) Per-sample top-k and expert MLPs (P2P alltoallv batched per owner-expert)
    // Build routing lists on host: for each owner r and its local expert e, collect (b, ew)
    struct Item { int b; float w; int e_local; };
    std::vector<std::vector<Item>> per_owner(MP_GROUP_SIZE);
    for (int r = 0; r < MP_GROUP_SIZE; ++r) per_owner[r].reserve(B * tr->config.experts_per_token);
    for (int b = 0; b < B; ++b) {
      int K = tr->config.experts_per_token;
      std::vector<int> idxs(E); for (int i=0;i<E;++i) idxs[i]=i;
      float* rrow = router.data() + (size_t)b * E;
      std::partial_sort(idxs.begin(), idxs.begin()+K, idxs.end(), [&](int a,int b2){return rrow[a]>rrow[b2];});
      std::vector<int> topk_idx(K);
      std::vector<float> topk_val(K);
      for (int i=0;i<K;++i){ topk_idx[i]=idxs[i]; topk_val[i]=rrow[idxs[i]]; }
      float maxv = -INFINITY;
      for (float v: topk_val) if (v > maxv) maxv = v;
      float sum = 0.f; for (float &v: topk_val) { v = expf(v - maxv); sum += v; }
      float inv = 1.f / sum; for (float &v: topk_val) v *= inv;
      for (int ki = 0; ki < K; ++ki) {
        int e_glob = topk_idx[ki]; float ew = topk_val[ki];
        int owner = -1;
        int owner_off = -1;
        for (int rr = base; rr < base + MP_GROUP_SIZE; ++rr) {
          if (e_glob >= g_expert_shard[rr].e_start && e_glob < g_expert_shard[rr].e_start + g_expert_shard[rr].e_count) {
            owner = rr; owner_off = rr - base; break; }
        }
        if (owner < 0) { fprintf(stderr,"EP owner not found in group %d\n", group_id); exit(EXIT_FAILURE); }
        int e_local = e_glob - g_expert_shard[owner].e_start;
        per_owner[owner_off].push_back({b, ew, e_local});
      }
    }

    // Zero root accum buffer once for all samples
    HIP_CHECK(hipSetDevice(root));
    HIP_CHECK(hipMemsetAsync(g_state[root].d_moe_accum, 0, (size_t)B * D * sizeof(float), g_state[root].comm_stream));
    HIP_CHECK(hipStreamSynchronize(g_state[root].comm_stream));

    // For each owner, group by local expert and process minibatches
    for (int r_off = 0; r_off < MP_GROUP_SIZE; ++r_off) {
      int owner = base + r_off;
      if (per_owner[r_off].empty()) continue;
      // Build per-expert map
      int E_local = g_expert_shard[owner].e_count;
      std::vector<std::vector<Item>> by_exp(E_local);
      for (auto &it : per_owner[r_off]) by_exp[it.e_local].push_back(it);

      for (int e_local = 0; e_local < E_local; ++e_local) {
        auto &v = by_exp[e_local];
        if (v.empty()) continue;
        int nb = (int)v.size();
        // Copy indices and weights to root device
        HIP_CHECK(hipSetDevice(root));
        std::vector<int> h_idx(nb);
        std::vector<float> h_w(nb);
        for (int i = 0; i < nb; ++i) { h_idx[i] = v[i].b; h_w[i] = v[i].w; }
        HIP_CHECK(hipMemcpy(g_state[root].d_idx_tmp, h_idx.data(), (size_t)nb * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(g_state[root].d_weight_tmp, h_w.data(), (size_t)nb * sizeof(float), hipMemcpyHostToDevice));
        // Gather inputs on root: in_batch[nb,D]
        dim3 gg((D+255)/256, nb);
        k_gather_rows_f32<<<gg, 256, 0, g_state[root].comm_stream>>>(
            g_state[root].d_t, g_state[root].d_idx_tmp, nb, D, g_state[root].d_in_batch);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(g_state[root].comm_stream));
        // P2P copy to owner
        HIP_CHECK(hipMemcpyPeerAsync(g_state[owner].d_in_batch, owner,
                                     g_state[root].d_in_batch, root,
                                     (size_t)nb * D * sizeof(float), g_state[owner].comm_stream));
        // Owner compute batched expert MLP
        HIP_CHECK(hipSetDevice(owner));
        auto& devw = g_weights[owner];
        auto& st   = g_state[owner];
        size_t mlp1_stride = (size_t)devw.mlp1_Mpad * (size_t)devw.mlp1_Kpad;
        const hip_bfloat16* W1 = devw.d_w_mlp1_packed + ((size_t)l* g_expert_shard[owner].e_count + e_local) * mlp1_stride;
        const float*        B1 = devw.d_b_mlp1 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)(2*I);
        dim3 b_mlp1(16, 4);
        dim3 g_mlp1((unsigned int)((2*I + 15) / 16), (unsigned int)((nb + 15) / 16));
        const float clampU = tr->config.swiglu_limit;
        const float clampL = -tr->config.swiglu_limit;
        k_mlp1_matvec_bf16_fp32_batched<<<g_mlp1, b_mlp1, 0, st.comm_stream>>>(
            W1, st.d_in_batch, st.d_gate_up, B1, D, 2*I, nb, clampU, clampL,
            devw.mlp1_Mpad, devw.mlp1_Kpad);
        HIP_CHECK(hipGetLastError());
        const hip_bfloat16* W2 = devw.d_w_mlp2 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)D * I;
        const float*        B2 = devw.d_b_mlp2 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)D;
        dim3 g2(D, nb), b2d(256);
        k_mlp2_matvec_bf16_fp32_batched<<<g2, b2d, 0, st.comm_stream>>>(W2, st.d_gate_up, st.d_ffn_out, I, D);
        HIP_CHECK(hipGetLastError());
        dim3 gb2((D+255)/256, nb);
        k_add_bias_batched<<<gb2, 256, 0, st.comm_stream>>>(st.d_ffn_out, B2, D);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(st.comm_stream));
        // Copy back to root and scatter-add with weights
        HIP_CHECK(hipSetDevice(root));
        HIP_CHECK(hipMemcpyPeerAsync(g_state[root].d_in_batch, root,
                                     g_state[owner].d_ffn_out, owner,
                                     (size_t)nb * D * sizeof(float), g_state[root].comm_stream));
        dim3 gsax((D+255)/256, nb);
        k_scatter_axpy_rows_f32<<<gsax, 256, 0, g_state[root].comm_stream>>>(
            g_state[root].d_moe_accum,
            g_state[root].d_idx_tmp,
            g_state[root].d_weight_tmp,
            g_state[root].d_in_batch,
            nb, D);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(g_state[root].comm_stream));
      }
    }

    // Add accumulated expert outputs into root x buffer (residual)
    HIP_CHECK(hipSetDevice(root));
    dim3 gadd_exp((D+255)/256, B);
    k_add_inplace_batched<<<gadd_exp, 256>>>(g_state[root].d_x, g_state[root].d_moe_accum, D);
    HIP_CHECK(hipGetLastError());
    // Update host xB for the next layer inputs
    memcpyDtoH_chunked(xB.data(), g_state[root].d_x, (size_t)B * D * sizeof(float));
  }

  // Final RMSNorm + logits (batched) and argmax on root
  HIP_CHECK(hipSetDevice(root));
  {
    auto& devwr = g_weights[root];
    auto& str   = g_state[root];
    dim3 gridN(1, B), blockN(256);
    k_rmsnorm_vec_batched<<<gridN, blockN>>>(str.d_x, devwr.d_rms_out_w, str.d_x, D);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(str.d_t, str.d_x, (size_t)B*D*sizeof(float), hipMemcpyDeviceToDevice));
    int V = tr->config.vocab_size;
    dim3 gridL(V, B), blockL(256);
    k_matvec_fp32_batched<<<gridL, blockL>>>(devwr.d_w_out, str.d_t, str.d_logits, D, V);
    HIP_CHECK(hipGetLastError());
    const size_t shmem_bytes = (size_t)1024 * (sizeof(float) + sizeof(int));
    k_argmax_batched<<<B, 1024, shmem_bytes>>>(str.d_logits, B, V, str.d_next_token);
    HIP_CHECK(hipGetLastError());
    next_tokens_out.resize(B);
    HIP_CHECK(hipMemcpy(next_tokens_out.data(), str.d_next_token, (size_t)B * sizeof(int), hipMemcpyDeviceToHost));
  }
}

// ---------------------------------------------
// Inference driver for getp_eval.cpp
// ---------------------------------------------
long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  // Continuous batching via DP_GROUPS worker threads. Each group runs its own
  // request stream and keeps pulling next request as soon as it finishes.
  std::atomic<int> next_idx{0};
  std::atomic<long long> total_tokens{0};

  auto worker = [&](int group_id) {
    while (true) {
      // Try to fill a new microbatch (up to g_group_batch)
      int idx_start = next_idx.fetch_add(g_group_batch);
      if (idx_start >= requests->num_reqs) break;
      int B = std::min(g_group_batch, requests->num_reqs - idx_start);

      // Per-slot state
      std::vector<const char*> inputs(B, nullptr);
      std::vector<int*> outputs(B, nullptr);
      std::vector<int> num_prompt_tokens(B, 0);
      std::vector<std::vector<int>> prompt_tokens(B);
      std::vector<int> token(B, 0);
      std::vector<int> pos(B, 0);
      std::vector<char> active(B, 0);

      for (int i = 0; i < B; ++i) {
        int ridx = idx_start + i;
        inputs[i] = get_str_req_ptr(requests, ridx);
        outputs[i] = get_tok_gen_ptr(requests, ridx);
        int npt = 0;
        prompt_tokens[i].resize(strlen(inputs[i]) + 3);
        encode(tokenizer, inputs[i], -1, -1, prompt_tokens[i].data(), &npt,
               transformer->config.initial_context_length);
        num_prompt_tokens[i] = npt;
        if (npt < 1) { fprintf(stderr, "empty prompt in batch item %d\n", i); exit(EXIT_FAILURE);}    
        token[i] = prompt_tokens[i][0];
        pos[i] = 0;
        active[i] = 1;
      }

      int D = transformer->config.hidden_dim;
      int steps = requests->max_seq_len;
      if (steps == 0 || steps > transformer->config.seq_len) steps = transformer->config.seq_len;
      std::vector<float> xB((size_t)B * D, 0.f);
      std::vector<float> tB((size_t)B * D, 0.f);
      std::vector<int> next(B, 0);

      long long tokens_out_total = 0;
      auto count_active = [&](){ int c=0; for(char a:active) if(a) ++c; return c; };

      while (true) {
        // If no active slots, attempt to refill
        if (count_active() == 0) {
          bool refilled = false;
          for (int b = 0; b < B; ++b) {
            if (active[b]) continue;
            int ridx = next_idx.fetch_add(1);
            if (ridx >= requests->num_reqs) break;
            inputs[b] = get_str_req_ptr(requests, ridx);
            outputs[b] = get_tok_gen_ptr(requests, ridx);
            int npt = 0;
            prompt_tokens[b].resize(strlen(inputs[b]) + 3);
            encode(tokenizer, inputs[b], -1, -1, prompt_tokens[b].data(), &npt,
                   transformer->config.initial_context_length);
            num_prompt_tokens[b] = npt;
            token[b] = prompt_tokens[b][0];
            pos[b] = 0;
            active[b] = 1;
            refilled = true;
          }
          if (!refilled) break; // no work left globally
        }

        // Build compact active batch
        std::vector<int> slot_ids_active;
        std::vector<int> tokens_active;
        std::vector<int> pos_active;
        slot_ids_active.reserve(B);
        tokens_active.reserve(B);
        pos_active.reserve(B);
        for (int b = 0; b < B; ++b) {
          if (!active[b]) continue;
          slot_ids_active.push_back(b);
          tokens_active.push_back(token[b]);
          pos_active.push_back(pos[b]);
        }
        int Bact = (int)slot_ids_active.size();
        if (Bact == 0) continue; // nothing active; try refill next iteration

        // Run compact-batched forward for active slots
        std::vector<float> xB_act((size_t)Bact * D, 0.f);
        std::vector<float> tB_act((size_t)Bact * D, 0.f);
        std::vector<int> next_act(Bact, 0);
        forward_one_group_batched_compact(transformer, group_id, Bact,
                                          tokens_active.data(), pos_active.data(), slot_ids_active.data(),
                                          xB_act, tB_act, next_act);

        // Advance active slots, finalize EOS, and refill as needed
        for (int i = 0; i < Bact; ++i) {
          int b = slot_ids_active[i];
          pos[b]++;
          int nt;
          if (pos[b] < num_prompt_tokens[b]) {
            nt = prompt_tokens[b][pos[b]];
          } else {
            nt = next_act[i];
            outputs[b][pos[b] - num_prompt_tokens[b]] = nt;
            tokens_out_total++;
          }
          bool hit_eos = (nt == 199999 || nt == 200002);
          bool hit_steps = (pos[b] >= steps);
          if (hit_eos || hit_steps) {
            // finalize current request
            outputs[b][pos[b] - num_prompt_tokens[b] + 1] = -1;
            active[b] = 0;
            // refill immediately if available
            int ridx = next_idx.fetch_add(1);
            if (ridx < requests->num_reqs) {
              inputs[b] = get_str_req_ptr(requests, ridx);
              outputs[b] = get_tok_gen_ptr(requests, ridx);
              int npt = 0;
              prompt_tokens[b].resize(strlen(inputs[b]) + 3);
              encode(tokenizer, inputs[b], -1, -1, prompt_tokens[b].data(), &npt,
                     transformer->config.initial_context_length);
              num_prompt_tokens[b] = npt;
              token[b] = prompt_tokens[b][0];
              pos[b] = 0;
              active[b] = 1;
            }
          } else {
            token[b] = nt;
          }
        }
      }
      total_tokens.fetch_add(tokens_out_total);
    }
  };

  // Launch DP_GROUPS workers (one per DP group)
  std::vector<std::thread> threads;
  int groups_to_launch = DP_GROUPS;
  for (int g = 0; g < groups_to_launch; ++g) {
    threads.emplace_back(worker, g);
  }
  for (auto& th : threads) th.join();
  return total_tokens.load();
}


// ---------------------------------------------
// Warm-up
// ---------------------------------------------
void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  transformer->config.seq_len = 1024; // override max seq len to 1024
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;

  printf("[WARMUP] Starting GPU warmup process...\n");
  printf("[WARMUP] Model config: %d layers, %d hidden_dim, %d head_dim, %d experts\n", 
         p->n_layers, p->hidden_dim, p->head_dim, p->n_experts);
  fflush(stdout);

  // Read GROUP_BATCH from environment if provided
  const char* env_gb = getenv("GROUP_BATCH");
  if (env_gb && *env_gb) {
    int v = atoi(env_gb);
    if (v > 0) g_group_batch = v;
  }
  printf("[WARMUP] Using GROUP_BATCH=%d (env override supported)\n", g_group_batch);

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < WORLD_GPUS) {
    fprintf(stderr, "This implementation expects at least %d GPUs, found %d.\n", WORLD_GPUS, device_count);
    exit(EXIT_FAILURE);
  }

  // Enable P2P access across all GPUs (best-effort)
  enable_peer_access_all();

  // Print initial memory state for all GPUs
  printf("[WARMUP] Initial GPU memory state:\n");
  for (int r = 0; r < WORLD_GPUS; ++r) {
    print_memory_usage(r, "Initial");
  }

  for (int r = 0; r < WORLD_GPUS; ++r) {
    int mp_rank = mp_rank_from_rank(r);
    g_head_shard[r] = compute_head_shard(p, mp_rank, MP_GROUP_SIZE);
    g_expert_shard[r] = compute_expert_shard(p, mp_rank, MP_GROUP_SIZE);
    printf("[GPU%d / G%d.R%d] Head shard: Q[%d:%d], KV[%d:%d] | Expert shard: [%d:%d]\n",
           r, group_id_from_rank(r), mp_rank,
           g_head_shard[r].q_start, g_head_shard[r].q_start + g_head_shard[r].q_count - 1,
           g_head_shard[r].kv_start, g_head_shard[r].kv_start + g_head_shard[r].kv_count - 1,
           g_expert_shard[r].e_start, g_expert_shard[r].e_start + g_expert_shard[r].e_count - 1);
  }

  // Legacy buffer kept allocated for compatibility (not used in group path)
  h_tb2_sum = (float*)malloc(p->hidden_dim * sizeof(float));
  for (int r = 0; r < WORLD_GPUS; ++r) {
    h_tb2_partials[r] = (float*)malloc(p->hidden_dim * sizeof(float));
  }

  const int L  = p->n_layers;
  const int D  = p->hidden_dim;
  const int hd = p->head_dim;
  const int seq = p->seq_len;

  printf("[WARMUP] Starting per-GPU weight loading...\n");
  fflush(stdout);

  for (int r = 0; r < WORLD_GPUS; ++r) {
    printf("\n[GPU%d] ========== Starting weight loading ==========\n", r);
    HIP_CHECK(hipSetDevice(r));
    print_memory_usage(r, "Before allocation");
    gpu_copy_weights(transformer, r);
    gpu_malloc_run_state(transformer, r);
    print_memory_usage(r, "After weight loading completed");
    printf("[GPU%d] ========== Weight loading completed ==========\n\n", r);
    fflush(stdout);
  }

  printf("[warm_up] Sharded weights loaded across %d GPUs (chunk %zu MB)\n", WORLD_GPUS, CHUNK_BYTES/1024/1024);
  fflush(stdout);
}

// ---------------------------------------------
// Finish
// ---------------------------------------------
void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (int r = 0; r < WORLD_GPUS; ++r) {
    HIP_CHECK(hipSetDevice(r));
    auto& devw = g_weights[r];
    auto& st   = g_state[r];
    // Weights
    safeHipFree(devw.d_w_qkv_rows);
    safeHipFree(devw.d_b_qkv_rows);
    safeHipFree(devw.d_w_o_cols);
    safeHipFree(devw.d_w_mlp1_packed);
    safeHipFree(devw.d_b_mlp1);
    safeHipFree(devw.d_w_mlp2);
    safeHipFree(devw.d_b_mlp2);
    safeHipFree(devw.d_w_router);
    safeHipFree(devw.d_b_router);
    safeHipFree(devw.d_rms_attn_w);
    safeHipFree(devw.d_rms_ffn_w);
    safeHipFree(devw.d_rms_out_w);
    safeHipFree(devw.d_attn_sinks);
    safeHipFree(devw.d_b_o);
#if KEEP_FP8_UTILS
    safeHipFree(devw.d_w_mlp1_u8);
    safeHipFree(devw.d_w_mlp1_scale);
#endif
    safeHipFree(devw.d_w_out);
    // Run state
    safeHipFree(st.d_kv_k);
    safeHipFree(st.d_kv_v);
    safeHipFree(st.d_x);
    safeHipFree(st.d_t);
    safeHipFree(st.d_qkv);
    safeHipFree(st.d_q_local);
    safeHipFree(st.d_tb_local);
    safeHipFree(st.d_tb2_partial);
    safeHipFree(st.d_gate_up);
    safeHipFree(st.d_ffn_out);
    safeHipFree(st.d_cos);
    safeHipFree(st.d_sin);
    safeHipFree(st.d_mask);
    safeHipFree(st.d_router);
    safeHipFree(st.d_topk);
    safeHipFree(st.d_pos_list);
    safeHipFree(st.d_slot_ids);
    safeHipFree(st.d_next_token);
    safeHipFree(st.d_logits);
    if (st.comm_stream) { HIP_CHECK(hipStreamDestroy(st.comm_stream)); st.comm_stream = nullptr; }
    safeHipFree(st.d_comm_tmp);
    safeHipFree(st.d_moe_accum);
    safeHipFree(st.d_in_batch);
    safeHipFree(st.d_idx_tmp);
    safeHipFree(st.d_weight_tmp);

  }
  for (int r = 0; r < WORLD_GPUS; ++r) { free(h_tb2_partials[r]); h_tb2_partials[r] = nullptr; }
  free(h_tb2_sum); h_tb2_sum = nullptr;
}

// ---------------------------------------------
// GPU allocation/copy helpers (weights vs run state)
// ---------------------------------------------
static void gpu_copy_weights(Transformer* transformer, int r) {
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;

  const int L  = p->n_layers;
  const int D  = p->hidden_dim;
  const int hd = p->head_dim;
  const int E = p->n_experts;

  auto& devw = g_weights[r];
  auto& hs   = g_head_shard[r];
  auto& es   = g_expert_shard[r];

  // Derived locals
  devw.rows_per_layer_local = (hs.q_count + 2*hs.kv_count)*hd;
  devw.cols_per_layer_local = hs.q_count * hd;
  devw.w_qkv_rows_stride    = (size_t)devw.rows_per_layer_local * D;

  size_t tot_rows = (size_t)L * devw.rows_per_layer_local;
  size_t tot_cols = (size_t)L * devw.cols_per_layer_local;

  // Allocate attention weights
  printf("[GPU%d] Allocating attention weights...\n", r);
  size_t qkv_weights_size = tot_rows * D * sizeof(hip_bfloat16);
  size_t qkv_bias_size    = tot_rows * sizeof(float);
  size_t wo_weights_size  = (size_t)D * tot_cols * sizeof(hip_bfloat16);
  printf("[GPU%d]   - QKV weights: %.2f MB\n", r, qkv_weights_size / (1024.0*1024.0));
  printf("[GPU%d]   - QKV bias: %.2f MB\n", r, qkv_bias_size / (1024.0*1024.0));
  printf("[GPU%d]   - W_o weights: %.2f MB\n", r, wo_weights_size / (1024.0*1024.0));

  HIP_CHECK(hipMalloc(&devw.d_w_qkv_rows, qkv_weights_size));
  HIP_CHECK(hipMalloc(&devw.d_b_qkv_rows, qkv_bias_size));
  HIP_CHECK(hipMalloc(&devw.d_w_o_cols,   wo_weights_size));

  print_memory_usage(r, "After attention allocation");

  // Load attention weights layer by layer
  printf("[GPU%d] Loading attention weights...\n", r);
  {
    std::vector<hip_bfloat16> tmp_rows(devw.rows_per_layer_local * D);
    std::vector<float>        tmp_bias(devw.rows_per_layer_local);
    std::vector<hip_bfloat16> tmp_cols((size_t)D * devw.cols_per_layer_local);

    for (int l = 0; l < L; ++l) {
      if (l % 10 == 0 || l == L-1) {
        printf("[GPU%d]   Loading attention layer %d/%d...\n", r, l+1, L);
        fflush(stdout);
      }
      pack_layer_w_qkv_local_bf16(w, p, l, hs, tmp_rows.data(), tmp_bias.data());
      pack_layer_w_o_colslice_bf16(w, p, l, hs, tmp_cols.data());

      hip_bfloat16* w_qkv_dst = devw.d_w_qkv_rows + (size_t)l * devw.w_qkv_rows_stride;
      float*        b_qkv_dst = devw.d_b_qkv_rows + (size_t)l * devw.rows_per_layer_local;
      hip_bfloat16* w_o_dst   = devw.d_w_o_cols   + (size_t)l * (size_t)D * devw.cols_per_layer_local;

      memcpyHtoD_chunked(w_qkv_dst, tmp_rows.data(), tmp_rows.size() * sizeof(hip_bfloat16));
      memcpyHtoD_chunked(b_qkv_dst, tmp_bias.data(), tmp_bias.size() * sizeof(float));
      memcpyHtoD_chunked(w_o_dst,   tmp_cols.data(), tmp_cols.size() * sizeof(hip_bfloat16));
    }
  }
  printf("[GPU%d] ✓ Attention weights loaded successfully\n", r);

  // Copy RMSNorm weights
  printf("[GPU%d] Copying RMSNorm weights...\n", r);
  HIP_CHECK(hipMalloc(&devw.d_rms_attn_w, (size_t)L * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&devw.d_rms_ffn_w,  (size_t)L * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&devw.d_rms_out_w,  D * sizeof(float)));
  memcpyHtoD_chunked(devw.d_rms_attn_w, transformer->weights.rms_attn_w, (size_t)L * D * sizeof(float));
  memcpyHtoD_chunked(devw.d_rms_ffn_w,  transformer->weights.rms_ffn_w,  (size_t)L * D * sizeof(float));
  memcpyHtoD_chunked(devw.d_rms_out_w,  transformer->weights.rms_out_w,  D * sizeof(float));


  // Copy final classifier (FP32) on group root only (mp_rank==0)
  if (mp_rank_from_rank(r) == 0) {
    int V = p->vocab_size;
    printf("[GPU%d] Copying OUT weights (vocab x hidden) FP32...\n", r);
    HIP_CHECK(hipMalloc(&devw.d_w_out, (size_t)V * (size_t)D * sizeof(float)));
    memcpyHtoD_chunked(devw.d_w_out, transformer->weights.out, (size_t)V * (size_t)D * sizeof(float));
  }

  // Copy Router weights (FP32)
  printf("[GPU%d] Copying Router weights...\n", r);
  HIP_CHECK(hipMalloc(&devw.d_w_router, (size_t)L * (size_t)E * (size_t)D * sizeof(float)));
  HIP_CHECK(hipMalloc(&devw.d_b_router, (size_t)L * (size_t)E * sizeof(float)));
  memcpyHtoD_chunked(devw.d_w_router, transformer->weights.w_router, (size_t)L * (size_t)E * (size_t)D * sizeof(float));
  memcpyHtoD_chunked(devw.d_b_router, transformer->weights.b_router, (size_t)L * (size_t)E * sizeof(float));

  // Copy W_o bias per layer (FP32)
  printf("[GPU%d] Copying W_o bias...\n", r);
  HIP_CHECK(hipMalloc(&devw.d_b_o, (size_t)L * (size_t)D * sizeof(float)));
  memcpyHtoD_chunked(devw.d_b_o, transformer->weights.b_o, (size_t)L * (size_t)D * sizeof(float));

  // Allocate and load MLP weights (per local expert)
  const int I = p->intermediate_dim;
  const int E_local = es.e_count;
  printf("[GPU%d] Allocating MLP weights for %d local experts...\n", r, E_local);
  size_t mlp1_rows_per_block = 2 * (size_t)I;
  int Mpad = ((2 * I + 15) / 16) * 16;
  int Kpad = ((D + 15) / 16) * 16;
  size_t mlp1_packed_elems = (size_t)Mpad * (size_t)Kpad;

  size_t mlp1_weights_size = (size_t)L * E_local * mlp1_packed_elems * sizeof(hip_bfloat16);
  size_t mlp1_bias_size    = (size_t)L * E_local * mlp1_rows_per_block * sizeof(float);
  size_t mlp2_weights_size = (size_t)L * E_local * (size_t)D * I * sizeof(hip_bfloat16);
  size_t mlp2_bias_size    = (size_t)L * E_local * (size_t)D * sizeof(float);
  printf("[GPU%d]   - MLP1 weights (BF16): %.2f MB\n", r, mlp1_weights_size / (1024.0*1024.0));
  printf("[GPU%d]   - MLP1 bias: %.2f MB\n", r, mlp1_bias_size / (1024.0*1024.0));
  printf("[GPU%d]   - MLP2 weights (BF16): %.2f MB\n", r, mlp2_weights_size / (1024.0*1024.0));
  printf("[GPU%d]   - MLP2 bias: %.2f MB\n", r, mlp2_bias_size / (1024.0*1024.0));

  HIP_CHECK(hipMalloc(&devw.d_w_mlp1_packed, mlp1_weights_size));
  devw.mlp1_Mpad = Mpad;
  devw.mlp1_Kpad = Kpad;
  HIP_CHECK(hipMalloc(&devw.d_b_mlp1, mlp1_bias_size));
  HIP_CHECK(hipMalloc(&devw.d_w_mlp2, mlp2_weights_size));
  HIP_CHECK(hipMalloc(&devw.d_b_mlp2, mlp2_bias_size));

  // Load MLP weights
  printf("[GPU%d] Loading MLP weights...\n", r);
  {
    const int M_mlp1 = 2 * I;
    const int K_mlp1 = D;
    HIP_CHECK(hipMemset(devw.d_w_mlp1_packed, 0, mlp1_weights_size));

    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024;
    size_t staging_elems = std::max((size_t)1, STAGING_BYTES / sizeof(float));
    const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
    if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;
    size_t max_rows = staging_elems / (size_t)K_mlp1;
    if (max_rows < 1) max_rows = 1;

    float *d_stage_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&d_stage_fp32, (size_t)max_rows * (size_t)K_mlp1 * sizeof(float)));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<float>        tmp_b1(mlp1_rows_per_block);
    std::vector<hip_bfloat16> tmp_w2((size_t)D * I);
    std::vector<float>        tmp_b2(D);

    const dim3 block(256);
    int total_experts_loaded = 0;
    for (int l = 0; l < L; ++l) {
      for (int ei = 0; ei < E_local; ++ei) {
        int e_global = es.e_start + ei;
        if (total_experts_loaded % 50 == 0 || (l == L-1 && ei == E_local-1)) {
          printf("[GPU%d]   Loading expert %d/%d (layer %d, local expert %d)...\n",
                 r, total_experts_loaded + 1, L * E_local, l+1, ei+1);
          fflush(stdout);
        }
        const float* W1 = w->w_mlp1 + 1ll*(l*p->n_experts + e_global) * (2ll*I) * D;
        const float* B1 = w->b_mlp1 + 1ll*(l*p->n_experts + e_global) * (2ll*I);

        hip_bfloat16* dst_w1 = devw.d_w_mlp1_packed + ((size_t)l*E_local + ei) * mlp1_packed_elems;
        float*        dst_b1 = devw.d_b_mlp1 + ((size_t)l*E_local + ei) * mlp1_rows_per_block;

        for (int row_off = 0; row_off < M_mlp1; row_off += (int)max_rows) {
          int rows = std::min((int)max_rows, M_mlp1 - row_off);
          size_t tile_bytes = (size_t)rows * (size_t)K_mlp1 * sizeof(float);
          HIP_CHECK(hipMemcpyAsync(d_stage_fp32,
                                   W1 + (size_t)row_off * (size_t)K_mlp1,
                                   tile_bytes,
                                   hipMemcpyHostToDevice,
                                   stream));
          size_t total = (size_t)rows * (size_t)K_mlp1;
          dim3 grid((unsigned int)((total + block.x - 1) / block.x));
          pack_convert_fp32_to_bf16_tiles16_kernel<<<grid, block, 0, stream>>>(
              d_stage_fp32, row_off, rows, M_mlp1, K_mlp1, dst_w1, Mpad, Kpad);
          HIP_CHECK(hipGetLastError());
        }
        std::copy(B1, B1 + 2ll*I, tmp_b1.begin());
        memcpyHtoD_chunked(dst_b1, tmp_b1.data(), tmp_b1.size() * sizeof(float));

        const float* W2 = w->w_mlp2 + 1ll*(l*p->n_experts + e_global) * (size_t)D * I;
        const float* B2 = w->b_mlp2 + 1ll*(l*p->n_experts + e_global) * D;
        for (int i = 0; i < D*I; ++i) tmp_w2[i] = fp32_to_bf16_host(W2[i]);
        std::copy(B2, B2 + D, tmp_b2.begin());

        hip_bfloat16* dst_w2 = devw.d_w_mlp2 + ((size_t)l*E_local + ei) * (size_t)D * I;
        float*        dst_b2 = devw.d_b_mlp2 + ((size_t)l*E_local + ei) * (size_t)D;
        memcpyHtoD_chunked(dst_w2, tmp_w2.data(), tmp_w2.size() * sizeof(hip_bfloat16));
        memcpyHtoD_chunked(dst_b2, tmp_b2.data(), tmp_b2.size() * sizeof(float));

        total_experts_loaded++;
      }
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_stage_fp32));
    printf("[GPU%d] ✓ MLP weights loaded successfully (%d experts)\n", r, total_experts_loaded);
  }

  // Allocate & load attention sinks for local Q-heads
  int q_local = hs.q_count;
  if (q_local > 0) {
    HIP_CHECK(hipMalloc(&devw.d_attn_sinks, (size_t)p->n_layers * q_local * sizeof(float)));
    std::vector<float> h_sinks((size_t)p->n_layers * q_local);
    for (int l = 0; l < p->n_layers; ++l) {
      for (int h = 0; h < q_local; ++h) {
        int h_global = hs.q_start + h;
        h_sinks[(size_t)l * q_local + h] = transformer->weights.attn_sinks[l * p->n_attn_heads + h_global];
      }
    }
    memcpyHtoD_chunked(devw.d_attn_sinks, h_sinks.data(), h_sinks.size() * sizeof(float));
  }
}

static void gpu_malloc_run_state(Transformer* transformer, int r) {
  Config* p = &transformer->config;
  const int L  = p->n_layers;
  const int D  = p->hidden_dim;
  const int hd = p->head_dim;
  const int seq = p->seq_len;
  auto& hs   = g_head_shard[r];
  auto& devw = g_weights[r];
  auto& st   = g_state[r];

  // Allocate KV cache
  printf("[GPU%d] Allocating KV cache...\n", r);
  int kv_dim_local = hs.kv_count * hd;
  size_t kv_cache_size_one = (size_t)L * seq * kv_dim_local * sizeof(float);
  size_t kv_cache_size_all = (size_t)g_group_batch * kv_cache_size_one;
  printf("[GPU%d]   - KV cache (K): %.2f MB x B=%d\n", r, kv_cache_size_one / (1024.0*1024.0), g_group_batch);
  printf("[GPU%d]   - KV cache (V): %.2f MB x B=%d\n", r, kv_cache_size_one / (1024.0*1024.0), g_group_batch);
  HIP_CHECK(hipMalloc(&st.d_kv_k, kv_cache_size_all));
  HIP_CHECK(hipMalloc(&st.d_kv_v, kv_cache_size_all));

  // Allocate working buffers
  printf("[GPU%d] Allocating working buffers...\n", r);
  HIP_CHECK(hipMalloc(&st.d_x, (size_t)g_group_batch * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_t, (size_t)g_group_batch * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_qkv, (size_t)g_group_batch * devw.rows_per_layer_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_q_local, (size_t)g_group_batch * hs.q_count * hd * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_tb_local, (size_t)g_group_batch * hs.q_count * hd * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_tb2_partial, (size_t)g_group_batch * D * sizeof(float)));

  const int I = p->intermediate_dim;
  HIP_CHECK(hipMalloc(&st.d_gate_up,  (size_t)g_group_batch * I * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_ffn_out,  (size_t)g_group_batch * D * sizeof(float)));

  // Rotary buffers
  printf("[GPU%d] Allocating rotary buffers...\n", r);
  HIP_CHECK(hipMalloc(&st.d_cos, (p->head_dim/2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_sin, (p->head_dim/2) * sizeof(float)));

  // Attention mask (optional)
  if (transformer->state.mask) {
    printf("[GPU%d] Allocating attention mask...\n", r);
    size_t mask_elems = (size_t)p->seq_len * p->seq_len;
    size_t mask_size = mask_elems * sizeof(float);
    printf("[GPU%d]   - Attention mask: %.2f MB\n", r, mask_size / (1024.0*1024.0));
    HIP_CHECK(hipMalloc(&st.d_mask, mask_size));
    memcpyHtoD_chunked(st.d_mask, transformer->state.mask, mask_size);
    printf("[GPU%d] ✓ Attention mask loaded\n", r);
  }

  // Router temporaries and logits buffer (group root)
  HIP_CHECK(hipMalloc(&st.d_router, (size_t)g_group_batch * (size_t)p->n_experts * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_topk,   (size_t)g_group_batch * (size_t)p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_pos_list, (size_t)g_group_batch * sizeof(int)));
  HIP_CHECK(hipMalloc(&st.d_slot_ids, (size_t)g_group_batch * sizeof(int)));
  // P2P comm allocations
  HIP_CHECK(hipMalloc(&st.d_comm_tmp, (size_t)g_group_batch * (size_t)D * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_moe_accum, (size_t)g_group_batch * (size_t)D * sizeof(float)));
  HIP_CHECK(hipStreamCreate(&st.comm_stream));
  // Batched MoE staging buffers
  HIP_CHECK(hipMalloc(&st.d_in_batch, (size_t)g_group_batch * (size_t)D * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_idx_tmp, (size_t)g_group_batch * sizeof(int)));
  HIP_CHECK(hipMalloc(&st.d_weight_tmp, (size_t)g_group_batch * sizeof(float)));
  if (mp_rank_from_rank(r) == 0) {
    HIP_CHECK(hipMalloc(&st.d_logits, (size_t)g_group_batch * (size_t)p->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&st.d_next_token, (size_t)g_group_batch * sizeof(int)));
  }
}

#endif // GETP_RUN
