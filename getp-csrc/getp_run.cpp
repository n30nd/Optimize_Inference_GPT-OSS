// getp-csrc/getp_run.cpp
// End-to-end "getp" path with 4x AMD GPUs, TP (Attention) + EP (MoE).
// - No external GPU libs. Naive HIP kernels focused on correctness.
// - Weights:    All stored as BF16 (MLP1 now BF16, previously FP8-like)
// - Activations: FP32 (GPURunState concept)
// - All-reduce: host-bridge (D2H sum H2D) for Attention W_o partials
// - Warm-up does sharding + chunked copy (512 MB default)
// - Inference runs one token stream per request (driver in getp_eval.cpp)
// - Router computed on host; MoE dispatch via host -> owner GPU -> host

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

#ifndef NUM_GPUS
#define NUM_GPUS 4
#endif

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

// ---------------------------------------------
// Very small, naive device kernels (correctness)
// ---------------------------------------------

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

__global__ void k_add_bias(float* y, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i];
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
static HeadShard compute_head_shard(const Config* p, int rank) {
  HeadShard hs{};
  split_even(p->n_attn_heads, rank, NUM_GPUS, hs.q_start, hs.q_count);
  split_even(p->n_kv_heads,  rank, NUM_GPUS, hs.kv_start, hs.kv_count);
  return hs;
}

struct ExpertShard { int e_start, e_count; };
static ExpertShard compute_expert_shard(const Config* p, int rank) {
  ExpertShard es{};
  split_even(p->n_experts, rank, NUM_GPUS, es.e_start, es.e_count);
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

  hip_bfloat16* d_w_mlp1 = nullptr; // MLP1 as BF16
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

  float* d_mlp1_out = nullptr;
  float* d_gate_up  = nullptr;
  float* d_ffn_out  = nullptr;

  float* d_cos = nullptr;   // size head_dim/2
  float* d_sin = nullptr;   // size head_dim/2
  float* d_mask = nullptr;  // size seq_len * seq_len  (nullable)

  // Router temporaries
  float* d_router = nullptr; // size E
  float* d_topk   = nullptr; // size K (experts_per_token)

  // Logits (GPU0 only)
  float* d_logits = nullptr; // size vocab_size
};

static GPUTransformerWeights g_weights[NUM_GPUS];
static GPURunState g_state[NUM_GPUS];
static HeadShard  g_head_shard[NUM_GPUS];
static ExpertShard g_expert_shard[NUM_GPUS];
static int g_world = NUM_GPUS;
static std::mutex g_reduce_mutex;

static float* h_tb2_partials[NUM_GPUS] = {nullptr, nullptr, nullptr, nullptr};
static float* h_tb2_sum = nullptr;

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

  // Copy Router weights (FP32)
  printf("[GPU%d] Copying Router weights...\n", r);
  HIP_CHECK(hipMalloc(&devw.d_w_router, (size_t)L * (size_t)E * (size_t)D * sizeof(float)));
  HIP_CHECK(hipMalloc(&devw.d_b_router, (size_t)L * (size_t)E * sizeof(float)));
  memcpyHtoD_chunked(devw.d_w_router, transformer->weights.w_router, (size_t)L * (size_t)E * (size_t)D * sizeof(float));
  memcpyHtoD_chunked(devw.d_b_router, transformer->weights.b_router, (size_t)L * (size_t)E * sizeof(float));

  // Copy final classifier (FP32) on GPU0 only
  if (r == 0) {
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

  // Allocate and load MLP weights (per local expert)
  const int I = p->intermediate_dim;
  const int E_local = es.e_count;
  printf("[GPU%d] Allocating MLP weights for %d local experts...\n", r, E_local);
  size_t mlp1_rows_per_block = 2 * (size_t)I;
  size_t mlp1_elems_per_block = mlp1_rows_per_block * (size_t)D;

  size_t mlp1_weights_size = (size_t)L * E_local * mlp1_elems_per_block * sizeof(hip_bfloat16);
  size_t mlp1_bias_size    = (size_t)L * E_local * mlp1_rows_per_block * sizeof(float);
  size_t mlp2_weights_size = (size_t)L * E_local * (size_t)D * I * sizeof(hip_bfloat16);
  size_t mlp2_bias_size    = (size_t)L * E_local * (size_t)D * sizeof(float);
  printf("[GPU%d]   - MLP1 weights (BF16): %.2f MB\n", r, mlp1_weights_size / (1024.0*1024.0));
  printf("[GPU%d]   - MLP1 bias: %.2f MB\n", r, mlp1_bias_size / (1024.0*1024.0));
  printf("[GPU%d]   - MLP2 weights (BF16): %.2f MB\n", r, mlp2_weights_size / (1024.0*1024.0));
  printf("[GPU%d]   - MLP2 bias: %.2f MB\n", r, mlp2_bias_size / (1024.0*1024.0));

  HIP_CHECK(hipMalloc(&devw.d_w_mlp1, mlp1_weights_size));
  HIP_CHECK(hipMalloc(&devw.d_b_mlp1, mlp1_bias_size));
  HIP_CHECK(hipMalloc(&devw.d_w_mlp2, mlp2_weights_size));
  HIP_CHECK(hipMalloc(&devw.d_b_mlp2, mlp2_bias_size));

  // Load MLP weights
  printf("[GPU%d] Loading MLP weights...\n", r);
  {
    std::vector<hip_bfloat16> tmp_w1(mlp1_elems_per_block);
    std::vector<float>        tmp_b1(mlp1_rows_per_block);
    std::vector<hip_bfloat16> tmp_w2((size_t)D * I);
    std::vector<float>        tmp_b2(D);
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
        for (size_t i = 0; i < mlp1_elems_per_block; ++i) tmp_w1[i] = fp32_to_bf16_host(W1[i]);
        std::copy(B1, B1 + 2ll*I, tmp_b1.begin());

        hip_bfloat16* dst_w1 = devw.d_w_mlp1 + ((size_t)l*E_local + ei) * mlp1_elems_per_block;
        float*        dst_b1 = devw.d_b_mlp1 + ((size_t)l*E_local + ei) * mlp1_rows_per_block;
        memcpyHtoD_chunked(dst_w1, tmp_w1.data(), tmp_w1.size() * sizeof(hip_bfloat16));
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
  size_t kv_cache_size = (size_t)L * seq * kv_dim_local * sizeof(float);
  printf("[GPU%d]   - KV cache (K): %.2f MB\n", r, kv_cache_size / (1024.0*1024.0));
  printf("[GPU%d]   - KV cache (V): %.2f MB\n", r, kv_cache_size / (1024.0*1024.0));
  HIP_CHECK(hipMalloc(&st.d_kv_k, kv_cache_size));
  HIP_CHECK(hipMalloc(&st.d_kv_v, kv_cache_size));

  // Allocate working buffers
  printf("[GPU%d] Allocating working buffers...\n", r);
  HIP_CHECK(hipMalloc(&st.d_x, D * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_t, D * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_qkv, devw.rows_per_layer_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_q_local, hs.q_count * hd * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_tb_local, hs.q_count * hd * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_tb2_partial, D * sizeof(float)));

  const int I = p->intermediate_dim;
  HIP_CHECK(hipMalloc(&st.d_mlp1_out, (size_t)(2*I) * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_gate_up,  I * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_ffn_out,  D * sizeof(float)));

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

  // Router temporaries and logits buffer (GPU0)
  HIP_CHECK(hipMalloc(&st.d_router, (size_t)p->n_experts * sizeof(float)));
  HIP_CHECK(hipMalloc(&st.d_topk,   (size_t)p->experts_per_token * sizeof(float)));
  if (r == 0) {
    HIP_CHECK(hipMalloc(&st.d_logits, (size_t)p->vocab_size * sizeof(float)));
  }
}

// ---------------------------------------------
// Warm-up
// ---------------------------------------------
void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;

  printf("[WARMUP] Starting GPU warmup process...\n");
  printf("[WARMUP] Model config: %d layers, %d hidden_dim, %d head_dim, %d experts\n", 
         p->n_layers, p->hidden_dim, p->head_dim, p->n_experts);
  fflush(stdout);

  if (NUM_GPUS < 4) {
    fprintf(stderr, "This implementation expects NUM_GPUS=4.\n");
    exit(EXIT_FAILURE);
  }

  // Print initial memory state for all GPUs
  printf("[WARMUP] Initial GPU memory state:\n");
  for (int r = 0; r < NUM_GPUS; ++r) {
    print_memory_usage(r, "Initial");
  }

  for (int r = 0; r < NUM_GPUS; ++r) {
    g_head_shard[r] = compute_head_shard(p, r);
    g_expert_shard[r] = compute_expert_shard(p, r);
    printf("[GPU%d] Head shard: Q[%d:%d], KV[%d:%d] | Expert shard: [%d:%d]\n",
           r, g_head_shard[r].q_start, g_head_shard[r].q_start + g_head_shard[r].q_count - 1,
           g_head_shard[r].kv_start, g_head_shard[r].kv_start + g_head_shard[r].kv_count - 1,
           g_expert_shard[r].e_start, g_expert_shard[r].e_start + g_expert_shard[r].e_count - 1);
  }

  h_tb2_sum = (float*)malloc(p->hidden_dim * sizeof(float));
  for (int r = 0; r < NUM_GPUS; ++r) {
    h_tb2_partials[r] = (float*)malloc(p->hidden_dim * sizeof(float));
  }

  const int L  = p->n_layers;
  const int D  = p->hidden_dim;
  const int hd = p->head_dim;
  const int seq = p->seq_len;

  printf("[WARMUP] Starting per-GPU weight loading...\n");
  fflush(stdout);

  for (int r = 0; r < NUM_GPUS; ++r) {
    printf("\n[GPU%d] ========== Starting weight loading ==========\n", r);
    HIP_CHECK(hipSetDevice(r));
    print_memory_usage(r, "Before allocation");
    gpu_copy_weights(transformer, r);
    gpu_malloc_run_state(transformer, r);
    print_memory_usage(r, "After weight loading completed");
    printf("[GPU%d] ========== Weight loading completed ==========\n\n", r);
    fflush(stdout);
  }

  printf("[warm_up] Sharded weights loaded across %d GPUs (chunk %zu MB)\n", NUM_GPUS, CHUNK_BYTES/1024/1024);
  fflush(stdout);
}

// ---------------------------------------------
// Finish
// ---------------------------------------------
void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (int r = 0; r < NUM_GPUS; ++r) {
    HIP_CHECK(hipSetDevice(r));
    auto& devw = g_weights[r];
    auto& st   = g_state[r];
    // Weights
    safeHipFree(devw.d_w_qkv_rows);
    safeHipFree(devw.d_b_qkv_rows);
    safeHipFree(devw.d_w_o_cols);
    safeHipFree(devw.d_w_mlp1);
    safeHipFree(devw.d_b_mlp1);
    safeHipFree(devw.d_w_mlp2);
    safeHipFree(devw.d_b_mlp2);
    safeHipFree(devw.d_w_router);
    safeHipFree(devw.d_b_router);
    safeHipFree(devw.d_rms_attn_w);
    safeHipFree(devw.d_rms_ffn_w);
    safeHipFree(devw.d_rms_out_w);
    safeHipFree(devw.d_attn_sinks);
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
    safeHipFree(st.d_mlp1_out);
    safeHipFree(st.d_gate_up);
    safeHipFree(st.d_ffn_out);
    safeHipFree(st.d_cos);
    safeHipFree(st.d_sin);
    safeHipFree(st.d_mask);
    safeHipFree(st.d_router);
    safeHipFree(st.d_topk);
    safeHipFree(st.d_logits);

  }
  for (int r = 0; r < NUM_GPUS; ++r) { free(h_tb2_partials[r]); h_tb2_partials[r] = nullptr; }
  free(h_tb2_sum); h_tb2_sum = nullptr;
}


// ---------------------------------------------
// One-token forward using TP + EP
// ---------------------------------------------
static int forward_one(Transformer* tr, Sampler* sampler, int token, int pos,
                       std::vector<float>& x,
                       std::vector<float>& t, std::vector<float>& tb2_sum) {
  Config* p = &tr->config;
  TransformerWeights* w = &tr->weights;

  const int L  = p->n_layers;
  const int D  = p->hidden_dim;
  const int hd = p->head_dim;
  const int kv_mul = p->n_attn_heads / p->n_kv_heads;
  const int I  = p->intermediate_dim;
  const int E  = p->n_experts;

  float* emb_row = w->token_embedding_table + token * D;
  std::copy(emb_row, emb_row + D, x.begin());

  for (int l = 0; l < L; ++l) {
    // Compute RMSNorm on GPU for each shard: t = RMSNorm(x, rms_attn_w[l])
    for (int r = 0; r < NUM_GPUS; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& devw = g_weights[r];
      auto& st   = g_state[r];
      HIP_CHECK(hipMemcpy(st.d_x, x.data(), D*sizeof(float), hipMemcpyHostToDevice));
      const float* w_attn = devw.d_rms_attn_w + (size_t)l * D;
      k_rmsnorm_vec<<<1,256>>>(st.d_x, w_attn, st.d_t, D);
      HIP_CHECK(hipGetLastError());
    }

    for (int r = 0; r < NUM_GPUS; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& devw = g_weights[r];
      auto& st   = g_state[r];
      int rows = devw.rows_per_layer_local;
      const hip_bfloat16* W = devw.d_w_qkv_rows + (size_t)l * devw.w_qkv_rows_stride;
      float* Y = st.d_qkv;
      dim3 grid(rows), block(256);
      k_matvec_bf16_fp32<<<grid, block>>>(W, st.d_t, Y, D, rows);
      HIP_CHECK(hipGetLastError());
      float* b = devw.d_b_qkv_rows + (size_t)l * rows;
      k_add_bias<<<(rows+255)/256, 256>>>(Y, b, rows);
      HIP_CHECK(hipGetLastError());
    }

    for (int r = 0; r < NUM_GPUS; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& hs = g_head_shard[r];
      auto& devw = g_weights[r];
      auto& st = g_state[r];
      // (AFTER) — đúng thứ tự: quay Q/K rồi mới lưu K vào KV-cache
      int q_rows = hs.q_count * hd;
      int kv_rows = hs.kv_count * hd;

      float* q_ptr = st.d_qkv;
      float* k_ptr = st.d_qkv + q_rows;
      float* v_ptr = st.d_qkv + q_rows + kv_rows;

      // 1) compute cos/sin on device for this position
      dim3 grid_cs(((p->head_dim/2) + 255)/256), block_cs(256);
      k_compute_cos_sin<<<grid_cs, block_cs>>>(
          pos,
          p->rope_theta,
          p->head_dim,
          p->rope_scaling_factor,
          p->initial_context_length,
          32.0f, 1.0f,
          st.d_cos,
          st.d_sin);
      HIP_CHECK(hipGetLastError());

      // 2) Quay RoPE cho Q và K (trên buffer q_ptr/k_ptr)
      dim3 grid_q(hs.q_count), block_q(hd/2);
      k_apply_rotary<<<grid_q, block_q>>>(q_ptr, st.d_cos, st.d_sin, hs.q_count, hd);
      HIP_CHECK(hipGetLastError());

      dim3 grid_k(hs.kv_count), block_k(hd/2);
      k_apply_rotary<<<grid_k, block_k>>>(k_ptr, st.d_cos, st.d_sin, hs.kv_count, hd);
      HIP_CHECK(hipGetLastError());

      // 3) Sau khi K đã quay, mới lưu K/V vào KV-cache
      int kv_dim_local = hs.kv_count * hd;
      float* k_cache_l = st.d_kv_k + (size_t)l * p->seq_len * kv_dim_local;
      float* v_cache_l = st.d_kv_v + (size_t)l * p->seq_len * kv_dim_local;
      HIP_CHECK(hipMemcpy(k_cache_l + (size_t)pos*kv_dim_local, k_ptr, kv_dim_local*sizeof(float), hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(v_cache_l + (size_t)pos*kv_dim_local, v_ptr, kv_dim_local*sizeof(float), hipMemcpyDeviceToDevice));

      // 4) Q local cho attention
      HIP_CHECK(hipMemcpy(st.d_q_local, q_ptr, q_rows*sizeof(float), hipMemcpyDeviceToDevice));
        
    
    }

    for (int r = 0; r < NUM_GPUS; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& hs = g_head_shard[r];
      auto& devw = g_weights[r];
      auto& st = g_state[r];
      int kv_dim_local = hs.kv_count * hd;
      float* k_cache_l = st.d_kv_k + (size_t)l * p->seq_len * kv_dim_local;
      float* v_cache_l = st.d_kv_v + (size_t)l * p->seq_len * kv_dim_local;

      const float* mask_row = nullptr;
      if (st.d_mask && (p->sliding_window > 0) && (l % 2 == 0)) {
        mask_row = st.d_mask + (size_t)pos * p->seq_len; // device pointer
      }
      const float* sinks_l = devw.d_attn_sinks ? (devw.d_attn_sinks + (size_t)l * g_head_shard[r].q_count) : nullptr;

      // shared memory needs (pos+1 + 1) floats
      size_t smem = (size_t)(pos + 2) * sizeof(float);
      dim3 grid(hs.q_count), block(hd);
      k_attention_step<<<grid, block, smem>>>(
          st.d_q_local, k_cache_l, v_cache_l, st.d_tb_local,
          pos+1, hd, hs.q_count, hs.kv_count, kv_mul,
          mask_row, sinks_l);
      HIP_CHECK(hipGetLastError());

    }

    for (int r = 0; r < NUM_GPUS; ++r) {
      HIP_CHECK(hipSetDevice(r));
      auto& devw = g_weights[r];
      auto& st = g_state[r];
      const hip_bfloat16* Wcols = devw.d_w_o_cols + (size_t)l * (size_t)D * devw.cols_per_layer_local;
      dim3 grid(D), block(256);
      k_matvec_colslice_bf16<<<grid, block>>>(Wcols, st.d_tb_local, st.d_tb2_partial, D, devw.cols_per_layer_local);
      HIP_CHECK(hipGetLastError());
    }

    for (int r = 0; r < NUM_GPUS; ++r) {
      HIP_CHECK(hipSetDevice(r));
      memcpyDtoH_chunked(h_tb2_partials[r], g_state[r].d_tb2_partial, D*sizeof(float));
    }
    for (int i = 0; i < D; ++i) {
      double s = 0.0;
      for (int r = 0; r < NUM_GPUS; ++r) s += h_tb2_partials[r][i];
      h_tb2_sum[i] = (float)s;
    }
    const float* b_o = tr->weights.b_o + 1ll*l*D;
    for (int i = 0; i < D; ++i) x[i] += (h_tb2_sum[i] + b_o[i]);

    // Compute RMSNorm for FFN on GPU0; get t back to host
    {
      HIP_CHECK(hipSetDevice(0));
      auto& devw0 = g_weights[0];
      auto& st0   = g_state[0];
      HIP_CHECK(hipMemcpy(st0.d_x, x.data(), D*sizeof(float), hipMemcpyHostToDevice));
      const float* w_ffn = devw0.d_rms_ffn_w + (size_t)l * D;
      k_rmsnorm_vec<<<1,256>>>(st0.d_x, w_ffn, st0.d_t, D);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(t.data(), st0.d_t, D*sizeof(float), hipMemcpyDeviceToHost));
    }

    // Router matvec on GPU0
    std::vector<float> router(E);
    {
      HIP_CHECK(hipSetDevice(0));
      auto& devw0 = g_weights[0];
      auto& st0   = g_state[0];
      const float* Wrouter_l = devw0.d_w_router + (size_t)l * (size_t)E * D;
      k_matvec_fp32<<<E, 256>>>(Wrouter_l, st0.d_t, st0.d_router, D, E);
      HIP_CHECK(hipGetLastError());
      const float* b_router_l = devw0.d_b_router + (size_t)l * E;
      k_add_bias<<<(E+255)/256, 256>>>(st0.d_router, b_router_l, E);
      HIP_CHECK(hipGetLastError());
      memcpyDtoH_chunked(router.data(), st0.d_router, E*sizeof(float));
    }

    int K = tr->config.experts_per_token;
    std::vector<int> idx(E); for (int i=0;i<E;++i) idx[i]=i;
    std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
                      [&](int a,int b){return router[a]>router[b];});
    std::vector<int> topk_idx(K);
    std::vector<float> topk_val(K);
    for (int i=0;i<K;++i){ topk_idx[i]=idx[i]; topk_val[i]=router[idx[i]]; }
    // Softmax over top-K on GPU0
    {
      HIP_CHECK(hipSetDevice(0));
      auto& st0 = g_state[0];
      HIP_CHECK(hipMemcpy(st0.d_topk, topk_val.data(), (size_t)K*sizeof(float), hipMemcpyHostToDevice));
      k_softmax_inplace<<<1,1>>>(st0.d_topk, K);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(topk_val.data(), st0.d_topk, (size_t)K*sizeof(float), hipMemcpyDeviceToHost));
    }

    std::vector<float> e_agg(D, 0.f);

    for (int ki = 0; ki < K; ++ki) {
      int e_glob = topk_idx[ki];
      float ew = topk_val[ki];
      int owner = -1;
      for (int r = 0; r < NUM_GPUS; ++r) {
        if (e_glob >= g_expert_shard[r].e_start &&
            e_glob <  g_expert_shard[r].e_start + g_expert_shard[r].e_count) { owner = r; break; }
      }
      if (owner < 0) { fprintf(stderr,"EP owner not found\n"); exit(EXIT_FAILURE); }
      int e_local = e_glob - g_expert_shard[owner].e_start;

      HIP_CHECK(hipSetDevice(owner));
      auto& devw = g_weights[owner];
      auto& st   = g_state[owner];
      HIP_CHECK(hipMemcpy(st.d_t, t.data(), D*sizeof(float), hipMemcpyHostToDevice));

      const int I = p->intermediate_dim;
      const hip_bfloat16* W1 = devw.d_w_mlp1 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)(2*I) * D;
      const float*        B1 = devw.d_b_mlp1 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)(2*I);

      dim3 g1((2*I), 1, 1), b1(256, 1, 1);
      k_matvec_bf16_fp32<<<g1, b1>>>(W1, st.d_t, st.d_mlp1_out, D, 2*I);
      HIP_CHECK(hipGetLastError());
      k_add_bias<<<((2*I)+255)/256, 256>>>(st.d_mlp1_out, B1, 2*I);
      HIP_CHECK(hipGetLastError());

      std::vector<float> h_mlp1(2*I);
      HIP_CHECK(hipMemcpy(h_mlp1.data(), st.d_mlp1_out, (size_t)2*I*sizeof(float), hipMemcpyDeviceToHost));
      std::vector<float> h_gate_up(I);
      const float clampU = tr->config.swiglu_limit;
      const float clampL = -tr->config.swiglu_limit;

      for (int i = 0; i < I; ++i) {
        float g = h_mlp1[2*i+0];
        float u = h_mlp1[2*i+1];
        // gate: only upper clamp (CPU ref)
        g = fminf(g, clampU);
        // up: symmetric clamp
        u = fminf(fmaxf(u, clampL), clampU);
        h_gate_up[i] = silu(g) * (u + 1.0f);
      }

      HIP_CHECK(hipMemcpy(st.d_gate_up, h_gate_up.data(), I*sizeof(float), hipMemcpyHostToDevice));

      const hip_bfloat16* W2 = devw.d_w_mlp2 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)D * I;
      const float*        B2 = devw.d_b_mlp2 + ((size_t)l* g_expert_shard[owner].e_count + e_local) * (size_t)D;

      dim3 g2(D, 1, 1), b2d(256, 1, 1);
      k_matvec_bf16_fp32<<<g2, b2d>>>(W2, st.d_gate_up, st.d_ffn_out, I, D);
      HIP_CHECK(hipGetLastError());
      k_add_bias<<<(D+255)/256, 256>>>(st.d_ffn_out, B2, D);
      HIP_CHECK(hipGetLastError());

      std::vector<float> h_ffn(D);
      memcpyDtoH_chunked(h_ffn.data(), st.d_ffn_out, D*sizeof(float));
      for (int i = 0; i < D; ++i) e_agg[i] += ew * h_ffn[i];
    }

    for (int i = 0; i < D; ++i) x[i] += e_agg[i];
  }

  // Final RMSNorm on GPU0
  {
    HIP_CHECK(hipSetDevice(0));
    auto& devw0 = g_weights[0];
    auto& st0   = g_state[0];
    HIP_CHECK(hipMemcpy(st0.d_x, x.data(), D*sizeof(float), hipMemcpyHostToDevice));
    k_rmsnorm_vec<<<1,256>>>(st0.d_x, devw0.d_rms_out_w, st0.d_x, D);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(x.data(), st0.d_x, D*sizeof(float), hipMemcpyDeviceToHost));
  }

  // Compute final logits on GPU0 and sample next token
  int next_token;
  {
    HIP_CHECK(hipSetDevice(0));
    auto& devw0 = g_weights[0];
    auto& st0   = g_state[0];
    int Dv = tr->config.hidden_dim;
    int V  = tr->config.vocab_size;
    HIP_CHECK(hipMemcpy(st0.d_t, x.data(), (size_t)Dv * sizeof(float), hipMemcpyHostToDevice));
    dim3 gL(V), bL(256);
    k_matvec_fp32<<<gL, bL>>>(devw0.d_w_out, st0.d_t, st0.d_logits, Dv, V);
    HIP_CHECK(hipGetLastError());
    std::vector<float> logits(V);
    memcpyDtoH_chunked(logits.data(), st0.d_logits, (size_t)V * sizeof(float));
    next_token = sample(sampler, logits.data());
  }

  return next_token;
}

// ---------------------------------------------
// Token generation for one request (GPU path)
// ---------------------------------------------
long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  const char *empty_prompt = "";
  if (input_seq == NULL) input_seq = empty_prompt;

  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  Config* p = &transformer->config;
  std::vector<float> x(p->hidden_dim, 0.f);
  std::vector<float> t(p->hidden_dim, 0.f);
  std::vector<float> tb2_sum(p->hidden_dim, 0.f);

  // RoPE cos/sin computed on device in forward_one

  int next;
  int token = prompt_tokens[0];
  int pos = 0;

  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece); fflush(stdout);

  while (pos < steps) {
    int sampled = forward_one(transformer, sampler, token, pos, x, t, tb2_sum);

    pos++;
    if (pos < num_prompt_tokens) {
      next = prompt_tokens[pos];
    } else {
      next = sampled;
      output_tokens[pos - num_prompt_tokens] = next;
    }
    if (next == 199999 || next == 200002) break;

    token = next;
  }
  printf("\n");

  output_tokens[pos - num_prompt_tokens + 1] = -1;
  free(prompt_tokens);
  return pos - num_prompt_tokens + 1;
}

// ---------------------------------------------
// Inference driver for getp_eval.cpp
// ---------------------------------------------
long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
