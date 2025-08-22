// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (MI250), BF16 weights/KV; FP32 accumulate.
// - Phase 0 correctness check via GETP_DEBUG=1.
// - Phase 1 single-GPU end-to-end.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#ifndef GETP_RUN
#define GETP_RUN

// ------------------------- Utils / BF16 -------------------------

#ifndef BF16_EPS
#define BF16_EPS 1e-5f
#endif

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d : %s\n", #x, __FILE__, __LINE__, hipGetErrorString(_e)); \
  abort(); } } while(0)

static inline uint16_t fp32_to_bf16_host(float x) {
  union { uint32_t u; float f; } v; v.f = x;
  uint32_t lsb = (v.u >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  uint16_t bf = (uint16_t)((v.u + rounding_bias) >> 16);
  return bf;
}
static inline float bf16_to_fp32_host(uint16_t h) {
  union { uint32_t u; float f; } v; v.u = ((uint32_t)h) << 16;
  return v.f;
}

__device__ __forceinline__ float bf16_to_fp32(uint16_t h) {
  uint32_t u = ((uint32_t)h) << 16;
  float f = __int_as_float((int)u);
  return f;
}

__device__ __forceinline__ uint16_t fp32_to_bf16(float x) {
  uint32_t u = __float_as_int(x);
  uint32_t lsb = (u >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  return (uint16_t)((u + rounding_bias) >> 16);
}

// ------------------------- GPU Context -------------------------

struct GPUWeightsBF16 {
  uint16_t* token_embedding_table; // [vocab, hidden]
  uint16_t* rms_attn_w;            // [layers, hidden]
  uint16_t* rms_ffn_w;             // [layers, hidden]
  uint16_t* w_qkv;                 // [layers, qkv_out, hidden]
  uint16_t* w_o;                   // [layers, hidden, q_out]
  uint16_t* b_qkv;                 // [layers, qkv_out]
  uint16_t* b_o;                   // [layers, hidden]
  float*    attn_sinks;            // [layers, n_heads] keep f32
  uint16_t* w_router;              // [layers, hidden, n_experts]
  uint16_t* b_router;              // [layers, n_experts]
  uint16_t* w_mlp1;                // [layers, n_experts, 2*inter, hidden]
  uint16_t* w_mlp2;                // [layers, n_experts, hidden, inter]
  uint16_t* b_mlp1;                // [layers, n_experts, 2*inter]
  uint16_t* b_mlp2;                // [layers, n_experts, hidden]
  uint16_t* rms_out_w;             // [hidden]
  uint16_t* out;                   // [vocab, hidden]
};

struct GPUState {
  // FP32 working buffers
  float* x;        // [hidden]
  float* t;        // [hidden]
  float* tb;       // [n_heads*head_dim]
  float* tb2;      // [hidden]
  float* router_score; // [n_experts]
  float* topk_v;       // [experts_per_token]
  int*   topk_i;       // [experts_per_token]
  float* mlp1_out;     // [2*inter]
  float* gate;         // [inter]
  float* up;           // [inter]
  float* gate_up;      // [inter]
  float* e_agg;        // [hidden]
  float* qkv;          // [qkv_out]
  float* q;            // [n_heads*head_dim]
  float* k;            // [n_kv_heads*head_dim]
  float* v;            // [n_kv_heads*head_dim]
  float* att;          // [n_heads, seq_len]
  float* logits;       // [vocab]
  float* mask;         // [seq_len, seq_len]
  float* d_cos;        // [head_dim/2]
  float* d_sin;        // [head_dim/2]
  // KV cache as BF16
  uint16_t* key_cache;   // [layers, seq_len, kv_dim]
  uint16_t* value_cache; // [layers, seq_len, kv_dim]
};

struct GPUContext {
  bool initialized = false;
  bool debug = false;
  Transformer* t = nullptr;
  GPUWeightsBF16 W{};
  GPUState S{};
};

static GPUContext g;

// ------------------------- Kernels: small helpers -------------------------

__global__ void k_vec_gather_row_bf16(float* __restrict__ out,
                                      const uint16_t* __restrict__ table,
                                      int row, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int off = row * cols;
  if (i < cols) out[i] = bf16_to_fp32(table[off + i]);
}

__global__ void k_vec_add_inplace(float* __restrict__ a,
                                  const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void k_vec_add_bias(float* __restrict__ x, const uint16_t* __restrict__ b_bf16, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += bf16_to_fp32(b_bf16[i]);
}

__global__ void k_vec_add_bias_f32(float* __restrict__ x, const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += b[i];
}

__global__ void k_zero(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 0.0f;
}

__global__ void k_zero_i(int* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 0;
}

__global__ void k_cast_f32_to_bf16(uint16_t* __restrict__ dst, const float* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = fp32_to_bf16(src[i]);
}

__global__ void k_cast_bf16_to_f32(float* __restrict__ dst, const uint16_t* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bf16_to_fp32(src[i]);
}

// ------------------------- RMSNorm -------------------------

__global__ void k_rmsnorm(float* __restrict__ o, const float* __restrict__ x,
                          const uint16_t* __restrict__ scale_bf16,
                          int n, float eps) {
  extern __shared__ float smem[];
  float v = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float xi = x[i];
    v += xi * xi;
  }
  smem[threadIdx.x] = v;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float ss = smem[0];
  float scale = rsqrtf(ss / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float g = bf16_to_fp32(scale_bf16[i]);
    o[i] = g * (x[i] * scale);
  }
}

// ------------------------- MatVec (W[out,in] @ x[in] -> y[out]) -------------------------

__global__ void k_matvec_bf16(const uint16_t* __restrict__ W,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int in, int out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) return;
  const uint16_t* row = W + (size_t)o * (size_t)in;
  float acc = 0.f;
  int i = 0;
  for (; i + 3 < in; i += 4) {
    acc += bf16_to_fp32(row[i+0]) * x[i+0];
    acc += bf16_to_fp32(row[i+1]) * x[i+1];
    acc += bf16_to_fp32(row[i+2]) * x[i+2];
    acc += bf16_to_fp32(row[i+3]) * x[i+3];
  }
  for (; i < in; ++i) acc += bf16_to_fp32(row[i]) * x[i];
  y[o] = acc;
}

// ------------------------- RoPE apply (Q & K) -------------------------

__global__ void k_apply_rope(float* __restrict__ x,
                             const float* __restrict__ cos,
                             const float* __restrict__ sin,
                             int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (h >= n_heads || i >= half) return;
  float x1 = x[h * head_dim + i];
  float x2 = x[h * head_dim + half + i];
  float c = cos[i];
  float s = sin[i];
  x[h * head_dim + i]         = x1 * c - x2 * s;
  x[h * head_dim + half + i]  = x2 * c + x1 * s;
}

// ------------------------- Attention (scores & context) -------------------------

__global__ void k_attn_scores(float* __restrict__ att, // [n_heads, seq_len]
                              const float* __restrict__ q, // [n_heads, d]
                              const uint16_t* __restrict__ key_cache, // bf16
                              int layer, int pos, int seq_len,
                              int n_heads, int n_kv_heads, int head_dim) {
  int h = blockIdx.x;
  int t = blockIdx.y;
  int i = threadIdx.x;
  if (h >= n_heads || t > pos || i >= head_dim) return;

  int kv_mul = n_heads / n_kv_heads;
  int g = h / kv_mul;
  int kv_dim = n_kv_heads * head_dim;

  size_t layer_off = (size_t)layer * (size_t)(seq_len * kv_dim);
  size_t t_off = (size_t)t * (size_t)kv_dim + (size_t)g * (size_t)head_dim;

  float qi = q[h * head_dim + i];
  float ki = bf16_to_fp32(key_cache[layer_off + t_off + i]);
  float v = qi * ki;

  extern __shared__ float ssum[];
  ssum[i] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (i < s) ssum[i] += ssum[i + s];
    __syncthreads();
  }
  if (i == 0) {
    float score = ssum[0] / sqrtf((float)head_dim);
    att[h * seq_len + t] = score;
  }
}

__global__ void k_attn_apply_mask_and_sink(float* __restrict__ att, // [n_heads, seq_len]
                                           const float* __restrict__ mask, // [seq_len, seq_len]
                                           const float* __restrict__ sinks, // [n_heads]
                                           int pos, int seq_len, int n_heads,
                                           int apply_mask) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  for (int t = tid; t <= pos; t += blockDim.x) {
    if (apply_mask) att[h * seq_len + t] += mask[pos * seq_len + t];
  }
  __syncthreads();
  if (tid == 0) {
    att[h * seq_len + (pos + 1)] = sinks[h];
  }
}

__global__ void k_softmax_len(float* __restrict__ att, int pos, int seq_len) {
  int h = blockIdx.x;
  float maxv = -INFINITY;
  for (int t = 0; t <= pos + 1; ++t) {
    float v = att[h * seq_len + t];
    if (v > maxv) maxv = v;
  }
  float sum = 0.f;
  for (int t = 0; t <= pos + 1; ++t) {
    float v = expf(att[h * seq_len + t] - maxv);
    att[h * seq_len + t] = v;
    sum += v;
  }
  float inv = 1.0f / sum;
  for (int t = 0; t <= pos + 1; ++t) {
    att[h * seq_len + t] *= inv;
  }
}

__global__ void k_attn_context(float* __restrict__ tb, // [n_heads*head_dim]
                               const float* __restrict__ att, // [n_heads, seq_len]
                               const uint16_t* __restrict__ value_cache,
                               int layer, int pos, int seq_len,
                               int n_heads, int n_kv_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  if (h >= n_heads || i >= head_dim) return;
  int kv_mul = n_heads / n_kv_heads;
  int g = h / kv_mul;
  int kv_dim = n_kv_heads * head_dim;
  size_t layer_off = (size_t)layer * (size_t)(seq_len * kv_dim);

  float acc = 0.f;
  for (int t = 0; t <= pos; ++t) {
    size_t t_off = (size_t)t * (size_t)kv_dim + (size_t)g * (size_t)head_dim;
    float vt = bf16_to_fp32(value_cache[layer_off + t_off + i]);
    float w = att[h * seq_len + t];
    acc += w * vt;
  }
  tb[h * head_dim + i] = acc;
}

__global__ void k_write_kv(uint16_t* __restrict__ key_cache,
                           uint16_t* __restrict__ value_cache,
                           const float* __restrict__ k, // [n_kv_heads*head_dim]
                           const float* __restrict__ v, // [n_kv_heads*head_dim]
                           int layer, int pos, int seq_len,
                           int n_kv_heads, int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_dim = n_kv_heads * head_dim;
  if (i >= kv_dim) return;
  size_t layer_off = (size_t)layer * (size_t)(seq_len * kv_dim);
  size_t off = layer_off + (size_t)pos * (size_t)kv_dim + (size_t)i;
  key_cache[off]   = fp32_to_bf16(k[i]);
  value_cache[off] = fp32_to_bf16(v[i]);
}

// ------------------------- Router (softmax + topk) -------------------------

// Simpler single-thread version (n_experts nhỏ) -> đủ nhanh, rõ ràng.
__global__ void k_softmax_f32(float* x, int n) {
  if (threadIdx.x == 0) {
    float maxv = -INFINITY;
    for (int i = 0; i < n; ++i) maxv = fmaxf(maxv, x[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    float inv = 1.0f / fmaxf(sum, 1e-30f);
    for (int i = 0; i < n; ++i) x[i] *= inv;
  }
}

__global__ void k_topk(const float* __restrict__ x, int n, int k,
                       float* __restrict__ out_v, int* __restrict__ out_i) {
  if (threadIdx.x == 0) {
    for (int j = 0; j < k; ++j) { out_v[j] = -INFINITY; out_i[j] = -1; }
    for (int i = 0; i < n; ++i) {
      float v = x[i];
      int idx = i;
      for (int j = 0; j < k; ++j) {
        if (v > out_v[j]) {
          for (int s = k-1; s > j; --s) { out_v[s] = out_v[s-1]; out_i[s] = out_i[s-1]; }
          out_v[j] = v; out_i[j] = idx;
          break;
        }
      }
    }
  }
}

// ------------------------- SwiGLU -------------------------

__global__ void k_swiglu(float* __restrict__ gate_up,
                         const float* __restrict__ gate,
                         const float* __restrict__ up,
                         int n, float limit, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  float u = up[i];
  if (g > limit) g = limit;
  if (u > limit) u = limit;
  if (u < -limit) u = -limit;
  g *= (1.0f / (1.0f + expf(-alpha * g)));
  gate_up[i] = (u + 1.0f) * g;
}

// ------------------------- AXPY -------------------------
__global__ void k_axpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

// ------------------------- Warm-up / Weights copy -------------------------

static void build_mask_and_alloc_kv(const Config* p) {
  const int S = p->seq_len;
  std::vector<float> hmask((size_t)S * (size_t)S, 0.f);
  if (p->sliding_window > 0) {
    int W = p->sliding_window;
    for (int r = 0; r < S; ++r) {
      for (int c = 0; c < S; ++c) {
        if (c < r - W) hmask[r * S + c] = -INFINITY;
      }
    }
  }
  HIP_CHECK(hipMalloc(&g.S.mask, sizeof(float)*S*S));
  HIP_CHECK(hipMemcpy(g.S.mask, hmask.data(), sizeof(float)*S*S, hipMemcpyHostToDevice));

  const int kv_dim = p->head_dim * p->n_kv_heads;
  size_t kv_size = (size_t)p->n_layers * (size_t)p->seq_len * (size_t)kv_dim;
  HIP_CHECK(hipMalloc(&g.S.key_cache,   sizeof(uint16_t)*kv_size));
  HIP_CHECK(hipMalloc(&g.S.value_cache, sizeof(uint16_t)*kv_size));
  HIP_CHECK(hipMemset(g.S.key_cache, 0, kv_size * sizeof(uint16_t)));
  HIP_CHECK(hipMemset(g.S.value_cache, 0, kv_size * sizeof(uint16_t)));
}

template<typename T>
static void dalloc(T** ptr, size_t n) { HIP_CHECK(hipMalloc(ptr, n)); }

static void alloc_state(const Config* p) {
  const int hidden = p->hidden_dim;
  const int d = p->head_dim;
  const int nh = p->n_attn_heads;
  const int S = p->seq_len;
  const int inter = p->intermediate_dim;
  const int kv_dim = p->head_dim * p->n_kv_heads;

  dalloc(&g.S.x,          sizeof(float)*hidden);
  dalloc(&g.S.t,          sizeof(float)*hidden);
  dalloc(&g.S.tb,         sizeof(float)*nh*d);
  dalloc(&g.S.tb2,        sizeof(float)*hidden);
  dalloc(&g.S.router_score, sizeof(float)*p->n_experts);
  dalloc(&g.S.topk_v,     sizeof(float)*p->experts_per_token);
  dalloc(&g.S.topk_i,     sizeof(int)*p->experts_per_token);
  dalloc(&g.S.mlp1_out,   sizeof(float)*2*inter);
  dalloc(&g.S.gate,       sizeof(float)*inter);
  dalloc(&g.S.up,         sizeof(float)*inter);
  dalloc(&g.S.gate_up,    sizeof(float)*inter);
  dalloc(&g.S.e_agg,      sizeof(float)*hidden);
  dalloc(&g.S.qkv,        sizeof(float)*(p->head_dim*(p->n_attn_heads + 2*p->n_kv_heads)));
  dalloc(&g.S.q,          sizeof(float)*nh*d);
  dalloc(&g.S.k,          sizeof(float)*kv_dim);
  dalloc(&g.S.v,          sizeof(float)*kv_dim);
  dalloc(&g.S.att,        sizeof(float)*nh*S);
  dalloc(&g.S.logits,     sizeof(float)*p->vocab_size);
  dalloc(&g.S.d_cos,      sizeof(float)*(d/2));
  dalloc(&g.S.d_sin,      sizeof(float)*(d/2));
}

static void to_bf16_and_copy(uint16_t** dst_d, const float* src_h, size_t n) {
  std::vector<uint16_t> tmp(n);
  for (size_t i=0;i<n;++i) tmp[i] = fp32_to_bf16_host(src_h[i]);
  HIP_CHECK(hipMalloc(dst_d, n*sizeof(uint16_t)));
  HIP_CHECK(hipMemcpy(*dst_d, tmp.data(), n*sizeof(uint16_t), hipMemcpyHostToDevice));
}

static void copy_weights_bf16(const Transformer* t) {
  const Config* p = &t->config;
  const TransformerWeights* w = &t->weights;

  to_bf16_and_copy(&g.W.token_embedding_table,
                   w->token_embedding_table,
                   (size_t)p->vocab_size * (size_t)p->hidden_dim);

  to_bf16_and_copy(&g.W.rms_attn_w, w->rms_attn_w,
                   (size_t)p->n_layers * (size_t)p->hidden_dim);
  to_bf16_and_copy(&g.W.rms_ffn_w,  w->rms_ffn_w,
                   (size_t)p->n_layers * (size_t)p->hidden_dim);

  const size_t qkv_out = (size_t)p->head_dim * (size_t)(p->n_attn_heads + 2*p->n_kv_heads);
  to_bf16_and_copy(&g.W.w_qkv, w->w_qkv, (size_t)p->n_layers * qkv_out * (size_t)p->hidden_dim);
  to_bf16_and_copy(&g.W.b_qkv, w->b_qkv, (size_t)p->n_layers * qkv_out);

  to_bf16_and_copy(&g.W.w_o, w->w_o, (size_t)p->n_layers * (size_t)p->hidden_dim * (size_t)(p->n_attn_heads * p->head_dim));
  to_bf16_and_copy(&g.W.b_o, w->b_o, (size_t)p->n_layers * (size_t)p->hidden_dim);

  HIP_CHECK(hipMalloc(&g.W.attn_sinks, sizeof(float)*(size_t)p->n_layers*(size_t)p->n_attn_heads));
  HIP_CHECK(hipMemcpy(g.W.attn_sinks, w->attn_sinks,
                      sizeof(float)*(size_t)p->n_layers*(size_t)p->n_attn_heads,
                      hipMemcpyHostToDevice));

  to_bf16_and_copy(&g.W.w_router, w->w_router, (size_t)p->n_layers*(size_t)p->hidden_dim*(size_t)p->n_experts);
  to_bf16_and_copy(&g.W.b_router, w->b_router, (size_t)p->n_layers*(size_t)p->n_experts);

  to_bf16_and_copy(&g.W.w_mlp1, w->w_mlp1, (size_t)p->n_layers*(size_t)p->n_experts*(size_t)(2*p->intermediate_dim)*(size_t)p->hidden_dim);
  to_bf16_and_copy(&g.W.b_mlp1, w->b_mlp1, (size_t)p->n_layers*(size_t)p->n_experts*(size_t)(2*p->intermediate_dim));
  to_bf16_and_copy(&g.W.w_mlp2, w->w_mlp2, (size_t)p->n_layers*(size_t)p->n_experts*(size_t)p->hidden_dim*(size_t)p->intermediate_dim);
  to_bf16_and_copy(&g.W.b_mlp2, w->b_mlp2, (size_t)p->n_layers*(size_t)p->n_experts*(size_t)p->hidden_dim);

  to_bf16_and_copy(&g.W.rms_out_w, w->rms_out_w, (size_t)p->hidden_dim);
  to_bf16_and_copy(&g.W.out,       w->out,       (size_t)p->vocab_size*(size_t)p->hidden_dim);
}

// Forward decl của CPU helper trong run.cpp (đúng chữ ký).
void compute_cos_sin(int pos,
                     float base, int head_dim, float scaling_factor,
                     float initial_context_length, float ntk_beta,
                     float ntk_alpha,
                     float* cos_out, float* sin_out);

// ------------------------- One-token GPU forward -------------------------

static float* gpu_forward(Transformer* transformer, int token, int pos) {
  Config* p = &transformer->config;
  const int hidden = p->hidden_dim;
  const int d = p->head_dim;
  const int nh = p->n_attn_heads;
  const int nkv = p->n_kv_heads;
  const int qkv_out = d * (nh + 2*nkv);
  const int S = p->seq_len;

  dim3 grid, blk;

  // x = embedding[token]
  blk = dim3(256);
  grid = dim3((hidden + blk.x - 1) / blk.x);
  k_vec_gather_row_bf16<<<grid, blk>>>(g.S.x, g.W.token_embedding_table, token, hidden);

  // For all layers
  for (int L = 0; L < p->n_layers; ++L) {
    // t = RMSNorm_attn(x)
    k_rmsnorm<<<1, 256, 256*sizeof(float)>>>(g.S.t, g.S.x,
      g.W.rms_attn_w + (size_t)L * hidden, hidden, BF16_EPS);

    // qkv = W_qkv[L] @ t + b_qkv[L]
    blk = dim3(256); grid = dim3((qkv_out + blk.x - 1)/blk.x);
    k_matvec_bf16<<<grid, blk>>>(g.W.w_qkv + (size_t)L * (size_t)qkv_out * (size_t)hidden,
                                 g.S.t, g.S.qkv, hidden, qkv_out);
    k_vec_add_bias<<<grid, blk>>>(g.S.qkv, g.W.b_qkv + (size_t)L * (size_t)qkv_out, qkv_out);

    // split q,k,v
    HIP_CHECK(hipMemcpy(g.S.q, g.S.qkv, sizeof(float)*nh*d, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(g.S.k, g.S.qkv + nh*d, sizeof(float)*nkv*d, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(g.S.v, g.S.qkv + nh*d + nkv*d, sizeof(float)*nkv*d, hipMemcpyDeviceToDevice));

    // RoPE cos/sin từ CPU helper (bé) -> copy xuống GPU
    std::vector<float> hcos(d/2), hsin(d/2);
    compute_cos_sin(pos, p->rope_theta, d, p->rope_scaling_factor,
                    (float)p->initial_context_length, /*ntk_beta*/32.0f, /*ntk_alpha*/1.0f,
                    hcos.data(), hsin.data());
    HIP_CHECK(hipMemcpy(g.S.d_cos, hcos.data(), sizeof(float)*(d/2), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g.S.d_sin, hsin.data(), sizeof(float)*(d/2), hipMemcpyHostToDevice));

    // apply_rope(q), apply_rope(k)
    k_apply_rope<<<nh, d/2>>>(g.S.q, g.S.d_cos, g.S.d_sin, nh, d);
    k_apply_rope<<<nkv, d/2>>>(g.S.k, g.S.d_cos, g.S.d_sin, nkv, d);

    // write KV cache at (L, pos)
    blk = dim3(256); grid = dim3(((nkv*d) + blk.x - 1)/blk.x);
    k_write_kv<<<grid, blk>>>(g.S.key_cache, g.S.value_cache, g.S.k, g.S.v,
                               L, pos, S, nkv, d);

    // Attention scores over 0..pos for each head
    {
      dim3 grid2(nh, std::max(1, pos+1));
      dim3 blk2(d);
      size_t sh = sizeof(float) * d;
      k_attn_scores<<<grid2, blk2, sh>>>(g.S.att, g.S.q, g.S.key_cache,
                                         L, pos, S, nh, nkv, d);
    }

    // mask (even layers only) + sink then softmax (len=pos+2)
    {
      int apply_mask = (p->sliding_window > 0 && (L % 2 == 0)) ? 1 : 0;
      dim3 gridh(nh); dim3 blkm(256);
      const float* sinksL = g.W.attn_sinks + (size_t)L * (size_t)nh;
      k_attn_apply_mask_and_sink<<<gridh, blkm>>>(g.S.att, g.S.mask, sinksL,
                                                  pos, S, nh, apply_mask);
      k_softmax_len<<<gridh, 1>>>(g.S.att, pos, S);
    }

    // context: tb[h,i]
    {
      dim3 gridh(nh), blkhd(d);
      k_attn_context<<<gridh, blkhd>>>(g.S.tb, g.S.att, g.S.value_cache,
                                       L, pos, S, nh, nkv, d);
    }

    // proj: tb2 = W_o[L] @ tb + b_o[L]
    {
      int outN = hidden;
      int inN = nh * d;
      blk = dim3(256); grid = dim3((outN + blk.x - 1)/blk.x);
      const uint16_t* W = g.W.w_o + (size_t)L * (size_t)hidden * (size_t)(nh*d);
      k_matvec_bf16<<<grid, blk>>>(W, g.S.tb, g.S.tb2, inN, outN);
      k_vec_add_bias<<<grid, blk>>>(g.S.tb2, g.W.b_o + (size_t)L * (size_t)hidden, outN);
    }

    // residual x += tb2
    {
      dim3 gr((hidden+255)/256), bl(256);
      k_vec_add_inplace<<<gr, bl>>>(g.S.x, g.S.tb2, hidden);
    }

    // FFN: t = RMSNorm_ffn(x)
    k_rmsnorm<<<1, 256, 256*sizeof(float)>>>(g.S.t, g.S.x,
      g.W.rms_ffn_w + (size_t)L*hidden, hidden, BF16_EPS);

    // router_score = W_router[L] @ t + b_router[L]
    {
      int outN = p->n_experts, inN = hidden;
      blk = dim3(256); grid = dim3((outN + blk.x - 1)/blk.x);
      const uint16_t* W = g.W.w_router + (size_t)L * (size_t)hidden * (size_t)outN;
      k_matvec_bf16<<<grid, blk>>>(W, g.S.t, g.S.router_score, inN, outN);
      k_vec_add_bias<<<grid, blk>>>(g.S.router_score, g.W.b_router + (size_t)L * (size_t)outN, outN);
      k_softmax_f32<<<1, 1>>>(g.S.router_score, outN);
      k_topk<<<1,1>>>(g.S.router_score, outN, p->experts_per_token, g.S.topk_v, g.S.topk_i);
      k_zero<<<(hidden+255)/256,256>>>(g.S.e_agg, hidden);
    }

    // Aggregate top-k experts
    for (int eidx = 0; eidx < p->experts_per_token; ++eidx) {
      int ei; HIP_CHECK(hipMemcpy(&ei, g.S.topk_i + eidx, sizeof(int), hipMemcpyDeviceToHost));
      if (ei < 0) continue;
      float ew; HIP_CHECK(hipMemcpy(&ew, g.S.topk_v + eidx, sizeof(float), hipMemcpyDeviceToHost));

      // mlp1_out = W_mlp1[L,ei] @ t + b_mlp1[L,ei]  -> [2*inter]
      {
        int outN = 2*p->intermediate_dim, inN = hidden;
        const uint16_t* W = g.W.w_mlp1
          + (size_t)(L * p->n_experts + ei) * (size_t)(2*p->intermediate_dim) * (size_t)hidden;
        const uint16_t* B = g.W.b_mlp1
          + (size_t)(L * p->n_experts + ei) * (size_t)(2*p->intermediate_dim);
        blk = dim3(256); grid = dim3((outN + blk.x - 1)/blk.x);
        k_matvec_bf16<<<grid, blk>>>(W, g.S.t, g.S.mlp1_out, inN, outN);
        k_vec_add_bias<<<grid, blk>>>(g.S.mlp1_out, B, outN);
      }
      // split gate/up
      HIP_CHECK(hipMemcpy(g.S.gate, g.S.mlp1_out, sizeof(float)*p->intermediate_dim, hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(g.S.up,   g.S.mlp1_out + p->intermediate_dim, sizeof(float)*p->intermediate_dim, hipMemcpyDeviceToDevice));

      // SwiGLU
      k_swiglu<<<(p->intermediate_dim+255)/256,256>>>(g.S.gate_up, g.S.gate, g.S.up, p->intermediate_dim,
                                                      p->swiglu_limit, 1.702f);

      // tb2 = W_mlp2[L,ei] @ gate_up + b_mlp2[L,ei] -> [hidden]
      {
        int outN = hidden, inN = p->intermediate_dim;
        const uint16_t* W = g.W.w_mlp2
          + (size_t)(L * p->n_experts + ei) * (size_t)hidden * (size_t)p->intermediate_dim;
        const uint16_t* B = g.W.b_mlp2
          + (size_t)(L * p->n_experts + ei) * (size_t)hidden;
        blk = dim3(256); grid = dim3((outN + blk.x - 1)/blk.x);
        k_matvec_bf16<<<grid, blk>>>(W, g.S.gate_up, g.S.tb2, inN, outN);
        k_vec_add_bias<<<grid, blk>>>(g.S.tb2, B, outN);
      }
      // e_agg += tb2 * ew
      dim3 b2(256), g2((hidden+255)/256);
      k_axpy<<<g2,b2>>>(g.S.e_agg, g.S.tb2, ew, hidden);
    }

    // x += e_agg
    {
      dim3 b(256), gdim((hidden+255)/256);
      k_vec_add_inplace<<<gdim,b>>>(g.S.x, g.S.e_agg, hidden);
    }
  }

  // final rmsnorm
  k_rmsnorm<<<1,256,256*sizeof(float)>>>(g.S.x, g.S.x, g.W.rms_out_w, hidden, BF16_EPS);

  // logits = W_out @ x  (out: vocab)
  {
    int outN = p->vocab_size, inN = hidden;
    dim3 b(256), gdim((outN + b.x - 1)/b.x);
    k_matvec_bf16<<<gdim, b>>>(g.W.out, g.S.x, g.S.logits, inN, outN);
  }
  return g.S.logits; // device ptr
}

// ------------------------- Exposed API -------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  (void)tokenizer;
  g.t = transformer;
  g.debug = (getenv("GETP_DEBUG") && atoi(getenv("GETP_DEBUG")) != 0);

  alloc_state(&transformer->config);
  build_mask_and_alloc_kv(&transformer->config);
  copy_weights_bf16(transformer);

  g.initialized = true;
  fprintf(stdout, "[GPU] Warm-up done. BF16 weights loaded to VRAM.\n");
  fflush(stdout);
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (!g.initialized) warm_up(transformer, tokenizer);
  (void)tokenizer;

  long long total_out = 0;
  const int S = transformer->config.seq_len;

  for (int ridx = 0; ridx < requests->num_reqs; ++ridx) {
    const char* input = get_str_req_ptr(requests, ridx);
    int* out_tokens   = get_tok_gen_ptr(requests, ridx);

    std::vector<int> prompt_tokens(strlen(input) + 3);
    int n_prompt = 0;
    encode(tokenizer, input, /*bos=*/1, /*eos=*/0,
           prompt_tokens.data(), &n_prompt,
           transformer->config.initial_context_length);

    int pos = 0;
    int next = 0;
    int token = (n_prompt > 0) ? prompt_tokens[0] : 0;

    std::vector<float> hlogits(transformer->config.vocab_size);

    for (int step = 0; step < requests->max_seq_len && pos < S-1; ++step) {
      float* d_logits = gpu_forward(transformer, token, pos);
      HIP_CHECK(hipMemcpy(hlogits.data(), d_logits,
                          sizeof(float)*hlogits.size(), hipMemcpyDeviceToHost));

      if (g.debug && pos < 8) {
        float* cpu_logits = forward(transformer, token, pos);
        double maxdiff = 0.0;
        for (size_t i=0;i<hlogits.size();++i) {
          double d = fabs((double)hlogits[i] - (double)cpu_logits[i]);
          if (d > maxdiff) maxdiff = d;
        }
        fprintf(stdout, "[DBG] pos=%d max|Δ|=%.6g\n", pos, maxdiff);
        fflush(stdout);
      }

      next = sample(sampler, hlogits.data());
      out_tokens[step] = next;
      total_out++;

      pos++;
      if (pos < n_prompt) token = prompt_tokens[pos];
      else token = next;

      if (pos >= n_prompt && next == 2) break; // <|end|>
    }
  }
  return total_out;
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  (void)transformer; (void)tokenizer;
  auto dfree = [](void* p){ if (p) hipFree(p); };
  dfree(g.W.token_embedding_table);
  dfree(g.W.rms_attn_w);
  dfree(g.W.rms_ffn_w);
  dfree(g.W.w_qkv);
  dfree(g.W.w_o);
  dfree(g.W.b_qkv);
  dfree(g.W.b_o);
  dfree(g.W.attn_sinks);
  dfree(g.W.w_router);
  dfree(g.W.b_router);
  dfree(g.W.w_mlp1);
  dfree(g.W.w_mlp2);
  dfree(g.W.b_mlp1);
  dfree(g.W.b_mlp2);
  dfree(g.W.rms_out_w);
  dfree(g.W.out);

  dfree(g.S.x); dfree(g.S.t); dfree(g.S.tb); dfree(g.S.tb2);
  dfree(g.S.router_score); dfree(g.S.topk_v); dfree(g.S.topk_i);
  dfree(g.S.mlp1_out); dfree(g.S.gate); dfree(g.S.up);
  dfree(g.S.gate_up); dfree(g.S.e_agg);
  dfree(g.S.qkv); dfree(g.S.q); dfree(g.S.k); dfree(g.S.v);
  dfree(g.S.att); dfree(g.S.logits); dfree(g.S.mask);
  dfree(g.S.d_cos); dfree(g.S.d_sin);
  dfree(g.S.key_cache); dfree(g.S.value_cache);

  g.initialized = false;
  fprintf(stdout, "[GPU] Finish: freed VRAM.\n");
  fflush(stdout);
}

#endif // GETP_RUN
