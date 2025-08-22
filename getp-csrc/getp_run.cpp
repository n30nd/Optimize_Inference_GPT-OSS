// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (MI250), BF16 weights/KV; FP32 accumulate.
// - Phase 0 correctness check via GETP_DEBUG=1.
// - Phase 1 single-GPU end-to-end.
//
// DEBUG USAGE:
// To enable detailed layer-by-layer comparison between GPU and CPU:
// 1. Set environment variable: export GETP_DEBUG=1
// 2. Run your program - it will print detailed comparisons for first 8 positions
// 3. Look for lines starting with [DBG] to see where differences occur
//
// Debug output format:
// [DBG] L{layer} pos={position} {stage}: max|Δ|={max_diff} (idx={index}) avg|Δ|={avg_diff}
// 
// Stages compared:
// - embedding: Token embedding lookup
// - rmsnorm_attn_out: After attention RMSNorm
// - qkv_out: After QKV projection + bias
// - q_after_rope, k_after_rope: After RoPE application
// - attn_output: After attention computation
// - x_after_attn: After attention residual connection  
// - router_scores: Router softmax output
// - topk_experts: Selected top-k experts
// - x_after_ffn: After FFN residual connection

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

// Helper to run CPU forward pass up to a specific layer
static void run_cpu_forward_to_layer(Transformer* cpu_transformer, int token, int pos, int target_layer) {
  Config* p = &cpu_transformer->config;
  TransformerWeights* w = &cpu_transformer->weights;
  RunState* s = &cpu_transformer->state;
  
  float* x = s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;
  
  // Copy token embedding
  float* content_row = w->token_embedding_table + token * hidden_dim;
  memcpy(x, content_row, hidden_dim * sizeof(*x));
  
  // Forward through layers up to target_layer
  for (int l = 0; l <= target_layer && l < p->n_layers; l++) {
    // Attention
    rmsnorm(s->t, x, w->rms_attn_w + (size_t)l * hidden_dim, hidden_dim);
    
    int loff = l * p->seq_len * kv_dim;
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;
    
    float* w_qkv = w->w_qkv + (size_t)l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    float* b_qkv = w->b_qkv + (size_t)l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    matmul(s->qkv, s->t, w_qkv, hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    
    for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
      s->qkv[i] += b_qkv[i];
    }
    
    memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float));
    memcpy(s->k, s->qkv + head_dim * p->n_attn_heads, head_dim * p->n_kv_heads * sizeof(float));
    memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads, head_dim * p->n_kv_heads * sizeof(float));
    
    // RoPE
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float* cos_vals = (float*)malloc((head_dim / 2) * sizeof(float));
    float* sin_vals = (float*)malloc((head_dim / 2) * sizeof(float));
    compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                    p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
    apply_rotary_emb(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);
    free(cos_vals);
    free(sin_vals);
    
    // Attention computation
    for (int h = 0; h < p->n_attn_heads; h++) {
      float* q = s->q + h * head_dim;
      float* att = s->att + h * p->seq_len;
      
      for (int t = 0; t <= pos; t++) {
        float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        double score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_dim);
        if (p->sliding_window > 0 && (l % 2 == 0)) {
          score += s->mask[pos * p->seq_len + t];
        }
        att[t] = score;
      }
      att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
      softmax(att, pos + 2);
      
      float* tb = s->tb + h * head_dim;
      memset(tb, 0, head_dim * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        float a = att[t];
        for (int i = 0; i < head_dim; i++) {
          tb[i] += a * v[i];
        }
      }
    }
    
    // Output projection
    float* w_o = w->w_o + (size_t)l * (head_dim * p->n_attn_heads) * hidden_dim;
    float* b_o = w->b_o + (size_t)l * hidden_dim;
    matmul(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
      s->tb2[i] += b_o[i];
    }
    
    // Residual
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->tb2[i];
    }
    
    // FFN
    rmsnorm(s->t, x, w->rms_ffn_w + (size_t)l * hidden_dim, hidden_dim);
    
    float* w_router = w->w_router + (size_t)l * hidden_dim * n_experts;
    float* b_router = w->b_router + (size_t)l * n_experts;
    matmul(s->router_score, s->t, w_router, hidden_dim, n_experts);
    for (int i = 0; i < n_experts; i++) {
      s->router_score[i] += b_router[i];
    }
    
    topk(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
    softmax(s->topk_v, p->experts_per_token);
    
    memset(s->e_agg, 0, hidden_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
      float expert_w = 0;
      int in_topk = 0;
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        if (s->topk_i[idx] == e) {
          in_topk = 1;
          expert_w = s->topk_v[idx];
          break;
        }
      }
      
      if (in_topk) {
        float* w_mlp1 = w->w_mlp1 + (size_t)(l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
        float* b_mlp1 = w->b_mlp1 + (size_t)(l * n_experts + e) * (2 * p->intermediate_dim);
        matmul(s->mlp1_out, s->t, w_mlp1, hidden_dim, 2 * p->intermediate_dim);
        for (int i = 0; i < 2 * p->intermediate_dim; i++) {
          s->mlp1_out[i] += b_mlp1[i];
        }
        
        for (int j = 0; j < p->intermediate_dim; j++) {
          s->gate[j] = s->mlp1_out[2 * j];
          s->up[j] = s->mlp1_out[2 * j + 1];
        }
        
        const float alpha = 1.702f;
        for (int i = 0; i < p->intermediate_dim; i++) {
          float val = s->gate[i];
          float up_val = s->up[i];
          if (val > p->swiglu_limit) val = p->swiglu_limit;
          if (up_val > p->swiglu_limit) up_val = p->swiglu_limit;
          if (up_val < -p->swiglu_limit) up_val = -p->swiglu_limit;
          val *= (1.0f / (1.0f + expf(-alpha * val)));
          val *= (up_val + 1.0f);
          s->gate_up[i] = val;
        }
        
        float* w_mlp2 = w->w_mlp2 + (size_t)(l * n_experts + e) * hidden_dim * p->intermediate_dim;
        float* b_mlp2 = w->b_mlp2 + (size_t)(l * n_experts + e) * hidden_dim;
        matmul(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
          s->tb2[i] += b_mlp2[i];
        }
        
        for (int i = 0; i < hidden_dim; i++) {
          s->e_agg[i] += s->tb2[i] * expert_w;
        }
      }
    }
    
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->e_agg[i];
    }
  }
}

// Simplified CPU forward for step-by-step comparison
static void run_cpu_step_by_step(Transformer* cpu_transformer, int token, int pos, int layer, const char* stage) {
  Config* p = &cpu_transformer->config;
  TransformerWeights* w = &cpu_transformer->weights;
  RunState* s = &cpu_transformer->state;
  
  float* x = s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int n_experts = p->n_experts;
  
  // Initialize with embedding if this is the start
  if (layer == 0 && strcmp(stage, "after_rmsnorm_attn") == 0) {
    float* content_row = w->token_embedding_table + token * hidden_dim;
    memcpy(x, content_row, hidden_dim * sizeof(*x));
  }
  
  // Run layers up to current layer
  for (int l = 0; l < layer; l++) {
    // Full layer forward pass (simplified)
    run_cpu_forward_to_layer(cpu_transformer, token, pos, l);
    x = s->x; // Update x pointer
  }
  
  // Run specific stage for current layer
  if (strcmp(stage, "after_rmsnorm_attn") == 0) {
    rmsnorm(s->t, x, w->rms_attn_w + (size_t)layer * hidden_dim, hidden_dim);
  }
  else if (strcmp(stage, "after_qkv") == 0) {
    // Run rmsnorm first
    rmsnorm(s->t, x, w->rms_attn_w + (size_t)layer * hidden_dim, hidden_dim);
    
    // Then QKV
    float* w_qkv = w->w_qkv + (size_t)layer * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    float* b_qkv = w->b_qkv + (size_t)layer * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    matmul(s->qkv, s->t, w_qkv, hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
      s->qkv[i] += b_qkv[i];
    }
  }
  // Add more stages as needed...
}

// ------------------------- Debug Helper Functions -------------------------

static void print_debug_status() {
  if (g.debug) {
    printf("[DBG] Layer-by-layer GPU vs CPU comparison ENABLED\n");
    printf("[DBG] Will compare intermediate results for positions 0-7\n");
    printf("[DBG] Look for lines starting with [DBG] to see differences\n");
    printf("[DBG] Large max|Δ| values indicate potential issues\n");
    fflush(stdout);
  }
}

static void compare_vectors(const char* name, float* gpu_result, float* cpu_result, 
                           int size, int pos = -1, int layer = -1) {
  // Add bounds checking
  if (size <= 0 || size > 100000) {
    printf("[DBG] WARNING: Invalid size for %s: %d\n", name, size);
    return;
  }
  if (!gpu_result || !cpu_result) {
    printf("[DBG] WARNING: Null pointer for %s\n", name);
    return;
  }
  
  double max_diff = 0.0;
  double avg_diff = 0.0;
  int max_idx = 0;
  
  for (int i = 0; i < size; i++) {
    // Check for invalid values
    if (!isfinite(gpu_result[i]) || !isfinite(cpu_result[i])) {
      printf("[DBG] WARNING: Non-finite value in %s at idx=%d: GPU=%.6g CPU=%.6g\n", 
             name, i, (double)gpu_result[i], (double)cpu_result[i]);
      continue;
    }
    
    double diff = fabs((double)gpu_result[i] - (double)cpu_result[i]);
    if (diff > max_diff) {
      max_diff = diff;
      max_idx = i;
    }
    avg_diff += diff;
  }
  avg_diff /= size;
  
  if (layer >= 0 && pos >= 0) {
    printf("[DBG] L%d pos=%d %s: max|Δ|=%.6g (idx=%d) avg|Δ|=%.6g GPU[%d]=%.6g CPU[%d]=%.6g\n", 
           layer, pos, name, max_diff, max_idx, avg_diff, 
           max_idx, (double)gpu_result[max_idx], max_idx, (double)cpu_result[max_idx]);
  } else if (pos >= 0) {
    printf("[DBG] pos=%d %s: max|Δ|=%.6g (idx=%d) avg|Δ|=%.6g GPU[%d]=%.6g CPU[%d]=%.6g\n", 
           pos, name, max_diff, max_idx, avg_diff,
           max_idx, (double)gpu_result[max_idx], max_idx, (double)cpu_result[max_idx]);
  } else {
    printf("[DBG] %s: max|Δ|=%.6g (idx=%d) avg|Δ|=%.6g GPU[%d]=%.6g CPU[%d]=%.6g\n", 
           name, max_diff, max_idx, avg_diff,
           max_idx, (double)gpu_result[max_idx], max_idx, (double)cpu_result[max_idx]);
  }
  fflush(stdout);
}

static void debug_compare_intermediate_results(Transformer* cpu_transformer, int token, int pos, int layer, const char* stage) {
  if (!g.debug || pos >= 8) return;
  
  Config* p = &cpu_transformer->config;
  const int hidden = p->hidden_dim;
  const int nh = p->n_attn_heads;
  const int nkv = p->n_kv_heads; 
  const int d = p->head_dim;
  const int qkv_out = d * (nh + 2*nkv);
  
  // Simplified: just run full CPU forward to this layer instead of step-by-step
  run_cpu_forward_to_layer(cpu_transformer, token, pos, layer);
  RunState* cpu_s = &cpu_transformer->state;
  
  if (strcmp(stage, "after_rmsnorm_attn") == 0) {
    std::vector<float> gpu_t(hidden);
    HIP_CHECK(hipMemcpy(gpu_t.data(), g.S.t, sizeof(float)*hidden, hipMemcpyDeviceToHost));
    compare_vectors("rmsnorm_attn_out", gpu_t.data(), cpu_s->t, hidden, pos, layer);
  }
  else if (strcmp(stage, "after_qkv") == 0) {
    std::vector<float> gpu_qkv(qkv_out);
    HIP_CHECK(hipMemcpy(gpu_qkv.data(), g.S.qkv, sizeof(float)*qkv_out, hipMemcpyDeviceToHost));
    compare_vectors("qkv_out", gpu_qkv.data(), cpu_s->qkv, qkv_out, pos, layer);
  }
  else if (strcmp(stage, "after_rope") == 0) {
    int q_size = nh * d;
    int k_size = nkv * d;
    
    // Add bounds checking
    if (q_size > 10000 || k_size > 10000) {
      printf("[DBG] WARNING: Skipping RoPE comparison - sizes too large: q=%d k=%d\n", q_size, k_size);
      return;
    }
    
    std::vector<float> gpu_q(q_size), gpu_k(k_size);
    HIP_CHECK(hipMemcpy(gpu_q.data(), g.S.q, sizeof(float)*q_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_k.data(), g.S.k, sizeof(float)*k_size, hipMemcpyDeviceToHost));
    
    // Only compare first 100 elements to avoid overwhelming output
    int compare_size_q = std::min(q_size, 100);
    int compare_size_k = std::min(k_size, 100);
    
    compare_vectors("q_after_rope", gpu_q.data(), cpu_s->q, compare_size_q, pos, layer);
    compare_vectors("k_after_rope", gpu_k.data(), cpu_s->k, compare_size_k, pos, layer);
  }
  else if (strcmp(stage, "after_attn") == 0) {
    int tb_size = nh * d;
    if (tb_size > 10000) {
      printf("[DBG] WARNING: Skipping attention comparison - size too large: %d\n", tb_size);
      return;
    }
    std::vector<float> gpu_tb(tb_size);
    HIP_CHECK(hipMemcpy(gpu_tb.data(), g.S.tb, sizeof(float)*tb_size, hipMemcpyDeviceToHost));
    compare_vectors("attn_output", gpu_tb.data(), cpu_s->tb, std::min(tb_size, 100), pos, layer);
  }
  else if (strcmp(stage, "after_router") == 0) {
    std::vector<float> gpu_router(p->n_experts);
    HIP_CHECK(hipMemcpy(gpu_router.data(), g.S.router_score, sizeof(float)*p->n_experts, hipMemcpyDeviceToHost));
    compare_vectors("router_scores", gpu_router.data(), cpu_s->router_score, p->n_experts, pos, layer);
    
    std::vector<float> gpu_topk_v(p->experts_per_token);
    std::vector<int> gpu_topk_i(p->experts_per_token);
    HIP_CHECK(hipMemcpy(gpu_topk_v.data(), g.S.topk_v, sizeof(float)*p->experts_per_token, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_topk_i.data(), g.S.topk_i, sizeof(int)*p->experts_per_token, hipMemcpyDeviceToHost));
    
    printf("[DBG] L%d pos=%d topk_experts: GPU=[", layer, pos);
    for (int i = 0; i < p->experts_per_token; i++) {
      printf("%d:%.4f", gpu_topk_i[i], gpu_topk_v[i]);
      if (i < p->experts_per_token-1) printf(", ");
    }
    printf("] CPU=[");
    for (int i = 0; i < p->experts_per_token; i++) {
      printf("%d:%.4f", cpu_s->topk_i[i], cpu_s->topk_v[i]);
      if (i < p->experts_per_token-1) printf(", ");
    }
    printf("]\n");
    fflush(stdout);
  }
}

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

  // Debug: Compare embedding with CPU
  if (g.debug && pos == 0) {
    run_cpu_forward_to_layer(transformer, token, pos, -1); // Just embedding
    std::vector<float> gpu_embedding(hidden);
    HIP_CHECK(hipMemcpy(gpu_embedding.data(), g.S.x, sizeof(float)*hidden, hipMemcpyDeviceToHost));
    compare_vectors("embedding", gpu_embedding.data(), transformer->state.x, hidden, pos);
  }

  // For all layers
  for (int L = 0; L < p->n_layers; ++L) {
    // t = RMSNorm_attn(x)
    k_rmsnorm<<<1, 256, 256*sizeof(float)>>>(g.S.t, g.S.x,
      g.W.rms_attn_w + (size_t)L * hidden, hidden, BF16_EPS);

    // Debug: Compare RMSNorm output
    if (g.debug && pos < 2) { // Only first 2 positions to be safe
      debug_compare_intermediate_results(transformer, token, pos, L, "after_rmsnorm_attn");
    }

    // qkv = W_qkv[L] @ t + b_qkv[L]
    blk = dim3(256); grid = dim3((qkv_out + blk.x - 1)/blk.x);
    k_matvec_bf16<<<grid, blk>>>(g.W.w_qkv + (size_t)L * (size_t)qkv_out * (size_t)hidden,
                                 g.S.t, g.S.qkv, hidden, qkv_out);
    k_vec_add_bias<<<grid, blk>>>(g.S.qkv, g.W.b_qkv + (size_t)L * (size_t)qkv_out, qkv_out);

    // Debug: Compare QKV output  
    if (g.debug && pos < 2) {
      debug_compare_intermediate_results(transformer, token, pos, L, "after_qkv");
    }

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

    // Debug: Compare Q,K after RoPE - TEMPORARILY DISABLED TO AVOID SEGFAULT
    if (g.debug && pos == 0 && L == 0) { // Only pos=0, layer=0 to be extra safe
      printf("[DBG] L%d pos=%d RoPE debug temporarily limited to avoid segfault\n", L, pos);
      // debug_compare_intermediate_results(transformer, token, pos, L, "after_rope");
    }

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

    // Debug: Compare attention output - TEMPORARILY DISABLED
    // debug_compare_intermediate_results(transformer, token, pos, L, "after_attn");

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

    // Debug: Compare attention output with CPU  
    if (g.debug && pos < 2 && L == 0) {
      run_cpu_forward_to_layer(transformer, token, pos, L);
      std::vector<float> gpu_x_after_attn(hidden);
      HIP_CHECK(hipMemcpy(gpu_x_after_attn.data(), g.S.x, sizeof(float)*hidden, hipMemcpyDeviceToHost));
      compare_vectors("x_after_attn", gpu_x_after_attn.data(), transformer->state.x, std::min(hidden, 100), pos, L);
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

    // Debug: Compare router output - TEMPORARILY DISABLED
    // debug_compare_intermediate_results(transformer, token, pos, L, "after_router");

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

    // Debug: Compare final layer output with CPU - DISABLED
    // if (g.debug && pos < 8) {
    //   run_cpu_forward_to_layer(transformer, token, pos, L);
    //   std::vector<float> gpu_x_after_ffn(hidden);
    //   HIP_CHECK(hipMemcpy(gpu_x_after_ffn.data(), g.S.x, sizeof(float)*hidden, hipMemcpyDeviceToHost));
    //   compare_vectors("x_after_ffn", gpu_x_after_ffn.data(), transformer->state.x, hidden, pos, L);
    // }
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
  print_debug_status();
  fflush(stdout);
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (!g.initialized) warm_up(transformer, tokenizer);
  (void)tokenizer;

  long long total_out = 0;
  const int S = transformer->config.seq_len;

  if (g.debug) {
    printf("[DBG] Starting inference with detailed GPU vs CPU comparison\n");
    printf("[DBG] This will be slower due to CPU reference computations\n");
    fflush(stdout);
  }

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
        printf("[DBG] pos=%d FINAL_LOGITS: max|Δ|=%.6g\n", pos, maxdiff);
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
  auto dfree = [](void* p){ if (p) { (void)hipFree(p); } };
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
