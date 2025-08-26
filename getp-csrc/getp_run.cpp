// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode, now with BATCH support.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (e.g., MI250), BF16 weights/KV; FP32 accumulate.
// - Warm-up loads weights with chunked copy+convert to avoid OOM.
// - Runtime fixes: bound pos to seq_len; guard attention sink write.
// - Batch-aware run state & kernels (grid.y = batch index).
// - MoE top-k on host per-sample; attention batched on device.
//
// Batch size:
//   - Set env GETP_MAX_BATCH (default 8). KV cache scales with max_batch.

#include "getp_eval.cpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <limits>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>

#ifndef GETP_RUN
#define GETP_RUN

#ifndef GB
#define GB (1024.0*1024.0*1024.0)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HIP_CHECK(cmd) \
  do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

static inline int getenv_int(const char* key, int defv) {
  const char* v = getenv(key);
  if (!v) return defv;
  int x = atoi(v);
  return x > 0 ? x : defv;
}

// ------------------------- GPU weights & run state -------------------------

typedef struct {
  hip_bfloat16 *token_embedding_table;
  hip_bfloat16 *out;
  hip_bfloat16 *rms_attn_w;
  hip_bfloat16 *rms_ffn_w;
  hip_bfloat16 *rms_out_w;
  hip_bfloat16 *w_qkv;
  hip_bfloat16 *b_qkv;
  hip_bfloat16 *w_o;
  hip_bfloat16 *b_o;
  hip_bfloat16 *attn_sinks;
  hip_bfloat16 *w_router;
  hip_bfloat16 *b_router;
  hip_bfloat16 *w_mlp1;
  hip_bfloat16 *w_mlp2;
  hip_bfloat16 *b_mlp1;
  hip_bfloat16 *b_mlp2;
} GPUTransformerWeights;

typedef struct {
  int batch; // max batch
  // Per-batch main buffers (leading dim = batch)
  hip_bfloat16 *x;        // [B, hidden]
  hip_bfloat16 *t;        // [B, hidden]
  hip_bfloat16 *tb;       // [B, n_heads*head_dim]
  hip_bfloat16 *tb2;      // [B, hidden]
  hip_bfloat16 *router_score; // [B, n_experts]
  hip_bfloat16 *topk_v;   // [B, experts_per_token]
  int          *topk_i;   // [B, experts_per_token]
  hip_bfloat16 *mlp1_out; // [B, 2*intermediate]
  hip_bfloat16 *gate;     // [B, intermediate]
  hip_bfloat16 *up;       // [B, intermediate]
  hip_bfloat16 *gate_up;  // [B, intermediate]
  hip_bfloat16 *e_agg;    // [B, hidden]
  hip_bfloat16 *qkv;      // [B, (n_heads + 2*n_kv_heads)*head_dim]
  hip_bfloat16 *q;        // [B, n_heads*head_dim]
  hip_bfloat16 *k;        // [B, n_kv_heads*head_dim]
  hip_bfloat16 *v;        // [B, n_kv_heads*head_dim]
  hip_bfloat16 *att;      // [B, n_attn_heads*seq_len]
  hip_bfloat16 *logits;   // [B, vocab] (unused — kept for compat)

  // KV caches per sample
  hip_bfloat16 *key_cache;   // [B, n_layers, seq_len, kv_dim]
  hip_bfloat16 *value_cache; // [B, n_layers, seq_len, kv_dim]

  // Shared (batch-invariant) mask SxS if sliding window enabled
  hip_bfloat16 *mask; // [S,S] or NULL

  // RoPE cos/sin per sample
  hip_bfloat16 *cos_vals; // [B, head_dim/2]
  hip_bfloat16 *sin_vals; // [B, head_dim/2]

  // FP32 buffers
  float *acc_buffer;   // [B, hidden]
  float *logits_fp32;  // [B, vocab]
} GPURunState;

// Per-GPU device context
typedef struct {
  int device_id;
  int max_batch;
  GPUTransformerWeights weights;
  GPURunState state;
} GPUContext;

static std::vector<GPUContext> gpu_contexts;
static thread_local GPUContext* t_ctx = nullptr;

// ------------------------- BF16 utils -------------------------

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
}

// ------------------------- Kernels (batched) -------------------------

// FP32 -> BF16 (device), contiguous
__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
}
// BF16 -> FP32 (device), contiguous
__global__ void convert_bf16_to_fp32_kernel(const hip_bfloat16 *bf16_data, float *fp32_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) fp32_data[tid] = bfloat16_to_float(bf16_data[tid]);
}

// Strided array convert: per-sample (grid.y = b)
__global__ void convert_bf16_to_fp32_strided(const hip_bfloat16 *in, float *out,
                                             int size, int in_stride, int out_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < size) {
    out[b * out_stride + i] = bfloat16_to_float(in[b * in_stride + i]);
  }
}

// RMSNorm batched: grid.x spans feature, grid.y spans batch
__global__ void rmsnorm_kernel_bf16_batch(hip_bfloat16 *o, const hip_bfloat16 *x,
                                          const hip_bfloat16 *weight,
                                          int size, int out_stride, int in_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // feature idx
  int b = blockIdx.y;                             // batch idx
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    const hip_bfloat16* xb = x + b * in_stride;
    for (int j = 0; j < size; j++) {
      float xv = bfloat16_to_float(xb[j]);
      ss += xv * xv;
    }
    ss = ss / size + 1e-5f;
    shared_ss = 1.0f / sqrtf(ss);
  }
  __syncthreads();
  if (i < size) {
    const hip_bfloat16 *xb = x + b * in_stride;
    hip_bfloat16 *ob       = o + b * out_stride;
    float xv = bfloat16_to_float(xb[i]);
    float wv = bfloat16_to_float(weight[i]);
    ob[i] = float_to_bfloat16(wv * (shared_ss * xv));
  }
}

// y = W * x  (W: [d,n], x: [B,n], y: [B,d])
__global__ void matmul_kernel_bf16_batch(hip_bfloat16 *y, const hip_bfloat16 *x, const hip_bfloat16 *w,
                                         int n, int d, int y_stride, int x_stride) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; // 0..d-1
  int b   = blockIdx.y;
  if (row < d) {
    float acc = 0.0f;
    const hip_bfloat16 *xb = x + b * x_stride;
    const hip_bfloat16 *wr = w + (size_t)row * (size_t)n;
    for (int j = 0; j < n; j++) {
      acc += bfloat16_to_float(wr[j]) * bfloat16_to_float(xb[j]);
    }
    (y + b * y_stride)[row] = float_to_bfloat16(acc);
  }
}

__global__ void add_bias_kernel_bf16_batch(hip_bfloat16 *y, const hip_bfloat16 *bias,
                                           int size, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < size) {
    hip_bfloat16 *yb = y + b * y_stride;
    float ov = bfloat16_to_float(yb[i]);
    float bv = bfloat16_to_float(bias[i]);
    yb[i] = float_to_bfloat16(ov + bv);
  }
}

__global__ void residual_add_kernel_bf16_batch(hip_bfloat16 *x, const hip_bfloat16 *residual,
                                               int size, int x_stride, int r_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < size) {
    hip_bfloat16 *xb = x + b * x_stride;
    const hip_bfloat16 *rb = residual + b * r_stride;
    float xv = bfloat16_to_float(xb[i]);
    float rv = bfloat16_to_float(rb[i]);
    xb[i] = float_to_bfloat16(xv + rv);
  }
}

// RoPE cos/sin per sample using pos[b]
__global__ void compute_cos_sin_kernel_bf16_batch(hip_bfloat16 *cos_out, hip_bfloat16 *sin_out,
                                                  const int *pos_b,
                                                  float base, int head_dim, float scaling_factor,
                                                  float initial_context_length, float ntk_beta, float ntk_alpha,
                                                  int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..head_dim/2-1
  int b = blockIdx.y;
  int d_half = head_dim / 2;
  if (i < d_half) {
    float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
    float inv_freq;
    if (scaling_factor > 1.0f) {
      float low  = d_half * logf(initial_context_length / (ntk_beta  * 2.0f * M_PI)) / logf(base);
      float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
      float interpolation = 1.0f / (scaling_factor * freq);
      float extrapolation = 1.0f / freq;
      float ramp = ((float)i - low) / (high - low);
      if (ramp < 0) ramp = 0;
      if (ramp > 1) ramp = 1;
      float mask = 1.0f - ramp;
      inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
    } else {
      inv_freq = 1.0f / freq;
    }
    float concentration = (scaling_factor > 1.0f) ? (0.1f * logf(scaling_factor) + 1.0f) : 1.0f;
    float val = (float)pos_b[b] * inv_freq;
    hip_bfloat16 *cb = cos_out + b * stride;
    hip_bfloat16 *sb = sin_out + b * stride;
    cb[i] = float_to_bfloat16(cosf(val) * concentration);
    sb[i] = float_to_bfloat16(sinf(val) * concentration);
  }
}

__global__ void apply_rotary_emb_kernel_bf16_batch(hip_bfloat16 *x,
                                                   const hip_bfloat16 *cos, const hip_bfloat16 *sin,
                                                   int n_heads, int head_dim, int x_stride, int cos_stride) {
  int h = blockIdx.x;             // head idx
  int b = blockIdx.y;             // batch idx
  int i = threadIdx.x;            // half-dim idx
  int half = head_dim / 2;
  if (h < n_heads && i < half) {
    hip_bfloat16 *xb = x + b * x_stride + h * head_dim;
    const hip_bfloat16 *cb = cos + b * cos_stride;
    const hip_bfloat16 *sb = sin + b * cos_stride;
    float x1 = bfloat16_to_float(xb[i]);
    float x2 = bfloat16_to_float(xb[half + i]);
    float c  = bfloat16_to_float(cb[i]);
    float s  = bfloat16_to_float(sb[i]);
    xb[i]          = float_to_bfloat16(x1 * c - x2 * s);
    xb[half + i]   = float_to_bfloat16(x2 * c + x1 * s);
  }
}

// Scores per head; att: [B, seq_len] for this head
__global__ void attention_scores_kernel_bf16_batch(hip_bfloat16 *att,
                                                   const hip_bfloat16 *q,
                                                   const hip_bfloat16 *key_cache,
                                                   const hip_bfloat16 *mask, // [S,S] or NULL
                                                   const int *pos_b,
                                                   int head_dim, int kv_dim, int kv_mul,
                                                   int h, int loff, int seq_len,
                                                   int att_stride, int q_stride,
                                                   int cache_batch_stride) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; // time
  int b = blockIdx.y;
  int pos = pos_b[b];
  if (t <= pos) {
    const hip_bfloat16 *qb = q + b * q_stride + (size_t)h * (size_t)head_dim;
    const hip_bfloat16 *kb = key_cache + b * cache_batch_stride;
    const hip_bfloat16 *k = kb + loff + (size_t)t * (size_t)kv_dim + (size_t)(h / kv_mul) * (size_t)head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += bfloat16_to_float(qb[i]) * bfloat16_to_float(k[i]);
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      float mv = bfloat16_to_float(mask[pos * seq_len + t]);
      score += mv;
    }
    (att + b * att_stride)[t] = float_to_bfloat16(score);
  }
}

__global__ void attention_weighted_sum_kernel_bf16_batch(hip_bfloat16 *tb,
                                                         const hip_bfloat16 *att,
                                                         const hip_bfloat16 *value_cache,
                                                         const int *pos_b,
                                                         int head_dim, int kv_dim, int kv_mul,
                                                         int h, int loff,
                                                         int tb_stride, int att_stride,
                                                         int cache_batch_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // head feature idx
  int b = blockIdx.y;
  int pos = pos_b[b];
  if (i < head_dim) {
    const hip_bfloat16 *vb = value_cache + b * cache_batch_stride;
    const hip_bfloat16 *attb = att + b * att_stride;
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const hip_bfloat16 *v = vb + loff + (size_t)t * (size_t)kv_dim + (size_t)(h / kv_mul) * (size_t)head_dim;
      sum += bfloat16_to_float(attb[t]) * bfloat16_to_float(v[i]);
    }
    (tb + b * tb_stride)[(size_t)h * (size_t)head_dim + i] = float_to_bfloat16(sum);
  }
}

// Split gate/up batched
__global__ void split_gate_up_kernel_bf16_batch(hip_bfloat16 *gate, hip_bfloat16 *up,
                                                const hip_bfloat16 *mlp1_out, int intermediate_dim,
                                                int g_stride, int u_stride, int m_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < intermediate_dim) {
    (gate + b * g_stride)[i] = (mlp1_out + b * m_stride)[2 * i];
    (up   + b * u_stride)[i] = (mlp1_out + b * m_stride)[2 * i + 1];
  }
}

__global__ void swiglu_kernel_bf16_batch(hip_bfloat16 *gate_up,
                                         const hip_bfloat16 *gate, const hip_bfloat16 *up,
                                         int intermediate_dim, float swiglu_limit,
                                         int gu_stride, int g_stride, int u_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < intermediate_dim) {
    float val   = bfloat16_to_float((gate + b * g_stride)[i]);
    float upval = bfloat16_to_float((up   + b * u_stride)[i]);
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    (gate_up + b * gu_stride)[i] = float_to_bfloat16(val);
  }
}

__global__ void weighted_sum_kernel_bf16_batch(hip_bfloat16 *e_agg, const hip_bfloat16 *tb2,
                                               float expert_w, int hidden_dim,
                                               int agg_stride, int tb2_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < hidden_dim) {
    float agg = bfloat16_to_float((e_agg + b * agg_stride)[i]);
    float add = bfloat16_to_float((tb2   + b * tb2_stride)[i]);
    (e_agg + b * agg_stride)[i] = float_to_bfloat16(agg + add * expert_w);
  }
}

// Final output: W(BF16)*x(FP32)->logits(FP32), batched
__global__ void matvec_wbf16_xfp32_outfp32_batch(float *out, const float *x,
                                                 const hip_bfloat16 *w, int n, int d,
                                                 int out_stride, int x_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row 0..d-1
  int b = blockIdx.y;
  if (i < d) {
    float sum = 0.0f;
    const float *xb = x + b * x_stride;
    const hip_bfloat16 *wr = w + (size_t)i * (size_t)n;
    for (int j = 0; j < n; ++j) sum += (float)wr[j] * xb[j];
    (out + b * out_stride)[i] = sum;
  }
}

// Serial softmax per-sample per-head for stability
__global__ void softmax_serial_kernel_bf16(hip_bfloat16 *x, int size) {
  __shared__ float max_val;
  __shared__ float sum_val;
  if (threadIdx.x == 0) {
    if (size <= 0) return;
    max_val = bfloat16_to_float(x[0]);
    for (int i = 1; i < size; i++) {
      float v = bfloat16_to_float(x[i]);
      if (v > max_val) max_val = v;
    }
    sum_val = 0.0f;
    for (int i = 0; i < size; i++) {
      float v = expf(bfloat16_to_float(x[i]) - max_val);
      x[i] = float_to_bfloat16(v);
      sum_val += v;
    }
    float inv = 1.0f / sum_val;
    for (int i = 0; i < size; i++) {
      float v = bfloat16_to_float(x[i]) * inv;
      x[i] = float_to_bfloat16(v);
    }
  }
}

// ------------------------- CPU top-k -------------------------

void topk_cpu(float *topk_values, int *topk_indices, const float *router_score, int num_experts, int experts_per_token) {
  struct Pair { float value; int index; };
  std::vector<Pair> pairs(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    pairs[i].value = router_score[i];
    pairs[i].index = i;
  }
  // partial selection for k
  for (int i = 0; i < experts_per_token; i++) {
    int max_idx = i;
    for (int j = i + 1; j < num_experts; j++) {
      if (pairs[j].value > pairs[max_idx].value) max_idx = j;
    }
    if (max_idx != i) std::swap(pairs[i], pairs[max_idx]);
    topk_values[i]  = pairs[i].value;
    topk_indices[i] = pairs[i].index;
  }
}

// ------------------------- Alloc/Free run state -------------------------
void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch);
void gpu_free_run_state(GPURunState *s);

// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
void gpu_free_weights(GPUTransformerWeights *gpu_w);

// ------------------------- Forward (batched) -------------------------

// Returns host pointer to [B_active, vocab] FP32 logits (thread-local, reused).
float *gpu_forward_batch(Transformer *transformer,
                         const int *tokens_host, // [B_active]
                         const int *pos_host,    // [B_active]
                         int B_active,
                         GPUContext *gpu_ctx) {
  Config *p = &transformer->config;
  const int B = B_active;
  const int head_dim         = p->head_dim;
  const int hidden_dim       = p->hidden_dim;
  const int kv_dim           = p->head_dim * p->n_kv_heads;
  const int kv_mul           = p->n_attn_heads / p->n_kv_heads;
  const int intermediate_dim = p->intermediate_dim;
  const int n_experts        = p->n_experts;
  const int S                = p->seq_len;

  // strides (leading dimensions)
  const int stride_hidden = hidden_dim;
  const int stride_heads  = head_dim * p->n_attn_heads;
  const int stride_qkv    = head_dim * (p->n_attn_heads + 2 * p->n_kv_heads);
  const int stride_kv     = head_dim * p->n_kv_heads;
  const int stride_att    = p->n_attn_heads * S;
  const int stride_inter  = intermediate_dim;
  const int stride_expk   = p->experts_per_token;
  const int cache_batch_stride = (size_t)p->n_layers * (size_t)S * (size_t)kv_dim;

  hipStream_t stream = nullptr; // default stream

  // Device scratch for positions
  static thread_local int *d_pos = nullptr;
  static thread_local int  d_pos_cap = 0;
  if (d_pos_cap < B) {
    if (d_pos) HIP_CHECK(hipFree(d_pos));
    HIP_CHECK(hipMalloc(&d_pos, B * sizeof(int)));
    d_pos_cap = B;
  }
  HIP_CHECK(hipMemcpy(d_pos, pos_host, B * sizeof(int), hipMemcpyHostToDevice));

  dim3 block256(256);
  dim3 block128(128);

  // 1) Token embedding lookup: x[b] = emb[token_b], in BF16
  for (int b = 0; b < B; ++b) {
    int tok = tokens_host[b];
    hip_bfloat16 *x_b = gpu_ctx->state.x + (size_t)b * (size_t)stride_hidden;
    const hip_bfloat16 *row = gpu_ctx->weights.token_embedding_table + (size_t)tok * (size_t)hidden_dim;
    HIP_CHECK(hipMemcpy(x_b, row, hidden_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
  }

  for (int l = 0; l < p->n_layers; l++) {
    // 2) Attention RMSNorm: t = RMS(x)
    {
      dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
      const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
      rmsnorm_kernel_bf16_batch<<<grid_hidden, block256>>>(gpu_ctx->state.t, gpu_ctx->state.x, rms_attn_w,
                                                           hidden_dim, stride_hidden, stride_hidden);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // 3) QKV projection (BF16)
    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    {
      const hip_bfloat16 *w_qkv = gpu_ctx->weights.w_qkv + (size_t)l * (size_t)hidden_dim * (size_t)qkv_out_dim;
      const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l * (size_t)qkv_out_dim;
      dim3 grid_qkv((qkv_out_dim + block256.x - 1) / block256.x, B);
      matmul_kernel_bf16_batch<<<grid_qkv, block256>>>(gpu_ctx->state.qkv, gpu_ctx->state.t, w_qkv,
                                                       hidden_dim, qkv_out_dim, stride_qkv, stride_hidden);
      add_bias_kernel_bf16_batch<<<grid_qkv, block256>>>(gpu_ctx->state.qkv, b_qkv, qkv_out_dim, stride_qkv);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // 4) Split Q,K,V and store K/V into cache at pos[b]
    {
      const int q_size = head_dim * p->n_attn_heads;
      const int k_size = head_dim * p->n_kv_heads;
      const int v_size = head_dim * p->n_kv_heads;
      for (int b = 0; b < B; ++b) {
        hip_bfloat16 *qb = gpu_ctx->state.q + (size_t)b * (size_t)q_size;
        hip_bfloat16 *kb = gpu_ctx->state.k + (size_t)b * (size_t)k_size;
        hip_bfloat16 *vb = gpu_ctx->state.v + (size_t)b * (size_t)v_size;
        hip_bfloat16 *qkvb = gpu_ctx->state.qkv + (size_t)b * (size_t)(q_size + k_size + v_size);
        HIP_CHECK(hipMemcpy(qb, qkvb,                     q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(kb, qkvb + q_size,            k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(vb, qkvb + q_size + k_size,   v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
      }

      // RoPE
      float ntk_beta  = 32.0f;
      float ntk_alpha = 1.0f;
      dim3 grid_rope(((head_dim / 2) + block256.x - 1) / block256.x, B);
      compute_cos_sin_kernel_bf16_batch<<<grid_rope, block256>>>(
        gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals, d_pos,
        p->rope_theta, head_dim, p->rope_scaling_factor,
        p->initial_context_length, ntk_beta, ntk_alpha, head_dim/2);
      HIP_CHECK(hipDeviceSynchronize());

      dim3 grid_apply_q(p->n_attn_heads, B);
      dim3 grid_apply_k(p->n_kv_heads,   B);
      apply_rotary_emb_kernel_bf16_batch<<<grid_apply_q, head_dim/2>>>(
        gpu_ctx->state.q, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
        p->n_attn_heads, head_dim, head_dim*p->n_attn_heads, head_dim/2);
      apply_rotary_emb_kernel_bf16_batch<<<grid_apply_k, head_dim/2>>>(
        gpu_ctx->state.k, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
        p->n_kv_heads, head_dim, head_dim*p->n_kv_heads, head_dim/2);
      HIP_CHECK(hipDeviceSynchronize());

      // Write K/V to cache
      const int loff_base = l * S * kv_dim;
      for (int b = 0; b < B; ++b) {
        int pos = pos_host[b];
        size_t loff = loff_base + (size_t)pos * (size_t)kv_dim;
        hip_bfloat16 *k_cache_b = gpu_ctx->state.key_cache   + (size_t)b * (size_t)cache_batch_stride;
        hip_bfloat16 *v_cache_b = gpu_ctx->state.value_cache + (size_t)b * (size_t)cache_batch_stride;
        HIP_CHECK(hipMemcpy(k_cache_b + loff, gpu_ctx->state.k + (size_t)b * (size_t)stride_kv,
                            (size_t)kv_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(v_cache_b + loff, gpu_ctx->state.v + (size_t)b * (size_t)stride_kv,
                            (size_t)kv_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
      }
    }

    // 5) Multi-head attention (batched across B; loop over heads on host)
    for (int h = 0; h < p->n_attn_heads; h++) {
      // Scores
      {
        // grid over time uses the *max* pos across batch for launch; kernel guards by pos[b]
        int max_pos = 0;
        for (int b = 0; b < B; ++b) max_pos = std::max(max_pos, pos_host[b]);
        dim3 grid_att(((max_pos + 1) + block256.x - 1) / block256.x, B);
        int loff = l * S * (p->head_dim * p->n_kv_heads);
        attention_scores_kernel_bf16_batch<<<grid_att, block256>>>(
          gpu_ctx->state.att + (size_t)h * (size_t)S, // base per-head, but kernel adds b*att_stride
          gpu_ctx->state.q,
          gpu_ctx->state.key_cache,
          (p->sliding_window > 0 && (l % 2 == 0)) ? gpu_ctx->state.mask : NULL,
          d_pos,
          p->head_dim, p->head_dim * p->n_kv_heads, p->n_attn_heads / p->n_kv_heads,
          h, loff, S,
          stride_att, head_dim * p->n_attn_heads,
          cache_batch_stride);
        HIP_CHECK(hipDeviceSynchronize());
      }

      // Optional attention sink per-sample (append at t = pos+1 when room)
      for (int b = 0; b < B; ++b) {
        int pos = pos_host[b];
        bool can_add_sink = (pos + 1) < S;
        if (can_add_sink) {
          hip_bfloat16 bf16_sink;
          HIP_CHECK(hipMemcpy(&bf16_sink,
                              gpu_ctx->weights.attn_sinks + (size_t)l * (size_t)p->n_attn_heads + (size_t)h,
                              sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
          hip_bfloat16 *att_bh = gpu_ctx->state.att + (size_t)b * (size_t)stride_att + (size_t)h * (size_t)S;
          HIP_CHECK(hipMemcpy(att_bh + pos + 1, &bf16_sink, sizeof(hip_bfloat16), hipMemcpyHostToDevice));
          // softmax length = pos+2
          softmax_serial_kernel_bf16<<<1,1>>>(att_bh, pos + 2);
        } else {
          hip_bfloat16 *att_bh = gpu_ctx->state.att + (size_t)b * (size_t)stride_att + (size_t)h * (size_t)S;
          softmax_serial_kernel_bf16<<<1,1>>>(att_bh, pos + 1);
        }
      }
      HIP_CHECK(hipDeviceSynchronize());

      // Weighted sum
      {
        dim3 grid_head_dim((p->head_dim + block256.x - 1) / block256.x, B);
        int loff = l * S * (p->head_dim * p->n_kv_heads);
        attention_weighted_sum_kernel_bf16_batch<<<grid_head_dim, block256>>>(
          gpu_ctx->state.tb, gpu_ctx->state.att + (size_t)h * (size_t)S, gpu_ctx->state.value_cache,
          d_pos, p->head_dim, p->head_dim * p->n_kv_heads, p->n_attn_heads / p->n_kv_heads,
          h, loff,
          stride_heads, S,
          cache_batch_stride);
        HIP_CHECK(hipDeviceSynchronize());
      }
    }

    // 6) Output projection of attention: tb2 = W_o * tb + b_o
    {
      const hip_bfloat16 *w_o = gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
      const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
      dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
      matmul_kernel_bf16_batch<<<grid_hidden, block256>>>(gpu_ctx->state.tb2, gpu_ctx->state.tb, w_o,
                                                          head_dim * p->n_attn_heads, hidden_dim,
                                                          stride_hidden, stride_heads);
      add_bias_kernel_bf16_batch<<<grid_hidden, block256>>>(gpu_ctx->state.tb2, b_o, hidden_dim, stride_hidden);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // Residual: x += tb2
    {
      dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
      residual_add_kernel_bf16_batch<<<grid_hidden, block256>>>(gpu_ctx->state.x, gpu_ctx->state.tb2,
                                                                hidden_dim, stride_hidden, stride_hidden);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // -------- MoE --------
    // 7) FFN RMSNorm: t = RMS(x)
    {
      const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
      dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
      rmsnorm_kernel_bf16_batch<<<grid_hidden, block256>>>(gpu_ctx->state.t, gpu_ctx->state.x, rms_ffn_w,
                                                           hidden_dim, stride_hidden, stride_hidden);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // 8) Router: score = W_router * t + b_router
    {
      const hip_bfloat16 *w_router = gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)n_experts;
      const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)n_experts;
      dim3 grid_experts((n_experts + block256.x - 1) / block256.x, B);
      matmul_kernel_bf16_batch<<<grid_experts, block256>>>(gpu_ctx->state.router_score, gpu_ctx->state.t, w_router,
                                                           hidden_dim, n_experts, n_experts, stride_hidden);
      add_bias_kernel_bf16_batch<<<grid_experts, block256>>>(gpu_ctx->state.router_score, b_router, n_experts, n_experts);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // Router scores BF16 -> FP32 device -> host (batched)
    std::vector<float> host_router_score((size_t)B * (size_t)n_experts);
    {
      dim3 grid_experts((n_experts + block256.x - 1) / block256.x, B);
      convert_bf16_to_fp32_strided<<<grid_experts, block256>>>(gpu_ctx->state.router_score,
                                                               gpu_ctx->state.logits_fp32, // reuse as staging
                                                               n_experts, n_experts, p->vocab_size /* any >= n_experts is fine*/);
      HIP_CHECK(hipDeviceSynchronize());
      // copy each sample's scores out (contiguous per sample in logits_fp32)
      for (int b = 0; b < B; ++b) {
        HIP_CHECK(hipMemcpy(host_router_score.data() + (size_t)b * (size_t)n_experts,
                            gpu_ctx->state.logits_fp32 + (size_t)b * (size_t)p->vocab_size,
                            (size_t)n_experts * sizeof(float), hipMemcpyDeviceToHost));
      }
    }

    // 9) Host top-k per sample + softmax over top-k; copy topk to device
    {
      std::vector<float> host_topk_v((size_t)B * (size_t)p->experts_per_token);
      std::vector<int>   host_topk_i((size_t)B * (size_t)p->experts_per_token);
      for (int b = 0; b < B; ++b) {
        float *topkv = host_topk_v.data() + (size_t)b * (size_t)p->experts_per_token;
        int   *topki = host_topk_i.data() + (size_t)b * (size_t)p->experts_per_token;
        topk_cpu(topkv, topki, host_router_score.data() + (size_t)b * (size_t)n_experts,
                 n_experts, p->experts_per_token);
        // softmax over top-k
        float mx = topkv[0];
        for (int i = 1; i < p->experts_per_token; ++i) mx = std::max(mx, topkv[i]);
        float sum = 0.f;
        for (int i = 0; i < p->experts_per_token; ++i) { topkv[i] = expf(topkv[i] - mx); sum += topkv[i]; }
        for (int i = 0; i < p->experts_per_token; ++i) topkv[i] /= (sum + 1e-9f);
      }
      // copy to device (fp32 then convert to bf16)
      float *d_topk_fp32 = nullptr;
      HIP_CHECK(hipMalloc(&d_topk_fp32, (size_t)B * (size_t)p->experts_per_token * sizeof(float)));
      HIP_CHECK(hipMemcpy(d_topk_fp32, host_topk_v.data(),
                          (size_t)B * (size_t)p->experts_per_token * sizeof(float), hipMemcpyHostToDevice));
      dim3 grid_topk((p->experts_per_token + block256.x - 1) / block256.x, B);
      convert_fp32_to_bf16_kernel<<<grid_topk, block256>>>(d_topk_fp32, gpu_ctx->state.topk_v,
                                                           B * p->experts_per_token);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipFree(d_topk_fp32));
      HIP_CHECK(hipMemcpy(gpu_ctx->state.topk_i, host_topk_i.data(),
                          (size_t)B * (size_t)p->experts_per_token * sizeof(int), hipMemcpyHostToDevice));

      // Init e_agg = 0
      HIP_CHECK(hipMemset(gpu_ctx->state.e_agg, 0, (size_t)B * (size_t)stride_hidden * sizeof(hip_bfloat16)));

      // 10) Experts: vì expert khác nhau giữa các sample → chạy theo sample
      for (int b = 0; b < B; ++b) {
        int *topki_b = host_topk_i.data() + (size_t)b * (size_t)p->experts_per_token;
        float *topkv_b = host_topk_v.data() + (size_t)b * (size_t)p->experts_per_token;

        for (int idx = 0; idx < p->experts_per_token; ++idx) {
          int e = topki_b[idx];
          float ew = topkv_b[idx];

          const hip_bfloat16 *w_mlp1 = gpu_ctx->weights.w_mlp1 +
            ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
          const hip_bfloat16 *b_mlp1 = gpu_ctx->weights.b_mlp1 +
            ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);

          // mlp1_out[b] = W_mlp1 * t[b] + b_mlp1
          {
            dim3 grid_mlp1(((2*intermediate_dim) + block256.x - 1) / block256.x, 1);
            matmul_kernel_bf16_batch<<<grid_mlp1, block256>>>(
              gpu_ctx->state.mlp1_out + (size_t)b * (size_t)(2*intermediate_dim),
              gpu_ctx->state.t + (size_t)b * (size_t)hidden_dim, w_mlp1,
              hidden_dim, 2 * intermediate_dim,
              2 * intermediate_dim, hidden_dim);
            add_bias_kernel_bf16_batch<<<grid_mlp1, block256>>>(
              gpu_ctx->state.mlp1_out + (size_t)b * (size_t)(2*intermediate_dim),
              b_mlp1, 2 * intermediate_dim, 2 * intermediate_dim);
            HIP_CHECK(hipDeviceSynchronize());
          }

          // split gate/up -> swiglu -> mlp2 -> accumulate e_agg
          {
            dim3 grid_inter((intermediate_dim + block256.x - 1) / block256.x, 1);
            split_gate_up_kernel_bf16_batch<<<grid_inter, block256>>>(
              gpu_ctx->state.gate + (size_t)b * (size_t)intermediate_dim,
              gpu_ctx->state.up   + (size_t)b * (size_t)intermediate_dim,
              gpu_ctx->state.mlp1_out + (size_t)b * (size_t)(2*intermediate_dim),
              intermediate_dim, intermediate_dim, intermediate_dim, 2*intermediate_dim);
            HIP_CHECK(hipDeviceSynchronize());

            swiglu_kernel_bf16_batch<<<grid_inter, block256>>>(
              gpu_ctx->state.gate_up + (size_t)b * (size_t)intermediate_dim,
              gpu_ctx->state.gate + (size_t)b * (size_t)intermediate_dim,
              gpu_ctx->state.up   + (size_t)b * (size_t)intermediate_dim,
              intermediate_dim, p->swiglu_limit,
              intermediate_dim, intermediate_dim, intermediate_dim);
            HIP_CHECK(hipDeviceSynchronize());
          }

          const hip_bfloat16 *w_mlp2 = gpu_ctx->weights.w_mlp2 +
            ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
          const hip_bfloat16 *b_mlp2 = gpu_ctx->weights.b_mlp2 +
            ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;

          {
            dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, 1);
            // tb2[b] = W_mlp2 * gate_up[b] + b_mlp2
            matmul_kernel_bf16_batch<<<grid_hidden, block256>>>(
              gpu_ctx->state.tb2 + (size_t)b * (size_t)hidden_dim,
              gpu_ctx->state.gate_up + (size_t)b * (size_t)intermediate_dim,
              w_mlp2,
              intermediate_dim, hidden_dim,
              hidden_dim, intermediate_dim);
            add_bias_kernel_bf16_batch<<<grid_hidden, block256>>>(
              gpu_ctx->state.tb2 + (size_t)b * (size_t)hidden_dim,
              b_mlp2, hidden_dim, hidden_dim);
            HIP_CHECK(hipDeviceSynchronize());

            // e_agg[b] += tb2[b] * ew
            weighted_sum_kernel_bf16_batch<<<grid_hidden, block256>>>(
              gpu_ctx->state.e_agg + (size_t)b * (size_t)hidden_dim,
              gpu_ctx->state.tb2 + (size_t)b * (size_t)hidden_dim,
              ew, hidden_dim,
              hidden_dim, hidden_dim);
            HIP_CHECK(hipDeviceSynchronize());
          }
        } // experts per token
      }   // batch loop

      // Residual: x += e_agg
      dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
      residual_add_kernel_bf16_batch<<<grid_hidden, block256>>>(
        gpu_ctx->state.x, gpu_ctx->state.e_agg, hidden_dim, stride_hidden, stride_hidden);
      HIP_CHECK(hipDeviceSynchronize());
    }
  } // layers

  // Final RMSNorm
  {
    dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
    rmsnorm_kernel_bf16_batch<<<grid_hidden, block256>>>(
      gpu_ctx->state.x, gpu_ctx->state.x, gpu_ctx->weights.rms_out_w,
      hidden_dim, stride_hidden, stride_hidden);
    HIP_CHECK(hipDeviceSynchronize());
  }

  // x(BF16)->acc_buffer(FP32)
  {
    dim3 grid_hidden((hidden_dim + block256.x - 1) / block256.x, B);
    // do per-sample contiguous conversion by indexing manually
    for (int b = 0; b < B; ++b) {
      convert_bf16_to_fp32_kernel<<<(hidden_dim + block256.x - 1)/block256.x, block256>>>(
        gpu_ctx->state.x + (size_t)b * (size_t)hidden_dim,
        gpu_ctx->state.acc_buffer + (size_t)b * (size_t)hidden_dim,
        hidden_dim);
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  // logits = W_out(BF16) * x_fp32(FP32) -> FP32
  {
    dim3 grid_vocab((p->vocab_size + block256.x - 1) / block256.x, B);
    matvec_wbf16_xfp32_outfp32_batch<<<grid_vocab, block256>>>(
      gpu_ctx->state.logits_fp32, gpu_ctx->state.acc_buffer, gpu_ctx->weights.out,
      /*n=*/hidden_dim, /*d=*/p->vocab_size, p->vocab_size, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
  }

  // Copy logits to host (thread-local, reused)
  static thread_local std::vector<float> host_logits;
  host_logits.resize((size_t)B * (size_t)p->vocab_size);
  HIP_CHECK(hipMemcpy(host_logits.data(), gpu_ctx->state.logits_fp32,
                      (size_t)B * (size_t)p->vocab_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host_logits.data();
}

// ------------------------- Warm-up & Finish -------------------------

void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch) {
  s->batch = max_batch;
  int B = s->batch;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int S      = p->seq_len;
  int L      = p->n_layers;

  // Per-batch tensors
  HIP_CHECK(hipMalloc(&s->x,    (size_t)B * p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->t,    (size_t)B * p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb,   (size_t)B * p->head_dim * p->n_attn_heads * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb2,  (size_t)B * p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->router_score, (size_t)B * p->n_experts * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_v,       (size_t)B * p->experts_per_token * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_i,       (size_t)B * p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->mlp1_out, (size_t)B * 2 * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate,     (size_t)B * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->up,       (size_t)B * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate_up,  (size_t)B * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->e_agg,    (size_t)B * p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->qkv, (size_t)B * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->q,   (size_t)B * p->n_attn_heads * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->k,   (size_t)B * p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->v,   (size_t)B * p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->key_cache,   (size_t)B * (size_t)L * (size_t)S * (size_t)kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->value_cache, (size_t)B * (size_t)L * (size_t)S * (size_t)kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->att,         (size_t)B * p->n_attn_heads * p->seq_len * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->logits,      (size_t)B * p->vocab_size * sizeof(hip_bfloat16)));

  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, (size_t)S * (size_t)S * sizeof(hip_bfloat16)));
  } else {
    s->mask = NULL;
  }

  HIP_CHECK(hipMalloc(&s->cos_vals, (size_t)B * (p->head_dim / 2) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->sin_vals, (size_t)B * (p->head_dim / 2) * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->acc_buffer,  (size_t)B * p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits_fp32, (size_t)B * p->vocab_size * sizeof(float)));
}

void gpu_free_run_state(GPURunState *s) {
  HIP_CHECK(hipFree(s->x));
  HIP_CHECK(hipFree(s->t));
  HIP_CHECK(hipFree(s->tb));
  HIP_CHECK(hipFree(s->tb2));
  HIP_CHECK(hipFree(s->router_score));
  HIP_CHECK(hipFree(s->topk_v));
  HIP_CHECK(hipFree(s->topk_i));
  HIP_CHECK(hipFree(s->mlp1_out));
  HIP_CHECK(hipFree(s->gate));
  HIP_CHECK(hipFree(s->up));
  HIP_CHECK(hipFree(s->gate_up));
  HIP_CHECK(hipFree(s->e_agg));
  HIP_CHECK(hipFree(s->qkv));
  HIP_CHECK(hipFree(s->q));
  HIP_CHECK(hipFree(s->k));
  HIP_CHECK(hipFree(s->v));
  HIP_CHECK(hipFree(s->key_cache));
  HIP_CHECK(hipFree(s->value_cache));
  HIP_CHECK(hipFree(s->att));
  HIP_CHECK(hipFree(s->logits));
  if (s->mask) HIP_CHECK(hipFree(s->mask));
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  HIP_CHECK(hipFree(s->acc_buffer));
  HIP_CHECK(hipFree(s->logits_fp32));
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  int max_batch = getenv_int("GETP_MAX_BATCH", 8);

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "No HIP devices found.\n");
    exit(EXIT_FAILURE);
  }
  int max_devices = device_count;
  gpu_contexts.clear();
  gpu_contexts.resize(max_devices);

  for (int d = 0; d < max_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    gpu_contexts[d].device_id = d;
    gpu_contexts[d].max_batch = max_batch;
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config, max_batch);

    // precompute and upload mask once
    if (transformer->config.sliding_window > 0 && gpu_contexts[d].state.mask != NULL) {
      int S = transformer->config.seq_len;
      std::vector<float> cpu_mask((size_t)S * (size_t)S);
      for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
          cpu_mask[(size_t)i * (size_t)S + (size_t)j] = (i - j >= transformer->config.sliding_window) ? -INFINITY : 0.0f;
        }
      }
      float *gpu_mask_fp32 = nullptr;
      HIP_CHECK(hipMalloc(&gpu_mask_fp32, (size_t)S * (size_t)S * sizeof(float)));
      HIP_CHECK(hipMemcpy(gpu_mask_fp32, cpu_mask.data(), (size_t)S * (size_t)S * sizeof(float), hipMemcpyHostToDevice));

      dim3 block(256);
      dim3 grid = dim3((((size_t)S * (size_t)S) + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block>>>(gpu_mask_fp32, gpu_contexts[d].state.mask, (int)((size_t)S * (size_t)S));
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipFree(gpu_mask_fp32));
    }
  }
  printf("[GPU] Warm-up done on %d device(s). BF16 weights loaded to VRAM. Max batch=%d\n", max_devices, max_batch);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx.state);
  }
  gpu_contexts.clear();
  printf("[GPU] Cleanup done.\n");
}

// ------------------------- Driver (batched generate) -------------------------

// Generate for a batch of inputs assigned to a GPU context.
// input_strs: array size B_act; out_tokens_ptrs: array size B_act (each big enough).
long long batched_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                                Sampler *base_sampler,
                                const char **input_strs, int **out_tokens_ptrs,
                                int B_act, int max_new_tokens, GPUContext *gpu_ctx) {
  const char *empty_prompt = "";
  std::vector<std::vector<int>> prompts(B_act);
  std::vector<int> prompt_lens(B_act, 0);

  for (int b = 0; b < B_act; ++b) {
    const char *inp = input_strs[b] ? input_strs[b] : empty_prompt;
    prompts[b].resize(strlen(inp) + 8);
    encode(tokenizer, inp, -1, -1, prompts[b].data(), &prompt_lens[b],
           transformer->config.initial_context_length);
    if (prompt_lens[b] < 1) {
      fprintf(stderr, "invalid prompt for sample %d\n", b);
      exit(EXIT_FAILURE);
    }
  }

  // Per-sample sampler
  std::vector<Sampler> samplers(B_act);
  for (int b = 0; b < B_act; ++b) {
    build_sampler(&samplers[b], base_sampler->vocab_size, base_sampler->temperature,
                  base_sampler->topp, base_sampler->rng_state + (12345 * (b+1)));
  }

  // Generation loop
  const int S = transformer->config.seq_len;
  const int max_steps = std::min(max_new_tokens, S);
  std::vector<int> pos(B_act, 0);
  std::vector<int> token(B_act, 0);
  std::vector<char> alive(B_act, 1);

  for (int b = 0; b < B_act; ++b) token[b] = prompts[b][0];

  long long total_out = 0;
  int finished = 0;

  while (finished < B_act) {
    // Prepare tokens/pos for active samples
    std::vector<int> tokens_now(B_act);
    std::vector<int> pos_now(B_act);
    for (int b = 0; b < B_act; ++b) {
      tokens_now[b] = token[b];
      pos_now[b]    = pos[b];
    }

    // Forward pass for all (even finished; pos stays, kernels guard by pos)
    float *logitsB = gpu_forward_batch(transformer, tokens_now.data(), pos_now.data(), B_act, gpu_ctx);

    // Sampling / teacher-forcing per sample
    for (int b = 0; b < B_act; ++b) {
      if (!alive[b]) continue;
      pos[b]++;
      int next;
      if (pos[b] < prompt_lens[b]) {
        next = prompts[b][pos[b]]; // follow prompt
      } else {
        next = sample(&samplers[b], logitsB + (size_t)b * (size_t)base_sampler->vocab_size);
        out_tokens_ptrs[b][pos[b] - prompt_lens[b]] = next;
      }
      if (next == 199999 || next == 200002 || pos[b] >= S || pos[b] >= max_steps) {
        out_tokens_ptrs[b][pos[b] - prompt_lens[b] + 1] = -1;
        alive[b] = 0;
        finished++;
        total_out += (long long)(pos[b] - prompt_lens[b] + 1);
      } else {
        token[b] = next;
      }
    }
  }

  for (int b = 0; b < B_act; ++b) free_sampler(&samplers[b]);
  return total_out;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  const int num_devices = (int)gpu_contexts.size();

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  // Split requests across devices round-robin, then process each device in mini-batches
  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[t].device_id));
      t_ctx = &gpu_contexts[t];

      // Collect indices for this device
      std::vector<int> idxs;
      idxs.reserve((requests->num_reqs + num_devices - 1)/num_devices);
      for (int i = t; i < requests->num_reqs; i += num_devices) idxs.push_back(i);

      // Iterate by batches of size <= max_batch
      Sampler base_sampler;
      build_sampler(&base_sampler, sampler->vocab_size, sampler->temperature, sampler->topp, sampler->rng_state + t + 1);

      const int Bcap = gpu_contexts[t].max_batch;
      long long local_out = 0;

      for (size_t off = 0; off < idxs.size(); off += Bcap) {
        int B_act = (int)std::min((size_t)Bcap, idxs.size() - off);

        std::vector<const char*> in_ptrs(B_act);
        std::vector<int*> out_ptrs(B_act);
        for (int b = 0; b < B_act; ++b) {
          int ridx = idxs[off + b];
          in_ptrs[b]  = get_str_req_ptr(requests, ridx);
          out_ptrs[b] = get_tok_gen_ptr(requests, ridx);
        }
        local_out += batched_getp_generate(transformer, tokenizer, &base_sampler,
                                           in_ptrs.data(), out_ptrs.data(),
                                           B_act, requests->max_seq_len, t_ctx);
      }

      total_out.fetch_add(local_out, std::memory_order_relaxed);
      free_sampler(&base_sampler);
    });
  }
  for (auto &th : workers) th.join();
  return total_out.load(std::memory_order_relaxed);
}

#endif // GETP_RUN


// ------------------------- Weights: copy/convert (unchanged) -------------------------

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int head_dim      = p->head_dim;
  int n_layers      = p->n_layers;
  int n_experts     = p->n_experts;
  int hidden_dim    = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size    = p->vocab_size;
  int n_attn_heads  = p->n_attn_heads;
  int n_kv_heads    = p->n_kv_heads;

  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    const size_t bf16_size = count * sizeof(hip_bfloat16);
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024;
    size_t staging_elems = std::max((size_t)1, STAGING_BYTES / sizeof(float));
    const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
    if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;

    float *d_stage_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&d_stage_fp32, staging_elems * sizeof(float)));

    size_t done = 0;
    const dim3 block(256);
    while (done < count) {
      size_t tile = std::min(staging_elems, count - done);
      size_t tile_bytes_fp32 = tile * sizeof(float);

      HIP_CHECK(hipMemcpyAsync(d_stage_fp32, cpu_data + done, tile_bytes_fp32,
                               hipMemcpyHostToDevice, stream));

      int n = static_cast<int>(tile);
      dim3 grid((n + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(d_stage_fp32, *gpu_data + done, n);
      HIP_CHECK(hipGetLastError());

      done += tile;
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_stage_fp32));
    HIP_CHECK(hipStreamDestroy(stream));
  };

  size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
  copy_and_convert(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, token_embed_count);
  copy_and_convert(cpu_w->out,                    &gpu_w->out,                    token_embed_count);

  size_t rms_count = (size_t)n_layers * (size_t)hidden_dim;
  copy_and_convert(cpu_w->rms_attn_w, &gpu_w->rms_attn_w, rms_count);
  copy_and_convert(cpu_w->rms_ffn_w,  &gpu_w->rms_ffn_w,  rms_count);

  size_t rms_out_count = (size_t)hidden_dim;
  copy_and_convert(cpu_w->rms_out_w,  &gpu_w->rms_out_w,  rms_out_count);

  size_t qkv_w_count = (size_t)n_layers * (size_t)hidden_dim *
                       ((size_t)head_dim * (size_t)n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
  copy_and_convert(cpu_w->w_qkv, &gpu_w->w_qkv, qkv_w_count);

  size_t qkv_b_count = (size_t)n_layers *
                       ((size_t)head_dim * (size_t)n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
  copy_and_convert(cpu_w->b_qkv, &gpu_w->b_qkv, qkv_b_count);

  size_t o_w_count = (size_t)n_layers * (size_t)head_dim * (size_t)n_attn_heads * (size_t)hidden_dim;
  copy_and_convert(cpu_w->w_o, &gpu_w->w_o, o_w_count);

  size_t o_b_count = (size_t)n_layers * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_o, &gpu_w->b_o, o_b_count);

  size_t attn_sinks_count = (size_t)n_layers * (size_t)n_attn_heads;
  copy_and_convert(cpu_w->attn_sinks, &gpu_w->attn_sinks, attn_sinks_count);

  size_t router_w_count = (size_t)n_layers * (size_t)hidden_dim * (size_t)n_experts;
  copy_and_convert(cpu_w->w_router, &gpu_w->w_router, router_w_count);

  size_t router_b_count = (size_t)n_layers * (size_t)n_experts;
  copy_and_convert(cpu_w->b_router, &gpu_w->b_router, router_b_count);

  size_t mlp1_w_count = (size_t)n_layers * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim * (size_t)hidden_dim;
  copy_and_convert(cpu_w->w_mlp1, &gpu_w->w_mlp1, mlp1_w_count);

  size_t mlp1_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->b_mlp1, &gpu_w->b_mlp1, mlp1_b_count);

  size_t mlp2_w_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->w_mlp2, &gpu_w->w_mlp2, mlp2_w_count);

  size_t mlp2_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_mlp2, &gpu_w->b_mlp2, mlp2_b_count);
}

void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  HIP_CHECK(hipFree(gpu_w->token_embedding_table));
  HIP_CHECK(hipFree(gpu_w->out));
  HIP_CHECK(hipFree(gpu_w->rms_attn_w));
  HIP_CHECK(hipFree(gpu_w->rms_ffn_w));
  HIP_CHECK(hipFree(gpu_w->rms_out_w));
  HIP_CHECK(hipFree(gpu_w->w_qkv));
  HIP_CHECK(hipFree(gpu_w->b_qkv));
  HIP_CHECK(hipFree(gpu_w->w_o));
  HIP_CHECK(hipFree(gpu_w->b_o));
  HIP_CHECK(hipFree(gpu_w->attn_sinks));
  HIP_CHECK(hipFree(gpu_w->w_router));
  HIP_CHECK(hipFree(gpu_w->b_router));
  HIP_CHECK(hipFree(gpu_w->w_mlp1));
  HIP_CHECK(hipFree(gpu_w->b_mlp1));
  HIP_CHECK(hipFree(gpu_w->w_mlp2));
  HIP_CHECK(hipFree(gpu_w->b_mlp2));
}
