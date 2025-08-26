// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (e.g., MI250), BF16 weights/KV; FP32 accumulate.
// - Warm-up loads weights with chunked copy+convert to avoid OOM.
// - Runtime fixes: bound pos to seq_len; guard attention sink write.

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

#define BATCH_SIZE 1
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
  hip_bfloat16 *x;
  hip_bfloat16 *t;
  hip_bfloat16 *tb;
  hip_bfloat16 *tb2;
  hip_bfloat16 *router_score;
  hip_bfloat16 *topk_v;
  int *topk_i;
  hip_bfloat16 *mlp1_out;
  hip_bfloat16 *gate;
  hip_bfloat16 *up;
  hip_bfloat16 *gate_up;
  hip_bfloat16 *e_agg;
  hip_bfloat16 *qkv;
  hip_bfloat16 *q;
  hip_bfloat16 *k;
  hip_bfloat16 *v;
  hip_bfloat16 *att;         // size: n_attn_heads * seq_len
  hip_bfloat16 *logits;      // (unused for final FP32 logits, but kept)
  hip_bfloat16 *key_cache;
  hip_bfloat16 *value_cache;
  hip_bfloat16 *mask;
  hip_bfloat16 *cos_vals;
  hip_bfloat16 *sin_vals;
  // FP32 buffers for accumulation and precision-sensitive ops
  float *acc_buffer;         // x in FP32 for output matvec
  float *logits_fp32;        // final logits
} GPURunState;

// Batched run-state
typedef struct {
  int batch_size;
  // current wave of activations for a batch of tokens
  hip_bfloat16 *x;      // [B, hidden_dim]
  hip_bfloat16 *t;      // [B, hidden_dim]
  hip_bfloat16 *tb;     // [B, head_dim * n_attn_heads]
  hip_bfloat16 *tb2;    // [B, hidden_dim]
  hip_bfloat16 *router_score; // [B, n_experts]
  hip_bfloat16 *topk_v;       // [B, experts_per_token]
  int *topk_i;                // [B, experts_per_token]
  hip_bfloat16 *mlp1_out;     // [B, 2 * intermediate_dim]
  hip_bfloat16 *gate;         // [B, intermediate_dim]
  hip_bfloat16 *up;           // [B, intermediate_dim]
  hip_bfloat16 *gate_up;      // [B, intermediate_dim]
  hip_bfloat16 *e_agg;        // [B, hidden_dim]
  hip_bfloat16 *qkv;          // [B, head_dim * (n_attn_heads + 2 * n_kv_heads)]
  hip_bfloat16 *q;            // [B, n_attn_heads * head_dim]
  hip_bfloat16 *k;            // [B, n_kv_heads * head_dim]
  hip_bfloat16 *v;            // [B, n_kv_heads * head_dim]
  hip_bfloat16 *att;          // [B, n_attn_heads * seq_len]
  hip_bfloat16 *key_cache;    // [B, n_layers * seq_len * kv_dim]
  hip_bfloat16 *value_cache;  // [B, n_layers * seq_len * kv_dim]
  hip_bfloat16 *cos_vals;     // [B, head_dim/2]
  hip_bfloat16 *sin_vals;     // [B, head_dim/2]
  float *acc_buffer;          // [B, hidden_dim] (FP32)
  float *logits_fp32;         // [B, vocab_size] (FP32)
} GPURunStateBatch;

// Per-GPU device context
typedef struct {
  int device_id;
  GPUTransformerWeights weights;
  GPURunState state;
  // optional batched state; when batch_size == 0, batching is disabled
  GPURunStateBatch state_batch;
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

// ------------------------- Kernels -------------------------

// FP32 -> BF16 (device)
__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
}

// BF16 -> FP32 (device)
__global__ void convert_bf16_to_fp32_kernel(const hip_bfloat16 *bf16_data, float *fp32_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) fp32_data[tid] = bfloat16_to_float(bf16_data[tid]);
}

__global__ void rmsnorm_kernel_bf16(hip_bfloat16 *o, const hip_bfloat16 *x, const hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
      float xv = bfloat16_to_float(x[j]);
      ss += xv * xv;
    }
    ss = ss / size + 1e-5f;
    shared_ss = 1.0f / sqrtf(ss);
  }
  __syncthreads();
  if (tid < size) {
    float xv = bfloat16_to_float(x[tid]);
    float wv = bfloat16_to_float(weight[tid]);
    o[tid] = float_to_bfloat16(wv * (shared_ss * xv));
  }
}

__global__ void matmul_kernel_bf16(hip_bfloat16 *xout, const hip_bfloat16 *x, const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      float wv = bfloat16_to_float(w[i * n + j]);
      float xv = bfloat16_to_float(x[j]);
      val += wv * xv;
    }
    xout[i] = float_to_bfloat16(val);
  }
}

__global__ void add_bias_kernel_bf16(hip_bfloat16 *output, const hip_bfloat16 *bias, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float ov = bfloat16_to_float(output[tid]);
    float bv = bfloat16_to_float(bias[tid]);
    output[tid] = float_to_bfloat16(ov + bv);
  }
}

__global__ void copy_kernel_bf16(hip_bfloat16 *dst, const hip_bfloat16 *src, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) dst[tid] = src[tid];
}

// NOTE: serial softmax (thread 0). OK for small sizes; simple & safe.
__global__ void softmax_kernel_bf16(hip_bfloat16 *x, int size) {
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

__global__ void compute_cos_sin_kernel_bf16(hip_bfloat16 *cos_out, hip_bfloat16 *sin_out, int pos,
                                            float base, int head_dim, float scaling_factor,
                                            float initial_context_length, float ntk_beta, float ntk_alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim / 2;
  if (tid < d_half) {
    float freq = powf(base, ((float)(2 * tid)) / (float)head_dim);
    float inv_freq;
    if (scaling_factor > 1.0f) {
      float low  = d_half * logf(initial_context_length / (ntk_beta  * 2.0f * M_PI)) / logf(base);
      float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
      float interpolation = 1.0f / (scaling_factor * freq);
      float extrapolation = 1.0f / freq;
      float ramp = ((float)tid - low) / (high - low);
      if (ramp < 0) ramp = 0;
      if (ramp > 1) ramp = 1;
      float mask = 1.0f - ramp;
      inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
    } else {
      inv_freq = 1.0f / freq;
    }
    float concentration = (scaling_factor > 1.0f) ? (0.1f * logf(scaling_factor) + 1.0f) : 1.0f;
    float val = (float)pos * inv_freq;
    cos_out[tid] = float_to_bfloat16(cosf(val) * concentration);
    sin_out[tid] = float_to_bfloat16(sinf(val) * concentration);
  }
}

__global__ void apply_rotary_emb_kernel_bf16(hip_bfloat16 *x, const hip_bfloat16 *cos, const hip_bfloat16 *sin,
                                             int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (h < n_heads && i < half) {
    float x1 = bfloat16_to_float(x[h * head_dim + i]);
    float x2 = bfloat16_to_float(x[h * head_dim + half + i]);
    float c  = bfloat16_to_float(cos[i]);
    float s  = bfloat16_to_float(sin[i]);
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    x[h * head_dim + i]           = float_to_bfloat16(o1);
    x[h * head_dim + half + i]    = float_to_bfloat16(o2);
  }
}

__global__ void attention_scores_kernel_bf16(hip_bfloat16 *att, const hip_bfloat16 *q, const hip_bfloat16 *key_cache,
                                             int head_dim, int loff, int kv_dim, int kv_mul,
                                             int h, int pos, int seq_len, const hip_bfloat16 *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const hip_bfloat16 *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      float qv = bfloat16_to_float(q[i]);
      float kv = bfloat16_to_float(k[i]);
      score += qv * kv;
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      float mv = bfloat16_to_float(mask[pos * seq_len + t]);
      score += mv;
    }
    att[t] = float_to_bfloat16(score);
  }
}

__global__ void attention_weighted_sum_kernel_bf16(hip_bfloat16 *tb, const hip_bfloat16 *att, const hip_bfloat16 *value_cache,
                                                   int head_dim, int loff, int kv_dim, int kv_mul,
                                                   int h, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < head_dim) {
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const hip_bfloat16 *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      float a  = bfloat16_to_float(att[t]);
      float vv = bfloat16_to_float(v[i]);
      sum += a * vv;
    }
    tb[h * head_dim + i] = float_to_bfloat16(sum);
  }
}

__global__ void swiglu_kernel_bf16(hip_bfloat16 *gate_up, const hip_bfloat16 *gate, const hip_bfloat16 *up,
                                   int intermediate_dim, float swiglu_limit) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    float val   = bfloat16_to_float(gate[tid]);
    float upval = bfloat16_to_float(up[tid]);
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    gate_up[tid] = float_to_bfloat16(val);
  }
}

__global__ void split_gate_up_kernel_bf16(hip_bfloat16 *gate, hip_bfloat16 *up, const hip_bfloat16 *mlp1_out, int intermediate_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    gate[tid] = mlp1_out[2 * tid];
    up[tid]   = mlp1_out[2 * tid + 1];
  }
}

__global__ void weighted_sum_kernel_bf16(hip_bfloat16 *e_agg, const hip_bfloat16 *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float agg = bfloat16_to_float(e_agg[tid]);
    float add = bfloat16_to_float(tb2[tid]);
    e_agg[tid] = float_to_bfloat16(agg + add * expert_w);
  }
}

__global__ void residual_connection_kernel_bf16(hip_bfloat16 *x, const hip_bfloat16 *residual, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float xv  = bfloat16_to_float(x[tid]);
    float rv  = bfloat16_to_float(residual[tid]);
    x[tid] = float_to_bfloat16(xv + rv);
  }
}

// Final output: W (BF16) * x (FP32) -> logits (FP32); avoid big w_out_fp32 buffer
__global__ void matvec_wbf16_xfp32_outfp32(float *out, const float *x,
                                           const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
      sum += (float)w[i * n + j] * x[j];
    }
    out[i] = sum;
  }
}

// ------------------------- Batched Kernels -------------------------

__global__ void rmsnorm_kernel_bf16_batched(hip_bfloat16 *o, const hip_bfloat16 *x, const hip_bfloat16 *weight,
                                            int batch_size, int size) {
  int b = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size) return;
  const hip_bfloat16 *xb = x + (size_t)b * size;
  hip_bfloat16 *ob = o + (size_t)b * size;
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
      float xv = bfloat16_to_float(xb[j]);
      ss += xv * xv;
    }
    ss = ss / size + 1e-5f;
    shared_ss = 1.0f / sqrtf(ss);
  }
  __syncthreads();
  if (tid < size) {
    float xv = bfloat16_to_float(xb[tid]);
    float wv = bfloat16_to_float(weight[tid]);
    ob[tid] = float_to_bfloat16(wv * (shared_ss * xv));
  }
}

__global__ void matmul_kernel_bf16_batched(hip_bfloat16 *xout, const hip_bfloat16 *x, const hip_bfloat16 *w,
                                           int batch_size, int n, int d) {
  int b = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row of output (d)
  if (b >= batch_size || i >= d) return;
  const hip_bfloat16 *xb = x + (size_t)b * n;
  hip_bfloat16 *xoutb = xout + (size_t)b * d;
  float val = 0.0f;
  for (int j = 0; j < n; j++) {
    float wv = bfloat16_to_float(w[i * n + j]);
    float xv = bfloat16_to_float(xb[j]);
    val += wv * xv;
  }
  xoutb[i] = float_to_bfloat16(val);
}

__global__ void add_bias_kernel_bf16_batched(hip_bfloat16 *output, const hip_bfloat16 *bias,
                                             int batch_size, int size) {
  int b = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size || tid >= size) return;
  hip_bfloat16 *outb = output + (size_t)b * size;
  float ov = bfloat16_to_float(outb[tid]);
  float bv = bfloat16_to_float(bias[tid]);
  outb[tid] = float_to_bfloat16(ov + bv);
}

__global__ void residual_connection_kernel_bf16_batched(hip_bfloat16 *x, const hip_bfloat16 *residual,
                                                        int batch_size, int size) {
  int b = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size || tid >= size) return;
  hip_bfloat16 *xb = x + (size_t)b * size;
  const hip_bfloat16 *rb = residual + (size_t)b * size;
  float xv  = bfloat16_to_float(xb[tid]);
  float rv  = bfloat16_to_float(rb[tid]);
  xb[tid] = float_to_bfloat16(xv + rv);
}

__global__ void convert_bf16_to_fp32_kernel_batched(const hip_bfloat16 *bf16_data, float *fp32_data,
                                                    int batch_size, int size) {
  int b = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size || tid >= size) return;
  const hip_bfloat16 *inb = bf16_data + (size_t)b * size;
  float *outb = fp32_data + (size_t)b * size;
  outb[tid] = bfloat16_to_float(inb[tid]);
}

__global__ void matvec_wbf16_xfp32_outfp32_batched(float *out, const float *x,
                                                   const hip_bfloat16 *w, int batch_size,
                                                   int n, int d) {
  int b = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (b >= batch_size || i >= d) return;
  const float *xb = x + (size_t)b * n;
  float *outb = out + (size_t)b * d;
  float sum = 0.0f;
  for (int j = 0; j < n; ++j) {
    sum += (float)w[i * n + j] * xb[j];
  }
  outb[i] = sum;
}

// ------------------------- CPU top-k -------------------------

void topk_cpu(float *topk_values, int *topk_indices, const float *router_score, int num_experts, int experts_per_token) {
  struct Pair { float value; int index; };
  Pair *pairs = (Pair*)malloc(num_experts * sizeof(Pair));
  for (int i = 0; i < num_experts; ++i) {
    pairs[i].value = router_score[i];
    pairs[i].index = i;
  }
  for (int i = 0; i < experts_per_token; i++) {
    int max_idx = i;
    for (int j = i + 1; j < num_experts; j++) {
      if (pairs[j].value > pairs[max_idx].value) max_idx = j;
    }
    if (max_idx != i) { Pair t = pairs[i]; pairs[i] = pairs[max_idx]; pairs[max_idx] = t; }
    topk_values[i]  = pairs[i].value;
    topk_indices[i] = pairs[i].index;
  }
  free(pairs);
}

// ------------------------- Alloc/Free run state -------------------------
void gpu_malloc_run_state(GPURunState *s, Config *p);
void gpu_free_run_state(GPURunState *s);

// Batched alloc/free
void gpu_malloc_run_state_batch(GPURunStateBatch *sb, Config *p, int batch_size);
void gpu_free_run_state_batch(GPURunStateBatch *sb);


// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
void gpu_free_weights(GPUTransformerWeights *gpu_w);



// ------------------------- Forward -------------------------

float *gpu_forward(Transformer *transformer, int token, int pos, GPUContext *gpu_ctx) {
  Config *p = &transformer->config;

  int head_dim         = p->head_dim;
  int hidden_dim       = p->hidden_dim;
  int kv_dim           = p->head_dim * p->n_kv_heads;
  int kv_mul           = p->n_attn_heads / p->n_kv_heads;
  int intermediate_dim = p->intermediate_dim;
  int n_experts        = p->n_experts;

  dim3 block(256);
  dim3 grid_hidden       = dim3((hidden_dim + block.x - 1) / block.x);
  dim3 grid_vocab        = dim3((p->vocab_size + block.x - 1) / block.x);
  dim3 grid_intermediate = dim3((intermediate_dim + block.x - 1) / block.x);
  dim3 grid_experts      = dim3((n_experts + block.x - 1) / block.x);

  // Token embedding
  hip_bfloat16 *token_row = gpu_ctx->weights.token_embedding_table + (size_t)token * (size_t)hidden_dim;
  HIP_CHECK(hipMemcpy(gpu_ctx->state.x, token_row, hidden_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));

  // Layers
  for (int l = 0; l < p->n_layers; l++) {
    // Attention RMSNorm
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.t, gpu_ctx->state.x, rms_attn_w, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // QKV projection
    int loff = l * p->seq_len * kv_dim;
    const hip_bfloat16 *w_qkv = gpu_ctx->weights.w_qkv + (size_t)l * (size_t)hidden_dim *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);

    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 grid_qkv = dim3((qkv_out_dim + block.x - 1) / block.x);
    matmul_kernel_bf16<<<grid_qkv, block>>>(gpu_ctx->state.qkv, gpu_ctx->state.t, w_qkv, hidden_dim, qkv_out_dim);
    add_bias_kernel_bf16<<<grid_qkv, block>>>(gpu_ctx->state.qkv, b_qkv, qkv_out_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Split Q, K, V
    int q_size = head_dim * p->n_attn_heads;
    int k_size = head_dim * p->n_kv_heads;
    int v_size = head_dim * p->n_kv_heads;

    HIP_CHECK(hipMemcpy(gpu_ctx->state.q, gpu_ctx->state.qkv,                 q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(gpu_ctx->state.k, gpu_ctx->state.qkv + q_size,        k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(gpu_ctx->state.v, gpu_ctx->state.qkv + q_size + k_size, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));

    // Store K, V in cache
    hip_bfloat16 *k_cache_pos = gpu_ctx->state.key_cache   + loff + pos * kv_dim;
    hip_bfloat16 *v_cache_pos = gpu_ctx->state.value_cache + loff + pos * kv_dim;
    HIP_CHECK(hipMemcpy(k_cache_pos, gpu_ctx->state.k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(v_cache_pos, gpu_ctx->state.v, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));

    // RoPE
    float ntk_beta  = 32.0f;
    float ntk_alpha = 1.0f;
    dim3 grid_rope = dim3(((head_dim / 2) + block.x - 1) / block.x);
    compute_cos_sin_kernel_bf16<<<grid_rope, block>>>(gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals, pos,
                                                      p->rope_theta, head_dim, p->rope_scaling_factor,
                                                      p->initial_context_length, ntk_beta, ntk_alpha);
    HIP_CHECK(hipDeviceSynchronize());

    apply_rotary_emb_kernel_bf16<<<p->n_attn_heads, head_dim/2>>>(gpu_ctx->state.q, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb_kernel_bf16<<<p->n_kv_heads,   head_dim/2>>>(gpu_ctx->state.k, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals, p->n_kv_heads,   head_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Update K cache after RoPE
    HIP_CHECK(hipMemcpy(k_cache_pos, gpu_ctx->state.k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));

    // Multi-head attention
    for (int h = 0; h < p->n_attn_heads; h++) {
      hip_bfloat16 *q = gpu_ctx->state.q + (size_t)h * (size_t)head_dim;
      hip_bfloat16 *att = gpu_ctx->state.att + (size_t)h * (size_t)p->seq_len;

      // Scores
      dim3 grid_att = dim3(((pos + 1) + block.x - 1) / block.x);
      attention_scores_kernel_bf16<<<grid_att, block>>>(att, q, gpu_ctx->state.key_cache, head_dim, loff,
                                                        kv_dim, kv_mul, h, pos, p->seq_len,
                                                        (p->sliding_window > 0 && l % 2 == 0) ? gpu_ctx->state.mask : NULL);
      HIP_CHECK(hipDeviceSynchronize());

      // Attention sink: chỉ thêm nếu còn chỗ (tránh OOB)
      int len_no_sink = pos + 1;    // t = 0..pos
      bool can_add_sink = (pos + 1) < p->seq_len;
      if (can_add_sink) {
        hip_bfloat16 bf16_sink;
        HIP_CHECK(hipMemcpy(&bf16_sink,
                            gpu_ctx->weights.attn_sinks + (size_t)l * (size_t)p->n_attn_heads + (size_t)h,
                            sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(att + pos + 1, &bf16_sink, sizeof(hip_bfloat16), hipMemcpyHostToDevice));
        softmax_kernel_bf16<<<1, 1>>>(att, len_no_sink + 1);
      } else {
        softmax_kernel_bf16<<<1, 1>>>(att, len_no_sink);
      }
      HIP_CHECK(hipDeviceSynchronize());

      // Weighted sum (chỉ 0..pos, không có v-sink)
      dim3 grid_head_dim = dim3((head_dim + block.x - 1) / block.x);
      attention_weighted_sum_kernel_bf16<<<grid_head_dim, block>>>(gpu_ctx->state.tb, att, gpu_ctx->state.value_cache,
                                                                   head_dim, loff, kv_dim, kv_mul, h, pos);
      HIP_CHECK(hipDeviceSynchronize());
    }

    // Output projection of attention
    const hip_bfloat16 *w_o = gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    matmul_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.tb2, gpu_ctx->state.tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    add_bias_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.tb2, b_o, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Residual
    residual_connection_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.x, gpu_ctx->state.tb2, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // -------- MoE --------
    // FFN RMSNorm
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.t, gpu_ctx->state.x, rms_ffn_w, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Router
    const hip_bfloat16 *w_router = gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)n_experts;
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)n_experts;
    matmul_kernel_bf16<<<grid_experts, block>>>(gpu_ctx->state.router_score, gpu_ctx->state.t, w_router, hidden_dim, n_experts);
    add_bias_kernel_bf16<<<grid_experts, block>>>(gpu_ctx->state.router_score, b_router, n_experts);
    HIP_CHECK(hipDeviceSynchronize());

    // Router scores: BF16 -> FP32 (device) -> host
    float *gpu_router_score_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&gpu_router_score_fp32, n_experts * sizeof(float)));
    convert_bf16_to_fp32_kernel<<<grid_experts, block>>>(gpu_ctx->state.router_score, gpu_router_score_fp32, n_experts);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> cpu_router_score(n_experts);
    HIP_CHECK(hipMemcpy(cpu_router_score.data(), gpu_router_score_fp32, n_experts * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> cpu_topk_v(p->experts_per_token);
    std::vector<int>   cpu_topk_i(p->experts_per_token);
    topk_cpu(cpu_topk_v.data(), cpu_topk_i.data(), cpu_router_score.data(), n_experts, p->experts_per_token);

    // Softmax top-k (host)
    float sum = 0.0f;
    float max_val = cpu_topk_v[0];
    for (int i = 1; i < p->experts_per_token; i++) if (cpu_topk_v[i] > max_val) max_val = cpu_topk_v[i];
    for (int i = 0; i < p->experts_per_token; i++) { cpu_topk_v[i] = expf(cpu_topk_v[i] - max_val); sum += cpu_topk_v[i]; }
    for (int i = 0; i < p->experts_per_token; i++) cpu_topk_v[i] /= sum;

    // Copy topk back to GPU (as BF16 for weights)
    hip_bfloat16 *gpu_topk_v_bf16 = nullptr;
    float *gpu_topk_v_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&gpu_topk_v_bf16, p->experts_per_token * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&gpu_topk_v_fp32, p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMemcpy(gpu_topk_v_fp32, cpu_topk_v.data(), p->experts_per_token * sizeof(float), hipMemcpyHostToDevice));

    dim3 grid_topk = dim3((p->experts_per_token + block.x - 1) / block.x);
    convert_fp32_to_bf16_kernel<<<grid_topk, block>>>(gpu_topk_v_fp32, gpu_topk_v_bf16, p->experts_per_token);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(gpu_ctx->state.topk_v, gpu_topk_v_bf16, p->experts_per_token * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(gpu_ctx->state.topk_i, cpu_topk_i.data(), p->experts_per_token * sizeof(int), hipMemcpyHostToDevice));

    // Init aggregation
    HIP_CHECK(hipMemset(gpu_ctx->state.e_agg, 0, hidden_dim * sizeof(hip_bfloat16)));

    // Experts
    for (int idx = 0; idx < p->experts_per_token; idx++) {
      int e = cpu_topk_i[idx];
      float expert_w = cpu_topk_v[idx];

      const hip_bfloat16 *w_mlp1 = gpu_ctx->weights.w_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
      const hip_bfloat16 *b_mlp1 = gpu_ctx->weights.b_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);

      dim3 grid_mlp1 = dim3(((2 * intermediate_dim) + block.x - 1) / block.x);
      matmul_kernel_bf16<<<grid_mlp1, block>>>(gpu_ctx->state.mlp1_out, gpu_ctx->state.t, w_mlp1, hidden_dim, 2 * intermediate_dim);
      add_bias_kernel_bf16<<<grid_mlp1, block>>>(gpu_ctx->state.mlp1_out, b_mlp1, 2 * intermediate_dim);
      HIP_CHECK(hipDeviceSynchronize());

      split_gate_up_kernel_bf16<<<grid_intermediate, block>>>(gpu_ctx->state.gate, gpu_ctx->state.up, gpu_ctx->state.mlp1_out, intermediate_dim);
      HIP_CHECK(hipDeviceSynchronize());

      swiglu_kernel_bf16<<<grid_intermediate, block>>>(gpu_ctx->state.gate_up, gpu_ctx->state.gate, gpu_ctx->state.up, intermediate_dim, p->swiglu_limit);
      HIP_CHECK(hipDeviceSynchronize());

      const hip_bfloat16 *w_mlp2 = gpu_ctx->weights.w_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
      const hip_bfloat16 *b_mlp2 = gpu_ctx->weights.b_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;

      matmul_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.tb2, gpu_ctx->state.gate_up, w_mlp2, intermediate_dim, hidden_dim);
      add_bias_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.tb2, b_mlp2, hidden_dim);
      HIP_CHECK(hipDeviceSynchronize());

      weighted_sum_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.e_agg, gpu_ctx->state.tb2, expert_w, hidden_dim);
      HIP_CHECK(hipDeviceSynchronize());
    }

    residual_connection_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.x, gpu_ctx->state.e_agg, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Cleanup temporaries
    HIP_CHECK(hipFree(gpu_router_score_fp32));
    HIP_CHECK(hipFree(gpu_topk_v_bf16));
    HIP_CHECK(hipFree(gpu_topk_v_fp32));
  }

  // Final RMSNorm
  rmsnorm_kernel_bf16<<<grid_hidden, block>>>(gpu_ctx->state.x, gpu_ctx->state.x, gpu_ctx->weights.rms_out_w, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());

  // Convert x to FP32
  convert_bf16_to_fp32_kernel<<<grid_hidden, block>>>(gpu_ctx->state.x, gpu_ctx->state.acc_buffer, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());

  // Output logits directly: W(BF16)*x(FP32)->logits(FP32)
  dim3 grid_vocab2 = dim3((p->vocab_size + block.x - 1) / block.x);
  matvec_wbf16_xfp32_outfp32<<<grid_vocab2, block>>>(gpu_ctx->state.logits_fp32, gpu_ctx->state.acc_buffer, gpu_ctx->weights.out,
                                                     /*n=*/hidden_dim, /*d=*/p->vocab_size);
  HIP_CHECK(hipDeviceSynchronize());

  // Copy logits to CPU (thread-local buffer across calls)
  thread_local float *cpu_logits = NULL;
  if (cpu_logits == NULL) {
    cpu_logits = (float*)malloc(p->vocab_size * sizeof(float));
  }
  HIP_CHECK(hipMemcpy(cpu_logits, gpu_ctx->state.logits_fp32, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
  return cpu_logits;
}

// Batched forward
float *gpu_forward_batch(Transformer *transformer, const int *tokens, int batch_size, int pos, GPUContext *gpu_ctx) {
  Config *p = &transformer->config;
  const int B = batch_size;
  if (B <= 0) return NULL;
  dim3 block(256);

  // 1) embed tokens -> state_batch.x
  for (int b = 0; b < B; ++b) {
    int token = tokens[b];
    hip_bfloat16 *dst = gpu_ctx->state_batch.x + (size_t)b * p->hidden_dim;
    hip_bfloat16 *src = gpu_ctx->weights.token_embedding_table + (size_t)token * (size_t)p->hidden_dim;
    HIP_CHECK(hipMemcpy(dst, src, p->hidden_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
  }

  // 2) per-layer minimal batched path: Norm -> QKV matmul + bias (batched)
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;
  int kv_dim = p->head_dim * p->n_kv_heads;

  dim3 grid_hidden((hidden_dim + block.x - 1) / block.x, B);

  for (int l = 0; l < p->n_layers; l++) {
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16_batched<<<grid_hidden, block>>>(gpu_ctx->state_batch.t, gpu_ctx->state_batch.x, rms_attn_w, B, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    const hip_bfloat16 *w_qkv = gpu_ctx->weights.w_qkv + (size_t)l * (size_t)hidden_dim *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 grid_qkv((qkv_out_dim + block.x - 1) / block.x, B);
    matmul_kernel_bf16_batched<<<grid_qkv, block>>>(gpu_ctx->state_batch.qkv, gpu_ctx->state_batch.t, w_qkv, B, hidden_dim, qkv_out_dim);
    add_bias_kernel_bf16_batched<<<grid_qkv, block>>>(gpu_ctx->state_batch.qkv, b_qkv, B, qkv_out_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Split Q, K, V per batch element
    int q_size = head_dim * p->n_attn_heads;
    int k_size = head_dim * p->n_kv_heads;
    int v_size = head_dim * p->n_kv_heads;
    for (int b = 0; b < B; ++b) {
      hip_bfloat16 *base = gpu_ctx->state_batch.qkv + (size_t)b * qkv_out_dim;
      HIP_CHECK(hipMemcpy(gpu_ctx->state_batch.q + (size_t)b * q_size, base,                 q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(gpu_ctx->state_batch.k + (size_t)b * k_size, base + q_size,        k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(gpu_ctx->state_batch.v + (size_t)b * v_size, base + q_size + k_size, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    }

    // Store K, V in per-batch caches at time pos
    int loff = l * p->seq_len * kv_dim;
    for (int b = 0; b < B; ++b) {
      hip_bfloat16 *k_cache_pos = gpu_ctx->state_batch.key_cache   + (size_t)b * p->n_layers * p->seq_len * kv_dim + loff + pos * kv_dim;
      hip_bfloat16 *v_cache_pos = gpu_ctx->state_batch.value_cache + (size_t)b * p->n_layers * p->seq_len * kv_dim + loff + pos * kv_dim;
      HIP_CHECK(hipMemcpy(k_cache_pos, gpu_ctx->state_batch.k + (size_t)b * k_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(v_cache_pos, gpu_ctx->state_batch.v + (size_t)b * v_size, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    }

    // RoPE per batch
    float ntk_beta  = 32.0f;
    float ntk_alpha = 1.0f;
    dim3 grid_rope_b(((head_dim / 2) + block.x - 1) / block.x, B);
    compute_cos_sin_kernel_bf16<<<grid_rope_b, block>>>(gpu_ctx->state_batch.cos_vals, gpu_ctx->state_batch.sin_vals, pos,
                                                        p->rope_theta, head_dim, p->rope_scaling_factor,
                                                        p->initial_context_length, ntk_beta, ntk_alpha);
    HIP_CHECK(hipDeviceSynchronize());
    for (int b = 0; b < B; ++b) {
      hip_bfloat16 *qb = gpu_ctx->state_batch.q + (size_t)b * q_size;
      hip_bfloat16 *kb = gpu_ctx->state_batch.k + (size_t)b * k_size;
      hip_bfloat16 *cosb = gpu_ctx->state_batch.cos_vals + (size_t)b * (head_dim/2);
      hip_bfloat16 *sinb = gpu_ctx->state_batch.sin_vals + (size_t)b * (head_dim/2);
      apply_rotary_emb_kernel_bf16<<<p->n_attn_heads, head_dim/2>>>(qb, cosb, sinb, p->n_attn_heads, head_dim);
      apply_rotary_emb_kernel_bf16<<<p->n_kv_heads,   head_dim/2>>>(kb, cosb, sinb, p->n_kv_heads,   head_dim);
      HIP_CHECK(hipDeviceSynchronize());
      // update K cache after RoPE
      hip_bfloat16 *k_cache_pos = gpu_ctx->state_batch.key_cache   + (size_t)b * p->n_layers * p->seq_len * kv_dim + loff + pos * kv_dim;
      HIP_CHECK(hipMemcpy(k_cache_pos, kb, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    }

    // Attention per head per batch
    for (int b = 0; b < B; ++b) {
      hip_bfloat16 *att_base = gpu_ctx->state_batch.att + (size_t)b * p->n_attn_heads * p->seq_len;
      for (int h = 0; h < p->n_attn_heads; h++) {
        hip_bfloat16 *q = gpu_ctx->state_batch.q + (size_t)b * q_size + (size_t)h * head_dim;
        hip_bfloat16 *att = att_base + (size_t)h * (size_t)p->seq_len;
        dim3 grid_att = dim3(((pos + 1) + block.x - 1) / block.x);
        const hip_bfloat16 *mask_ptr = NULL;
        attention_scores_kernel_bf16<<<grid_att, block>>>(att, q, gpu_ctx->state_batch.key_cache + (size_t)b * p->n_layers * p->seq_len * kv_dim,
                                                          head_dim, loff, kv_dim, p->n_attn_heads / p->n_kv_heads,
                                                          h, pos, p->seq_len, mask_ptr);
        HIP_CHECK(hipDeviceSynchronize());
        int len_no_sink = pos + 1;
        bool can_add_sink = (pos + 1) < p->seq_len;
        if (can_add_sink) {
          hip_bfloat16 bf16_sink;
          HIP_CHECK(hipMemcpy(&bf16_sink,
                              gpu_ctx->weights.attn_sinks + (size_t)l * (size_t)p->n_attn_heads + (size_t)h,
                              sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
          HIP_CHECK(hipMemcpy(att + pos + 1, &bf16_sink, sizeof(hip_bfloat16), hipMemcpyHostToDevice));
          softmax_kernel_bf16<<<1, 1>>>(att, len_no_sink + 1);
        } else {
          softmax_kernel_bf16<<<1, 1>>>(att, len_no_sink);
        }
        HIP_CHECK(hipDeviceSynchronize());
        dim3 grid_head_dim = dim3((head_dim + block.x - 1) / block.x);
        attention_weighted_sum_kernel_bf16<<<grid_head_dim, block>>>(gpu_ctx->state_batch.tb + (size_t)b * p->n_attn_heads * head_dim,
                                                                     att, gpu_ctx->state_batch.value_cache + (size_t)b * p->n_layers * p->seq_len * kv_dim,
                                                                     head_dim, loff, kv_dim, p->n_attn_heads / p->n_kv_heads, h, pos);
        HIP_CHECK(hipDeviceSynchronize());
      }
    }

    // Output projection of attention (batched)
    const hip_bfloat16 *w_o = gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    matmul_kernel_bf16_batched<<<grid_hidden, block>>>(gpu_ctx->state_batch.tb2, gpu_ctx->state_batch.tb, w_o, B, head_dim * p->n_attn_heads, hidden_dim);
    add_bias_kernel_bf16_batched<<<grid_hidden, block>>>(gpu_ctx->state_batch.tb2, b_o, B, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    residual_connection_kernel_bf16_batched<<<grid_hidden, block>>>(gpu_ctx->state_batch.x, gpu_ctx->state_batch.tb2, B, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // MoE
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16_batched<<<grid_hidden, block>>>(gpu_ctx->state_batch.t, gpu_ctx->state_batch.x, rms_ffn_w, B, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    const hip_bfloat16 *w_router = gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)n_experts;
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)n_experts;
    dim3 grid_experts_b((n_experts + block.x - 1) / block.x, B);
    matmul_kernel_bf16_batched<<<grid_experts_b, block>>>(gpu_ctx->state_batch.router_score, gpu_ctx->state_batch.t, w_router, B, hidden_dim, n_experts);
    add_bias_kernel_bf16_batched<<<grid_experts_b, block>>>(gpu_ctx->state_batch.router_score, b_router, B, n_experts);
    HIP_CHECK(hipDeviceSynchronize());

    // CPU top-k per sample (simple baseline)
    for (int b = 0; b < B; ++b) {
      std::vector<float> cpu_router_score(n_experts);
      float *gpu_router_score_fp32 = nullptr;
      HIP_CHECK(hipMalloc(&gpu_router_score_fp32, n_experts * sizeof(float)));
      dim3 grid_x((n_experts + block.x - 1) / block.x);
      convert_bf16_to_fp32_kernel<<<grid_x, block>>>(gpu_ctx->state_batch.router_score + (size_t)b * n_experts, gpu_router_score_fp32, n_experts);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(cpu_router_score.data(), gpu_router_score_fp32, n_experts * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipFree(gpu_router_score_fp32));

      std::vector<float> cpu_topk_v(p->experts_per_token);
      std::vector<int>   cpu_topk_i(p->experts_per_token);
      topk_cpu(cpu_topk_v.data(), cpu_topk_i.data(), cpu_router_score.data(), n_experts, p->experts_per_token);
      float tsum = 0.0f;
      float tmax = cpu_topk_v[0];
      for (int i = 1; i < p->experts_per_token; i++) if (cpu_topk_v[i] > tmax) tmax = cpu_topk_v[i];
      for (int i = 0; i < p->experts_per_token; i++) { cpu_topk_v[i] = expf(cpu_topk_v[i] - tmax); tsum += cpu_topk_v[i]; }
      for (int i = 0; i < p->experts_per_token; i++) cpu_topk_v[i] /= tsum;

      hip_bfloat16 *gpu_topk_v_bf16 = nullptr;
      float *gpu_topk_v_fp32 = nullptr;
      HIP_CHECK(hipMalloc(&gpu_topk_v_bf16, p->experts_per_token * sizeof(hip_bfloat16)));
      HIP_CHECK(hipMalloc(&gpu_topk_v_fp32, p->experts_per_token * sizeof(float)));
      HIP_CHECK(hipMemcpy(gpu_topk_v_fp32, cpu_topk_v.data(), p->experts_per_token * sizeof(float), hipMemcpyHostToDevice));
      dim3 grid_topk((p->experts_per_token + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid_topk, block>>>(gpu_topk_v_fp32, gpu_topk_v_bf16, p->experts_per_token);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(gpu_ctx->state_batch.topk_v + (size_t)b * p->experts_per_token, gpu_topk_v_bf16, p->experts_per_token * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(gpu_ctx->state_batch.topk_i + (size_t)b * p->experts_per_token, cpu_topk_i.data(), p->experts_per_token * sizeof(int), hipMemcpyHostToDevice));
      HIP_CHECK(hipFree(gpu_topk_v_bf16));
      HIP_CHECK(hipFree(gpu_topk_v_fp32));

      // zero e_agg for this sample
      HIP_CHECK(hipMemset(gpu_ctx->state_batch.e_agg + (size_t)b * hidden_dim, 0, hidden_dim * sizeof(hip_bfloat16)));

      // Experts per sample
      dim3 grid_hidden1((hidden_dim + block.x - 1) / block.x);
      dim3 grid_intermediate1((intermediate_dim + block.x - 1) / block.x);
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        int e = cpu_topk_i[idx];
        float expert_w = cpu_topk_v[idx];
        const hip_bfloat16 *w_mlp1 = gpu_ctx->weights.w_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
        const hip_bfloat16 *b_mlp1 = gpu_ctx->weights.b_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);
        matmul_kernel_bf16<<<dim3(((2 * intermediate_dim) + block.x - 1) / block.x), block>>>(gpu_ctx->state_batch.mlp1_out + (size_t)b * 2 * intermediate_dim,
                                                                                               gpu_ctx->state_batch.t + (size_t)b * hidden_dim,
                                                                                               w_mlp1, hidden_dim, 2 * intermediate_dim);
        add_bias_kernel_bf16<<<dim3(((2 * intermediate_dim) + block.x - 1) / block.x), block>>>(gpu_ctx->state_batch.mlp1_out + (size_t)b * 2 * intermediate_dim, b_mlp1, 2 * intermediate_dim);
        HIP_CHECK(hipDeviceSynchronize());
        split_gate_up_kernel_bf16<<<grid_intermediate1, block>>>(gpu_ctx->state_batch.gate + (size_t)b * intermediate_dim,
                                                                 gpu_ctx->state_batch.up + (size_t)b * intermediate_dim,
                                                                 gpu_ctx->state_batch.mlp1_out + (size_t)b * 2 * intermediate_dim, intermediate_dim);
        HIP_CHECK(hipDeviceSynchronize());
        swiglu_kernel_bf16<<<grid_intermediate1, block>>>(gpu_ctx->state_batch.gate_up + (size_t)b * intermediate_dim,
                                                          gpu_ctx->state_batch.gate + (size_t)b * intermediate_dim,
                                                          gpu_ctx->state_batch.up + (size_t)b * intermediate_dim,
                                                          intermediate_dim, p->swiglu_limit);
        HIP_CHECK(hipDeviceSynchronize());
        const hip_bfloat16 *w_mlp2 = gpu_ctx->weights.w_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
        const hip_bfloat16 *b_mlp2 = gpu_ctx->weights.b_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;
        matmul_kernel_bf16<<<grid_hidden1, block>>>(gpu_ctx->state_batch.tb2 + (size_t)b * hidden_dim,
                                                    gpu_ctx->state_batch.gate_up + (size_t)b * intermediate_dim,
                                                    w_mlp2, intermediate_dim, hidden_dim);
        add_bias_kernel_bf16<<<grid_hidden1, block>>>(gpu_ctx->state_batch.tb2 + (size_t)b * hidden_dim, b_mlp2, hidden_dim);
        HIP_CHECK(hipDeviceSynchronize());
        weighted_sum_kernel_bf16<<<grid_hidden1, block>>>(gpu_ctx->state_batch.e_agg + (size_t)b * hidden_dim,
                                                          gpu_ctx->state_batch.tb2 + (size_t)b * hidden_dim,
                                                          expert_w, hidden_dim);
        HIP_CHECK(hipDeviceSynchronize());
      }
      residual_connection_kernel_bf16<<<grid_hidden1, block>>>(gpu_ctx->state_batch.x + (size_t)b * hidden_dim,
                                                               gpu_ctx->state_batch.e_agg + (size_t)b * hidden_dim,
                                                               hidden_dim);
      HIP_CHECK(hipDeviceSynchronize());
    }
  }
  // Final RMSNorm (batched)
  dim3 grid_hiddenB((hidden_dim + block.x - 1) / block.x, B);
  rmsnorm_kernel_bf16_batched<<<grid_hiddenB, block>>>(gpu_ctx->state_batch.x, gpu_ctx->state_batch.x, gpu_ctx->weights.rms_out_w, B, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());

  // Convert x to FP32 and compute logits (batched)
  dim3 grid_hiddenB2((hidden_dim + block.x - 1) / block.x, B);
  convert_bf16_to_fp32_kernel_batched<<<grid_hiddenB2, block>>>(gpu_ctx->state_batch.x, gpu_ctx->state_batch.acc_buffer, B, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());
  dim3 grid_vocabB((p->vocab_size + block.x - 1) / block.x, B);
  matvec_wbf16_xfp32_outfp32_batched<<<grid_vocabB, block>>>(gpu_ctx->state_batch.logits_fp32, gpu_ctx->state_batch.acc_buffer, gpu_ctx->weights.out,
                                                             B, /*n=*/hidden_dim, /*d=*/p->vocab_size);
  HIP_CHECK(hipDeviceSynchronize());

  // Copy all B logits to CPU contiguous buffer [B, vocab]
  static thread_local float *cpu_logits_batch = NULL;
  int needed = B * p->vocab_size;
  if (cpu_logits_batch == NULL) {
    cpu_logits_batch = (float*)malloc((size_t)needed * sizeof(float));
  }
  HIP_CHECK(hipMemcpy(cpu_logits_batch, gpu_ctx->state_batch.logits_fp32,
                      (size_t)needed * sizeof(float), hipMemcpyDeviceToHost));
  return cpu_logits_batch;
}

// ------------------------- Warm-up & Finish -------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
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
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config);
    // initialize batched state with batch size 4 (can be changed later)
    gpu_malloc_run_state_batch(&gpu_contexts[d].state_batch, &transformer->config, /*batch_size=*/BATCH_SIZE);

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
  printf("[GPU] Warm-up done on %d device(s). BF16 weights loaded to VRAM.\n", max_devices);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx.state);
    gpu_free_run_state_batch(&gpu_ctx.state_batch);
  }
  gpu_contexts.clear();
  printf("[GPU] Cleanup done.\n");
}

// ------------------------- Driver (generate & inference) -------------------------

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                              Sampler *sampler, const char *input_seq,
                              int *output_tokens, int steps, GPUContext *gpu_ctx) {
  const char *empty_prompt = "";
  if (input_seq == NULL) input_seq = empty_prompt;

  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  int next;
  int token = prompt_tokens[0];
  int pos = 0;
  const int max_ctx = transformer->config.seq_len;
  const int max_steps = std::min(steps, max_ctx); // bound pos to seq_len

  while (pos < max_steps) {
    float *logits = gpu_forward(transformer, token, pos, gpu_ctx);

    pos++;
    if (pos < num_prompt_tokens) {
      next = prompt_tokens[pos];
    } else {
      next = sample(sampler, logits);
      output_tokens[pos - num_prompt_tokens] = next;
    }
    if (next == 199999 || next == 200002) break;
    if (pos >= max_ctx) break; // guard against OOB in caches/att
    token = next;
  }

  output_tokens[pos - num_prompt_tokens + 1] = -1;
  free(prompt_tokens);
  return pos - num_prompt_tokens + 1;
}

// Batched generation: lockstep decoding for B samples on one GPU
static long long simple_getp_generate_batch(Transformer *transformer, Tokenizer *tokenizer,
                                            Sampler *sampler, const char **input_seqs,
                                            int **output_tokens_arr, int B, int steps, GPUContext *gpu_ctx) {
  // Encode prompts
  std::vector<std::vector<int>> prompts(B);
  std::vector<int> prompt_lens(B, 0);
  for (int b = 0; b < B; ++b) {
    const char *inp = (input_seqs[b] ? input_seqs[b] : "");
    prompts[b].resize(strlen(inp) + 3);
    encode(tokenizer, inp, -1, -1, prompts[b].data(), &prompt_lens[b], transformer->config.initial_context_length);
    if (prompt_lens[b] < 1) { fprintf(stderr, "batch prompt empty\n"); exit(EXIT_FAILURE); }
  }

  int pos = 0;
  const int max_ctx = transformer->config.seq_len;
  const int max_steps = std::min(steps, max_ctx);

  std::vector<int> tokens(B);
  for (int b = 0; b < B; ++b) tokens[b] = prompts[b][0];
  long long total_out = 0;

  while (pos < max_steps) {
    float *logits_batch = gpu_forward_batch(transformer, tokens.data(), B, pos, gpu_ctx);
    pos++;

    for (int b = 0; b < B; ++b) {
      int next;
      if (pos < prompt_lens[b]) {
        next = prompts[b][pos];
      } else {
        float *logits = logits_batch + (size_t)b * transformer->config.vocab_size;
        next = sample(sampler, logits);
        output_tokens_arr[b][pos - prompt_lens[b]] = next;
      }
      if (next == 199999 || next == 200002) {
        output_tokens_arr[b][pos - prompt_lens[b] + 1] = -1;
      }
      tokens[b] = next;
    }

    if (pos >= max_ctx) break;
  }

  for (int b = 0; b < B; ++b) {
    output_tokens_arr[b][pos - prompt_lens[b] + 1] = -1;
    total_out += (pos - prompt_lens[b] + 1);
  }
  return total_out;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  const int num_devices = (int)gpu_contexts.size();

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[t].device_id));
      t_ctx = &gpu_contexts[t];
      Sampler local_sampler;
      build_sampler(&local_sampler, sampler->vocab_size, sampler->temperature, sampler->topp, sampler->rng_state + t + 1);
      long long local_out = 0;
      const int B = std::max(1, gpu_contexts[t].state_batch.batch_size);
      for (int idx = t; idx < requests->num_reqs; idx += num_devices * B) {
        int batch_count = 0;
        const char *input_ptrs[8];
        int *out_ptrs[8];
        for (int b = 0; b < B; ++b) {
          int ridx = idx + b * num_devices;
          if (ridx < requests->num_reqs) {
            input_ptrs[b] = get_str_req_ptr(requests, ridx);
            out_ptrs[b]   = get_tok_gen_ptr(requests, ridx);
            batch_count++;
          }
        }
        local_out += simple_getp_generate_batch(transformer, tokenizer, &local_sampler,
                                                  input_ptrs, out_ptrs, B, requests->max_seq_len, t_ctx);
      }
      total_out.fetch_add(local_out, std::memory_order_relaxed);
      free_sampler(&local_sampler);
    });
  }
  for (auto &th : workers) th.join();
  return total_out.load(std::memory_order_relaxed);
}

#endif // GETP_RUN




void gpu_malloc_run_state(GPURunState *s, Config *p) {
  int kv_dim = p->head_dim * p->n_kv_heads;

  HIP_CHECK(hipMalloc(&s->x,   p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->t,   p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb,  p->head_dim * p->n_attn_heads * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb2, p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->router_score,      p->n_experts * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_v,            p->experts_per_token * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_i,            p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->mlp1_out, 2 * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate,     p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->up,       p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate_up,  p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->e_agg,    p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->qkv, p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->q,   p->n_attn_heads * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->k,   p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->v,   p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->key_cache,   p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->att,         p->n_attn_heads * p->seq_len * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->logits,      p->vocab_size * sizeof(hip_bfloat16)));

  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(hip_bfloat16)));
  } else {
    s->mask = NULL;
  }

  HIP_CHECK(hipMalloc(&s->cos_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->sin_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->acc_buffer,  p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits_fp32, p->vocab_size * sizeof(float)));
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
  HIP_CHECK(hipFree(s->mask));
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  HIP_CHECK(hipFree(s->acc_buffer));
  HIP_CHECK(hipFree(s->logits_fp32));
}

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int head_dim      = p->head_dim;
  int n_layers      = p->n_layers;
  int n_experts     = p->n_experts;
  int hidden_dim    = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size    = p->vocab_size;
  int n_attn_heads  = p->n_attn_heads;
  int n_kv_heads    = p->n_kv_heads;

  // Helper: host FP32 -> device staging FP32 (tile) -> device BF16 dest
  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    const size_t bf16_size = count * sizeof(hip_bfloat16);
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));

    auto toGB = [](size_t bytes){ return (double)bytes / GB; };
    // printf("BF16 SIZE: %.2f GB\n", toGB(bf16_size));
    // printf("FP32 SIZE: %.2f GB\n", toGB(count * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024; // 512 MB
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

  // Counts as size_t to avoid overflow
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

void gpu_malloc_run_state_batch(GPURunStateBatch *sb, Config *p, int batch_size) {
  if (batch_size <= 0) { sb->batch_size = 0; return; }
  int kv_dim = p->head_dim * p->n_kv_heads;
  sb->batch_size = batch_size;
  size_t B = (size_t)batch_size;

  HIP_CHECK(hipMalloc(&sb->x,   B * p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->t,   B * p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->tb,  B * p->head_dim * p->n_attn_heads * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->tb2, B * p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&sb->router_score, B * p->n_experts * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->topk_v,       B * p->experts_per_token * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->topk_i,       B * p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&sb->mlp1_out, B * 2 * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->gate,     B * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->up,       B * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->gate_up,  B * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->e_agg,    B * p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&sb->qkv, B * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->q,   B * p->n_attn_heads * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->k,   B * p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->v,   B * p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&sb->key_cache,   B * p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->value_cache, B * p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->att,         B * p->n_attn_heads * p->seq_len * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&sb->cos_vals, B * (p->head_dim / 2) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&sb->sin_vals, B * (p->head_dim / 2) * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&sb->acc_buffer,  B * p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&sb->logits_fp32, B * p->vocab_size * sizeof(float)));
}

void gpu_free_run_state_batch(GPURunStateBatch *sb) {
  HIP_CHECK(hipFree(sb->x));
  HIP_CHECK(hipFree(sb->t));
  HIP_CHECK(hipFree(sb->tb));
  HIP_CHECK(hipFree(sb->tb2));
  HIP_CHECK(hipFree(sb->router_score));
  HIP_CHECK(hipFree(sb->topk_v));
  HIP_CHECK(hipFree(sb->topk_i));
  HIP_CHECK(hipFree(sb->mlp1_out));
  HIP_CHECK(hipFree(sb->gate));
  HIP_CHECK(hipFree(sb->up));
  HIP_CHECK(hipFree(sb->gate_up));
  HIP_CHECK(hipFree(sb->e_agg));
  HIP_CHECK(hipFree(sb->qkv));
  HIP_CHECK(hipFree(sb->q));
  HIP_CHECK(hipFree(sb->k));
  HIP_CHECK(hipFree(sb->v));
  HIP_CHECK(hipFree(sb->key_cache));
  HIP_CHECK(hipFree(sb->value_cache));
  HIP_CHECK(hipFree(sb->att));
  HIP_CHECK(hipFree(sb->cos_vals));
  HIP_CHECK(hipFree(sb->sin_vals));
  HIP_CHECK(hipFree(sb->acc_buffer));
  HIP_CHECK(hipFree(sb->logits_fp32));
 }