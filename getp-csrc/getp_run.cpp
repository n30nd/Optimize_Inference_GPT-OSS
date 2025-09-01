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
#include <mutex>
#include <condition_variable>
#include <deque>

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

#ifndef MAX_EXPERT_STREAMS
#define MAX_EXPERT_STREAMS 4
#endif

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
  float *x;
  float *t;
  float *tb;
  float *tb2;
  float *router_score;
  float *topk_v;
  int *topk_i;
  float *mlp1_out;
  float *gate;
  float *up;
  float *gate_up;
  float *e_agg;
  float *qkv;
  float *q;
  float *k;
  float *v;
  float *att;         // size: n_attn_heads * seq_len
  float *logits;      // (unused for final FP32 logits, but kept)
  float *key_cache;
  float *value_cache;
  float *mask;
  float *cos_vals;
  float *sin_vals;
  // FP32 buffers for accumulation and precision-sensitive ops
  float *acc_buffer;         // x in FP32 for output matvec
  float *logits_fp32;        // final logits
  // Per-stream scratch buffers for expert parallelization
  float *mlp1_out_s;   // [MAX_EXPERT_STREAMS][2 * intermediate_dim]
  float *gate_s;       // [MAX_EXPERT_STREAMS][intermediate_dim]
  float *up_s;         // [MAX_EXPERT_STREAMS][intermediate_dim]
  float *gate_up_s;    // [MAX_EXPERT_STREAMS][intermediate_dim]
  float *tb2_s;        // [MAX_EXPERT_STREAMS][hidden_dim]
  // Batched MoE staging (for grouped GEMMs across slots/experts)
  int   *sel_sids;        // [max_batch * experts_per_token]
  int   *sel_eids;        // [max_batch * experts_per_token]
  float *sel_w;           // [max_batch * experts_per_token]
  float *mlp1_out_all;    // [max_batch * experts_per_token][2 * intermediate_dim]
  float *gate_up_all;     // [max_batch * experts_per_token][intermediate_dim]
} GPURunState;

// Per-GPU device context
typedef struct {
  int device_id;
  GPUTransformerWeights weights;
  GPURunState state;
  hipStream_t expert_streams[MAX_EXPERT_STREAMS];
  int num_expert_streams;
  int max_batch; // number of concurrent slots per GPU for continuous batching
  std::vector<hipStream_t> slot_streams; // persistent stream per slot
  hipStream_t batch_stream;              // shared stream for batched ops
  // Staging device buffers for batched indices and positions
  int *d_slot_ids;
  int *d_pos_list;
  int *d_token_ids;
} GPUContext;

static std::vector<GPUContext> gpu_contexts;
static thread_local GPUContext* t_ctx = nullptr;



// ------------------------- BF16 utils -------------------------

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __host__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
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

// ========================= FLOAT KERNELS =========================
// New kernels for float RunState buffers

__global__ void rmsnorm_kernel_float(float *o, const float *x, const hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
      float xv = x[j];
      ss += xv * xv;
    }
    ss = ss / size + 1e-5f;
    shared_ss = 1.0f / sqrtf(ss);
  }
  __syncthreads();
  if (tid < size) {
    float xv = x[tid];
    float wv = bfloat16_to_float(weight[tid]);
    o[tid] = wv * (shared_ss * xv);
  }
}

__global__ void matmul_kernel_float(float *xout, const float *x, const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      float wv = bfloat16_to_float(w[i * n + j]);
      float xv = x[j];
      val += wv * xv;
    }
    xout[i] = val;
  }
}

// Batched variant: same weights across batch, different inputs/outputs
__global__ void matmul_kernel_float_batched(float *xout, const float *x,
                                            const hip_bfloat16 *w, int n, int d,
                                            int batch, int out_stride, int in_stride,
                                            const int *slot_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  int b = blockIdx.y; // batch index
  if (i < d && b < batch) {
    float val = 0.0f;
    int sid = slot_ids[b];
    const float *x_b = x + (size_t)sid * (size_t)in_stride;
    for (int j = 0; j < n; j++) {
      float wv = bfloat16_to_float(w[i * n + j]);
      float xv = x_b[j];
      val += wv * xv;
    }
    float *out_s = xout + (size_t)sid * (size_t)out_stride;
    out_s[i] = val;
  }
}

__global__ void add_bias_kernel_float(float *output, const hip_bfloat16 *bias, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float ov = output[tid];
    float bv = bfloat16_to_float(bias[tid]);
    output[tid] = ov + bv;
  }
}

__global__ void add_bias_kernel_float_batched(float *output, const hip_bfloat16 *bias,
                                              int size, int batch, int out_stride,
                                              const int *slot_ids) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (tid < size && b < batch) {
    int sid = slot_ids[b];
    float *out_s = output + (size_t)sid * (size_t)out_stride;
    float ov = out_s[tid];
    float bv = bfloat16_to_float(bias[tid]);
    out_s[tid] = ov + bv;
  }
}

__global__ void copy_kernel_float(float *dst, const float *src, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) dst[tid] = src[tid];
}

__global__ void softmax_kernel_float(float *x, int size) {
  __shared__ float max_val;
  __shared__ float sum_val;
  if (threadIdx.x == 0) {
    if (size <= 0) return;
    max_val = x[0];
    for (int i = 1; i < size; i++) {
      float v = x[i];
      if (v > max_val) max_val = v;
    }
    sum_val = 0.0f;
    for (int i = 0; i < size; i++) {
      float v = expf(x[i] - max_val);
      x[i] = v;
      sum_val += v;
    }
    float inv = 1.0f / sum_val;
    for (int i = 0; i < size; i++) {
      x[i] *= inv;
    }
  }
}

// Serial top-k selection on device for a single vector of length n.
// Keeps K best (value, index) pairs in descending order.
__global__ void topk_kernel_float(const float *scores, int n, int k, float *topv, int *topi) {
  if (threadIdx.x != 0) return;
  // Initialize outputs
  for (int i = 0; i < k; ++i) {
    topv[i] = -INFINITY;
    topi[i] = -1;
  }
  // One-pass insertion into top-k buffers
  for (int j = 0; j < n; ++j) {
    float v = scores[j];
    // find position where v should be inserted among current topk
    int pos = -1;
    for (int t = 0; t < k; ++t) {
      if (v > topv[t]) { pos = t; break; }
    }
    if (pos >= 0) {
      // shift down [pos..k-2] -> [pos+1..k-1]
      for (int t = k - 1; t > pos; --t) {
        topv[t] = topv[t - 1];
        topi[t] = topi[t - 1];
      }
      topv[pos] = v;
      topi[pos] = j;
    }
  }
}

__global__ void compute_cos_sin_kernel_float(float *cos_out, float *sin_out, int pos,
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
    cos_out[tid] = cosf(val) * concentration;
    sin_out[tid] = sinf(val) * concentration;
  }
}

__global__ void apply_rotary_emb_kernel_float(float *x, const float *cos, const float *sin,
                                             int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (h < n_heads && i < half) {
    float x1 = x[h * head_dim + i];
    float x2 = x[h * head_dim + half + i];
    float c  = cos[i];
    float s  = sin[i];
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    x[h * head_dim + i]           = o1;
    x[h * head_dim + half + i]    = o2;
  }
}

// Batched RoPE: x is [B, n_heads*head_dim], cos/sin per batch
__global__ void apply_rotary_emb_kernel_float_batched(float *x, const float *cos, const float *sin,
                                                     int n_heads, int head_dim, int batch, int x_stride, int cs_stride,
                                                     const int *slot_ids) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int b = blockIdx.y;
  int half = head_dim / 2;
  if (b < batch && h < n_heads && i < half) {
    int sid = slot_ids[b];
    float *x_b = x + (size_t)sid * (size_t)x_stride;
    const float *cos_b = cos + (size_t)sid * (size_t)cs_stride;
    const float *sin_b = sin + (size_t)sid * (size_t)cs_stride;
    float x1 = x_b[h * head_dim + i];
    float x2 = x_b[h * head_dim + half + i];
    float c  = cos_b[i];
    float s  = sin_b[i];
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    x_b[h * head_dim + i]           = o1;
    x_b[h * head_dim + half + i]    = o2;
  }
}

__global__ void attention_scores_kernel_float(float *att, const float *q, const float *key_cache,
                                             int head_dim, int loff, int kv_dim, int kv_mul,
                                             int h, int pos, int seq_len, const float *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      float qv = q[i];
      float kv = k[i];
      score += qv * kv;
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      float mv = mask[pos * seq_len + t];
      score += mv;
    }
    att[t] = score;
  }
}

__global__ void attention_weighted_sum_kernel_float(float *tb, const float *att, const float *value_cache,
                                                   int head_dim, int loff, int kv_dim, int kv_mul,
                                                   int h, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < head_dim) {
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      float a  = att[t];
      float vv = v[i];
      sum += a * vv;
    }
    tb[h * head_dim + i] = sum;
  }
}

// Write attention sink value directly on device: reads BF16 sink and writes float
__global__ void add_attn_sink_from_bf16_kernel_float(float *att, int write_index,
                                                     const hip_bfloat16 *attn_sinks,
                                                     int layer, int n_attn_heads, int head) {
  if (threadIdx.x == 0) {
    size_t sink_idx = (size_t)layer * (size_t)n_attn_heads + (size_t)head;
    att[write_index] = bfloat16_to_float(attn_sinks[sink_idx]);
  }
}

__global__ void swiglu_kernel_float(float *gate_up, const float *gate, const float *up,
                                   int intermediate_dim, float swiglu_limit) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    float val   = gate[tid];
    float upval = up[tid];
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    gate_up[tid] = val;
  }
}

__global__ void split_gate_up_kernel_float(float *gate, float *up, const float *mlp1_out, int intermediate_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    gate[tid] = mlp1_out[2 * tid];
    up[tid]   = mlp1_out[2 * tid + 1];
  }
}

__global__ void weighted_sum_kernel_float(float *e_agg, const float *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float agg = e_agg[tid];
    float add = tb2[tid];
    e_agg[tid] = agg + add * expert_w;
  }
}

// Atomic variant for parallel accumulation across experts/streams
__global__ void weighted_sum_atomic_kernel_float(float *e_agg, const float *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float add = tb2[tid] * expert_w;
    atomicAdd(&e_agg[tid], add);
  }
}

__global__ void residual_connection_kernel_float(float *x, const float *residual, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float xv  = x[tid];
    float rv  = residual[tid];
    x[tid] = xv + rv;
  }
}

__global__ void residual_connection_kernel_float_batched(float *x, const float *residual, int size,
                                                         int batch, int stride, const int *slot_ids) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (tid < size && b < batch) {
    int sid = slot_ids[b];
    float *x_s = x + (size_t)sid * (size_t)stride;
    const float *r_s = residual + (size_t)sid * (size_t)stride;
    x_s[tid] = x_s[tid] + r_s[tid];
  }
}

// Zero e_agg for selected slots
__global__ void zero_selected_slots_kernel_float(float *buf, int stride, const int *slot_ids, int batch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (b < batch) {
    int sid = slot_ids[b];
    float *dst = buf + (size_t)sid * (size_t)stride;
    if (i < stride) dst[i] = 0.0f;
  }
}

// Fused TopK + Softmax for each slot in batch, serial per-slot (thread 0)
__global__ void topk_softmax_batched_kernel_float(const float *router_scores,
                                                  int n, int k,
                                                  float *topv, int *topi,
                                                  int score_stride, int topk_stride,
                                                  const int *slot_ids, int batch) {
  int b = blockIdx.x;
  if (b >= batch) return;
  if (threadIdx.x != 0) return;
  int sid = slot_ids[b];
  const float *scores = router_scores + (size_t)sid * (size_t)score_stride;
  float *topv_s = topv + (size_t)sid * (size_t)topk_stride;
  int   *topi_s = topi + (size_t)sid * (size_t)topk_stride;
  // init
  for (int i = 0; i < k; ++i) { topv_s[i] = -INFINITY; topi_s[i] = -1; }
  // top-k selection
  for (int j = 0; j < n; ++j) {
    float v = scores[j];
    int pos = -1;
    for (int t = 0; t < k; ++t) { if (v > topv_s[t]) { pos = t; break; } }
    if (pos >= 0) {
      for (int t = k - 1; t > pos; --t) { topv_s[t] = topv_s[t - 1]; topi_s[t] = topi_s[t - 1]; }
      topv_s[pos] = v; topi_s[pos] = j;
    }
  }
  // softmax over top-k
  float maxv = topv_s[0];
  for (int i = 1; i < k; ++i) if (topv_s[i] > maxv) maxv = topv_s[i];
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) { topv_s[i] = expf(topv_s[i] - maxv); sum += topv_s[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < k; ++i) topv_s[i] *= inv;
}

// Gather per-slot top-k into flattened [batch*k] lists
__global__ void gather_topk_pairs_kernel(const float *topv, const int *topi,
                                         const int *slot_ids, int batch, int k,
                                         float *sel_w, int *sel_eids, int *sel_sids) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int M = batch * k;
  if (idx < M) {
    int b = idx / k;
    int t = idx % k;
    int sid = slot_ids[b];
    sel_w[idx]    = topv[(size_t)sid * (size_t)k + t];
    sel_eids[idx] = topi[(size_t)sid * (size_t)k + t];
    sel_sids[idx] = sid;
  }
}

// Grouped MLP1: for each (sid,e) pair compute y = W_mlp1[e] * t[sid] + b_mlp1[e]
__global__ void mlp1_grouped_kernel_float(float *mlp1_out_all,
                                          const float *t, const hip_bfloat16 *w_mlp1, const hip_bfloat16 *b_mlp1,
                                          const int *sel_sids, const int *sel_eids,
                                          int M, int hidden_dim, int intermediate_dim,
                                          int layer, int n_experts) {
  int o = blockIdx.x * blockDim.x + threadIdx.x; // output idx in [0, 2*intermediate_dim)
  int m = blockIdx.y; // selection idx
  int out_dim = 2 * intermediate_dim;
  if (m < M && o < out_dim) {
    int sid = sel_sids[m];
    int eid = sel_eids[m];
    const float *x = t + (size_t)sid * (size_t)hidden_dim;
    size_t base = ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)out_dim * (size_t)hidden_dim;
    const hip_bfloat16 *w = w_mlp1 + base + (size_t)o * (size_t)hidden_dim;
    float sum = 0.0f;
    for (int j = 0; j < hidden_dim; ++j) {
      sum += bfloat16_to_float(w[j]) * x[j];
    }
    const hip_bfloat16 *b = b_mlp1 + ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)out_dim;
    mlp1_out_all[(size_t)m * (size_t)out_dim + (size_t)o] = sum + bfloat16_to_float(b[o]);
  }
}

// Split + SwiGLU batched for all selections
__global__ void swiglu_grouped_kernel_float(float *gate_up_all, const float *mlp1_out_all,
                                            int M, int intermediate_dim, float swiglu_limit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // [0, intermediate_dim)
  int m = blockIdx.y;
  if (m < M && i < intermediate_dim) {
    int out_dim = 2 * intermediate_dim;
    const float *row = mlp1_out_all + (size_t)m * (size_t)out_dim;
    float gate = row[2 * i + 0];
    float up   = row[2 * i + 1];
    if (gate >  swiglu_limit) gate =  swiglu_limit;
    if (up   >  swiglu_limit) up   =  swiglu_limit;
    if (up   < -swiglu_limit) up   = -swiglu_limit;
    const float alpha = 1.702f;
    gate *= (1.0f / (1.0f + expf(-alpha * gate)));
    float v = gate * (up + 1.0f);
    gate_up_all[(size_t)m * (size_t)intermediate_dim + (size_t)i] = v;
  }
}

// Grouped MLP2 + accumulate: for each (sid,e) pair compute y = W_mlp2[e] * gate_up[m] + b_mlp2[e]
// and atomically accumulate into e_agg[sid]
__global__ void mlp2_accum_grouped_kernel_float(float *e_agg,
                                                const float *gate_up_all,
                                                const hip_bfloat16 *w_mlp2, const hip_bfloat16 *b_mlp2,
                                                const int *sel_sids, const int *sel_eids, const float *sel_w,
                                                int M, int hidden_dim, int intermediate_dim,
                                                int layer, int n_experts, int eagg_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // output idx in [0, hidden_dim)
  int m = blockIdx.y;
  if (m < M && i < hidden_dim) {
    int sid = sel_sids[m];
    int eid = sel_eids[m];
    float ew = sel_w[m];
    const float *g = gate_up_all + (size_t)m * (size_t)intermediate_dim;
    size_t base = ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)hidden_dim * (size_t)intermediate_dim;
    const hip_bfloat16 *w = w_mlp2 + base + (size_t)i * (size_t)intermediate_dim;
    float sum = 0.0f;
    for (int j = 0; j < intermediate_dim; ++j) {
      sum += bfloat16_to_float(w[j]) * g[j];
    }
    const hip_bfloat16 *b = b_mlp2 + ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)hidden_dim;
    float y = sum + bfloat16_to_float(b[i]);
    atomicAdd(&e_agg[(size_t)sid * (size_t)eagg_stride + (size_t)i], ew * y);
  }
}

__global__ void bf16_to_float_kernel(float *out, const hip_bfloat16 *in, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    out[tid] = bfloat16_to_float(in[tid]);
  }
}

// Fused Multi-Head Attention Kernel: xử lý tất cả heads song song
__global__ void fused_multihead_attention_kernel(
    const float *all_q,           // Q cho tất cả heads [n_heads * head_dim] 
    const float *key_cache,       // K cache [layers * seq_len * kv_dim]
    const float *value_cache,     // V cache [layers * seq_len * kv_dim]
    const hip_bfloat16 *attn_sinks, // Attention sinks [layers * n_heads]
    float *all_att,               // Attention scores buffer [n_heads * seq_len]
    float *all_output,            // Output [n_heads * head_dim]
    const float *mask,            // Sliding window mask (optional)
    int head_dim, int loff, int kv_dim, int kv_mul,
    int n_heads, int pos, int seq_len, int layer,
    bool can_add_sink) {
  
  int h = blockIdx.y;  // Head index
  if (h >= n_heads) return;
  
  const float *q_head = all_q + h * head_dim;
  float *att = all_att + h * seq_len;
  float *output = all_output + h * head_dim;
  
  // Phase 1: Compute attention scores (parallel across time steps)
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += q_head[i] * k[i];
    }
    score /= sqrtf((float)head_dim);
    
    // Apply sliding window mask if enabled
    if (mask != NULL) {
      score += mask[pos * seq_len + t];
    }
    att[t] = score;
  }
  
  // Sync within each head's block before proceeding
  __syncthreads();
  
  // Phase 2: Add attention sink (only first thread of first block for each head)
  if (threadIdx.x == 0 && blockIdx.x == 0 && can_add_sink) {
    size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    att[pos + 1] = bfloat16_to_float(attn_sinks[sink_idx]);
  }
  
  // Global sync: wait for all blocks to finish scores + sink
  __syncthreads();
  
  // Phase 3: Softmax (only first thread of first block for each head)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int len = can_add_sink ? (pos + 2) : (pos + 1);
    if (len <= 0) return;
    
    // Find max
    float max_val = att[0];
    for (int i = 1; i < len; i++) {
      if (att[i] > max_val) max_val = att[i];
    }
    
    // Exp and sum
    float sum_val = 0.0f;
    for (int i = 0; i < len; i++) {
      float v = expf(att[i] - max_val);
      att[i] = v;
      sum_val += v;
    }
    
    // Normalize
    float inv = 1.0f / sum_val;
    for (int i = 0; i < len; i++) {
      att[i] *= inv;
    }
  }
  
  // Global sync: wait for softmax to complete
  __syncthreads();
  
  // Phase 4: Weighted sum with values
  // Each thread handles one output dimension
  if (blockIdx.x == 0 && threadIdx.x < head_dim) {
    int i = threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      sum += att[t] * v[i];
    }
    output[i] = sum;
  }
}

// Batched fused attention: process [batch, heads] where each batch has its own pos
__global__ void fused_multihead_attention_kernel_batched(
    const float *all_q,           // [B, n_heads*head_dim]
    const float *key_cache,       // [B, layers*seq_len*kv_dim]
    const float *value_cache,     // [B, layers*seq_len*kv_dim]
    const hip_bfloat16 *attn_sinks, // [layers * n_heads]
    float *all_att,               // [B, n_heads*seq_len]
    float *all_output,            // [B, n_heads*head_dim]
    const float *mask,            // Sliding window mask (optional)
    int head_dim, int loff, int kv_dim, int kv_mul,
    int n_heads, int seq_len, int layer,
    const int *pos_list,          // [B]
    bool can_add_sink,            // same rule per slot
    int batch,
    int q_stride,
    int kv_block_size,
    int att_block_size,
    const int *slot_ids) {

  int h = blockIdx.y;   // head index
  int b = blockIdx.z;   // batch index
  if (h >= n_heads || b >= batch) return;

  int pos = pos_list[b];
  int sid = slot_ids[b];
  const float *q_head = all_q + (size_t)sid * (size_t)q_stride + (size_t)h * (size_t)head_dim;
  float *att = all_att + (size_t)sid * (size_t)att_block_size + (size_t)h * (size_t)seq_len;
  float *output = all_output + (size_t)sid * (size_t)(n_heads * head_dim) + (size_t)h * (size_t)head_dim;

  const float *key_cache_base   = key_cache   + (size_t)sid * (size_t)kv_block_size;
  const float *value_cache_base = value_cache + (size_t)sid * (size_t)kv_block_size;

  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const float *k = key_cache_base + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += q_head[i] * k[i];
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      score += mask[pos * seq_len + t];
    }
    att[t] = score;
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0 && can_add_sink && (pos + 1) < seq_len) {
    size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    att[pos + 1] = bfloat16_to_float(attn_sinks[sink_idx]);
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int len = can_add_sink ? (pos + 2) : (pos + 1);
    if (len > 0) {
      float max_val = att[0];
      for (int i = 1; i < len; i++) max_val = fmaxf(max_val, att[i]);
      float sum_val = 0.0f;
      for (int i = 0; i < len; i++) { float v = expf(att[i] - max_val); att[i] = v; sum_val += v; }
      float inv = 1.0f / sum_val;
      for (int i = 0; i < len; i++) att[i] *= inv;
    }
  }
  __syncthreads();

  if (blockIdx.x == 0 && threadIdx.x < head_dim) {
    int i = threadIdx.x;
    float sum = 0.0f;
    for (int t2 = 0; t2 <= pos; t2++) {
      const float *v = value_cache_base + loff + t2 * kv_dim + (h / kv_mul) * head_dim;
      sum += att[t2] * v[i];
    }
    output[i] = sum;
  }
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
void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch);
void gpu_free_run_state(GPURunState *s);


// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
void gpu_free_weights(GPUTransformerWeights *gpu_w);



// ------------------------- Forward -------------------------

// slot: which per-GPU slot (0..max_batch-1) this sequence uses for KV-cache/att buffers
float *gpu_forward(Transformer *transformer, int token, int pos, GPUContext *gpu_ctx, int slot, hipStream_t stream) {
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

  // Per-slot activation base pointers
  float *x_slot   = gpu_ctx->state.x   + (size_t)slot * (size_t)hidden_dim;
  float *t_slot   = gpu_ctx->state.t   + (size_t)slot * (size_t)hidden_dim;
  float *tb_slot  = gpu_ctx->state.tb  + (size_t)slot * (size_t)(head_dim * p->n_attn_heads);
  float *tb2_slot = gpu_ctx->state.tb2 + (size_t)slot * (size_t)hidden_dim;

  float *router_score_slot = gpu_ctx->state.router_score + (size_t)slot * (size_t)n_experts;
  float *topk_v_slot       = gpu_ctx->state.topk_v       + (size_t)slot * (size_t)p->experts_per_token;
  int   *topk_i_slot       = gpu_ctx->state.topk_i       + (size_t)slot * (size_t)p->experts_per_token;

  float *mlp1_out_slot = gpu_ctx->state.mlp1_out + (size_t)slot * (size_t)(2 * intermediate_dim);
  float *gate_slot     = gpu_ctx->state.gate     + (size_t)slot * (size_t)intermediate_dim;
  float *up_slot       = gpu_ctx->state.up       + (size_t)slot * (size_t)intermediate_dim;
  float *gate_up_slot  = gpu_ctx->state.gate_up  + (size_t)slot * (size_t)intermediate_dim;
  float *e_agg_slot    = gpu_ctx->state.e_agg    + (size_t)slot * (size_t)hidden_dim;

  float *qkv_slot = gpu_ctx->state.qkv + (size_t)slot * (size_t)(head_dim * (p->n_attn_heads + 2 * p->n_kv_heads));

  // Token embedding: convert BF16 weights to float state
  hip_bfloat16 *token_row = gpu_ctx->weights.token_embedding_table + (size_t)token * (size_t)hidden_dim;
  bf16_to_float_kernel<<<grid_hidden, block, 0, stream>>>(x_slot, token_row, hidden_dim);

  // RoPE precomputation: tính cos/sin một lần cho tất cả layers (chỉ phụ thuộc pos và head_dim)
  float ntk_beta  = 32.0f;
  float ntk_alpha = 1.0f;
  dim3 grid_rope = dim3(((head_dim / 2) + block.x - 1) / block.x);
  float *cos_slot = gpu_ctx->state.cos_vals + (size_t)slot * (size_t)(head_dim/2);
  float *sin_slot = gpu_ctx->state.sin_vals + (size_t)slot * (size_t)(head_dim/2);
  compute_cos_sin_kernel_float<<<grid_rope, block, 0, stream>>>(cos_slot, sin_slot, pos,
                                                    p->rope_theta, head_dim, p->rope_scaling_factor,
                                                    p->initial_context_length, ntk_beta, ntk_alpha);

  // Compute base offsets for this slot
  const size_t kv_block_size  = (size_t)p->n_layers * (size_t)p->seq_len * (size_t)kv_dim;
  const size_t att_block_size = (size_t)p->n_attn_heads * (size_t)p->seq_len;
  float *key_cache_base   = gpu_ctx->state.key_cache   + (size_t)slot * kv_block_size;
  float *value_cache_base = gpu_ctx->state.value_cache + (size_t)slot * kv_block_size;
  float *att_base         = gpu_ctx->state.att         + (size_t)slot * att_block_size;

  // Layers
  for (int l = 0; l < p->n_layers; l++) {
    // Attention RMSNorm
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_float<<<grid_hidden, block, 0, stream>>>(t_slot, x_slot, rms_attn_w, hidden_dim);

    // QKV projection
    int loff = l * p->seq_len * kv_dim;
    const hip_bfloat16 *w_qkv = gpu_ctx->weights.w_qkv + (size_t)l * (size_t)hidden_dim *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);

    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 grid_qkv = dim3((qkv_out_dim + block.x - 1) / block.x);
    matmul_kernel_float<<<grid_qkv, block, 0, stream>>>(qkv_slot, t_slot, w_qkv, hidden_dim, qkv_out_dim);
    add_bias_kernel_float<<<grid_qkv, block, 0, stream>>>(qkv_slot, b_qkv, qkv_out_dim);

    // Q/K/V aliases: tránh D2D copy bằng cách dùng pointer offsets
    int q_size = head_dim * p->n_attn_heads;
    int k_size = head_dim * p->n_kv_heads;
    int v_size = head_dim * p->n_kv_heads;

    float *q = qkv_slot;                    // Q starts at beginning
    float *k = qkv_slot + q_size;           // K starts after Q
    float *v = qkv_slot + q_size + k_size;  // V starts after Q+K

    // Store K, V in cache trực tiếp từ qkv buffer (before RoPE)
    float *k_cache_pos = key_cache_base   + loff + pos * kv_dim;
    float *v_cache_pos = value_cache_base + loff + pos * kv_dim;
    HIP_CHECK(hipMemcpyAsync(k_cache_pos, k, k_size * sizeof(float), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(v_cache_pos, v, v_size * sizeof(float), hipMemcpyDeviceToDevice, stream));

    // RoPE
    apply_rotary_emb_kernel_float<<<p->n_attn_heads, head_dim/2, 0, stream>>>(q, cos_slot, sin_slot, p->n_attn_heads, head_dim);
    apply_rotary_emb_kernel_float<<<p->n_kv_heads,   head_dim/2, 0, stream>>>(k, cos_slot, sin_slot, p->n_kv_heads,   head_dim);

    // Update K cache after RoPE
    HIP_CHECK(hipMemcpyAsync(k_cache_pos, k, k_size * sizeof(float), hipMemcpyDeviceToDevice, stream));

    // Fused Multi-head attention: xử lý tất cả heads song song
    bool can_add_sink = (pos + 1) < p->seq_len;
    const float *mask = (p->sliding_window > 0 && l % 2 == 0) ? gpu_ctx->state.mask : NULL;
    
    // Launch fused kernel với grid 2D: [time_steps, heads]
    dim3 grid_fused(((pos + 1) + block.x - 1) / block.x, p->n_attn_heads);
    fused_multihead_attention_kernel<<<grid_fused, block, 0, stream>>>(
        q,                           // All Q heads
        key_cache_base,              // K cache (slot base)
        value_cache_base,            // V cache (slot base)
        gpu_ctx->weights.attn_sinks, // Attention sinks
        att_base,                    // Attention scores buffer (slot base)
        tb_slot,                     // Output buffer
        mask,                        // Sliding window mask
        head_dim, loff, kv_dim, kv_mul,
        p->n_attn_heads, pos, p->seq_len, l,
        can_add_sink);

    // Output projection of attention
    const hip_bfloat16 *w_o = gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    matmul_kernel_float<<<grid_hidden, block, 0, stream>>>(tb2_slot, tb_slot, w_o, head_dim * p->n_attn_heads, hidden_dim);
    add_bias_kernel_float<<<grid_hidden, block, 0, stream>>>(tb2_slot, b_o, hidden_dim);

    // Residual
    residual_connection_kernel_float<<<grid_hidden, block, 0, stream>>>(x_slot, tb2_slot, hidden_dim);

    // -------- MoE --------
    // FFN RMSNorm
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_float<<<grid_hidden, block, 0, stream>>>(t_slot, x_slot, rms_ffn_w, hidden_dim);

    // Router
    const hip_bfloat16 *w_router = gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)n_experts;
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)n_experts;
    matmul_kernel_float<<<grid_experts, block, 0, stream>>>(router_score_slot, t_slot, w_router, hidden_dim, n_experts);
    add_bias_kernel_float<<<grid_experts, block, 0, stream>>>(router_score_slot, b_router, n_experts);

    // GPU Router top-k: find top-k over router_score on device
    topk_kernel_float<<<1, 1, 0, stream>>>(router_score_slot, n_experts, p->experts_per_token,
                                topk_v_slot, topk_i_slot);
    // Softmax the selected top-k (device)
    softmax_kernel_float<<<1, 1, 0, stream>>>(topk_v_slot, p->experts_per_token);
    HIP_CHECK(hipStreamSynchronize(stream));

    // Fetch only K results to host for control flow (no H2D copy back)
    std::vector<float> cpu_topk_v(p->experts_per_token);
    std::vector<int>   cpu_topk_i(p->experts_per_token);
    HIP_CHECK(hipMemcpy(cpu_topk_v.data(), topk_v_slot, p->experts_per_token * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cpu_topk_i.data(), topk_i_slot, p->experts_per_token * sizeof(int),   hipMemcpyDeviceToHost));

    // Init aggregation
    HIP_CHECK(hipMemsetAsync(e_agg_slot, 0, hidden_dim * sizeof(float), stream));

    // Per-slot expert computation on the slot stream
    {
      dim3 grid_mlp1 = dim3(((2 * intermediate_dim) + block.x - 1) / block.x);
      for (int sel_idx = 0; sel_idx < p->experts_per_token; ++sel_idx) {
        int e = cpu_topk_i[sel_idx];
        float expert_w = cpu_topk_v[sel_idx];
        const hip_bfloat16 *w_mlp1 = gpu_ctx->weights.w_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
        const hip_bfloat16 *b_mlp1 = gpu_ctx->weights.b_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);

        // MLP1
        matmul_kernel_float<<<grid_mlp1, block, 0, stream>>>(mlp1_out_slot, t_slot, w_mlp1, hidden_dim, 2 * intermediate_dim);
        add_bias_kernel_float<<<grid_mlp1, block, 0, stream>>>(mlp1_out_slot, b_mlp1, 2 * intermediate_dim);

        // Split + SwiGLU
        split_gate_up_kernel_float<<<grid_intermediate, block, 0, stream>>>(gate_slot, up_slot, mlp1_out_slot, intermediate_dim);
        swiglu_kernel_float<<<grid_intermediate, block, 0, stream>>>(gate_up_slot, gate_slot, up_slot, intermediate_dim, p->swiglu_limit);

        // MLP2
        const hip_bfloat16 *w_mlp2 = gpu_ctx->weights.w_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
        const hip_bfloat16 *b_mlp2 = gpu_ctx->weights.b_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;
        matmul_kernel_float<<<grid_hidden, block, 0, stream>>>(tb2_slot, gate_up_slot, w_mlp2, intermediate_dim, hidden_dim);
        add_bias_kernel_float<<<grid_hidden, block, 0, stream>>>(tb2_slot, b_mlp2, hidden_dim);

        // Accumulate into e_agg
        weighted_sum_kernel_float<<<grid_hidden, block, 0, stream>>>(e_agg_slot, tb2_slot, expert_w, hidden_dim);
      }
    }

    residual_connection_kernel_float<<<grid_hidden, block, 0, stream>>>(x_slot, e_agg_slot, hidden_dim);
  }

  // Final RMSNorm
  rmsnorm_kernel_float<<<grid_hidden, block, 0, stream>>>(x_slot, x_slot, gpu_ctx->weights.rms_out_w, hidden_dim);

  // Output logits directly: W(BF16)*x(FP32)->logits(FP32)
  dim3 grid_vocab2 = dim3((p->vocab_size + block.x - 1) / block.x);
  float *logits_slot = gpu_ctx->state.logits_fp32 + (size_t)slot * (size_t)p->vocab_size;
  matvec_wbf16_xfp32_outfp32<<<grid_vocab2, block, 0, stream>>>(logits_slot, x_slot, gpu_ctx->weights.out,
                                                     /*n=*/hidden_dim, /*d=*/p->vocab_size);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy logits to CPU (thread-local buffer across calls)
  thread_local float *cpu_logits = NULL;
  if (cpu_logits == NULL) {
    cpu_logits = (float*)malloc(p->vocab_size * sizeof(float));
  }
  HIP_CHECK(hipMemcpy(cpu_logits, logits_slot, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
  return cpu_logits;
}

// Batched attention path over all active slots; includes embedding, per-layer
// attention (RMSNorm->QKV->RoPE->FusedAttn->W_o+residual). MoE is batched & grouped across slots/experts.
// host_logits_out: optional, if not null and do_final=true, copies logits of each slot
// into host buffer packed as [batch, vocab].
void gpu_forward_batch_step(Transformer *transformer, GPUContext *gpu_ctx,
                            const int *h_tokens, const int *h_pos_list, const int *h_slot_ids,
                            int batch, float *host_logits_out) {
  Config *p = &transformer->config;
  int head_dim         = p->head_dim;
  int hidden_dim       = p->hidden_dim;
  int kv_dim           = p->head_dim * p->n_kv_heads;
  int kv_mul           = p->n_attn_heads / p->n_kv_heads;

  // Copy indices to device staging
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_slot_ids,  h_slot_ids,  batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_pos_list,  h_pos_list,  batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_token_ids, h_tokens,    batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));

  // 1) Embedding gather: x_slot[sid] = Emb[token]
  // Kernel: for b in batch and i in hidden_dim, copy embedding row
  // Implement simple gather using a kernel
  auto launch_embed = [&]() {
    struct Kernel {
      static __global__ void run(float *x, const hip_bfloat16 *emb, const int *tok, const int *slot_ids,
                                 int hidden_dim, int batch, int x_stride) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (i < hidden_dim && b < batch) {
          int sid = slot_ids[b];
          int t = tok[b];
          const hip_bfloat16 *row = emb + (size_t)t * (size_t)hidden_dim;
          float *x_s = x + (size_t)sid * (size_t)x_stride;
          x_s[i] = bfloat16_to_float(row[i]);
        }
      }
    };
    dim3 block(256);
    dim3 grid((hidden_dim + block.x - 1) / block.x, batch);
    Kernel::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.x, gpu_ctx->weights.token_embedding_table,
                                                           gpu_ctx->d_token_ids, gpu_ctx->d_slot_ids,
                                                           hidden_dim, batch, hidden_dim);
  };
  launch_embed();

  // 2) RoPE precompute per slot
  {
    int d_half = head_dim / 2;
    dim3 block(256);
    dim3 grid((d_half + block.x - 1) / block.x, batch);
    // Batched cos/sin compute kernel
    struct KernelCS {
      static __global__ void run(float *cos_out, float *sin_out, const int *pos_list, const int *slot_ids,
                                 int batch, float base, int head_dim, float scaling_factor,
                                 float initial_context_length, float ntk_beta, float ntk_alpha, int cs_stride) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (b >= batch) return;
        int half = head_dim / 2;
        if (tid < half) {
          int sid = slot_ids[b];
          float freq = powf(base, ((float)(2 * tid)) / (float)head_dim);
          float inv_freq;
          if (scaling_factor > 1.0f) {
            float low  = half * logf(initial_context_length / (ntk_beta  * 2.0f * M_PI)) / logf(base);
            float high = half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
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
          float val = (float)pos_list[b] * inv_freq;
          float *cos_b = cos_out + (size_t)sid * (size_t)cs_stride;
          float *sin_b = sin_out + (size_t)sid * (size_t)cs_stride;
          cos_b[tid] = cosf(val) * concentration;
          sin_b[tid] = sinf(val) * concentration;
        }
      }
    };
    KernelCS::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
                                                             gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids,
                                                             batch, p->rope_theta, head_dim, p->rope_scaling_factor,
                                                             p->initial_context_length, 32.0f, 1.0f, d_half);
  }

  // Precompute constants for caches
  const size_t kv_block_size  = (size_t)p->n_layers * (size_t)p->seq_len * (size_t)kv_dim;
  const size_t att_block_size = (size_t)p->n_attn_heads * (size_t)p->seq_len;
  const int q_size = head_dim * p->n_attn_heads;
  const int k_size = head_dim * p->n_kv_heads;
  const int v_size = head_dim * p->n_kv_heads;

  // Helper kernels to write caches from qkv buffers
  struct KVWriterV {
    static __global__ void run(const float *qkv, float *value_cache, int batch, int qkv_stride,
                               int v_offset, int v_size, int kv_dim, int loff,
                               size_t kv_block_size,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < v_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        const float *v_src = qkv + (size_t)sid * (size_t)qkv_stride + (size_t)v_offset;
        float *v_dst = value_cache + (size_t)sid * kv_block_size + (size_t)loff + (size_t)pos * (size_t)kv_dim;
        v_dst[i] = v_src[i];
      }
    }
  };
  struct KVWriterK {
    static __global__ void run(const float *qkv, float *key_cache, int batch, int qkv_stride,
                               int k_offset, int k_size, int kv_dim, int loff,
                               size_t kv_block_size,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < k_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        const float *k_src = qkv + (size_t)sid * (size_t)qkv_stride + (size_t)k_offset;
        float *k_dst = key_cache + (size_t)sid * kv_block_size + (size_t)loff + (size_t)pos * (size_t)kv_dim;
        k_dst[i] = k_src[i];
      }
    }
  };

  // 3) Loop over layers with batched Attention and W_o
  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm per slot (still per-slot; can be batched later)
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    for (int bi = 0; bi < batch; ++bi) {
      int sid = h_slot_ids[bi];
      dim3 block(256), grid_hidden((hidden_dim + block.x - 1) / block.x);
      rmsnorm_kernel_float<<<grid_hidden, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t + (size_t)sid * (size_t)hidden_dim,
        gpu_ctx->state.x + (size_t)sid * (size_t)hidden_dim,
        rms_attn_w, hidden_dim);
    }

    // QKV projection (batched)
    int loff = l * p->seq_len * kv_dim;
    const hip_bfloat16 *w_qkv = gpu_ctx->weights.w_qkv + (size_t)l * (size_t)hidden_dim *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 block_mm(256), grid_mm((qkv_out_dim + block_mm.x - 1) / block_mm.x, batch);
    matmul_kernel_float_batched<<<grid_mm, block_mm, 0, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.qkv, gpu_ctx->state.t, w_qkv, hidden_dim, qkv_out_dim,
      batch, qkv_out_dim, hidden_dim, gpu_ctx->d_slot_ids);
    add_bias_kernel_float_batched<<<grid_mm, block_mm, 0, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.qkv, b_qkv, qkv_out_dim, batch, qkv_out_dim, gpu_ctx->d_slot_ids);

    // Write V to cache (before RoPE)
    {
      dim3 block(256), grid((v_size + block.x - 1) / block.x, batch);
      KVWriterV::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.qkv, gpu_ctx->state.value_cache,
        batch, qkv_out_dim, q_size + k_size, v_size, kv_dim, loff, kv_block_size, gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
    }

    // Apply RoPE to Q and K (batched)
    apply_rotary_emb_kernel_float_batched<<<p->n_attn_heads, head_dim/2, 0, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.qkv, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
      p->n_attn_heads, head_dim, batch, qkv_out_dim, head_dim/2, gpu_ctx->d_slot_ids);
    apply_rotary_emb_kernel_float_batched<<<p->n_kv_heads, head_dim/2, 0, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.qkv + q_size, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
      p->n_kv_heads, head_dim, batch, qkv_out_dim, head_dim/2, gpu_ctx->d_slot_ids);

    // Write K (after RoPE)
    {
      dim3 block(256), grid((k_size + block.x - 1) / block.x, batch);
      KVWriterK::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.qkv, gpu_ctx->state.key_cache,
        batch, qkv_out_dim, q_size, k_size, kv_dim, loff, kv_block_size, gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
    }

    // Fused attention (batched) -> tb
    {
      dim3 block(256);
      int max_pos = 0; for (int bi = 0; bi < batch; ++bi) max_pos = std::max(max_pos, h_pos_list[bi]);
      dim3 grid(((max_pos + 1) + block.x - 1) / block.x, p->n_attn_heads, batch); // conservative grid on time; pos varies
      bool can_add_sink = true;
      const float *layer_mask = (p->sliding_window > 0 && (l % 2 == 0)) ? gpu_ctx->state.mask : NULL;
      fused_multihead_attention_kernel_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.qkv, gpu_ctx->state.key_cache, gpu_ctx->state.value_cache, gpu_ctx->weights.attn_sinks,
        gpu_ctx->state.att, gpu_ctx->state.tb, layer_mask, head_dim, loff, kv_dim, kv_mul,
        p->n_attn_heads, p->seq_len, l, gpu_ctx->d_pos_list, can_add_sink, batch,
        qkv_out_dim, kv_block_size, att_block_size, gpu_ctx->d_slot_ids);
    }

    // Output projection W_o (batched) tb2 = tb @ W_o + b_o
    const hip_bfloat16 *w_o = gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      matmul_kernel_float_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.tb2, gpu_ctx->state.tb, w_o, head_dim * p->n_attn_heads, hidden_dim,
        batch, hidden_dim, head_dim * p->n_attn_heads, gpu_ctx->d_slot_ids);
      add_bias_kernel_float_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.tb2, b_o, hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }

    // Residual add: x += tb2
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      residual_connection_kernel_float_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.x, gpu_ctx->state.tb2, hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }

    // -------- MoE batched & grouped across slots and experts --------
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    const hip_bfloat16 *w_router = gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)p->n_experts;
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)p->n_experts;
    // 1) FFN RMSNorm per slot
    for (int bi = 0; bi < batch; ++bi) {
      int sid = h_slot_ids[bi];
      dim3 block(256), grid_hidden((hidden_dim + block.x - 1) / block.x);
      rmsnorm_kernel_float<<<grid_hidden, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t + (size_t)sid * (size_t)hidden_dim,
        gpu_ctx->state.x + (size_t)sid * (size_t)hidden_dim,
        rms_ffn_w, hidden_dim);
    }
    // 2) Router matmul + bias (batched across slots)
    {
      dim3 block(256), grid((p->n_experts + block.x - 1) / block.x, batch);
      matmul_kernel_float_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.router_score, gpu_ctx->state.t, w_router, hidden_dim, p->n_experts,
        batch, p->n_experts, hidden_dim, gpu_ctx->d_slot_ids);
      add_bias_kernel_float_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.router_score, b_router, p->n_experts, batch, p->n_experts, gpu_ctx->d_slot_ids);
    }
    // 3) Top-k + softmax on device for all slots (fused per-slot kernel)
    {
      dim3 grid(batch), block(1);
      topk_softmax_batched_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.router_score, p->n_experts, p->experts_per_token,
        gpu_ctx->state.topk_v, gpu_ctx->state.topk_i,
        p->n_experts, p->experts_per_token,
        gpu_ctx->d_slot_ids, batch);
    }
    // 4) Zero e_agg for selected slots
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      zero_selected_slots_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.e_agg, hidden_dim, gpu_ctx->d_slot_ids, batch);
    }
    // 5) Gather top-k selections into flattened arrays [batch * K]
    int M = batch * p->experts_per_token;
    {
      dim3 block(256), grid((M + block.x - 1) / block.x);
      gather_topk_pairs_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.topk_v, gpu_ctx->state.topk_i,
        gpu_ctx->d_slot_ids, batch, p->experts_per_token,
        gpu_ctx->state.sel_w, gpu_ctx->state.sel_eids, gpu_ctx->state.sel_sids);
    }
    // 6) Grouped MLP1 across all selections
    {
      dim3 block(256), grid((2 * p->intermediate_dim + block.x - 1) / block.x, M);
      mlp1_grouped_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.mlp1_out_all,
        gpu_ctx->state.t, gpu_ctx->weights.w_mlp1, gpu_ctx->weights.b_mlp1,
        gpu_ctx->state.sel_sids, gpu_ctx->state.sel_eids,
        M, hidden_dim, p->intermediate_dim,
        l, p->n_experts);
    }
    // 7) Split + SwiGLU batched across all selections
    {
      dim3 block(256), grid((p->intermediate_dim + block.x - 1) / block.x, M);
      swiglu_grouped_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.gate_up_all, gpu_ctx->state.mlp1_out_all,
        M, p->intermediate_dim, p->swiglu_limit);
    }
    // 8) Grouped MLP2 and accumulate into e_agg per slot with expert weights
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, M);
      mlp2_accum_grouped_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.e_agg,
        gpu_ctx->state.gate_up_all,
        gpu_ctx->weights.w_mlp2, gpu_ctx->weights.b_mlp2,
        gpu_ctx->state.sel_sids, gpu_ctx->state.sel_eids, gpu_ctx->state.sel_w,
        M, hidden_dim, p->intermediate_dim,
        l, p->n_experts, hidden_dim);
    }
    // 9) Residual x += e_agg for selected slots
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      residual_connection_kernel_float_batched<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.x, gpu_ctx->state.e_agg, hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }
  }

  // Final RMSNorm and logits per slot (sequential for now)
  if (host_logits_out != nullptr) {
    for (int bi = 0; bi < batch; ++bi) {
      int sid = h_slot_ids[bi];
      dim3 block(256), grid_hidden((hidden_dim + block.x - 1) / block.x);
      rmsnorm_kernel_float<<<grid_hidden, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.x + (size_t)sid * (size_t)hidden_dim,
        gpu_ctx->state.x + (size_t)sid * (size_t)hidden_dim,
        gpu_ctx->weights.rms_out_w, hidden_dim);
      dim3 grid_vocab((p->vocab_size + block.x - 1) / block.x);
      matvec_wbf16_xfp32_outfp32<<<grid_vocab, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.logits_fp32 + (size_t)sid * (size_t)p->vocab_size,
        gpu_ctx->state.x + (size_t)sid * (size_t)hidden_dim,
        gpu_ctx->weights.out, hidden_dim, p->vocab_size);
      HIP_CHECK(hipMemcpyAsync(host_logits_out + (size_t)bi * (size_t)p->vocab_size,
                               gpu_ctx->state.logits_fp32 + (size_t)sid * (size_t)p->vocab_size,
                               p->vocab_size * sizeof(float), hipMemcpyDeviceToHost, gpu_ctx->batch_stream));
    }
    HIP_CHECK(hipStreamSynchronize(gpu_ctx->batch_stream));
  }
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
    // Configure per-GPU max batch from env, default 8
    const char *env_mb = getenv("GETP_MAX_BATCH");
    int mb = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 8;
    gpu_contexts[d].max_batch = mb;
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config, gpu_contexts[d].max_batch);
    // Create per-slot persistent streams
    gpu_contexts[d].slot_streams.resize(gpu_contexts[d].max_batch);
    for (int s = 0; s < gpu_contexts[d].max_batch; ++s) {
      HIP_CHECK(hipStreamCreate(&gpu_contexts[d].slot_streams[s]));
    }
    // Create batch stream and staging device buffers
    HIP_CHECK(hipStreamCreate(&gpu_contexts[d].batch_stream));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_slot_ids,  gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_pos_list,  gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_token_ids, gpu_contexts[d].max_batch * sizeof(int)));
    // Create expert streams (up to MAX_EXPERT_STREAMS)
    gpu_contexts[d].num_expert_streams = MAX_EXPERT_STREAMS;
    for (int s = 0; s < gpu_contexts[d].num_expert_streams; ++s) {
      HIP_CHECK(hipStreamCreate(&gpu_contexts[d].expert_streams[s]));
    }

    if (transformer->config.sliding_window > 0 && gpu_contexts[d].state.mask != NULL) {
      int S = transformer->config.seq_len;
      std::vector<float> cpu_mask((size_t)S * (size_t)S);
      for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
          cpu_mask[(size_t)i * (size_t)S + (size_t)j] = (i - j >= transformer->config.sliding_window) ? -INFINITY : 0.0f;
        }
      }
      // Copy mask directly as float (no conversion needed)
      HIP_CHECK(hipMemcpy(gpu_contexts[d].state.mask, cpu_mask.data(), (size_t)S * (size_t)S * sizeof(float), hipMemcpyHostToDevice));
    }
  }
  printf("[GPU] Warm-up done on %d device(s). BF16 weights loaded to VRAM.\n", max_devices);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    // Destroy per-slot streams
    for (auto &st : gpu_ctx.slot_streams) {
      if (st != nullptr) HIP_CHECK(hipStreamDestroy(st));
    }
    gpu_ctx.slot_streams.clear();
    if (gpu_ctx.batch_stream) HIP_CHECK(hipStreamDestroy(gpu_ctx.batch_stream));
    if (gpu_ctx.d_slot_ids)  HIP_CHECK(hipFree(gpu_ctx.d_slot_ids));
    if (gpu_ctx.d_pos_list)  HIP_CHECK(hipFree(gpu_ctx.d_pos_list));
    if (gpu_ctx.d_token_ids) HIP_CHECK(hipFree(gpu_ctx.d_token_ids));
    for (int s = 0; s < gpu_ctx.num_expert_streams; ++s) {
      if (gpu_ctx.expert_streams[s] != nullptr) HIP_CHECK(hipStreamDestroy(gpu_ctx.expert_streams[s]));
    }
    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx.state);
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
    float *logits = gpu_forward(transformer, token, pos, gpu_ctx, /*slot=*/0, gpu_ctx->slot_streams[0]);

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

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  const int num_devices = (int)gpu_contexts.size();
  if (requests->num_reqs <= 0 || num_devices <= 0) return 0;
  // Global request index for all workers (lock-free producer)
  std::atomic<int> next_idx{0};

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  // Continuous batching per GPU worker
  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[t].device_id));
      GPUContext *ctx = &gpu_contexts[t];
      t_ctx = ctx;

      // Per-thread sampler
      Sampler local_sampler;
      build_sampler(&local_sampler,
                    sampler->vocab_size,
                    sampler->temperature,
                    sampler->topp,
                    sampler->rng_state + 0x9E3779B97f4A7C15ULL * (t + 1));

      struct Slot {
        bool active = false;
        int req_idx = -1;
        std::vector<int> prompt;
        int num_prompt_tokens = 0;
        int pos = 0;
        int token = 0;
        int *out_ptr = nullptr;
        int max_steps = 0; // bounded by seq_len
      };

      std::vector<Slot> slots(ctx->max_batch);
      long long local_out = 0;

      auto acquire_slot = [&](int sidx) {
        int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= requests->num_reqs) return false;
        Slot &s = slots[sidx];
        s.active = true;
        s.req_idx = idx;
        const char *input_seq = get_str_req_ptr(requests, idx);
        s.out_ptr = get_tok_gen_ptr(requests, idx);
        // Encode prompt
        int alloc = (int)strlen(input_seq) + 3;
        s.prompt.resize(alloc);
        encode(tokenizer, input_seq, -1, -1, s.prompt.data(), &s.num_prompt_tokens,
               transformer->config.initial_context_length);
        // Initialize state
        s.pos = 0;
        s.token = s.prompt[0];
        s.max_steps = std::min(requests->max_seq_len, transformer->config.seq_len);
        return true;
      };

      // Prime initial batch
      for (int i = 0; i < ctx->max_batch; ++i) {
        acquire_slot(i);
      }

      int active_cnt = 0;
      for (auto &s : slots) if (s.active) active_cnt++;

      while (active_cnt > 0) {
        // Build batch of current active slots
        std::vector<int> slot_ids; slot_ids.reserve(ctx->max_batch);
        std::vector<int> tokens;   tokens.reserve(ctx->max_batch);
        std::vector<int> poses;    poses.reserve(ctx->max_batch);
        for (int sidx = 0; sidx < (int)slots.size(); ++sidx) {
          if (!slots[sidx].active) continue;
          slot_ids.push_back(sidx);
          tokens.push_back(slots[sidx].token);
          poses.push_back(slots[sidx].pos);
        }
        int B = (int)slot_ids.size();
        if (B == 0) break;

        // Run batched forward for all active slots and fetch logits
        std::vector<float> logits_host((size_t)B * (size_t)transformer->config.vocab_size);
        gpu_forward_batch_step(transformer, ctx, tokens.data(), poses.data(), slot_ids.data(), B, logits_host.data());

        // Advance all slots with sampling and book-keeping
        for (int bi = 0; bi < B; ++bi) {
          int sidx = slot_ids[bi];
          Slot &s = slots[sidx];
          float *logits = logits_host.data() + (size_t)bi * (size_t)transformer->config.vocab_size;

          s.pos++;
          int next;
          if (s.pos < s.num_prompt_tokens) {
            next = s.prompt[s.pos];
          } else {
            next = sample(&local_sampler, logits);
            int gen_index = s.pos - s.num_prompt_tokens;
            s.out_ptr[gen_index] = next;
          }

          bool finished = false;
          if (next == 199999 || next == 200002) finished = true;
          if (s.pos >= s.max_steps) finished = true; // bound pos to seq_len
          if (s.pos >= transformer->config.seq_len) finished = true;

          if (finished) {
            int gen_tokens = std::max(0, s.pos - s.num_prompt_tokens);
            s.out_ptr[gen_tokens] = -1;
            local_out += (long long)(gen_tokens + 1); // include sentinel
            s.active = false;
            s.req_idx = -1;
            s.prompt.clear();
            active_cnt--;
            if (acquire_slot(sidx)) active_cnt++;
          } else {
            s.token = next;
          }
        }
      }

      total_out.fetch_add(local_out, std::memory_order_relaxed);
      free_sampler(&local_sampler);
    });
  }

  for (auto &th : workers) th.join();
  return total_out.load(std::memory_order_relaxed);
}


#endif // GETP_RUN




void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch) {
  int kv_dim = p->head_dim * p->n_kv_heads;
  size_t B = (size_t)max_batch;

  HIP_CHECK(hipMalloc(&s->x,   B * (size_t)p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->t,   B * (size_t)p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb,  B * (size_t)p->head_dim * (size_t)p->n_attn_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2, B * (size_t)p->hidden_dim * sizeof(float)));

  HIP_CHECK(hipMalloc(&s->router_score,      B * (size_t)p->n_experts * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_v,            B * (size_t)p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_i,            B * (size_t)p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->mlp1_out, B * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate,     B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up,       B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up,  B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->e_agg,    B * (size_t)p->hidden_dim * sizeof(float)));

  HIP_CHECK(hipMalloc(&s->qkv, B * (size_t)p->head_dim * (size_t)(p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->q,   B * (size_t)p->n_attn_heads * (size_t)p->head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->k,   B * (size_t)p->n_kv_heads   * (size_t)p->head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->v,   B * (size_t)p->n_kv_heads   * (size_t)p->head_dim * sizeof(float)));

  // Allocate KV cache and att per-slot for continuous batching
  size_t kv_cache_elems = (size_t)max_batch * (size_t)p->n_layers * (size_t)p->seq_len * (size_t)kv_dim;
  size_t att_elems      = (size_t)max_batch * (size_t)p->n_attn_heads * (size_t)p->seq_len;
  HIP_CHECK(hipMalloc(&s->key_cache,   kv_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->value_cache, kv_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->att,         att_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits,      B * (size_t)p->vocab_size * sizeof(float)));

  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(float)));
  } else {
    s->mask = NULL;
  }

  HIP_CHECK(hipMalloc(&s->cos_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sin_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));

  // Note: acc_buffer no longer needed since we use float state directly
  s->acc_buffer = nullptr;
  HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));

  // Per-stream expert scratch buffers
  HIP_CHECK(hipMalloc(&s->mlp1_out_s, MAX_EXPERT_STREAMS * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_s,     MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up_s,       MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_s,  MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2_s,      MAX_EXPERT_STREAMS * (size_t)p->hidden_dim * sizeof(float)));

  // Grouped MoE staging buffers
  size_t Mmax = (size_t)max_batch * (size_t)p->experts_per_token;
  HIP_CHECK(hipMalloc(&s->sel_sids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_eids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_w,        Mmax * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)p->intermediate_dim * sizeof(float)));
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
  if (s->acc_buffer) HIP_CHECK(hipFree(s->acc_buffer));  // acc_buffer may be nullptr
  HIP_CHECK(hipFree(s->logits_fp32));
  HIP_CHECK(hipFree(s->mlp1_out_s));
  HIP_CHECK(hipFree(s->gate_s));
  HIP_CHECK(hipFree(s->up_s));
  HIP_CHECK(hipFree(s->gate_up_s));
  HIP_CHECK(hipFree(s->tb2_s));
  // Grouped MoE staging buffers
  HIP_CHECK(hipFree(s->sel_sids));
  HIP_CHECK(hipFree(s->sel_eids));
  HIP_CHECK(hipFree(s->sel_w));
  HIP_CHECK(hipFree(s->mlp1_out_all));
  HIP_CHECK(hipFree(s->gate_up_all));
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
