// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode with CONTINUOUS BATCHING.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (e.g., MI250), BF16 weights/KV; FP32 accumulate.
// - Warm-up loads weights with chunked copy+convert to avoid OOM.
// - Runtime fixes: bound pos to seq_len; guard attention sink write.
// - Continuous batching manages a dynamic batch of requests to maximize GPU utilization.

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
#include <deque>
#include <condition_variable>
#include <memory> // Added for std::unique_ptr

#ifndef GETP_RUN
#define GETP_RUN

#ifndef GB
#define GB (1024.0*1024.0*1024.0)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define a max batch size for pre-allocation. This is critical for VRAM management.
#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 4
#endif

#define HIP_CHECK(cmd) \
  do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

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

// Batched run state for continuous batching
typedef struct {
  // --- Batch input/control buffers ---
  int *d_input_tokens;    // [max_batch_size]
  int *d_positions;       // [max_batch_size]

  // --- Batched intermediate buffers ---
  hip_bfloat16 *x_batch;
  hip_bfloat16 *t_batch;
  hip_bfloat16 *tb_batch;
  hip_bfloat16 *tb2_batch;
  hip_bfloat16 *router_score_batch;
  hip_bfloat16 *topk_v;       // (not batched, used per-request on CPU)
  int *topk_i;               // (not batched, used per-request on CPU)
  hip_bfloat16 *mlp1_out_batch;
  hip_bfloat16 *gate_batch;
  hip_bfloat16 *up_batch;
  hip_bfloat16 *gate_up_batch;
  hip_bfloat16 *e_agg_batch;
  hip_bfloat16 *qkv_batch;
  hip_bfloat16 *q_batch;
  hip_bfloat16 *k_batch;
  hip_bfloat16 *v_batch;
  hip_bfloat16 *att_batch;        // size: max_batch_size * n_attn_heads * (seq_len + 1)
  hip_bfloat16 *logits_batch;     // (unused for final FP32 logits)
  hip_bfloat16 *key_cache;        // size: max_batch_size * n_layers * seq_len * kv_dim
  hip_bfloat16 *value_cache;      // size: max_batch_size * n_layers * seq_len * kv_dim
  hip_bfloat16 *mask;
  hip_bfloat16 *cos_vals_batch;
  hip_bfloat16 *sin_vals_batch;

  // --- Batched FP32 buffers ---
  float *acc_buffer_batch;
  float *logits_fp32_batch;   // final batched logits
} GPURunState;

// Per-GPU device context
typedef struct {
  int device_id;
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

// ------------------------- KERNELS (NOW BATCHED) -------------------------
// Each kernel now takes a batch_size and operates on a batch index (blockIdx.y or .z)

__global__ void gather_token_embeddings_kernel_bf16(hip_bfloat16 *x_batch, const hip_bfloat16 *token_embedding_table, const int* tokens, int hidden_dim, int batch_size) {
    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size && tid < hidden_dim) {
        int token = tokens[batch_idx];
        const hip_bfloat16* token_row = token_embedding_table + (size_t)token * hidden_dim;
        x_batch[batch_idx * hidden_dim + tid] = token_row[tid];
    }
}

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

__global__ void rmsnorm_kernel_bf16_batch(hip_bfloat16 *o, const hip_bfloat16 *x, const hip_bfloat16 *weight, int size, int batch_size) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    const hip_bfloat16* x_i = x + (size_t)batch_idx * size;
    hip_bfloat16* o_i = o + (size_t)batch_idx * size;
    
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            float xv = bfloat16_to_float(x_i[j]);
            ss += xv * xv;
        }
        ss = ss / size + 1e-5f;
        shared_ss = 1.0f / sqrtf(ss);
    }
    __syncthreads();

    int tid = threadIdx.x;
    if (tid < size) {
        float xv = bfloat16_to_float(x_i[tid]);
        float wv = bfloat16_to_float(weight[tid]);
        o_i[tid] = float_to_bfloat16(wv * (shared_ss * xv));
    }
}

__global__ void matmul_kernel_bf16_batch(hip_bfloat16 *xout, const hip_bfloat16 *x, const hip_bfloat16 *w, int n, int d, int batch_size) {
    int batch_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    if (batch_idx >= batch_size || i >= d) return;

    const hip_bfloat16* x_i = x + (size_t)batch_idx * n;
    hip_bfloat16* xout_i = xout + (size_t)batch_idx * d;
    
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += bfloat16_to_float(w[i * n + j]) * bfloat16_to_float(x_i[j]);
    }
    xout_i[i] = float_to_bfloat16(val);
}

__global__ void add_bias_kernel_bf16_batch(hip_bfloat16 *output, const hip_bfloat16 *bias, int size, int batch_size) {
    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || tid >= size) return;
    
    hip_bfloat16* output_i = output + (size_t)batch_idx * size;
    float ov = bfloat16_to_float(output_i[tid]);
    float bv = bfloat16_to_float(bias[tid]);
    output_i[tid] = float_to_bfloat16(ov + bv);
}

__global__ void softmax_kernel_bf16_batch(hip_bfloat16 *x, int size, int batch_size) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    hip_bfloat16* x_i = x + (size_t)batch_idx * size;

    __shared__ float max_val;
    __shared__ float sum_val;
    if (threadIdx.x == 0) {
        if (size <= 0) return;
        max_val = bfloat16_to_float(x_i[0]);
        for (int i = 1; i < size; i++) {
            float v = bfloat16_to_float(x_i[i]);
            if (v > max_val) max_val = v;
        }
        sum_val = 0.0f;
        for (int i = 0; i < size; i++) {
            float v = expf(bfloat16_to_float(x_i[i]) - max_val);
            x_i[i] = float_to_bfloat16(v); // Store intermediate
            sum_val += v;
        }
        float inv = 1.0f / sum_val;
        for (int i = 0; i < size; i++) {
            float v = bfloat16_to_float(x_i[i]) * inv;
            x_i[i] = float_to_bfloat16(v);
        }
    }
}


__global__ void apply_rotary_emb_kernel_bf16_batch(hip_bfloat16 *x, const int* positions,
                                                   int n_heads, int head_dim, float base, float scaling_factor,
                                                   float initial_context_length, float ntk_beta, float ntk_alpha,
                                                   int batch_size, int qkv_stride) {
    int batch_idx = blockIdx.z;
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half = head_dim / 2;

    if (batch_idx >= batch_size || h >= n_heads || i >= half) return;

    int pos = positions[batch_idx];
    hip_bfloat16* x_i = x + (size_t)batch_idx * qkv_stride;

    // --- Recompute cos/sin on the fly ---
    float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
    float inv_freq;
    if (scaling_factor > 1.0f) {
      float low  = half * logf(initial_context_length / (ntk_beta  * 2.0f * M_PI)) / logf(base);
      float high = half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
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
    float val = (float)pos * inv_freq;
    float c = cosf(val) * concentration;
    float s = sinf(val) * concentration;

    float x1 = bfloat16_to_float(x_i[h * head_dim + i]);
    float x2 = bfloat16_to_float(x_i[h * head_dim + half + i]);
    
    x_i[h * head_dim + i] = float_to_bfloat16(x1 * c - x2 * s);
    x_i[h * head_dim + half + i] = float_to_bfloat16(x2 * c + x1 * s);
}


__global__ void attention_scores_kernel_bf16_batch(hip_bfloat16 *att_batch, const hip_bfloat16 *q_batch, const hip_bfloat16 *key_cache,
                                                   const int* positions, int head_dim, int loff_layer, int kv_dim, int kv_mul,
                                                   int n_attn_heads, int seq_len, const hip_bfloat16 *mask,
                                                   int batch_size, size_t batch_kv_cache_stride) {
    int batch_idx = blockIdx.z;
    int h = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    int pos = positions[batch_idx];
    if (t > pos) return;

    const hip_bfloat16* q = q_batch + (size_t)batch_idx * n_attn_heads * head_dim + h * head_dim;
    const hip_bfloat16* k_cache_layer_head = key_cache + (size_t)batch_idx * batch_kv_cache_stride + loff_layer;
    const hip_bfloat16* k = k_cache_layer_head + t * kv_dim + (h / kv_mul) * head_dim;

    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        score += bfloat16_to_float(q[i]) * bfloat16_to_float(k[i]);
    }
    score /= sqrtf((float)head_dim);
    
    if (mask != NULL) {
        score += bfloat16_to_float(mask[(size_t)pos * seq_len + t]);
    }

    int att_stride = seq_len + 1; // leave one slot for the sink
    att_batch[(size_t)batch_idx * n_attn_heads * att_stride + (size_t)h * att_stride + t] = float_to_bfloat16(score);
}

__global__ void attention_weighted_sum_kernel_bf16_batch(hip_bfloat16 *tb_batch, const hip_bfloat16 *att_batch, const hip_bfloat16 *value_cache,
                                                         const int* positions, int head_dim, int loff_layer, int kv_dim, int kv_mul,
                                                         int n_attn_heads, int h, int batch_size, size_t batch_kv_cache_stride, int att_stride) { // att_stride = seq_len + 1
    int batch_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || i >= head_dim) return;
    
    int pos = positions[batch_idx];
    const hip_bfloat16* att = att_batch + (size_t)batch_idx * n_attn_heads * att_stride + (size_t)h * att_stride;
    const hip_bfloat16* v_cache_layer_head = value_cache + (size_t)batch_idx * batch_kv_cache_stride + loff_layer;

    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const hip_bfloat16 *v = v_cache_layer_head + t * kv_dim + (h / kv_mul) * head_dim;
        sum += bfloat16_to_float(att[t]) * bfloat16_to_float(v[i]);
    }
    
    hip_bfloat16* tb_out = tb_batch + (size_t)batch_idx * n_attn_heads * head_dim;
    tb_out[h * head_dim + i] = float_to_bfloat16(sum);
}

__global__ void swiglu_kernel_bf16_batch(hip_bfloat16 *gate_up, const hip_bfloat16 *gate, const hip_bfloat16 *up,
                                         int intermediate_dim, float swiglu_limit, int batch_size) {
    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch_idx >= batch_size || tid >= intermediate_dim) return;

    size_t offset = (size_t)batch_idx * intermediate_dim + tid;
    float val   = bfloat16_to_float(gate[offset]);
    float upval = bfloat16_to_float(up[offset]);
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    gate_up[offset] = float_to_bfloat16(val);
}

__global__ void split_gate_up_kernel_bf16_batch(hip_bfloat16 *gate, hip_bfloat16 *up, const hip_bfloat16 *mlp1_out, int intermediate_dim, int batch_size) {
    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch_idx >= batch_size || tid >= intermediate_dim) return;
    
    size_t offset_mlp = (size_t)batch_idx * 2 * intermediate_dim;
    size_t offset_gu = (size_t)batch_idx * intermediate_dim;
    gate[offset_gu + tid] = mlp1_out[offset_mlp + 2 * tid];
    up[offset_gu + tid]   = mlp1_out[offset_mlp + 2 * tid + 1];
}

__global__ void weighted_sum_kernel_bf16_batch(hip_bfloat16 *e_agg, const hip_bfloat16 *tb2, float expert_w, int hidden_dim, int batch_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < hidden_dim) {
        size_t offset = (size_t)batch_idx * hidden_dim + tid;
        float agg = bfloat16_to_float(e_agg[offset]);
        float add = bfloat16_to_float(tb2[offset]);
        e_agg[offset] = float_to_bfloat16(agg + add * expert_w);
    }
}

__global__ void residual_connection_kernel_bf16_batch(hip_bfloat16 *x, const hip_bfloat16 *residual, int size, int batch_size) {
    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || tid >= size) return;
    size_t offset = (size_t)batch_idx * size + tid;
    float xv = bfloat16_to_float(x[offset]);
    float rv = bfloat16_to_float(residual[offset]);
    x[offset] = float_to_bfloat16(xv + rv);
}

__global__ void matvec_wbf16_xfp32_outfp32_batch(float *out, const float *x,
                                                 const hip_bfloat16 *w, int n, int d, int batch_size) {
    int batch_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    if (batch_idx >= batch_size || i >= d) return;

    const float* x_i = x + (size_t)batch_idx * n;
    float* out_i = out + (size_t)batch_idx * d;
    
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        sum += (float)w[i * n + j] * x_i[j];
    }
    out_i[i] = sum;
}

// ------------------------- CPU top-k -------------------------

void topk_cpu(float *topk_values, int *topk_indices, const float *router_score, int num_experts, int experts_per_token) {
  struct Pair { float value; int index; };
  std::vector<Pair> pairs(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    pairs[i] = {router_score[i], i};
  }
  std::partial_sort(pairs.begin(), pairs.begin() + experts_per_token, pairs.end(), 
                    [](const Pair& a, const Pair& b) { return a.value > b.value; });
  for (int i = 0; i < experts_per_token; i++) {
    topk_values[i]  = pairs[i].value;
    topk_indices[i] = pairs[i].index;
  }
}

// ... Function declarations for forward, alloc, free etc.
void gpu_malloc_run_state(GPURunState *s, Config *p);
void gpu_free_run_state(GPURunState *s);
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
void gpu_free_weights(GPUTransformerWeights *gpu_w);
float* gpu_forward_batch(Transformer *transformer, int batch_size, GPUContext *gpu_ctx);


// ------------------------- BATCH FORWARD -------------------------

float* gpu_forward_batch(Transformer *transformer, int batch_size, GPUContext *gpu_ctx) {
    Config *p = &transformer->config;
    GPURunState *s = &gpu_ctx->state;
    GPUTransformerWeights *w = &gpu_ctx->weights;

    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    
    dim3 block(256);
    dim3 grid_hidden( (hidden_dim + block.x - 1) / block.x, batch_size);
    dim3 grid_intermediate( (intermediate_dim + block.x - 1) / block.x, batch_size);
    dim3 grid_experts( (n_experts + block.x - 1) / block.x, batch_size);

    // Token embedding
    gather_token_embeddings_kernel_bf16<<<grid_hidden, block>>>(s->x_batch, w->token_embedding_table, s->d_input_tokens, hidden_dim, batch_size);
    HIP_CHECK(hipDeviceSynchronize());

    for (int l = 0; l < p->n_layers; l++) {
        // Attention RMSNorm
        const hip_bfloat16 *rms_attn_w = w->rms_attn_w + (size_t)l * hidden_dim;
        rmsnorm_kernel_bf16_batch<<<grid_hidden, block>>>(s->t_batch, s->x_batch, rms_attn_w, hidden_dim, batch_size);
        HIP_CHECK(hipDeviceSynchronize());
        
        // QKV projection
        int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        dim3 grid_qkv((qkv_out_dim + block.x - 1) / block.x, batch_size);
        const hip_bfloat16 *w_qkv = w->w_qkv + (size_t)l * hidden_dim * qkv_out_dim;
        const hip_bfloat16 *b_qkv = w->b_qkv + (size_t)l * qkv_out_dim;
        matmul_kernel_bf16_batch<<<grid_qkv, block>>>(s->qkv_batch, s->t_batch, w_qkv, hidden_dim, qkv_out_dim, batch_size);
        add_bias_kernel_bf16_batch<<<grid_qkv, block>>>(s->qkv_batch, b_qkv, qkv_out_dim, batch_size);
        HIP_CHECK(hipDeviceSynchronize());
        
        // Split Q, K, V and store in cache
        int q_size = head_dim * p->n_attn_heads;
        int k_size = head_dim * p->n_kv_heads;
        // This part needs a custom kernel for efficiency, but hipMemcpy works for correctness.
        // A single kernel would be: kernel(q, k, v, qkv, d_positions, k_cache, v_cache)
        for(int b=0; b<batch_size; ++b) {
            int h_pos;
            HIP_CHECK(hipMemcpy(&h_pos, s->d_positions + b, sizeof(int), hipMemcpyDeviceToHost));
            size_t batch_offset_qkv = (size_t)b * qkv_out_dim;
            size_t batch_offset_kv_cache = (size_t)b * p->n_layers * p->seq_len * kv_dim;
            size_t loff = (size_t)l * p->seq_len * kv_dim;
            
            hip_bfloat16 *k_cache_pos = s->key_cache   + batch_offset_kv_cache + loff + (size_t)h_pos * kv_dim;
            hip_bfloat16 *v_cache_pos = s->value_cache + batch_offset_kv_cache + loff + (size_t)h_pos * kv_dim;

            HIP_CHECK(hipMemcpy(s->q_batch + (size_t)b * q_size, s->qkv_batch + batch_offset_qkv, q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(s->k_batch + (size_t)b * k_size, s->qkv_batch + batch_offset_qkv + q_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(v_cache_pos, s->qkv_batch + batch_offset_qkv + q_size + k_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // RoPE
        apply_rotary_emb_kernel_bf16_batch<<<dim3(p->n_attn_heads, 1, batch_size), head_dim/2>>>(s->q_batch, s->d_positions, p->n_attn_heads, head_dim, p->rope_theta, p->rope_scaling_factor, p->initial_context_length, 32.0f, 1.0f, batch_size, q_size);
        apply_rotary_emb_kernel_bf16_batch<<<dim3(p->n_kv_heads, 1, batch_size), head_dim/2>>>(s->k_batch, s->d_positions, p->n_kv_heads, head_dim, p->rope_theta, p->rope_scaling_factor, p->initial_context_length, 32.0f, 1.0f, batch_size, k_size);
        HIP_CHECK(hipDeviceSynchronize());

        // Update K cache after RoPE
        for(int b=0; b<batch_size; ++b) {
            int h_pos;
            HIP_CHECK(hipMemcpy(&h_pos, s->d_positions + b, sizeof(int), hipMemcpyDeviceToHost));
            size_t batch_offset_kv_cache = (size_t)b * p->n_layers * p->seq_len * kv_dim;
            size_t loff = (size_t)l * p->seq_len * kv_dim;
            hip_bfloat16 *k_cache_pos = s->key_cache + batch_offset_kv_cache + loff + (size_t)h_pos * kv_dim;
            HIP_CHECK(hipMemcpy(k_cache_pos, s->k_batch + (size_t)b * k_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Multi-head attention
        std::vector<int> h_positions(batch_size);
        HIP_CHECK(hipMemcpy(h_positions.data(), s->d_positions, batch_size * sizeof(int), hipMemcpyDeviceToHost));
        int max_pos = 0;
        for(int p_ : h_positions) if (p_ > max_pos) max_pos = p_;

        dim3 grid_att( (max_pos + 1 + block.x - 1) / block.x, p->n_attn_heads, batch_size);
        size_t batch_kv_cache_stride = (size_t)p->n_layers * p->seq_len * kv_dim;
        attention_scores_kernel_bf16_batch<<<grid_att, block>>>(s->att_batch, s->q_batch, s->key_cache, s->d_positions, 
                                                                head_dim, l * p->seq_len * kv_dim, kv_dim, kv_mul, p->n_attn_heads, p->seq_len, 
                                                                (p->sliding_window > 0 && l % 2 == 0) ? s->mask : NULL,
                                                                batch_size, batch_kv_cache_stride);
        HIP_CHECK(hipDeviceSynchronize());

        // Softmax (can be optimized with a batched kernel, but this works)
        for(int b=0; b<batch_size; ++b) {
            for(int h=0; h<p->n_attn_heads; ++h) {
                hip_bfloat16* att = s->att_batch + (size_t)b * p->n_attn_heads * (p->seq_len + 1) + (size_t)h * (p->seq_len + 1);
                int len_no_sink = h_positions[b] + 1;
                bool can_add_sink = (len_no_sink < p->seq_len);
                if (can_add_sink) {
                    // Copy sink score directly device-to-device into the tail slot
                    HIP_CHECK(hipMemcpy(att + len_no_sink,
                                        w->attn_sinks + (size_t)l * p->n_attn_heads + h,
                                        sizeof(hip_bfloat16),
                                        hipMemcpyDeviceToDevice));
                    // Run softmax over [0..pos] plus the sink slot
                    softmax_kernel_bf16_batch<<<dim3(1,1), 1>>>(att, len_no_sink + 1, 1);
                } else {
                    // FIXED: Called correct batch kernel with batch_size=1
                    softmax_kernel_bf16_batch<<<dim3(1,1), 1>>>(att, len_no_sink, 1);
                }
            }
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Weighted sum
        for (int h = 0; h < p->n_attn_heads; h++) {
            dim3 grid_head_dim((head_dim + block.x - 1) / block.x, batch_size);
            attention_weighted_sum_kernel_bf16_batch<<<grid_head_dim, block>>>(s->tb_batch, s->att_batch, s->value_cache, s->d_positions,
                                                                              head_dim, l * p->seq_len * kv_dim, kv_dim, kv_mul,
                                                                              p->n_attn_heads, h, batch_size, batch_kv_cache_stride, p->seq_len + 1);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Output projection and residual
        const hip_bfloat16 *w_o = w->w_o + (size_t)l * (head_dim * p->n_attn_heads) * hidden_dim;
        const hip_bfloat16 *b_o = w->b_o + (size_t)l * hidden_dim;
        dim3 grid_out_proj( (hidden_dim + block.x - 1) / block.x, batch_size);
        matmul_kernel_bf16_batch<<<grid_out_proj, block>>>(s->tb2_batch, s->tb_batch, w_o, head_dim * p->n_attn_heads, hidden_dim, batch_size);
        add_bias_kernel_bf16_batch<<<grid_out_proj, block>>>(s->tb2_batch, b_o, hidden_dim, batch_size);
        residual_connection_kernel_bf16_batch<<<grid_out_proj, block>>>(s->x_batch, s->tb2_batch, hidden_dim, batch_size);
        HIP_CHECK(hipDeviceSynchronize());

        // -------- MoE --------
        const hip_bfloat16 *rms_ffn_w = w->rms_ffn_w + (size_t)l * hidden_dim;
        rmsnorm_kernel_bf16_batch<<<grid_hidden, block>>>(s->t_batch, s->x_batch, rms_ffn_w, hidden_dim, batch_size);
        
        const hip_bfloat16 *w_router = w->w_router + (size_t)l * hidden_dim * n_experts;
        const hip_bfloat16 *b_router = w->b_router + (size_t)l * n_experts;
        matmul_kernel_bf16_batch<<<grid_experts, block>>>(s->router_score_batch, s->t_batch, w_router, hidden_dim, n_experts, batch_size);
        add_bias_kernel_bf16_batch<<<grid_experts, block>>>(s->router_score_batch, b_router, n_experts, batch_size);
        HIP_CHECK(hipDeviceSynchronize());

        // Router scores: BF16 -> FP32 (device) -> host for topk
        float *gpu_router_score_fp32 = nullptr;
        HIP_CHECK(hipMalloc(&gpu_router_score_fp32, (size_t)batch_size * n_experts * sizeof(float)));
        convert_bf16_to_fp32_kernel<<<dim3(((size_t)batch_size * n_experts + block.x - 1) / block.x), block>>>(s->router_score_batch, gpu_router_score_fp32, batch_size * n_experts);
        
        std::vector<float> cpu_router_scores( (size_t)batch_size * n_experts );
        HIP_CHECK(hipMemcpy(cpu_router_scores.data(), gpu_router_score_fp32, (size_t)batch_size * n_experts * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(gpu_router_score_fp32));

        HIP_CHECK(hipMemset(s->e_agg_batch, 0, (size_t)batch_size * hidden_dim * sizeof(hip_bfloat16)));
        
        // This loop over the batch is a bottleneck. In a production system, top-k would be a GPU kernel.
        for (int b = 0; b < batch_size; b++) {
            std::vector<float> topk_v(p->experts_per_token);
            std::vector<int>   topk_i(p->experts_per_token);
            topk_cpu(topk_v.data(), topk_i.data(), cpu_router_scores.data() + (size_t)b * n_experts, n_experts, p->experts_per_token);

            float sum = 0.0f, max_val = topk_v[0];
            for (int i = 1; i < p->experts_per_token; i++) if (topk_v[i] > max_val) max_val = topk_v[i];
            for (int i = 0; i < p->experts_per_token; i++) { topk_v[i] = expf(topk_v[i] - max_val); sum += topk_v[i]; }
            for (int i = 0; i < p->experts_per_token; i++) topk_v[i] /= sum;

            for (int idx = 0; idx < p->experts_per_token; idx++) {
                int e = topk_i[idx];
                float expert_w = topk_v[idx];
                
                // FIXED: Cast dimensions to size_t before multiplication to prevent integer overflow on large models.
                size_t expert_l_offset = (size_t)l * n_experts + e;
                const hip_bfloat16 *w_mlp1 = w->w_mlp1 + expert_l_offset * (size_t)2 * (size_t)intermediate_dim * (size_t)hidden_dim;
                const hip_bfloat16 *b_mlp1 = w->b_mlp1 + expert_l_offset * (size_t)2 * (size_t)intermediate_dim;

                dim3 grid_mlp1_s( (2 * intermediate_dim + block.x - 1) / block.x, 1);
                matmul_kernel_bf16_batch<<<grid_mlp1_s, block>>>(s->mlp1_out_batch + (size_t)b * 2 * intermediate_dim, s->t_batch + (size_t)b * hidden_dim, w_mlp1, hidden_dim, 2 * intermediate_dim, 1);
                add_bias_kernel_bf16_batch<<<grid_mlp1_s, block>>>(s->mlp1_out_batch + (size_t)b * 2 * intermediate_dim, b_mlp1, 2 * intermediate_dim, 1);
                
                dim3 grid_inter_s( (intermediate_dim + block.x - 1) / block.x, 1);
                split_gate_up_kernel_bf16_batch<<<grid_inter_s, block>>>(s->gate_batch + (size_t)b * intermediate_dim, s->up_batch + (size_t)b * intermediate_dim, s->mlp1_out_batch + (size_t)b * 2 * intermediate_dim, intermediate_dim, 1);
                swiglu_kernel_bf16_batch<<<grid_inter_s, block>>>(s->gate_up_batch + (size_t)b * intermediate_dim, s->gate_batch + (size_t)b * intermediate_dim, s->up_batch + (size_t)b * intermediate_dim, intermediate_dim, p->swiglu_limit, 1);
                
                // FIXED: Cast dimensions to size_t before multiplication to prevent integer overflow on large models.
                const hip_bfloat16 *w_mlp2 = w->w_mlp2 + expert_l_offset * (size_t)hidden_dim * (size_t)intermediate_dim;
                const hip_bfloat16 *b_mlp2 = w->b_mlp2 + expert_l_offset * (size_t)hidden_dim;
                
                dim3 grid_hidden_s( (hidden_dim + block.x - 1) / block.x, 1);
                matmul_kernel_bf16_batch<<<grid_hidden_s, block>>>(s->tb2_batch + (size_t)b * hidden_dim, s->gate_up_batch + (size_t)b * intermediate_dim, w_mlp2, intermediate_dim, hidden_dim, 1);
                add_bias_kernel_bf16_batch<<<grid_hidden_s, block>>>(s->tb2_batch + (size_t)b * hidden_dim, b_mlp2, hidden_dim, 1);
                
                weighted_sum_kernel_bf16_batch<<<grid_hidden_s, block>>>(s->e_agg_batch, s->tb2_batch, expert_w, hidden_dim, b);
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        residual_connection_kernel_bf16_batch<<<grid_hidden, block>>>(s->x_batch, s->e_agg_batch, hidden_dim, batch_size);
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Final RMSNorm and Logits
    rmsnorm_kernel_bf16_batch<<<grid_hidden, block>>>(s->x_batch, s->x_batch, w->rms_out_w, hidden_dim, batch_size);
    convert_bf16_to_fp32_kernel<<<dim3(((size_t)batch_size * hidden_dim + block.x - 1) / block.x), block>>>(s->x_batch, s->acc_buffer_batch, batch_size * hidden_dim);
    
    dim3 grid_vocab( (p->vocab_size + block.x - 1) / block.x, batch_size);
    matvec_wbf16_xfp32_outfp32_batch<<<grid_vocab, block>>>(s->logits_fp32_batch, s->acc_buffer_batch, w->out, hidden_dim, p->vocab_size, batch_size);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy logits to CPU (thread-local buffer across calls)
    thread_local float *cpu_logits = NULL;
    if (cpu_logits == NULL) {
        cpu_logits = (float*)malloc((size_t)MAX_BATCH_SIZE * p->vocab_size * sizeof(float));
    }
    HIP_CHECK(hipMemcpy(cpu_logits, s->logits_fp32_batch, (size_t)batch_size * p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return cpu_logits;
}

// ------------------------- Continuous Batching Management -------------------------

enum RequestStatus { WAITING, RUNNING, DONE };

struct RequestState {
    int req_id;
    std::vector<int> prompt_tokens;
    std::vector<int> output_tokens;
    int pos = 0;
    RequestStatus status = WAITING;
    Sampler sampler;
    int* user_output_ptr;
    int max_steps;
};

class BatchManager {
public:
    BatchManager(Transformer* tr, Tokenizer* tk, Sampler* samp, GPUContext* ctx)
        : transformer(tr), tokenizer(tk), base_sampler(samp), gpu_ctx(ctx), stop_flag(false) {}

    void add_request(RequestState* req) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        waiting_requests.push_back(req);
        cv.notify_one();
    }

    void stop() {
        stop_flag.store(true);
        cv.notify_all();
    }

    void run_loop() {
        HIP_CHECK(hipSetDevice(gpu_ctx->device_id));
        t_ctx = gpu_ctx; // Set thread-local context

        std::vector<int> h_input_tokens(MAX_BATCH_SIZE);
        std::vector<int> h_positions(MAX_BATCH_SIZE);

        while (!stop_flag.load()) {
            // 1. Schedule: fill active_requests from waiting_requests
            update_batch();

            if (active_requests.empty()) {
                 std::unique_lock<std::mutex> lock(queue_mutex);
                 cv.wait(lock, [&]{ return !waiting_requests.empty() || stop_flag.load(); });
                 continue;
            }

            // 2. Prepare batch for GPU
            int current_batch_size = active_requests.size();
            for (int i = 0; i < current_batch_size; ++i) {
                RequestState* req = active_requests[i];
                if (req->pos < req->prompt_tokens.size()) {
                    h_input_tokens[i] = req->prompt_tokens[req->pos];
                } else {
                    h_input_tokens[i] = req->output_tokens.back();
                }
                h_positions[i] = req->pos;
            }
            
            HIP_CHECK(hipMemcpy(gpu_ctx->state.d_input_tokens, h_input_tokens.data(), current_batch_size * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gpu_ctx->state.d_positions, h_positions.data(), current_batch_size * sizeof(int), hipMemcpyHostToDevice));

            // 3. Forward pass
            float* batched_logits = gpu_forward_batch(transformer, current_batch_size, gpu_ctx);

            // 4. Post-process: sample and update requests
            int vocab_size = transformer->config.vocab_size;
            for (int i = 0; i < current_batch_size; ++i) {
                RequestState* req = active_requests[i];

                // Check if we have just finished processing the prompt.
                // The logits from the current forward pass correspond to the token AFTER the last prompt token.
                bool is_generating = (req->pos >= req->prompt_tokens.size() - 1);

                req->pos++;

                if (is_generating) {
                    // We are in the generation phase. Sample a token from the logits.
                    float* logits = batched_logits + (size_t)i * vocab_size;
                    int next_token = sample(&req->sampler, logits);

                    // Check for termination conditions
                    if (next_token == 199999 || next_token == 200002 || req->pos >= req->max_steps) {
                        req->status = DONE;
                    } else {
                        // This is a generated token, so add it to the output.
                        req->output_tokens.push_back(next_token);
                    }
                } else {
                    // We are still ingesting the prompt. We don't sample or add to the output.
                    // The next token for input will be handled by the logic at the top of the loop.
                    // We only check if we've hit the max length limit.
                    if (req->pos >= req->max_steps) {
                        req->status = DONE;
                    }
                }
            }
        }
    }

private:
    void update_batch() {
        // Remove finished requests
        active_requests.erase(
            std::remove_if(active_requests.begin(), active_requests.end(), 
                [this](RequestState* req) {
                    if (req->status == DONE) {
                        // Copy final output
                        int out_len = req->output_tokens.size();
                        if (out_len > 0) {
                            memcpy(req->user_output_ptr, req->output_tokens.data(), out_len * sizeof(int));
                        }
                        req->user_output_ptr[out_len] = -1; // Terminator
                        return true;
                    }
                    return false;
                }), 
            active_requests.end());

        // Add new requests if there is space
        std::lock_guard<std::mutex> lock(queue_mutex);
        while (active_requests.size() < MAX_BATCH_SIZE && !waiting_requests.empty()) {
            RequestState* req = waiting_requests.front();
            waiting_requests.pop_front();
            req->status = RUNNING;
            active_requests.push_back(req);
        }
    }

    Transformer* transformer;
    Tokenizer* tokenizer;
    Sampler* base_sampler;
    GPUContext* gpu_ctx;
    
    std::vector<RequestState*> active_requests;
    std::deque<RequestState*> waiting_requests;

    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_flag;
};

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

  printf("[GPU] Using MAX_BATCH_SIZE = %d\n", MAX_BATCH_SIZE);

  for (int d = 0; d < max_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    gpu_contexts[d].device_id = d;
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config);

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
      dim3 grid((((size_t)S * (size_t)S) + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block>>>(gpu_mask_fp32, gpu_contexts[d].state.mask, (int)((size_t)S * (size_t)S));
      HIP_CHECK(hipDeviceSynchronize());

      HIP_CHECK(hipFree(gpu_mask_fp32));
    }
  }
  printf("[GPU] Warm-up done on %d device(s). BF16 weights and batched buffers loaded to VRAM.\n", max_devices);
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


// ------------------------- BATCHING DRIVER -------------------------

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
    if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
    const int num_devices = (int)gpu_contexts.size();

    std::vector<RequestState> all_requests(requests->num_reqs);
    // FIXED: Use vector of unique_ptr to avoid issues with non-movable members (mutex, atomic)
    std::vector<std::unique_ptr<BatchManager>> managers;
    for(int i=0; i<num_devices; ++i) {
        managers.push_back(std::make_unique<BatchManager>(transformer, tokenizer, sampler, &gpu_contexts[i]));
    }
    
    // Create and enqueue all requests
    for (int i = 0; i < requests->num_reqs; ++i) {
        RequestState& req = all_requests[i];
        req.req_id = i;
        req.user_output_ptr = get_tok_gen_ptr(requests, i);
        req.max_steps = std::min(requests->max_seq_len, transformer->config.seq_len);
        
        const char *input_seq = get_str_req_ptr(requests, i);
        int num_prompt_tokens = 0;
        int *prompt_tokens_buf = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
        encode(tokenizer, input_seq, -1, -1, prompt_tokens_buf, &num_prompt_tokens, transformer->config.initial_context_length);
        req.prompt_tokens.assign(prompt_tokens_buf, prompt_tokens_buf + num_prompt_tokens);
        free(prompt_tokens_buf);
        
        build_sampler(&req.sampler, sampler->vocab_size, sampler->temperature, sampler->topp, sampler->rng_state + i + 1);

        managers[i % num_devices]->add_request(&req);
    }
    
    // Start manager threads
    std::vector<std::thread> workers;
    for(int i=0; i<num_devices; ++i) {
        workers.emplace_back(&BatchManager::run_loop, managers[i].get());
    }

    // Wait for all requests to be processed
    bool all_done = false;
    while(!all_done) {
        all_done = true;
        for(int i=0; i<requests->num_reqs; ++i) {
            if (all_requests[i].status != DONE) {
                all_done = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Stop managers and join threads
    for(auto& manager : managers) {
        manager->stop();
    }
    for(auto& th : workers) {
        th.join();
    }
    
    long long total_out = 0;
    for(auto& req : all_requests) {
        total_out += req.output_tokens.size();
        free_sampler(&req.sampler);
    }
    
    return total_out;
}

#endif // GETP_RUN


void gpu_malloc_run_state(GPURunState *s, Config *p) {
    size_t B = MAX_BATCH_SIZE;
    int kv_dim = p->head_dim * p->n_kv_heads;

    HIP_CHECK(hipMalloc(&s->d_input_tokens, B * sizeof(int)));
    HIP_CHECK(hipMalloc(&s->d_positions, B * sizeof(int)));

    HIP_CHECK(hipMalloc(&s->x_batch, B * p->hidden_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->t_batch, B * p->hidden_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->tb_batch, B * p->head_dim * p->n_attn_heads * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->tb2_batch, B * p->hidden_dim * sizeof(hip_bfloat16)));

    HIP_CHECK(hipMalloc(&s->router_score_batch, B * p->n_experts * sizeof(hip_bfloat16)));
    
    HIP_CHECK(hipMalloc(&s->mlp1_out_batch, B * 2 * p->intermediate_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->gate_batch, B * p->intermediate_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->up_batch, B * p->intermediate_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->gate_up_batch, B * p->intermediate_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->e_agg_batch, B * p->hidden_dim * sizeof(hip_bfloat16)));

    HIP_CHECK(hipMalloc(&s->qkv_batch, B * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->q_batch, B * p->n_attn_heads * p->head_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->k_batch, B * p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));
    // V is not needed as a separate buffer, it goes straight to cache.

    HIP_CHECK(hipMalloc(&s->key_cache, B * p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->value_cache, B * p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&s->att_batch, (size_t)B * (size_t)p->n_attn_heads * (size_t)(p->seq_len + 1) * sizeof(hip_bfloat16)));
    
    if (p->sliding_window > 0) {
        HIP_CHECK(hipMalloc(&s->mask, (size_t)p->seq_len * p->seq_len * sizeof(hip_bfloat16)));
    } else {
        s->mask = NULL;
    }

    HIP_CHECK(hipMalloc(&s->acc_buffer_batch, B * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->logits_fp32_batch, B * p->vocab_size * sizeof(float)));
}


void gpu_free_run_state(GPURunState *s) {
    HIP_CHECK(hipFree(s->d_input_tokens));
    HIP_CHECK(hipFree(s->d_positions));
    HIP_CHECK(hipFree(s->x_batch));
    HIP_CHECK(hipFree(s->t_batch));
    HIP_CHECK(hipFree(s->tb_batch));
    HIP_CHECK(hipFree(s->tb2_batch));
    HIP_CHECK(hipFree(s->router_score_batch));
    HIP_CHECK(hipFree(s->mlp1_out_batch));
    HIP_CHECK(hipFree(s->gate_batch));
    HIP_CHECK(hipFree(s->up_batch));
    HIP_CHECK(hipFree(s->gate_up_batch));
    HIP_CHECK(hipFree(s->e_agg_batch));
    HIP_CHECK(hipFree(s->qkv_batch));
    HIP_CHECK(hipFree(s->q_batch));
    HIP_CHECK(hipFree(s->k_batch));
    HIP_CHECK(hipFree(s->key_cache));
    HIP_CHECK(hipFree(s->value_cache));
    HIP_CHECK(hipFree(s->att_batch));
    HIP_CHECK(hipFree(s->mask));
    HIP_CHECK(hipFree(s->acc_buffer_batch));
    HIP_CHECK(hipFree(s->logits_fp32_batch));
}

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int head_dim       = p->head_dim;
  int n_layers       = p->n_layers;
  int n_experts      = p->n_experts;
  int hidden_dim     = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size     = p->vocab_size;
  int n_attn_heads   = p->n_attn_heads;
  int n_kv_heads     = p->n_kv_heads;

  // Helper: host FP32 -> device staging FP32 (tile) -> device BF16 dest
  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    const size_t bf16_size = count * sizeof(hip_bfloat16);
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));

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
