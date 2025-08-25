// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#ifndef GETP_RUN
#define GETP_RUN

#define HIP_CHECK(cmd) \
  do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// BF16 GPU versions of weights and state
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
  hip_bfloat16 *att;
  hip_bfloat16 *logits;
  hip_bfloat16 *key_cache;
  hip_bfloat16 *value_cache;
  hip_bfloat16 *mask;
  hip_bfloat16 *cos_vals;
  hip_bfloat16 *sin_vals;
  // FP32 buffers for accumulation and precision-sensitive operations
  float *acc_buffer;  // For accumulating results
  float *logits_fp32; // For final logits in FP32
} GPURunState;

// Global GPU data structures
GPUTransformerWeights gpu_weights;
GPURunState gpu_state;

// Conversion utilities
__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}

__device__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
}

// Convert FP32 array to BF16 on device
__global__ void convert_fp32_to_bf16_kernel(float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
  }
}

// Convert BF16 array to FP32 on device
__global__ void convert_bf16_to_fp32_kernel(hip_bfloat16 *bf16_data, float *fp32_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    fp32_data[tid] = bfloat16_to_float(bf16_data[tid]);
  }
}

// HIP Kernels with BF16
__global__ void rmsnorm_kernel_bf16(hip_bfloat16 *o, hip_bfloat16 *x, hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
      shared_ss = 0.0f;
      for (int j = 0; j < size; j++) {
        float x_val = bfloat16_to_float(x[j]);
        shared_ss += x_val * x_val;
      }
      shared_ss /= size;
      shared_ss += 1e-5f;
      shared_ss = 1.0f / sqrtf(shared_ss);
    }
    __syncthreads();
    
    float x_val = bfloat16_to_float(x[tid]);
    float w_val = bfloat16_to_float(weight[tid]);
    o[tid] = float_to_bfloat16(w_val * (shared_ss * x_val));
  }
}

__global__ void matmul_kernel_bf16(hip_bfloat16 *xout, hip_bfloat16 *x, hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    float val = 0.0f;  // Use FP32 for accumulation
    for (int j = 0; j < n; j++) {
      float w_val = bfloat16_to_float(w[i * n + j]);
      float x_val = bfloat16_to_float(x[j]);
      val += w_val * x_val;
    }
    xout[i] = float_to_bfloat16(val);
  }
}

__global__ void add_bias_kernel_bf16(hip_bfloat16 *output, hip_bfloat16 *bias, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float out_val = bfloat16_to_float(output[tid]);
    float bias_val = bfloat16_to_float(bias[tid]);
    output[tid] = float_to_bfloat16(out_val + bias_val);
  }
}

__global__ void copy_kernel_bf16(hip_bfloat16 *dst, hip_bfloat16 *src, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    dst[tid] = src[tid];
  }
}

__global__ void softmax_kernel_bf16(hip_bfloat16 *x, int size) {
  __shared__ float max_val;
  __shared__ float sum_val;
  
  if (threadIdx.x == 0) {
    max_val = bfloat16_to_float(x[0]);
    for (int i = 1; i < size; i++) {
      float val = bfloat16_to_float(x[i]);
      if (val > max_val) {
        max_val = val;
      }
    }
    
    sum_val = 0.0f;
    for (int i = 0; i < size; i++) {
      float val = expf(bfloat16_to_float(x[i]) - max_val);
      x[i] = float_to_bfloat16(val);
      sum_val += val;
    }
    
    for (int i = 0; i < size; i++) {
      float val = bfloat16_to_float(x[i]) / sum_val;
      x[i] = float_to_bfloat16(val);
    }
  }
  __syncthreads();
}

__global__ void compute_cos_sin_kernel_bf16(hip_bfloat16 *cos_out, hip_bfloat16 *sin_out, int pos, 
                                           float base, int head_dim, float scaling_factor,
                                           float initial_context_length, float ntk_beta, 
                                           float ntk_alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim / 2;
  
  if (tid < d_half) {
    // Compute in FP32 for precision
    float freq = powf(base, ((float)(2 * tid)) / (float)head_dim);
    float inv_freq;
    
    if (scaling_factor > 1.0f) {
      float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
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

__global__ void apply_rotary_emb_kernel_bf16(hip_bfloat16 *x, hip_bfloat16 *cos, hip_bfloat16 *sin, int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  
  if (h < n_heads && i < half) {
    float x1 = bfloat16_to_float(x[h * head_dim + i]);
    float x2 = bfloat16_to_float(x[h * head_dim + half + i]);
    
    float c = bfloat16_to_float(cos[i]);
    float s = bfloat16_to_float(sin[i]);
    
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    
    x[h * head_dim + i] = float_to_bfloat16(o1);
    x[h * head_dim + half + i] = float_to_bfloat16(o2);
  }
}

__global__ void attention_scores_kernel_bf16(hip_bfloat16 *att, hip_bfloat16 *q, hip_bfloat16 *key_cache, 
                                            int head_dim, int loff, int kv_dim, int kv_mul,
                                            int h, int pos, int seq_len, hip_bfloat16 *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    hip_bfloat16 *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;  // Use FP32 for accumulation
    for (int i = 0; i < head_dim; i++) {
      float q_val = bfloat16_to_float(q[i]);
      float k_val = bfloat16_to_float(k[i]);
      score += q_val * k_val;
    }
    score /= sqrtf(head_dim);
    
    // Apply sliding window mask if enabled
    if (mask != NULL) {
      float mask_val = bfloat16_to_float(mask[pos * seq_len + t]);
      score += mask_val;
    }
    
    att[t] = float_to_bfloat16(score);
  }
}

__global__ void attention_weighted_sum_kernel_bf16(hip_bfloat16 *tb, hip_bfloat16 *att, hip_bfloat16 *value_cache,
                                                   int head_dim, int loff, int kv_dim, int kv_mul,
                                                   int h, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < head_dim) {
    float sum = 0.0f;  // Use FP32 for accumulation
    for (int t = 0; t <= pos; t++) {
      hip_bfloat16 *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      float a = bfloat16_to_float(att[t]);
      float v_val = bfloat16_to_float(v[i]);
      sum += a * v_val;
    }
    tb[h * head_dim + i] = float_to_bfloat16(sum);
  }
}

__global__ void swiglu_kernel_bf16(hip_bfloat16 *gate_up, hip_bfloat16 *gate, hip_bfloat16 *up, int intermediate_dim, float swiglu_limit) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    float val = bfloat16_to_float(gate[tid]);
    float up_val = bfloat16_to_float(up[tid]);
    
    // Clamping
    if (val > swiglu_limit) val = swiglu_limit;
    if (up_val > swiglu_limit) up_val = swiglu_limit;
    if (up_val < -swiglu_limit) up_val = -swiglu_limit;
    
    // SwiGLU: silu(x) = x * sigmoid(alpha * x)
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (up_val + 1.0f);
    
    gate_up[tid] = float_to_bfloat16(val);
  }
}

__global__ void split_gate_up_kernel_bf16(hip_bfloat16 *gate, hip_bfloat16 *up, hip_bfloat16 *mlp1_out, int intermediate_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    gate[tid] = mlp1_out[2 * tid];
    up[tid] = mlp1_out[2 * tid + 1];
  }
}

__global__ void weighted_sum_kernel_bf16(hip_bfloat16 *e_agg, hip_bfloat16 *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float agg_val = bfloat16_to_float(e_agg[tid]);
    float tb2_val = bfloat16_to_float(tb2[tid]);
    e_agg[tid] = float_to_bfloat16(agg_val + tb2_val * expert_w);
  }
}

__global__ void residual_connection_kernel_bf16(hip_bfloat16 *x, hip_bfloat16 *residual, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float x_val = bfloat16_to_float(x[tid]);
    float res_val = bfloat16_to_float(residual[tid]);
    x[tid] = float_to_bfloat16(x_val + res_val);
  }
}

// FP32 matmul kernel for final output
__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

// CPU implementations for topk (will call from host)
void topk_cpu(float *topk_values, int *topk_indices, float *router_score, int num_experts, int experts_per_token) {
  typedef struct { float value; int index; } Pair;
  
  Pair *pairs = (Pair*)malloc(num_experts * sizeof(Pair));
  for (int i = 0; i < num_experts; ++i) {
    pairs[i].value = router_score[i];
    pairs[i].index = i;
  }
  
  // Simple selection sort for top-k
  for (int i = 0; i < experts_per_token; i++) {
    int max_idx = i;
    for (int j = i + 1; j < num_experts; j++) {
      if (pairs[j].value > pairs[max_idx].value) {
        max_idx = j;
      }
    }
    if (max_idx != i) {
      Pair temp = pairs[i];
      pairs[i] = pairs[max_idx];
      pairs[max_idx] = temp;
    }
    topk_values[i] = pairs[i].value;
    topk_indices[i] = pairs[i].index;
  }
  
  free(pairs);
}

void gpu_malloc_run_state(GPURunState *s, Config *p) {
  int kv_dim = p->head_dim * p->n_kv_heads;
  
  // Allocate BF16 buffers
  HIP_CHECK(hipMalloc(&s->x, p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->t, p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb, p->head_dim * p->n_attn_heads * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb2, p->hidden_dim * sizeof(hip_bfloat16)));
  
  HIP_CHECK(hipMalloc(&s->router_score, p->n_experts * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_v, p->experts_per_token * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_i, p->experts_per_token * sizeof(int)));
  
  HIP_CHECK(hipMalloc(&s->mlp1_out, 2 * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate, p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->up, p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate_up, p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->e_agg, p->hidden_dim * sizeof(hip_bfloat16)));
  
  HIP_CHECK(hipMalloc(&s->qkv, p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->q, p->n_attn_heads * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->k, p->n_kv_heads * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->v, p->n_kv_heads * p->head_dim * sizeof(hip_bfloat16)));
  
  HIP_CHECK(hipMalloc(&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->att, p->n_attn_heads * p->seq_len * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->logits, p->vocab_size * sizeof(hip_bfloat16)));
  
  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(hip_bfloat16)));
  } else {
    s->mask = NULL;
  }
  
  HIP_CHECK(hipMalloc(&s->cos_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->sin_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));
  
  // Allocate FP32 buffers for precision-sensitive operations
  HIP_CHECK(hipMalloc(&s->acc_buffer, p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits_fp32, p->vocab_size * sizeof(float)));
}

void gpu_free_run_state(GPURunState *s) {
  hipFree(s->x);
  hipFree(s->t);
  hipFree(s->tb);
  hipFree(s->tb2);
  hipFree(s->router_score);
  hipFree(s->topk_v);
  hipFree(s->topk_i);
  hipFree(s->mlp1_out);
  hipFree(s->gate);
  hipFree(s->up);
  hipFree(s->gate_up);
  hipFree(s->e_agg);
  hipFree(s->qkv);
  hipFree(s->q);
  hipFree(s->k);
  hipFree(s->v);
  hipFree(s->key_cache);
  hipFree(s->value_cache);
  hipFree(s->att);
  hipFree(s->logits);
  if (s->mask) hipFree(s->mask);
  hipFree(s->cos_vals);
  hipFree(s->sin_vals);
  hipFree(s->acc_buffer);
  hipFree(s->logits_fp32);
}

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int head_dim = p->head_dim;
  int n_layers = p->n_layers;
  int n_experts = p->n_experts;
  int hidden_dim = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size = p->vocab_size;
  int n_attn_heads = p->n_attn_heads;
  int n_kv_heads = p->n_kv_heads;
  
  dim3 block(256);
  
  // Helper function to copy and convert FP32 to BF16
  auto copy_and_convert = [&](float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    size_t bf16_size = count * sizeof(hip_bfloat16);
    size_t fp32_size = count * sizeof(float);
    
    // Allocate BF16 buffer
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));
    
    // Allocate temporary FP32 buffer on GPU
    float *temp_fp32;
    HIP_CHECK(hipMalloc(&temp_fp32, fp32_size));
    
    // Copy FP32 data to GPU
    HIP_CHECK(hipMemcpy(temp_fp32, cpu_data, fp32_size, hipMemcpyHostToDevice));
    
    // Convert FP32 to BF16 on GPU
    dim3 grid = dim3((count + block.x - 1) / block.x);
    convert_fp32_to_bf16_kernel<<<grid, block>>>(temp_fp32, *gpu_data, count);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Free temporary buffer
    hipFree(temp_fp32);
  };
  
  // Convert and copy all weights
  size_t token_embed_count = vocab_size * hidden_dim;
  copy_and_convert(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, token_embed_count);
  copy_and_convert(cpu_w->out, &gpu_w->out, token_embed_count);
  
  size_t rms_count = n_layers * hidden_dim;
  copy_and_convert(cpu_w->rms_attn_w, &gpu_w->rms_attn_w, rms_count);
  copy_and_convert(cpu_w->rms_ffn_w, &gpu_w->rms_ffn_w, rms_count);
  
  size_t rms_out_count = hidden_dim;
  copy_and_convert(cpu_w->rms_out_w, &gpu_w->rms_out_w, rms_out_count);
  
  size_t qkv_w_count = n_layers * hidden_dim * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads);
  copy_and_convert(cpu_w->w_qkv, &gpu_w->w_qkv, qkv_w_count);
  
  size_t qkv_b_count = n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads);
  copy_and_convert(cpu_w->b_qkv, &gpu_w->b_qkv, qkv_b_count);
  
  size_t o_w_count = n_layers * (head_dim * n_attn_heads) * hidden_dim;
  copy_and_convert(cpu_w->w_o, &gpu_w->w_o, o_w_count);
  
  size_t o_b_count = n_layers * hidden_dim;
  copy_and_convert(cpu_w->b_o, &gpu_w->b_o, o_b_count);
  
  size_t attn_sinks_count = n_layers * n_attn_heads;
  copy_and_convert(cpu_w->attn_sinks, &gpu_w->attn_sinks, attn_sinks_count);
  
  size_t router_w_count = n_layers * hidden_dim * n_experts;
  copy_and_convert(cpu_w->w_router, &gpu_w->w_router, router_w_count);
  
  size_t router_b_count = n_layers * n_experts;
  copy_and_convert(cpu_w->b_router, &gpu_w->b_router, router_b_count);
  
  size_t mlp1_w_count = n_layers * n_experts * 2 * intermediate_dim * hidden_dim;
  copy_and_convert(cpu_w->w_mlp1, &gpu_w->w_mlp1, mlp1_w_count);
  
  size_t mlp1_b_count = n_layers * n_experts * 2 * intermediate_dim;
  copy_and_convert(cpu_w->b_mlp1, &gpu_w->b_mlp1, mlp1_b_count);
  
  size_t mlp2_w_count = n_layers * n_experts * hidden_dim * intermediate_dim;
  copy_and_convert(cpu_w->w_mlp2, &gpu_w->w_mlp2, mlp2_w_count);
  
  size_t mlp2_b_count = n_layers * n_experts * hidden_dim;
  copy_and_convert(cpu_w->b_mlp2, &gpu_w->b_mlp2, mlp2_b_count);
}

void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  hipFree(gpu_w->token_embedding_table);
  hipFree(gpu_w->out);
  hipFree(gpu_w->rms_attn_w);
  hipFree(gpu_w->rms_ffn_w);
  hipFree(gpu_w->rms_out_w);
  hipFree(gpu_w->w_qkv);
  hipFree(gpu_w->b_qkv);
  hipFree(gpu_w->w_o);
  hipFree(gpu_w->b_o);
  hipFree(gpu_w->attn_sinks);
  hipFree(gpu_w->w_router);
  hipFree(gpu_w->b_router);
  hipFree(gpu_w->w_mlp1);
  hipFree(gpu_w->b_mlp1);
  hipFree(gpu_w->w_mlp2);
  hipFree(gpu_w->b_mlp2);
}

float *gpu_forward(Transformer *transformer, int token, int pos) {
  Config *p = &transformer->config;
  
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;
  
  dim3 block(256);
  dim3 grid_hidden = dim3((hidden_dim + block.x - 1) / block.x);
  dim3 grid_vocab = dim3((p->vocab_size + block.x - 1) / block.x);
  dim3 grid_intermediate = dim3((intermediate_dim + block.x - 1) / block.x);
  dim3 grid_experts = dim3((n_experts + block.x - 1) / block.x);
  
  // Copy token embedding
  hip_bfloat16 *token_row = gpu_weights.token_embedding_table + token * hidden_dim;
  HIP_CHECK(hipMemcpy(gpu_state.x, token_row, hidden_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
  
  // Forward through layers
  for (int l = 0; l < p->n_layers; l++) {
    // Attention RMSNorm
    hip_bfloat16 *rms_attn_w = gpu_weights.rms_attn_w + l * hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block>>>(gpu_state.t, gpu_state.x, rms_attn_w, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // QKV projection
    int loff = l * p->seq_len * kv_dim;
    hip_bfloat16 *w_qkv = gpu_weights.w_qkv + l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    hip_bfloat16 *b_qkv = gpu_weights.b_qkv + l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    
    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 grid_qkv = dim3((qkv_out_dim + block.x - 1) / block.x);
    matmul_kernel_bf16<<<grid_qkv, block>>>(gpu_state.qkv, gpu_state.t, w_qkv, hidden_dim, qkv_out_dim);
    add_bias_kernel_bf16<<<grid_qkv, block>>>(gpu_state.qkv, b_qkv, qkv_out_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Separate Q, K, V
    int q_size = head_dim * p->n_attn_heads;
    int k_size = head_dim * p->n_kv_heads;
    int v_size = head_dim * p->n_kv_heads;
    
    HIP_CHECK(hipMemcpy(gpu_state.q, gpu_state.qkv, q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(gpu_state.k, gpu_state.qkv + q_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(gpu_state.v, gpu_state.qkv + q_size + k_size, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    
    // Store K, V in cache
    hip_bfloat16 *k_cache_pos = gpu_state.key_cache + loff + pos * kv_dim;
    hip_bfloat16 *v_cache_pos = gpu_state.value_cache + loff + pos * kv_dim;
    HIP_CHECK(hipMemcpy(k_cache_pos, gpu_state.k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(v_cache_pos, gpu_state.v, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    
    // RoPE
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    dim3 grid_rope = dim3((head_dim / 2 + block.x - 1) / block.x);
    compute_cos_sin_kernel_bf16<<<grid_rope, block>>>(gpu_state.cos_vals, gpu_state.sin_vals, pos,
                                                     p->rope_theta, head_dim, p->rope_scaling_factor,
                                                     p->initial_context_length, ntk_beta, ntk_alpha);
    HIP_CHECK(hipDeviceSynchronize());
    
    apply_rotary_emb_kernel_bf16<<<p->n_attn_heads, head_dim/2>>>(gpu_state.q, gpu_state.cos_vals, gpu_state.sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb_kernel_bf16<<<p->n_kv_heads, head_dim/2>>>(gpu_state.k, gpu_state.cos_vals, gpu_state.sin_vals, p->n_kv_heads, head_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Update K cache after RoPE
    HIP_CHECK(hipMemcpy(k_cache_pos, gpu_state.k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    
    // Multi-head attention
    for (int h = 0; h < p->n_attn_heads; h++) {
      hip_bfloat16 *q = gpu_state.q + h * head_dim;
      hip_bfloat16 *att = gpu_state.att + h * p->seq_len;
      
      // Compute attention scores
      dim3 grid_att = dim3((pos + 1 + block.x - 1) / block.x);
      attention_scores_kernel_bf16<<<grid_att, block>>>(att, q, gpu_state.key_cache, head_dim, loff, 
                                                       kv_dim, kv_mul, h, pos, p->seq_len,
                                                       (p->sliding_window > 0 && l % 2 == 0) ? gpu_state.mask : NULL);
      HIP_CHECK(hipDeviceSynchronize());
      
      // Add attention sink
      hip_bfloat16 bf16_sink;
      HIP_CHECK(hipMemcpy(&bf16_sink, gpu_weights.attn_sinks + l * p->n_attn_heads + h, sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(att + pos + 1, &bf16_sink, sizeof(hip_bfloat16), hipMemcpyHostToDevice));
      
      // Softmax
      softmax_kernel_bf16<<<1, 1>>>(att, pos + 2);
      HIP_CHECK(hipDeviceSynchronize());
      
      // Weighted sum
      dim3 grid_head_dim = dim3((head_dim + block.x - 1) / block.x);
      attention_weighted_sum_kernel_bf16<<<grid_head_dim, block>>>(gpu_state.tb, att, gpu_state.value_cache,
                                                                  head_dim, loff, kv_dim, kv_mul, h, pos);
      HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Output projection
    hip_bfloat16 *w_o = gpu_weights.w_o + l * (head_dim * p->n_attn_heads) * hidden_dim;
    hip_bfloat16 *b_o = gpu_weights.b_o + l * hidden_dim;
    matmul_kernel_bf16<<<grid_hidden, block>>>(gpu_state.tb2, gpu_state.tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    add_bias_kernel_bf16<<<grid_hidden, block>>>(gpu_state.tb2, b_o, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Residual connection
    residual_connection_kernel_bf16<<<grid_hidden, block>>>(gpu_state.x, gpu_state.tb2, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // MoE
    // FFN RMSNorm
    hip_bfloat16 *rms_ffn_w = gpu_weights.rms_ffn_w + l * hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block>>>(gpu_state.t, gpu_state.x, rms_ffn_w, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Router
    hip_bfloat16 *w_router = gpu_weights.w_router + l * hidden_dim * n_experts;
    hip_bfloat16 *b_router = gpu_weights.b_router + l * n_experts;
    matmul_kernel_bf16<<<grid_experts, block>>>(gpu_state.router_score, gpu_state.t, w_router, hidden_dim, n_experts);
    add_bias_kernel_bf16<<<grid_experts, block>>>(gpu_state.router_score, b_router, n_experts);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Convert router scores to FP32 for topk
    float *cpu_router_score = (float*)malloc(n_experts * sizeof(float));
    float *cpu_topk_v = (float*)malloc(p->experts_per_token * sizeof(float));
    int *cpu_topk_i = (int*)malloc(p->experts_per_token * sizeof(int));
    
    // Convert BF16 to FP32 on GPU first
    float *gpu_router_score_fp32;
    HIP_CHECK(hipMalloc(&gpu_router_score_fp32, n_experts * sizeof(float)));
    convert_bf16_to_fp32_kernel<<<grid_experts, block>>>(gpu_state.router_score, gpu_router_score_fp32, n_experts);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(cpu_router_score, gpu_router_score_fp32, n_experts * sizeof(float), hipMemcpyDeviceToHost));
    topk_cpu(cpu_topk_v, cpu_topk_i, cpu_router_score, n_experts, p->experts_per_token);
    
    // Softmax on topk values
    float sum = 0.0f;
    float max_val = cpu_topk_v[0];
    for (int i = 1; i < p->experts_per_token; i++) {
      if (cpu_topk_v[i] > max_val) max_val = cpu_topk_v[i];
    }
    for (int i = 0; i < p->experts_per_token; i++) {
      cpu_topk_v[i] = expf(cpu_topk_v[i] - max_val);
      sum += cpu_topk_v[i];
    }
    for (int i = 0; i < p->experts_per_token; i++) {
      cpu_topk_v[i] /= sum;
    }
    
    // Convert back to BF16 and copy to GPU
    hip_bfloat16 *gpu_topk_v_bf16;
    HIP_CHECK(hipMalloc(&gpu_topk_v_bf16, p->experts_per_token * sizeof(hip_bfloat16)));
    float *gpu_topk_v_fp32;
    HIP_CHECK(hipMalloc(&gpu_topk_v_fp32, p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMemcpy(gpu_topk_v_fp32, cpu_topk_v, p->experts_per_token * sizeof(float), hipMemcpyHostToDevice));
    
    dim3 grid_topk = dim3((p->experts_per_token + block.x - 1) / block.x);
    convert_fp32_to_bf16_kernel<<<grid_topk, block>>>(gpu_topk_v_fp32, gpu_topk_v_bf16, p->experts_per_token);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(gpu_state.topk_v, gpu_topk_v_bf16, p->experts_per_token * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(gpu_state.topk_i, cpu_topk_i, p->experts_per_token * sizeof(int), hipMemcpyHostToDevice));
    
    // Initialize expert aggregation
    HIP_CHECK(hipMemset(gpu_state.e_agg, 0, hidden_dim * sizeof(hip_bfloat16)));
    
    // Process selected experts
    for (int idx = 0; idx < p->experts_per_token; idx++) {
      int e = cpu_topk_i[idx];
      float expert_w = cpu_topk_v[idx];
      
      // MLP1 (gate_up projection)
      hip_bfloat16 *w_mlp1 = gpu_weights.w_mlp1 + (l * n_experts + e) * (2 * intermediate_dim) * hidden_dim;
      hip_bfloat16 *b_mlp1 = gpu_weights.b_mlp1 + (l * n_experts + e) * (2 * intermediate_dim);
      
      dim3 grid_mlp1 = dim3((2 * intermediate_dim + block.x - 1) / block.x);
      matmul_kernel_bf16<<<grid_mlp1, block>>>(gpu_state.mlp1_out, gpu_state.t, w_mlp1, hidden_dim, 2 * intermediate_dim);
      add_bias_kernel_bf16<<<grid_mlp1, block>>>(gpu_state.mlp1_out, b_mlp1, 2 * intermediate_dim);
      HIP_CHECK(hipDeviceSynchronize());
      
      // Split gate and up
      split_gate_up_kernel_bf16<<<grid_intermediate, block>>>(gpu_state.gate, gpu_state.up, gpu_state.mlp1_out, intermediate_dim);
      HIP_CHECK(hipDeviceSynchronize());
      
      // SwiGLU
      swiglu_kernel_bf16<<<grid_intermediate, block>>>(gpu_state.gate_up, gpu_state.gate, gpu_state.up, intermediate_dim, p->swiglu_limit);
      HIP_CHECK(hipDeviceSynchronize());
      
      // MLP2 (down projection)
      hip_bfloat16 *w_mlp2 = gpu_weights.w_mlp2 + (l * n_experts + e) * hidden_dim * intermediate_dim;
      hip_bfloat16 *b_mlp2 = gpu_weights.b_mlp2 + (l * n_experts + e) * hidden_dim;
      
      matmul_kernel_bf16<<<grid_hidden, block>>>(gpu_state.tb2, gpu_state.gate_up, w_mlp2, intermediate_dim, hidden_dim);
      add_bias_kernel_bf16<<<grid_hidden, block>>>(gpu_state.tb2, b_mlp2, hidden_dim);
      HIP_CHECK(hipDeviceSynchronize());
      
      // Weighted sum to expert aggregation
      weighted_sum_kernel_bf16<<<grid_hidden, block>>>(gpu_state.e_agg, gpu_state.tb2, expert_w, hidden_dim);
      HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Final residual connection
    residual_connection_kernel_bf16<<<grid_hidden, block>>>(gpu_state.x, gpu_state.e_agg, hidden_dim);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Cleanup temporary buffers
    hipFree(gpu_router_score_fp32);
    hipFree(gpu_topk_v_bf16);
    hipFree(gpu_topk_v_fp32);
    free(cpu_router_score);
    free(cpu_topk_v);
    free(cpu_topk_i);
  }
  
  // Final RMSNorm
  rmsnorm_kernel_bf16<<<grid_hidden, block>>>(gpu_state.x, gpu_state.x, gpu_weights.rms_out_w, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());
  
  // Output projection - convert to FP32 for final logits
  // First convert x to FP32
  convert_bf16_to_fp32_kernel<<<grid_hidden, block>>>(gpu_state.x, gpu_state.acc_buffer, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());
  
  // Convert output weights to FP32 for final computation
  float *w_out_fp32;
  HIP_CHECK(hipMalloc(&w_out_fp32, p->vocab_size * hidden_dim * sizeof(float)));
  dim3 grid_out_weights = dim3((p->vocab_size * hidden_dim + block.x - 1) / block.x);
  convert_bf16_to_fp32_kernel<<<grid_out_weights, block>>>(gpu_weights.out, w_out_fp32, p->vocab_size * hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());
  
  // Final matmul in FP32
  matmul_kernel<<<grid_vocab, block>>>(gpu_state.logits_fp32, gpu_state.acc_buffer, w_out_fp32, hidden_dim, p->vocab_size);
  HIP_CHECK(hipDeviceSynchronize());
  
  // Copy logits back to CPU
  static float *cpu_logits = NULL;
  if (cpu_logits == NULL) {
    cpu_logits = (float*)malloc(p->vocab_size * sizeof(float));
  }
  HIP_CHECK(hipMemcpy(cpu_logits, gpu_state.logits_fp32, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
  
  hipFree(w_out_fp32);
  return cpu_logits;
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Initialize GPU
  HIP_CHECK(hipSetDevice(0));
  
  // Allocate GPU memory for weights and state
  gpu_copy_weights(&gpu_weights, &transformer->weights, &transformer->config);
  gpu_malloc_run_state(&gpu_state, &transformer->config);
  
  // Initialize mask for sliding window if needed
  if (transformer->config.sliding_window > 0 && gpu_state.mask != NULL) {
    float *cpu_mask = (float*)malloc(transformer->config.seq_len * transformer->config.seq_len * sizeof(float));
    for (int i = 0; i < transformer->config.seq_len; i++) {
      for (int j = 0; j < transformer->config.seq_len; j++) {
        if (i - j >= transformer->config.sliding_window) {
          cpu_mask[i * transformer->config.seq_len + j] = -INFINITY;
        } else {
          cpu_mask[i * transformer->config.seq_len + j] = 0.0f;
        }
      }
    }
    
    // Convert to BF16 and copy
    float *gpu_mask_fp32;
    HIP_CHECK(hipMalloc(&gpu_mask_fp32, transformer->config.seq_len * transformer->config.seq_len * sizeof(float)));
    HIP_CHECK(hipMemcpy(gpu_mask_fp32, cpu_mask, 
                       transformer->config.seq_len * transformer->config.seq_len * sizeof(float), 
                       hipMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid = dim3((transformer->config.seq_len * transformer->config.seq_len + block.x - 1) / block.x);
    convert_fp32_to_bf16_kernel<<<grid, block>>>(gpu_mask_fp32, gpu_state.mask, 
                                                transformer->config.seq_len * transformer->config.seq_len);
    HIP_CHECK(hipDeviceSynchronize());
    
    hipFree(gpu_mask_fp32);
    free(cpu_mask);
  }
  
  printf("GPU warm-up completed successfully with BF16 precision.\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Free GPU memory
  gpu_free_weights(&gpu_weights);
  gpu_free_run_state(&gpu_state);
  
  printf("GPU cleanup completed successfully.\n");
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  while (pos < steps) {
    // forward the transformer to get logits for the next token
    float *logits = gpu_forward(transformer, token, pos);  // Use GPU version

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the EOS (=199999 or =200002) token
    // delimits sequences
    if (next == 199999 || next == 200002) {
      break;
    }

    token = next;
  }

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

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