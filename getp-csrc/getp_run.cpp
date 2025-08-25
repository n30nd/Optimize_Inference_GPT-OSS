// TODO: GPU optimized version for AMD MI250 using HIP (no HIPBLAS)
#include "getp_eval.cpp"
#include <hip/hip_runtime.h>

#ifndef GETP_RUN
#define GETP_RUN

// GPU memory structures
typedef struct {
    // GPU weights pointers
    float *d_token_embedding_table;
    float *d_rms_attn_w;
    float *d_rms_ffn_w;
    float *d_w_qkv;
    float *d_w_o;
    float *d_b_qkv;
    float *d_b_o;
    float *d_attn_sinks;
    float *d_w_router;
    float *d_b_router;
    float *d_w_mlp1;
    float *d_w_mlp2;
    float *d_b_mlp1;
    float *d_b_mlp2;
    float *d_rms_out_w;
    float *d_out;
} GPUWeights;

typedef struct {
    // GPU state pointers
    float *d_x;
    float *d_t;
    float *d_tb;
    float *d_tb2;
    float *d_router_score;
    float *d_topk_v;
    int *d_topk_i;
    float *d_mlp1_out;
    float *d_gate;
    float *d_up;
    float *d_gate_up;
    float *d_e_agg;
    float *d_qkv;
    float *d_q;
    float *d_k;
    float *d_v;
    float *d_att;
    float *d_logits;
    float *d_key_cache;
    float *d_value_cache;
    float *d_mask;
    
    // Working buffers
    float *d_cos_vals;
    float *d_sin_vals;
} GPUState;

// Global GPU structures
static GPUWeights g_gpu_weights;
static GPUState g_gpu_state;
static float *g_cpu_logits = NULL; // Dynamic allocation

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple RMSNorm kernel without complex reduction
__global__ void rmsnorm_simple_kernel(float *output, const float *input, const float *weight, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute sum of squares across all threads
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += input[i] * input[i];
    }
    
    float rms = sqrtf(sum_sq / size + 1e-5f);
    
    if (idx < size) {
        output[idx] = (input[idx] / rms) * weight[idx];
    }
}

// Simple softmax kernel
__global__ void softmax_simple_kernel(float *x, int size) {
    int idx = threadIdx.x;
    
    // Find max value
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

__global__ void add_bias_kernel(float *output, const float *bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += bias[idx];
    }
}

__global__ void copy_embeddings_kernel(float *output, const float *embeddings, int token, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        output[idx] = embeddings[token * hidden_dim + idx];
    }
}

__global__ void element_add_kernel(float *a, const float *b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

// Simple matrix-vector multiplication kernel
__global__ void matvec_simple_kernel(float *output, const float *input, const float *weight, 
                                    int in_features, int out_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < out_features) {
        float sum = 0.0f;
        for (int col = 0; col < in_features; col++) {
            sum += weight[row * in_features + col] * input[col];
        }
        output[row] = sum;
    }
}

// GPU matrix multiplication wrapper
void gpu_matmul(float *output, const float *input, const float *weight, int in_features, int out_features) {
    int block_size = 256;
    dim3 grid((out_features + block_size - 1) / block_size);
    dim3 block(block_size);
    matvec_simple_kernel<<<grid, block>>>(output, input, weight, in_features, out_features);
}

// Simple kernels for MLP operations
__global__ void extract_gate_up_kernel(float *gate, float *up, const float *gate_up_interleaved, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gate[idx] = gate_up_interleaved[2 * idx];
        up[idx] = gate_up_interleaved[2 * idx + 1];
    }
}

__global__ void swiglu_kernel(float *gate_up, const float *gate, const float *up, int size, float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = gate[idx];
        float up_val = up[idx];
        
        // Clamping
        val = fminf(fmaxf(val, -limit), limit);
        up_val = fminf(fmaxf(up_val, -limit), limit);
        
        // SwiGLU: silu(x) * (up(x) + 1)
        const float alpha = 1.702f;
        val *= (1.0f / (1.0f + expf(-alpha * val)));
        val *= (up_val + 1.0f);
        
        gate_up[idx] = val;
    }
}

__global__ void copy_qkv_kernel(float *q, float *k, float *v, const float *qkv, 
                               int n_attn_heads, int n_kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_size = n_attn_heads * head_dim;
    int k_size = n_kv_heads * head_dim;
    int v_size = n_kv_heads * head_dim;
    
    if (idx < q_size) {
        q[idx] = qkv[idx];
    } else if (idx < q_size + k_size) {
        k[idx - q_size] = qkv[idx];
    } else if (idx < q_size + k_size + v_size) {
        v[idx - q_size - k_size] = qkv[idx];
    }
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
    printf("Initializing GPU resources...\n");
    
    // Initialize HIP
    HIP_CHECK(hipSetDevice(0));
    
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    
    int hidden_dim = p->hidden_dim;
    int vocab_size = p->vocab_size;
    int n_layers = p->n_layers;
    int n_experts = p->n_experts;
    int head_dim = p->head_dim;
    int n_attn_heads = p->n_attn_heads;
    int n_kv_heads = p->n_kv_heads;
    int seq_len = p->seq_len;
    int intermediate_dim = p->intermediate_dim;
    int experts_per_token = p->experts_per_token;
    int kv_dim = head_dim * n_kv_heads;
    
    printf("Allocating GPU memory for vocab_size=%d, hidden_dim=%d...\n", vocab_size, hidden_dim);
    
    // Allocate CPU buffer for logits
    g_cpu_logits = (float*)malloc(vocab_size * sizeof(float));
    if (!g_cpu_logits) {
        fprintf(stderr, "Failed to allocate CPU logits buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate GPU memory for weights
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_token_embedding_table, (size_t)vocab_size * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_rms_attn_w, (size_t)n_layers * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_rms_ffn_w, (size_t)n_layers * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_w_qkv, (size_t)n_layers * hidden_dim * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_w_o, (size_t)n_layers * (head_dim * n_attn_heads) * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_b_qkv, (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_b_o, (size_t)n_layers * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_attn_sinks, (size_t)n_layers * n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_w_router, (size_t)n_layers * hidden_dim * n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_b_router, (size_t)n_layers * n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_w_mlp1, (size_t)n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_w_mlp2, (size_t)n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_b_mlp1, (size_t)n_layers * n_experts * 2 * intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_b_mlp2, (size_t)n_layers * n_experts * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_rms_out_w, (size_t)hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_weights.d_out, (size_t)vocab_size * hidden_dim * sizeof(float)));
    
    // Allocate GPU memory for state
    HIP_CHECK(hipMalloc(&g_gpu_state.d_x, (size_t)hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_t, (size_t)hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_tb, (size_t)head_dim * n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_tb2, (size_t)hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_router_score, (size_t)n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_topk_v, (size_t)experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_topk_i, (size_t)experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_mlp1_out, (size_t)2 * intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_gate, (size_t)intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_up, (size_t)intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_gate_up, (size_t)intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_e_agg, (size_t)hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_qkv, (size_t)head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_q, (size_t)n_attn_heads * head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_k, (size_t)n_kv_heads * head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_v, (size_t)n_kv_heads * head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_att, (size_t)n_attn_heads * seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_logits, (size_t)vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_key_cache, (size_t)n_layers * seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_value_cache, (size_t)n_layers * seq_len * kv_dim * sizeof(float)));
    
    if (p->sliding_window > 0) {
        HIP_CHECK(hipMalloc(&g_gpu_state.d_mask, (size_t)seq_len * seq_len * sizeof(float)));
    }
    
    // Working buffers
    HIP_CHECK(hipMalloc(&g_gpu_state.d_cos_vals, (size_t)(head_dim / 2) * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_state.d_sin_vals, (size_t)(head_dim / 2) * sizeof(float)));
    
    printf("Copying weights to GPU...\n");
    
    // Copy weights from CPU to GPU
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_token_embedding_table, w->token_embedding_table, (size_t)vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_rms_attn_w, w->rms_attn_w, (size_t)n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_rms_ffn_w, w->rms_ffn_w, (size_t)n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_w_qkv, w->w_qkv, (size_t)n_layers * hidden_dim * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_w_o, w->w_o, (size_t)n_layers * (head_dim * n_attn_heads) * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_b_qkv, w->b_qkv, (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_b_o, w->b_o, (size_t)n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_attn_sinks, w->attn_sinks, (size_t)n_layers * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_w_router, w->w_router, (size_t)n_layers * hidden_dim * n_experts * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_b_router, w->b_router, (size_t)n_layers * n_experts * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_w_mlp1, w->w_mlp1, (size_t)n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_w_mlp2, w->w_mlp2, (size_t)n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_b_mlp1, w->b_mlp1, (size_t)n_layers * n_experts * 2 * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_b_mlp2, w->b_mlp2, (size_t)n_layers * n_experts * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_rms_out_w, w->rms_out_w, (size_t)hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_gpu_weights.d_out, w->out, (size_t)vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    // Initialize KV cache to zero
    HIP_CHECK(hipMemset(g_gpu_state.d_key_cache, 0, (size_t)n_layers * seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(g_gpu_state.d_value_cache, 0, (size_t)n_layers * seq_len * kv_dim * sizeof(float)));
    
    HIP_CHECK(hipDeviceSynchronize());
    printf("GPU initialization completed.\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
    printf("Cleaning up GPU resources...\n");
    
    // Free CPU buffer
    if (g_cpu_logits) {
        free(g_cpu_logits);
        g_cpu_logits = NULL;
    }
    
    // Free GPU weights
    hipFree(g_gpu_weights.d_token_embedding_table);
    hipFree(g_gpu_weights.d_rms_attn_w);
    hipFree(g_gpu_weights.d_rms_ffn_w);
    hipFree(g_gpu_weights.d_w_qkv);
    hipFree(g_gpu_weights.d_w_o);
    hipFree(g_gpu_weights.d_b_qkv);
    hipFree(g_gpu_weights.d_b_o);
    hipFree(g_gpu_weights.d_attn_sinks);
    hipFree(g_gpu_weights.d_w_router);
    hipFree(g_gpu_weights.d_b_router);
    hipFree(g_gpu_weights.d_w_mlp1);
    hipFree(g_gpu_weights.d_w_mlp2);
    hipFree(g_gpu_weights.d_b_mlp1);
    hipFree(g_gpu_weights.d_b_mlp2);
    hipFree(g_gpu_weights.d_rms_out_w);
    hipFree(g_gpu_weights.d_out);
    
    // Free GPU state
    hipFree(g_gpu_state.d_x);
    hipFree(g_gpu_state.d_t);
    hipFree(g_gpu_state.d_tb);
    hipFree(g_gpu_state.d_tb2);
    hipFree(g_gpu_state.d_router_score);
    hipFree(g_gpu_state.d_topk_v);
    hipFree(g_gpu_state.d_topk_i);
    hipFree(g_gpu_state.d_mlp1_out);
    hipFree(g_gpu_state.d_gate);
    hipFree(g_gpu_state.d_up);
    hipFree(g_gpu_state.d_gate_up);
    hipFree(g_gpu_state.d_e_agg);
    hipFree(g_gpu_state.d_qkv);
    hipFree(g_gpu_state.d_q);
    hipFree(g_gpu_state.d_k);
    hipFree(g_gpu_state.d_v);
    hipFree(g_gpu_state.d_att);
    hipFree(g_gpu_state.d_logits);
    hipFree(g_gpu_state.d_key_cache);
    hipFree(g_gpu_state.d_value_cache);
    if (g_gpu_state.d_mask) hipFree(g_gpu_state.d_mask);
    hipFree(g_gpu_state.d_cos_vals);
    hipFree(g_gpu_state.d_sin_vals);
    
    printf("GPU cleanup completed.\n");
}

// Simplified GPU forward function
float *gpu_forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    
    int hidden_dim = p->hidden_dim;
    int head_dim = p->head_dim;
    int n_attn_heads = p->n_attn_heads;
    int n_kv_heads = p->n_kv_heads;
    int n_layers = p->n_layers;
    int vocab_size = p->vocab_size;
    
    // Copy token embedding to GPU
    dim3 block(256);
    dim3 grid((hidden_dim + block.x - 1) / block.x);
    copy_embeddings_kernel<<<grid, block>>>(g_gpu_state.d_x, g_gpu_weights.d_token_embedding_table, token, hidden_dim);
    
    // Forward through all layers (simplified version)
    for (int l = 0; l < n_layers; l++) {
        // RMSNorm for attention
        rmsnorm_simple_kernel<<<grid, block>>>(
            g_gpu_state.d_t, g_gpu_state.d_x, 
            g_gpu_weights.d_rms_attn_w + l * hidden_dim, hidden_dim);
        
        // QKV projection
        gpu_matmul(g_gpu_state.d_qkv, g_gpu_state.d_t, 
                  g_gpu_weights.d_w_qkv + l * hidden_dim * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads),
                  hidden_dim, (n_attn_heads + 2 * n_kv_heads) * head_dim);
        
        // Add bias
        dim3 bias_grid(((n_attn_heads + 2 * n_kv_heads) * head_dim + block.x - 1) / block.x);
        add_bias_kernel<<<bias_grid, block>>>(g_gpu_state.d_qkv, 
                                            g_gpu_weights.d_b_qkv + l * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads),
                                            (n_attn_heads + 2 * n_kv_heads) * head_dim);
        
        // Copy Q, K, V
        dim3 qkv_grid(((n_attn_heads + 2 * n_kv_heads) * head_dim + block.x - 1) / block.x);
        copy_qkv_kernel<<<qkv_grid, block>>>(g_gpu_state.d_q, g_gpu_state.d_k, g_gpu_state.d_v, 
                                            g_gpu_state.d_qkv, n_attn_heads, n_kv_heads, head_dim);
        
        // Update KV cache (simplified - copy current K,V to cache)
        int kv_dim = head_dim * n_kv_heads;
        int cache_offset = l * p->seq_len * kv_dim + pos * kv_dim;
        HIP_CHECK(hipMemcpy(g_gpu_state.d_key_cache + cache_offset, g_gpu_state.d_k, 
                           kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(g_gpu_state.d_value_cache + cache_offset, g_gpu_state.d_v, 
                           kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
        
        // Simplified attention: just use identity for now (skip complex attention computation)
        HIP_CHECK(hipMemcpy(g_gpu_state.d_tb, g_gpu_state.d_q, 
                           head_dim * n_attn_heads * sizeof(float), hipMemcpyDeviceToDevice));
        
        // Output projection
        gpu_matmul(g_gpu_state.d_tb2, g_gpu_state.d_tb, 
                  g_gpu_weights.d_w_o + l * (head_dim * n_attn_heads) * hidden_dim,
                  head_dim * n_attn_heads, hidden_dim);
        
        // Add bias and residual
        add_bias_kernel<<<grid, block>>>(g_gpu_state.d_tb2, 
                                        g_gpu_weights.d_b_o + l * hidden_dim, hidden_dim);
        element_add_kernel<<<grid, block>>>(g_gpu_state.d_x, g_gpu_state.d_tb2, hidden_dim);
        
        // FFN 
        rmsnorm_simple_kernel<<<grid, block>>>(
            g_gpu_state.d_t, g_gpu_state.d_x, 
            g_gpu_weights.d_rms_ffn_w + l * hidden_dim, hidden_dim);
        
        // Simplified MoE: use first expert only
        int expert_idx = 0;
        
        // MLP1 (gate_up projection)
        gpu_matmul(g_gpu_state.d_mlp1_out, g_gpu_state.d_t,
                  g_gpu_weights.d_w_mlp1 + (l * p->n_experts + expert_idx) * 2 * p->intermediate_dim * hidden_dim,
                  hidden_dim, 2 * p->intermediate_dim);
        
        // Add bias
        dim3 mlp_grid((2 * p->intermediate_dim + block.x - 1) / block.x);
        add_bias_kernel<<<mlp_grid, block>>>(g_gpu_state.d_mlp1_out,
                                            g_gpu_weights.d_b_mlp1 + (l * p->n_experts + expert_idx) * 2 * p->intermediate_dim,
                                            2 * p->intermediate_dim);
        
        // Extract gate and up parts
        dim3 gate_grid((p->intermediate_dim + block.x - 1) / block.x);
        extract_gate_up_kernel<<<gate_grid, block>>>(g_gpu_state.d_gate, g_gpu_state.d_up, 
                                                    g_gpu_state.d_mlp1_out, p->intermediate_dim);
        
        swiglu_kernel<<<gate_grid, block>>>(g_gpu_state.d_gate_up, g_gpu_state.d_gate, 
                                          g_gpu_state.d_up, p->intermediate_dim, p->swiglu_limit);
        
        // MLP2 (down projection)
        gpu_matmul(g_gpu_state.d_tb2, g_gpu_state.d_gate_up,
                  g_gpu_weights.d_w_mlp2 + (l * p->n_experts + expert_idx) * hidden_dim * p->intermediate_dim,
                  p->intermediate_dim, hidden_dim);
        
        // Add bias
        add_bias_kernel<<<grid, block>>>(g_gpu_state.d_tb2,
                                        g_gpu_weights.d_b_mlp2 + (l * p->n_experts + expert_idx) * hidden_dim,
                                        hidden_dim);
        
        // Residual connection
        element_add_kernel<<<grid, block>>>(g_gpu_state.d_x, g_gpu_state.d_tb2, hidden_dim);
    }
    
    // Final RMSNorm
    rmsnorm_simple_kernel<<<grid, block>>>(
        g_gpu_state.d_x, g_gpu_state.d_x, g_gpu_weights.d_rms_out_w, hidden_dim);
    
    // Output projection
    gpu_matmul(g_gpu_state.d_logits, g_gpu_state.d_x, g_gpu_weights.d_out, hidden_dim, vocab_size);
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy logits back to CPU for sampling
    HIP_CHECK(hipMemcpy(g_cpu_logits, g_gpu_state.d_logits, vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    
    return g_cpu_logits;
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
    int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
    encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
           transformer->config.initial_context_length);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    // print the very first token
    const char *first_piece = decode_piece(tokenizer, 200006, token);
    safe_printf(first_piece);
    fflush(stdout);

    while (pos < steps) {
        // forward the transformer to get logits for the next token using GPU
        float *logits = gpu_forward(transformer, token, pos);

        // advance the state machine
        pos++;
        if (pos < num_prompt_tokens) {
            next = prompt_tokens[pos];
        } else {
            next = sample(sampler, logits);
            output_tokens[pos - num_prompt_tokens] = next;
        }

        // data-dependent terminating condition
        if (next == 199999 || next == 200002) {
            break;
        }

        // print the token as string
        const char *piece = decode_piece(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);

        token = next;
    }

    printf("\n");
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
