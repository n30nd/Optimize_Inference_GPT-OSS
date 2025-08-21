#pragma once
#include <hip/hip_runtime.h>
#include <vector>

// Re-declare Config to allow use without including the whole CPU implementation.
typedef struct {
  int vocab_size;             // vocabulary size
  int hidden_dim;             // model dimension
  int n_experts;              // number of experts
  int experts_per_token;      // number of experts per token
  int intermediate_dim;       // feed-forward dimension
  int n_layers;               // number of transformer layers
  int head_dim;               // dimension of each head
  int n_attn_heads;           // number of attention heads
  int n_kv_heads;             // number of key/value heads
  int seq_len;                // maximum sequence length
  int initial_context_length; // initial context length
  float rope_theta;           // RoPE base
  float rope_scaling_factor;  // RoPE scaling factor
  int sliding_window;         // sliding window
  float swiglu_limit;         // SwiGLU limit
} Config;

struct GpuWeights {
  float *token_embedding_table;
  float *rms_attn_w;
  float *rms_ffn_w;
  float *w_qkv;
  float *w_o;
  float *b_qkv;
  float *b_o;
  float *attn_sinks;
  float *w_router;
  float *b_router;
  float *w_mlp1;
  float *w_mlp2;
  float *b_mlp1;
  float *b_mlp2;
  float *rms_out_w;
  float *out;
};

struct GpuState {
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
  float *att;
  float *logits;
  float *key_cache;
  float *value_cache;
  float *mask;
};

struct GpuTransformer {
  Config config;
  GpuWeights weights;
  GpuState state;
};

extern std::vector<GpuTransformer> g_gpu_transformers;

