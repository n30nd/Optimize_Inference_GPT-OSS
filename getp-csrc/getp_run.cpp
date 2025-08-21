// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

#ifndef GETP_RUN
#define GETP_RUN

#include <hip/hip_runtime.h>
#include <vector>

// -----------------------------------------------------------------------------
// GPU data structures
// -----------------------------------------------------------------------------

typedef struct {
  hipDeviceptr_t token_embedding_table;
  hipDeviceptr_t rms_attn_w;
  hipDeviceptr_t rms_ffn_w;
  hipDeviceptr_t w_qkv;
  hipDeviceptr_t w_o;
  hipDeviceptr_t b_qkv;
  hipDeviceptr_t b_o;
  hipDeviceptr_t attn_sinks;
  hipDeviceptr_t w_router;
  hipDeviceptr_t b_router;
  hipDeviceptr_t w_mlp1;
  hipDeviceptr_t w_mlp2;
  hipDeviceptr_t b_mlp1;
  hipDeviceptr_t b_mlp2;
  hipDeviceptr_t rms_out_w;
  hipDeviceptr_t out;
} GPUWeights;

typedef struct {
  hipDeviceptr_t x;
  hipDeviceptr_t t;
  hipDeviceptr_t tb;
  hipDeviceptr_t tb2;
  hipDeviceptr_t router_score;
  hipDeviceptr_t topk_v;
  hipDeviceptr_t topk_i;
  hipDeviceptr_t mlp1_out;
  hipDeviceptr_t gate;
  hipDeviceptr_t up;
  hipDeviceptr_t gate_up;
  hipDeviceptr_t e_agg;
  hipDeviceptr_t qkv;
  hipDeviceptr_t q;
  hipDeviceptr_t k;
  hipDeviceptr_t v;
  hipDeviceptr_t att;
  hipDeviceptr_t logits;
  hipDeviceptr_t key_cache;
  hipDeviceptr_t value_cache;
  hipDeviceptr_t mask;
} GPUState;

typedef struct {
  Config config;
  GPUWeights weights;
  GPUState state;
} GPUTransformer;

static GPUTransformer *g_gpu = nullptr;

// -----------------------------------------------------------------------------
// Helpers for copying weights to the GPU
// -----------------------------------------------------------------------------

static inline uint16_t float_to_half_bits(float f) {
  uint32_t x = *(uint32_t *)&f;
  uint16_t sign = (x >> 16) & 0x8000u;
  uint32_t mant = x & 0x7fffffu;
  int exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10)
      return sign;
    mant = (mant | 0x800000u) >> (1 - exp);
    return sign | (mant >> 13);
  } else if (exp >= 31) {
    return sign | 0x7c00u;
  } else {
    return sign | (exp << 10) | (mant >> 13);
  }
}

static inline uint16_t float_to_bf16_bits(float f) {
  uint32_t x = *(uint32_t *)&f;
  return (uint16_t)(x >> 16);
}

static void copy_to_device(hipDeviceptr_t *dst, const float *src, size_t count,
                           bool use_bf16) {
  std::vector<uint16_t> tmp(count);
  if (use_bf16) {
    for (size_t i = 0; i < count; ++i)
      tmp[i] = float_to_bf16_bits(src[i]);
  } else {
    for (size_t i = 0; i < count; ++i)
      tmp[i] = float_to_half_bits(src[i]);
  }
  hipMalloc(dst, count * sizeof(uint16_t));
  hipMemcpy((void *)*dst, tmp.data(), count * sizeof(uint16_t),
            hipMemcpyHostToDevice);
}

// -----------------------------------------------------------------------------
// Convert CPU Transformer to GPU representation
// -----------------------------------------------------------------------------

Transformer *to_gpu_transformer(const Transformer *cpu) {
  GPUTransformer *gpu =
      reinterpret_cast<GPUTransformer *>(calloc(1, sizeof(GPUTransformer)));
  gpu->config = cpu->config;

  bool use_bf16 = true; // default to bf16 copies
  const Config *p = &cpu->config;
  const TransformerWeights *cw = &cpu->weights;
  GPUWeights *gw = &gpu->weights;

  size_t n;

  n = (size_t)p->vocab_size * p->hidden_dim;
  copy_to_device(&gw->token_embedding_table, cw->token_embedding_table, n,
                 use_bf16);

  n = (size_t)p->n_layers * p->hidden_dim;
  copy_to_device(&gw->rms_attn_w, cw->rms_attn_w, n, use_bf16);
  copy_to_device(&gw->rms_ffn_w, cw->rms_ffn_w, n, use_bf16);

  n = (size_t)p->n_layers *
      (p->head_dim * p->n_attn_heads + 2 * p->head_dim * p->n_kv_heads) *
      p->hidden_dim;
  copy_to_device(&gw->w_qkv, cw->w_qkv, n, use_bf16);

  n = (size_t)p->n_layers * p->hidden_dim *
      (p->head_dim * p->n_attn_heads);
  copy_to_device(&gw->w_o, cw->w_o, n, use_bf16);

  n = (size_t)p->n_layers *
      (p->head_dim * p->n_attn_heads + 2 * p->head_dim * p->n_kv_heads);
  copy_to_device(&gw->b_qkv, cw->b_qkv, n, use_bf16);

  n = (size_t)p->n_layers * p->hidden_dim;
  copy_to_device(&gw->b_o, cw->b_o, n, use_bf16);

  n = (size_t)p->n_layers * p->n_attn_heads;
  copy_to_device(&gw->attn_sinks, cw->attn_sinks, n, use_bf16);

  n = (size_t)p->n_layers * p->hidden_dim * p->n_experts;
  copy_to_device(&gw->w_router, cw->w_router, n, use_bf16);

  n = (size_t)p->n_layers * p->n_experts;
  copy_to_device(&gw->b_router, cw->b_router, n, use_bf16);

  n = (size_t)p->n_layers * p->n_experts * 2 * p->intermediate_dim *
      p->hidden_dim;
  copy_to_device(&gw->w_mlp1, cw->w_mlp1, n, use_bf16);

  n = (size_t)p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
  copy_to_device(&gw->w_mlp2, cw->w_mlp2, n, use_bf16);

  n = (size_t)p->n_layers * p->n_experts * 2 * p->intermediate_dim;
  copy_to_device(&gw->b_mlp1, cw->b_mlp1, n, use_bf16);

  n = (size_t)p->n_layers * p->n_experts * p->hidden_dim;
  copy_to_device(&gw->b_mlp2, cw->b_mlp2, n, use_bf16);

  n = (size_t)p->hidden_dim;
  copy_to_device(&gw->rms_out_w, cw->rms_out_w, n, use_bf16);

  n = (size_t)p->vocab_size * p->hidden_dim;
  copy_to_device(&gw->out, cw->out, n, use_bf16);

  // Allocate state buffers on device
  GPUState *gs = &gpu->state;
  size_t kv_dim = (size_t)p->head_dim * p->n_kv_heads;

  hipMalloc(&gs->x, p->hidden_dim * sizeof(float));
  hipMalloc(&gs->t, p->hidden_dim * sizeof(float));
  hipMalloc(&gs->tb, p->head_dim * p->n_attn_heads * sizeof(float));
  hipMalloc(&gs->tb2, p->hidden_dim * sizeof(float));
  hipMalloc(&gs->router_score, p->n_experts * sizeof(float));
  hipMalloc(&gs->topk_v, p->experts_per_token * sizeof(float));
  hipMalloc(&gs->topk_i, p->experts_per_token * sizeof(int));
  hipMalloc(&gs->mlp1_out, 2 * p->intermediate_dim * sizeof(float));
  hipMalloc(&gs->gate, p->intermediate_dim * sizeof(float));
  hipMalloc(&gs->up, p->intermediate_dim * sizeof(float));
  hipMalloc(&gs->gate_up, p->intermediate_dim * sizeof(float));
  hipMalloc(&gs->e_agg, p->hidden_dim * sizeof(float));
  hipMalloc(&gs->qkv, p->head_dim *
                         (p->n_attn_heads + 2 * p->n_kv_heads) *
                         sizeof(float));
  hipMalloc(&gs->q, p->n_attn_heads * p->head_dim * sizeof(float));
  hipMalloc(&gs->k, kv_dim * sizeof(float));
  hipMalloc(&gs->v, kv_dim * sizeof(float));
  hipMalloc(&gs->att, p->n_attn_heads * p->seq_len * sizeof(float));
  hipMalloc(&gs->logits, p->vocab_size * sizeof(float));
  hipMalloc(&gs->key_cache,
            (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float));
  hipMalloc(&gs->value_cache,
            (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float));
  if (p->sliding_window > 0) {
    hipMalloc(&gs->mask, (size_t)p->seq_len * p->seq_len * sizeof(float));
  } else {
    gs->mask = 0;
  }

  return reinterpret_cast<Transformer *>(gpu);
}

static void free_gpu_transformer(GPUTransformer *gpu) {
  GPUWeights *w = &gpu->weights;
  GPUState *s = &gpu->state;

  // weights
#define HIPFREE(x)                                                                \
  if (x)                                                                          \
    hipFree((void *)x);
  HIPFREE(w->token_embedding_table);
  HIPFREE(w->rms_attn_w);
  HIPFREE(w->rms_ffn_w);
  HIPFREE(w->w_qkv);
  HIPFREE(w->w_o);
  HIPFREE(w->b_qkv);
  HIPFREE(w->b_o);
  HIPFREE(w->attn_sinks);
  HIPFREE(w->w_router);
  HIPFREE(w->b_router);
  HIPFREE(w->w_mlp1);
  HIPFREE(w->w_mlp2);
  HIPFREE(w->b_mlp1);
  HIPFREE(w->b_mlp2);
  HIPFREE(w->rms_out_w);
  HIPFREE(w->out);

  // state
  HIPFREE(s->x);
  HIPFREE(s->t);
  HIPFREE(s->tb);
  HIPFREE(s->tb2);
  HIPFREE(s->router_score);
  HIPFREE(s->topk_v);
  HIPFREE(s->topk_i);
  HIPFREE(s->mlp1_out);
  HIPFREE(s->gate);
  HIPFREE(s->up);
  HIPFREE(s->gate_up);
  HIPFREE(s->e_agg);
  HIPFREE(s->qkv);
  HIPFREE(s->q);
  HIPFREE(s->k);
  HIPFREE(s->v);
  HIPFREE(s->att);
  HIPFREE(s->logits);
  HIPFREE(s->key_cache);
  HIPFREE(s->value_cache);
  HIPFREE(s->mask);

#undef HIPFREE

  free(gpu);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Convert the CPU transformer to its GPU representation once.
  if (g_gpu == nullptr && transformer != nullptr) {
    g_gpu =
        reinterpret_cast<GPUTransformer *>(to_gpu_transformer(transformer));
  }
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Release GPU resources created during warm_up.
  if (g_gpu != nullptr) {
    free_gpu_transformer(g_gpu);
    g_gpu = nullptr;
  }
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
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
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);

    token = next;
  }

  // should be removed
  printf("\n");

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
