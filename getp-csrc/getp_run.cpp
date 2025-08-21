// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include "gpu_transformer.hpp"
#include <cstring>

// Global array of GPU transformers, one per device
std::vector<GpuTransformer> g_gpu_transformers;

#ifndef GETP_RUN
#define GETP_RUN

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Allocate device memory and move model weights to each GPU
  int num_devices = 0;
  hipGetDeviceCount(&num_devices);
  if (num_devices <= 0)
    return;

  g_gpu_transformers.resize(num_devices);

  for (int i = 0; i < num_devices; ++i) {
    hipSetDevice(i);

    GpuTransformer &gt = g_gpu_transformers[i];
    gt.config = transformer->config; // copy configuration for later use

    Config *cfg = &gt.config;
    GpuWeights &gw = gt.weights;
    GpuState &gs = gt.state;

    size_t size;
    // --- weights ---
    size = 1ll * cfg->vocab_size * cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.token_embedding_table, size);
    hipMemcpy(gw.token_embedding_table,
              transformer->weights.token_embedding_table, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.rms_attn_w, size);
    hipMemcpy(gw.rms_attn_w, transformer->weights.rms_attn_w, size,
              hipMemcpyHostToDevice);
    hipMalloc(&gw.rms_ffn_w, size);
    hipMemcpy(gw.rms_ffn_w, transformer->weights.rms_ffn_w, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.rms_out_w, size);
    hipMemcpy(gw.rms_out_w, transformer->weights.rms_out_w, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->vocab_size * cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.out, size);
    hipMemcpy(gw.out, transformer->weights.out, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->hidden_dim *
           (cfg->head_dim * cfg->n_attn_heads +
            2 * cfg->head_dim * cfg->n_kv_heads) *
           sizeof(float);
    hipMalloc(&gw.w_qkv, size);
    hipMemcpy(gw.w_qkv, transformer->weights.w_qkv, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers *
           (cfg->head_dim * cfg->n_attn_heads +
            2 * cfg->head_dim * cfg->n_kv_heads) *
           sizeof(float);
    hipMalloc(&gw.b_qkv, size);
    hipMemcpy(gw.b_qkv, transformer->weights.b_qkv, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * (cfg->head_dim * cfg->n_attn_heads) *
           cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.w_o, size);
    hipMemcpy(gw.w_o, transformer->weights.w_o, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.b_o, size);
    hipMemcpy(gw.b_o, transformer->weights.b_o, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->n_attn_heads * sizeof(float);
    hipMalloc(&gw.attn_sinks, size);
    hipMemcpy(gw.attn_sinks, transformer->weights.attn_sinks, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->hidden_dim * cfg->n_experts *
           sizeof(float);
    hipMalloc(&gw.w_router, size);
    hipMemcpy(gw.w_router, transformer->weights.w_router, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->n_experts * sizeof(float);
    hipMalloc(&gw.b_router, size);
    hipMemcpy(gw.b_router, transformer->weights.b_router, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->n_experts * 2 *
           cfg->intermediate_dim * cfg->hidden_dim * sizeof(float);
    hipMalloc(&gw.w_mlp1, size);
    hipMemcpy(gw.w_mlp1, transformer->weights.w_mlp1, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->n_experts * 2 *
           cfg->intermediate_dim * sizeof(float);
    hipMalloc(&gw.b_mlp1, size);
    hipMemcpy(gw.b_mlp1, transformer->weights.b_mlp1, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->n_experts * cfg->hidden_dim *
           cfg->intermediate_dim * sizeof(float);
    hipMalloc(&gw.w_mlp2, size);
    hipMemcpy(gw.w_mlp2, transformer->weights.w_mlp2, size,
              hipMemcpyHostToDevice);

    size = 1ll * cfg->n_layers * cfg->n_experts * cfg->hidden_dim *
           sizeof(float);
    hipMalloc(&gw.b_mlp2, size);
    hipMemcpy(gw.b_mlp2, transformer->weights.b_mlp2, size,
              hipMemcpyHostToDevice);

    // --- state ---
    int kv_dim = cfg->head_dim * cfg->n_kv_heads;
    hipMalloc(&gs.x, cfg->hidden_dim * sizeof(float));
    hipMalloc(&gs.t, cfg->hidden_dim * sizeof(float));
    hipMalloc(&gs.tb, cfg->head_dim * cfg->n_attn_heads * sizeof(float));
    hipMalloc(&gs.tb2, cfg->hidden_dim * sizeof(float));
    hipMalloc(&gs.router_score, cfg->n_experts * sizeof(float));
    hipMalloc(&gs.topk_v, cfg->experts_per_token * sizeof(float));
    hipMalloc(&gs.topk_i, cfg->experts_per_token * sizeof(int));
    hipMalloc(&gs.mlp1_out, 2 * cfg->intermediate_dim * sizeof(float));
    hipMalloc(&gs.gate, cfg->intermediate_dim * sizeof(float));
    hipMalloc(&gs.up, cfg->intermediate_dim * sizeof(float));
    hipMalloc(&gs.gate_up, cfg->intermediate_dim * sizeof(float));
    hipMalloc(&gs.e_agg, cfg->hidden_dim * sizeof(float));
    hipMalloc(&gs.qkv, cfg->head_dim *
                         (cfg->n_attn_heads + 2 * cfg->n_kv_heads) *
                         sizeof(float));
    hipMalloc(&gs.q, cfg->n_attn_heads * cfg->head_dim * sizeof(float));
    hipMalloc(&gs.k, kv_dim * sizeof(float));
    hipMalloc(&gs.v, kv_dim * sizeof(float));
    hipMalloc(&gs.att, cfg->n_attn_heads * cfg->seq_len * sizeof(float));
    hipMalloc(&gs.logits, cfg->vocab_size * sizeof(float));
    hipMalloc(&gs.key_cache,
              1ll * cfg->n_layers * cfg->seq_len * kv_dim * sizeof(float));
    hipMalloc(&gs.value_cache,
              1ll * cfg->n_layers * cfg->seq_len * kv_dim * sizeof(float));
    if (cfg->sliding_window > 0) {
      hipMalloc(&gs.mask,
                1ll * cfg->seq_len * cfg->seq_len * sizeof(float));
    } else {
      gs.mask = nullptr;
    }
  }

  // Free CPU state and weights now that they reside on device
  free_run_state(&transformer->state);
  memset(&transformer->state, 0, sizeof(transformer->state));

  if (transformer->data != MAP_FAILED) {
    munmap(transformer->data, transformer->file_size);
    transformer->data = (float *)MAP_FAILED;
  }
  transformer->fd = -1;
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
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
