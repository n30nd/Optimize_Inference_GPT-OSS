// Loads gpt-oss straight from a Hugging Face model folder: the `original/`
// safetensors files OpenAI ships (bf16 tensors, expert weights in MXFP4).
//
// Small tensors are converted to fp32 into the same TransformerWeights layout
// run.cpp reads. Expert weights (~95% of the model) stay in MXFP4 inside the
// memory-mapped files and are decoded when used, so the 120b model needs
// ~70 GB of RAM instead of 467 GB.
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "safetensors.hpp"

// The course's exported .bin files used seq_len 2048; the HF config has no
// such field.
#define HF_DEFAULT_MAX_SEQ_LEN 2048

// MXFP4: each byte holds two 4-bit codes (low nibble first), each group of 32
// values shares one scale byte s meaning 2^(s - 127).
static const float kFp4Values[16] = {+0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f,
                                     +4.0f, +6.0f, -0.0f, -0.5f, -1.0f, -1.5f,
                                     -2.0f, -3.0f, -4.0f, -6.0f};

struct HfLayerExperts {
  const uint8_t *mlp1_blocks; // (n_experts, 2 * intermediate_dim, hidden_dim / 32, 16)
  const uint8_t *mlp1_scales; // (n_experts, 2 * intermediate_dim, hidden_dim / 32)
  const uint8_t *mlp2_blocks; // (n_experts, hidden_dim, intermediate_dim / 32, 16)
  const uint8_t *mlp2_scales; // (n_experts, hidden_dim, intermediate_dim / 32)
};

struct HfModel {
  bool loaded = false;
  std::string dir; // folder holding the safetensors files
  StModel st;
  std::vector<HfLayerExperts> experts; // one entry per layer
  std::vector<float *> owned;          // fp32 buffers allocated by the loader
};

static HfModel g_hf;

static inline float bf16_to_f32(const uint8_t *p) {
  uint16_t h;
  memcpy(&h, p, 2);
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static void bf16_to_f32_array(float *dst, const uint8_t *src, size_t n) {
  long long i;
#pragma omp parallel for private(i)
  for (i = 0; i < (long long)n; i++)
    dst[i] = bf16_to_f32(src + 2 * i);
}

// Decodes `rows` MXFP4 rows of `cols` values each into fp32. Gives the same
// values as dequantize_fp4() in export_model_bin/*/export_model_bin.py.
static void mxfp4_to_f32(float *dst, const uint8_t *blocks,
                         const uint8_t *scales, size_t rows, int cols) {
  const int groups = cols / 32;
  long long r;
#pragma omp parallel for private(r)
  for (r = 0; r < (long long)rows; r++) {
    const uint8_t *brow = blocks + (size_t)r * groups * 16;
    const uint8_t *srow = scales + (size_t)r * groups;
    float *out = dst + (size_t)r * cols;
    for (int g = 0; g < groups; g++) {
      float scale = ldexpf(1.0f, (int)srow[g] - 127);
      for (int k = 0; k < 16; k++) {
        uint8_t byte = brow[g * 16 + k];
        out[g * 32 + 2 * k] = kFp4Values[byte & 0x0F] * scale;
        out[g * 32 + 2 * k + 1] = kFp4Values[byte >> 4] * scale;
      }
    }
  }
}

static float *hf_alloc_f32(size_t n) {
  float *p = (float *)malloc(n * sizeof(float));
  if (!p) {
    fprintf(stderr, "weights: out of memory allocating %zu floats\n", n);
    exit(EXIT_FAILURE);
  }
  g_hf.owned.push_back(p);
  return p;
}

static size_t hf_numel(const std::vector<int64_t> &shape) {
  size_t n = 1;
  for (auto d : shape)
    n *= (size_t)d;
  return n;
}

static void hf_load_bf16(float *dst, const std::string &name,
                         const std::vector<int64_t> &shape) {
  const StTensor &t = st_require(&g_hf.st, name, "BF16", shape);
  bf16_to_f32_array(dst, t.data, hf_numel(shape));
}

static bool hf_is_dir(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool hf_is_file(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// `path` may be the model folder (…/gpt-oss-20b) or its original/ subfolder.
static std::string hf_weights_dir(const std::string &path) {
  if (hf_is_file(path + "/original/config.json"))
    return path + "/original";
  if (hf_is_file(path + "/config.json"))
    return path;
  fprintf(stderr,
          "weights: %s has neither original/config.json nor config.json\n",
          path.c_str());
  exit(EXIT_FAILURE);
}

static double hf_config_number(const JsonValue &cfg, const char *key) {
  const JsonValue *v = cfg.get(key);
  if (!v || v->type != JsonValue::Number) {
    fprintf(stderr, "weights: config.json has no number '%s'\n", key);
    exit(EXIT_FAILURE);
  }
  return v->number;
}

static void hf_read_config(const std::string &dir, Config *c) {
  JsonValue cfg;
  std::string path = dir + "/config.json";
  if (!json_parse_file(path.c_str(), &cfg) || cfg.type != JsonValue::Object) {
    fprintf(stderr, "weights: cannot read %s\n", path.c_str());
    exit(EXIT_FAILURE);
  }
  c->vocab_size = (int)hf_config_number(cfg, "vocab_size");
  c->hidden_dim = (int)hf_config_number(cfg, "hidden_size");
  c->n_experts = (int)hf_config_number(cfg, "num_experts");
  c->experts_per_token = (int)hf_config_number(cfg, "experts_per_token");
  c->intermediate_dim = (int)hf_config_number(cfg, "intermediate_size");
  c->n_layers = (int)hf_config_number(cfg, "num_hidden_layers");
  c->head_dim = (int)hf_config_number(cfg, "head_dim");
  c->n_attn_heads = (int)hf_config_number(cfg, "num_attention_heads");
  c->n_kv_heads = (int)hf_config_number(cfg, "num_key_value_heads");
  const JsonValue *seq = cfg.get("max_seq_len");
  c->seq_len = seq ? (int)seq->number : HF_DEFAULT_MAX_SEQ_LEN;
  c->initial_context_length =
      (int)hf_config_number(cfg, "initial_context_length");
  c->rope_theta = (float)hf_config_number(cfg, "rope_theta");
  c->rope_scaling_factor = (float)hf_config_number(cfg, "rope_scaling_factor");
  c->sliding_window = (int)hf_config_number(cfg, "sliding_window");
  c->swiglu_limit = (float)hf_config_number(cfg, "swiglu_limit");

  // run.cpp hard-codes ntk_beta = 32 and ntk_alpha = 1 in forward().
  const JsonValue *alpha = cfg.get("rope_ntk_alpha");
  const JsonValue *beta = cfg.get("rope_ntk_beta");
  if ((alpha && alpha->number != 1.0) || (beta && beta->number != 32.0))
    fprintf(stderr, "warning: config rope_ntk_alpha/beta differ from the "
                    "values run.cpp uses (1, 32)\n");
  if (c->hidden_dim % 32 != 0 || c->intermediate_dim % 32 != 0) {
    fprintf(stderr, "weights: MXFP4 needs sizes divisible by 32\n");
    exit(EXIT_FAILURE);
  }
}

static void hf_print_config(const Config *c) {
  printf("vocab_size: %d\n", c->vocab_size);
  printf("hidden_dim: %d\n", c->hidden_dim);
  printf("n_experts: %d\n", c->n_experts);
  printf("experts_per_token: %d\n", c->experts_per_token);
  printf("intermediate_dim: %d\n", c->intermediate_dim);
  printf("n_layers: %d\n", c->n_layers);
  printf("head_dim: %d\n", c->head_dim);
  printf("n_attn_heads: %d\n", c->n_attn_heads);
  printf("n_kv_heads: %d\n", c->n_kv_heads);
  printf("max_seq_len: %d\n", c->seq_len);
  printf("init context len: %d\n", c->initial_context_length);
  printf("rope theta: %f\n", c->rope_theta);
  printf("rope_scaling_factor: %f\n", c->rope_scaling_factor);
  printf("sliding window: %d\n", c->sliding_window);
  printf("swiglu_limit: %f\n", c->swiglu_limit);
}

// Fills t->config and t->weights from a Hugging Face folder and allocates the
// run state, like build_transformer() does for a .bin file.
static void hf_build_transformer(Transformer *t, const char *path) {
  long start = time_in_ms();
  Config *c = &t->config;
  TransformerWeights *w = &t->weights;

  g_hf.dir = hf_weights_dir(path);
  hf_read_config(g_hf.dir, c);
  hf_print_config(c);
  if (!st_open_dir(&g_hf.st, g_hf.dir))
    exit(EXIT_FAILURE);

  const int64_t V = c->vocab_size, H = c->hidden_dim, L = c->n_layers;
  const int64_t E = c->n_experts, I = c->intermediate_dim;
  const int64_t QKV = (int64_t)c->head_dim * (c->n_attn_heads + 2 * c->n_kv_heads);
  const int64_t ATT = (int64_t)c->head_dim * c->n_attn_heads;

  w->token_embedding_table = hf_alloc_f32(V * H);
  hf_load_bf16(w->token_embedding_table, "embedding.weight", {V, H});
  w->out = hf_alloc_f32(V * H);
  hf_load_bf16(w->out, "unembedding.weight", {V, H});
  w->rms_out_w = hf_alloc_f32(H);
  hf_load_bf16(w->rms_out_w, "norm.scale", {H});

  w->rms_attn_w = hf_alloc_f32(L * H);
  w->rms_ffn_w = hf_alloc_f32(L * H);
  w->w_qkv = hf_alloc_f32(L * QKV * H);
  w->b_qkv = hf_alloc_f32(L * QKV);
  w->w_o = hf_alloc_f32(L * H * ATT);
  w->b_o = hf_alloc_f32(L * H);
  w->attn_sinks = hf_alloc_f32(L * c->n_attn_heads);
  w->w_router = hf_alloc_f32(L * E * H);
  w->b_router = hf_alloc_f32(L * E);
  w->b_mlp1 = hf_alloc_f32(L * E * 2 * I);
  w->b_mlp2 = hf_alloc_f32(L * E * H);
  // Expert weights are not expanded to fp32; see g_hf.experts.
  w->w_mlp1 = NULL;
  w->w_mlp2 = NULL;

  g_hf.experts.resize(L);
  for (int64_t l = 0; l < L; l++) {
    std::string b = "block." + std::to_string(l) + ".";
    hf_load_bf16(w->rms_attn_w + l * H, b + "attn.norm.scale", {H});
    hf_load_bf16(w->rms_ffn_w + l * H, b + "mlp.norm.scale", {H});
    hf_load_bf16(w->w_qkv + l * QKV * H, b + "attn.qkv.weight", {QKV, H});
    hf_load_bf16(w->b_qkv + l * QKV, b + "attn.qkv.bias", {QKV});
    hf_load_bf16(w->w_o + l * H * ATT, b + "attn.out.weight", {H, ATT});
    hf_load_bf16(w->b_o + l * H, b + "attn.out.bias", {H});
    hf_load_bf16(w->attn_sinks + l * c->n_attn_heads, b + "attn.sinks",
                 {c->n_attn_heads});
    hf_load_bf16(w->w_router + l * E * H, b + "mlp.gate.weight", {E, H});
    hf_load_bf16(w->b_router + l * E, b + "mlp.gate.bias", {E});
    hf_load_bf16(w->b_mlp1 + l * E * 2 * I, b + "mlp.mlp1_bias", {E, 2 * I});
    hf_load_bf16(w->b_mlp2 + l * E * H, b + "mlp.mlp2_bias", {E, H});

    HfLayerExperts &x = g_hf.experts[l];
    x.mlp1_blocks =
        st_require(&g_hf.st, b + "mlp.mlp1_weight.blocks", "U8", {E, 2 * I, H / 32, 16}).data;
    x.mlp1_scales =
        st_require(&g_hf.st, b + "mlp.mlp1_weight.scales", "U8", {E, 2 * I, H / 32}).data;
    x.mlp2_blocks =
        st_require(&g_hf.st, b + "mlp.mlp2_weight.blocks", "U8", {E, H, I / 32, 16}).data;
    x.mlp2_scales =
        st_require(&g_hf.st, b + "mlp.mlp2_weight.scales", "U8", {E, H, I / 32}).data;
  }

  // run.cpp's free_transformer() skips munmap/close for these values.
  t->data = (float *)MAP_FAILED;
  t->fd = -1;
  t->file_size = 0;
  malloc_run_state(&t->state, c);
  g_hf.loaded = true;

  printf("weights: Hugging Face folder %s (%zu files, %.1f GB), loaded in "
         "%.1f s\n",
         g_hf.dir.c_str(), g_hf.st.files.size(), g_hf.st.total_bytes / 1e9,
         (time_in_ms() - start) / 1000.0);
  fflush(stdout);
}

static void hf_free() {
  for (float *p : g_hf.owned)
    free(p);
  g_hf.owned.clear();
  g_hf.experts.clear();
  st_close(&g_hf.st);
  g_hf.loaded = false;
}
