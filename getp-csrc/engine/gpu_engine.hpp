// getp on GPUs. Every visible GCD holds a full copy of the model and serves an
// equal, contiguous share of the requests (data parallel). On a GCD, up to
// `batch` requests run together and advance one position per step (static
// batching); finished requests stay in the batch until the wave ends.
//
// Weights: bf16 for the dense matrices, MXFP4 for the experts, fp32 for norms,
// biases and sinks, exactly as stored in the checkpoint. Activations and the
// KV cache are fp32. All kernels are the *_exact versions, so a GPU step
// reproduces hf_cpu_forward() bit for bit (tests/gpu_vs_cpu.cpp).
#pragma once

#include <stdlib.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "../kernels/attention.hpp"
#include "../kernels/gemm.hpp"
#include "../kernels/misc.hpp"
#include "../kernels/moe.hpp"
#include "hf_model.hpp"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err_ = (call);                                                  \
    if (err_ != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(err_),  \
              __FILE__, __LINE__, #call);                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define GPU_DEFAULT_BATCH 512
#define GPU_DEFAULT_MAX_POS 1024

struct GpuLayer {
  const uint16_t *w_qkv, *w_o, *w_router;
  const float *b_qkv, *b_o, *b_router, *sinks, *rms_attn, *rms_ffn;
  const uint8_t *mlp1_blocks, *mlp1_scales, *mlp2_blocks, *mlp2_scales;
  const float *b_mlp1, *b_mlp2;
  float *kc, *vc; // KV cache (batch, slots, kv_dim)
  int slots;      // positions kept: max_pos, or the sliding window
  int window;     // 0 for full attention
};

struct GpuWorker {
  int device = 0;
  hipStream_t stream;
  Config cfg;
  int batch = 0;   // requests per wave
  int max_pos = 0; // positions the KV cache holds
  std::vector<GpuLayer> layers;
  const uint16_t *emb = nullptr, *unemb = nullptr;
  const float *rms_out = nullptr;
  float *cos_tab = nullptr, *sin_tab = nullptr; // (max_pos, head_dim / 2)
  int *tokens = nullptr, *next = nullptr;
  float *x, *t, *qkv, *att, *proj, *router, *topk_w, *Xg, *Yg, *Gg, *Zg, *logits;
  int *topk_e, *offsets, *row_pair, *pair_row;
  std::vector<void *> weight_allocs, state_allocs;
};

static std::vector<GpuWorker *> g_workers;

template <class T>
static T *gpu_alloc(std::vector<void *> &owner, size_t n) {
  void *p = nullptr;
  HIP_CHECK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  owner.push_back(p);
  return (T *)p;
}

template <class T>
static T *gpu_upload(std::vector<void *> &owner, const void *src, size_t n) {
  T *p = gpu_alloc<T>(owner, n);
  HIP_CHECK(hipMemcpy(p, src, n * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

static int env_int(const char *name, int fallback) {
  const char *v = getenv(name);
  return v && *v ? atoi(v) : fallback;
}

// Weights come from the safetensors mapping (bf16, MXFP4) and from the fp32
// copies hf_build_transformer() made of the small tensors.
static void gpu_upload_weights(GpuWorker *w, const Transformer *t) {
  const Config &c = t->config;
  const TransformerWeights &tw = t->weights;
  const size_t H = c.hidden_dim, E = c.n_experts, I = c.intermediate_dim;
  const size_t QKV = (size_t)c.head_dim * (c.n_attn_heads + 2 * c.n_kv_heads);
  auto &own = w->weight_allocs;
  auto raw = [&](const std::string &name) -> const StTensor & {
    auto it = g_hf.st.tensors.find(name);
    if (it == g_hf.st.tensors.end()) {
      fprintf(stderr, "weights: tensor %s not found\n", name.c_str());
      exit(EXIT_FAILURE);
    }
    return it->second;
  };
  auto bf16 = [&](const std::string &name) {
    const StTensor &s = raw(name);
    return (const uint16_t *)gpu_upload<uint16_t>(own, s.data, s.nbytes / 2);
  };
  auto bytes = [&](const std::string &name) {
    const StTensor &s = raw(name);
    return (const uint8_t *)gpu_upload<uint8_t>(own, s.data, s.nbytes);
  };
  auto f32 = [&](const float *src, size_t n) {
    return (const float *)gpu_upload<float>(own, src, n);
  };

  w->emb = bf16("embedding.weight");
  w->unemb = bf16("unembedding.weight");
  w->rms_out = f32(tw.rms_out_w, H);
  w->layers.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; l++) {
    GpuLayer &g = w->layers[l];
    const std::string b = "block." + std::to_string(l) + ".";
    g.w_qkv = bf16(b + "attn.qkv.weight");
    g.b_qkv = f32(tw.b_qkv + l * QKV, QKV);
    g.w_o = bf16(b + "attn.out.weight");
    g.b_o = f32(tw.b_o + l * H, H);
    g.sinks = f32(tw.attn_sinks + (size_t)l * c.n_attn_heads, c.n_attn_heads);
    g.rms_attn = f32(tw.rms_attn_w + l * H, H);
    g.rms_ffn = f32(tw.rms_ffn_w + l * H, H);
    g.w_router = bf16(b + "mlp.gate.weight");
    g.b_router = f32(tw.b_router + l * E, E);
    g.mlp1_blocks = bytes(b + "mlp.mlp1_weight.blocks");
    g.mlp1_scales = bytes(b + "mlp.mlp1_weight.scales");
    g.b_mlp1 = f32(tw.b_mlp1 + l * E * 2 * I, E * 2 * I);
    g.mlp2_blocks = bytes(b + "mlp.mlp2_weight.blocks");
    g.mlp2_scales = bytes(b + "mlp.mlp2_weight.scales");
    g.b_mlp2 = f32(tw.b_mlp2 + l * E * H, E * H);
  }
}

static size_t gpu_state_bytes(const Config &c, int batch, int max_pos) {
  const size_t B = batch, H = c.hidden_dim, K = c.experts_per_token;
  const size_t QKV = (size_t)c.head_dim * (c.n_attn_heads + 2 * c.n_kv_heads);
  const size_t ATT = (size_t)c.head_dim * c.n_attn_heads;
  const size_t KVD = (size_t)c.head_dim * c.n_kv_heads;
  size_t kv_positions = 0;
  for (int l = 0; l < c.n_layers; l++)
    kv_positions += (c.sliding_window > 0 && l % 2 == 0)
                        ? std::min(c.sliding_window, max_pos)
                        : max_pos;
  size_t floats = 2 * B * kv_positions * KVD                  // KV cache
                  + B * (2 * H + QKV + ATT + H + c.n_experts) // x t qkv att proj router
                  + B * K * (1 + H + 2 * c.intermediate_dim + c.intermediate_dim + H)
                  + B * (size_t)c.vocab_size                  // logits
                  + 2 * (size_t)max_pos * c.head_dim / 2;     // cos, sin
  return floats * sizeof(float) + B * K * 3 * sizeof(int);
}

static void gpu_alloc_state(GpuWorker *w, int batch, int max_pos) {
  const Config &c = w->cfg;
  const size_t B = batch, H = c.hidden_dim, K = c.experts_per_token;
  const size_t QKV = (size_t)c.head_dim * (c.n_attn_heads + 2 * c.n_kv_heads);
  const size_t ATT = (size_t)c.head_dim * c.n_attn_heads;
  const size_t KVD = (size_t)c.head_dim * c.n_kv_heads;
  auto &own = w->state_allocs;
  w->batch = batch;
  w->max_pos = max_pos;
  for (int l = 0; l < c.n_layers; l++) {
    GpuLayer &g = w->layers[l];
    g.window = (c.sliding_window > 0 && l % 2 == 0) ? c.sliding_window : 0;
    g.slots = g.window > 0 ? std::min(g.window, max_pos) : max_pos;
    g.kc = gpu_alloc<float>(own, B * g.slots * KVD);
    g.vc = gpu_alloc<float>(own, B * g.slots * KVD);
  }
  w->tokens = gpu_alloc<int>(own, B);
  w->next = gpu_alloc<int>(own, B);
  w->x = gpu_alloc<float>(own, B * H);
  w->t = gpu_alloc<float>(own, B * H);
  w->qkv = gpu_alloc<float>(own, B * QKV);
  w->att = gpu_alloc<float>(own, B * ATT);
  w->proj = gpu_alloc<float>(own, B * H);
  w->router = gpu_alloc<float>(own, B * c.n_experts);
  w->topk_w = gpu_alloc<float>(own, B * K);
  w->topk_e = gpu_alloc<int>(own, B * K);
  w->offsets = gpu_alloc<int>(own, c.n_experts + 1);
  w->row_pair = gpu_alloc<int>(own, B * K);
  w->pair_row = gpu_alloc<int>(own, B * K);
  w->Xg = gpu_alloc<float>(own, B * K * H);
  w->Yg = gpu_alloc<float>(own, B * K * 2 * c.intermediate_dim);
  w->Gg = gpu_alloc<float>(own, B * K * c.intermediate_dim);
  w->Zg = gpu_alloc<float>(own, B * K * H);
  w->logits = gpu_alloc<float>(own, B * c.vocab_size);

  // RoPE angles from run.cpp's own compute_cos_sin(), one row per position.
  const int half = c.head_dim / 2;
  std::vector<float> cos_h((size_t)max_pos * half), sin_h((size_t)max_pos * half);
  for (int pos = 0; pos < max_pos; pos++)
    compute_cos_sin(pos, c.rope_theta, c.head_dim, c.rope_scaling_factor,
                    c.initial_context_length, 32.0f, 1.0f,
                    &cos_h[(size_t)pos * half], &sin_h[(size_t)pos * half]);
  w->cos_tab = gpu_upload<float>(own, cos_h.data(), cos_h.size());
  w->sin_tab = gpu_upload<float>(own, sin_h.data(), sin_h.size());
}

static void gpu_free_state(GpuWorker *w) {
  for (void *p : w->state_allocs)
    HIP_CHECK(hipFree(p));
  w->state_allocs.clear();
}

static GpuWorker *gpu_create_worker(int device, const Transformer *t, int batch,
                                    int max_pos) {
  HIP_CHECK(hipSetDevice(device));
  GpuWorker *w = new GpuWorker();
  w->device = device;
  w->cfg = t->config;
  HIP_CHECK(hipStreamCreate(&w->stream));
  gpu_upload_weights(w, t);
  size_t free_b = 0, total_b = 0;
  HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  const size_t reserve = (size_t)1 << 30;
  while (batch > 1 && gpu_state_bytes(w->cfg, batch, max_pos) + reserve > free_b)
    batch /= 2;
  gpu_alloc_state(w, batch, max_pos);
  return w;
}

// One decode step for rows 0..B-1 at position `pos`: tokens in w->tokens,
// greedy next tokens to w->next. Logits stay in w->logits.
static void gpu_step(GpuWorker *w, int B, int pos) {
  const Config &c = w->cfg;
  const int H = c.hidden_dim, E = c.n_experts, K = c.experts_per_token;
  const int I = c.intermediate_dim, D = c.head_dim, V = c.vocab_size;
  const int ATT = D * c.n_attn_heads, KVD = D * c.n_kv_heads;
  const int QKV = ATT + 2 * KVD;
  const int half = D / 2;
  const int R = B * K;
  hipStream_t s = w->stream;

  embed_kernel<<<B, 256, 0, s>>>(w->emb, w->tokens, w->x, H);
  for (int l = 0; l < c.n_layers; l++) {
    GpuLayer &g = w->layers[l];
    rmsnorm_exact<<<B, 256, H * sizeof(float), s>>>(w->x, g.rms_attn, w->t, H);
    gemm_bf16_exact(s, w->t, B, g.w_qkv, g.b_qkv, w->qkv, QKV, H);
    rope_exact<<<B, 256, 0, s>>>(w->qkv, QKV, w->cos_tab + (size_t)pos * half,
                                 w->sin_tab + (size_t)pos * half,
                                 c.n_attn_heads + c.n_kv_heads, D);
    const int slot = g.window > 0 ? pos % g.window : pos;
    kv_write_kernel<<<B, 256, 0, s>>>(w->qkv, QKV, ATT, ATT + KVD, g.kc, g.vc,
                                      g.slots, slot, KVD);
    const int n_pos = g.window > 0 ? std::min(pos + 1, g.window) : pos + 1;
    attention_exact<<<dim3(c.n_attn_heads, B), 64, (n_pos + 1) * sizeof(float), s>>>(
        w->qkv, QKV, g.kc, g.vc, g.slots, KVD, c.n_attn_heads / c.n_kv_heads,
        g.sinks, w->att, ATT, pos, g.window);
    gemm_bf16_exact(s, w->att, B, g.w_o, g.b_o, w->proj, H, ATT);
    add_inplace_kernel<<<((size_t)B * H + 255) / 256, 256, 0, s>>>(w->x, w->proj, (size_t)B * H);

    rmsnorm_exact<<<B, 256, H * sizeof(float), s>>>(w->x, g.rms_ffn, w->t, H);
    gemm_bf16_exact(s, w->t, B, g.w_router, g.b_router, w->router, E, H);
    router_topk_exact<<<(B + 63) / 64, 64, 0, s>>>(w->router, E, K, w->topk_w, w->topk_e, B);
    moe_group_kernel<<<1, ((E + 63) / 64) * 64, 0, s>>>(w->topk_e, R, E, w->offsets,
                                                       w->row_pair, w->pair_row);
    moe_gather_kernel<<<R, 256, 0, s>>>(w->t, w->row_pair, K, w->Xg, H);
    gemm_grouped_mxfp4_exact(s, w->Xg, w->offsets, B, E, g.mlp1_blocks,
                             g.mlp1_scales, g.b_mlp1, w->Yg, 2 * I, H);
    swiglu_exact<<<((size_t)R * I + 255) / 256, 256, 0, s>>>(w->Yg, w->Gg, R, I, c.swiglu_limit);
    gemm_grouped_mxfp4_exact(s, w->Gg, w->offsets, B, E, g.mlp2_blocks,
                             g.mlp2_scales, g.b_mlp2, w->Zg, H, I);
    moe_combine_exact<<<B, 256, 0, s>>>(w->x, w->Zg, w->pair_row, w->topk_w,
                                        w->topk_e, K, H);
  }
  rmsnorm_exact<<<B, 256, H * sizeof(float), s>>>(w->x, w->rms_out, w->t, H);
  gemm_bf16_exact(s, w->t, B, w->unemb, nullptr, w->logits, V, H);
  argmax_kernel<<<B, 256, 0, s>>>(w->logits, V, w->next);
  HIP_CHECK(hipGetLastError());
}

// Serves requests first .. first+n-1 together, mirroring simple_getp_generate()
// in getp_run.cpp for each of them. Returns the number of generated tokens.
static long long gpu_run_wave(GpuWorker *w, Tokenizer *tok, Requests *req,
                              int first, int n, int steps) {
  const Config &c = w->cfg;
  std::vector<std::vector<int>> prompts(n);
  for (int b = 0; b < n; b++) {
    const char *text = get_str_req_ptr(req, first + b);
    std::vector<int> buf(strlen(text) + 3);
    int np = 0;
    encode(tok, text, -1, -1, buf.data(), &np, c.initial_context_length);
    if (np < 1) {
      fprintf(stderr, "request %d: expected at least 1 prompt token\n", first + b);
      exit(EXIT_FAILURE);
    }
    buf.resize(np);
    prompts[b] = std::move(buf);
  }

  std::vector<int> cur(n), next(n), done(n, 0);
  for (int b = 0; b < n; b++)
    cur[b] = prompts[b][0];
  long long generated = 0;
  int active = n, pos = 0;
  while (pos < steps && active > 0) {
    HIP_CHECK(hipMemcpyAsync(w->tokens, cur.data(), n * sizeof(int),
                             hipMemcpyHostToDevice, w->stream));
    gpu_step(w, n, pos);
    HIP_CHECK(hipMemcpyAsync(next.data(), w->next, n * sizeof(int),
                             hipMemcpyDeviceToHost, w->stream));
    HIP_CHECK(hipStreamSynchronize(w->stream));
    pos++;
    for (int b = 0; b < n; b++) {
      if (done[b])
        continue;
      const int np = (int)prompts[b].size();
      int *out = get_tok_gen_ptr(req, first + b);
      int token;
      if (pos < np) {
        token = prompts[b][pos];
      } else {
        token = next[b];
        out[pos - np] = token;
      }
      if (token == 199999 || token == 200002) {
        done[b] = 1;
        active--;
        if (pos - np + 1 >= 0)
          out[pos - np + 1] = -1;
        generated += pos - np + 1;
      }
      cur[b] = token;
    }
  }
  for (int b = 0; b < n; b++) {
    if (done[b])
      continue;
    const int np = (int)prompts[b].size();
    if (pos - np + 1 >= 0)
      get_tok_gen_ptr(req, first + b)[pos - np + 1] = -1;
    generated += pos - np + 1;
  }
  return generated;
}

static void gpu_warm_up(const Transformer *t) {
  int n_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&n_dev));
  const int batch = env_int("GPTOSS_BATCH", GPU_DEFAULT_BATCH);
  const int max_pos = std::min(env_int("GPTOSS_MAX_POS", GPU_DEFAULT_MAX_POS),
                               t->config.seq_len);
  g_workers.assign(n_dev, nullptr);
  std::vector<std::thread> threads;
  for (int d = 0; d < n_dev; d++)
    threads.emplace_back([&, d] { g_workers[d] = gpu_create_worker(d, t, batch, max_pos); });
  for (auto &th : threads)
    th.join();
  printf("getp: %d GPU(s), %d requests per GPU per wave, KV cache for %d positions\n",
         n_dev, g_workers[0]->batch, max_pos);
  fflush(stdout);
}

static long long gpu_inference(Tokenizer *tok, Requests *req) {
  const int n_dev = (int)g_workers.size();
  const int steps = req->max_seq_len;
  const int per_dev = (req->num_reqs + n_dev - 1) / n_dev;
  std::vector<long long> generated(n_dev, 0);
  std::vector<std::thread> threads;
  for (int d = 0; d < n_dev; d++) {
    threads.emplace_back([&, d] {
      GpuWorker *w = g_workers[d];
      HIP_CHECK(hipSetDevice(w->device));
      if (steps > w->max_pos) { // KV cache too short for -n: grow it
        const int batch = w->batch;
        gpu_free_state(w);
        gpu_alloc_state(w, batch, steps);
      }
      const int begin = std::min(d * per_dev, req->num_reqs);
      const int end = std::min(begin + per_dev, req->num_reqs);
      for (int first = begin; first < end; first += w->batch)
        generated[d] += gpu_run_wave(w, tok, req, first,
                                     std::min(w->batch, end - first), steps);
    });
  }
  for (auto &th : threads)
    th.join();
  long long total = 0;
  for (long long g : generated)
    total += g;
  return total;
}

static void gpu_finish() {
  for (GpuWorker *w : g_workers) {
    HIP_CHECK(hipSetDevice(w->device));
    gpu_free_state(w);
    for (void *p : w->weight_allocs)
      HIP_CHECK(hipFree(p));
    HIP_CHECK(hipStreamDestroy(w->stream));
    delete w;
  }
  g_workers.clear();
}
