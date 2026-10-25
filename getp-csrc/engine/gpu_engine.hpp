// getp on GPUs.
//
// The visible GCDs are split into groups. A group is one GCD holding the whole
// model (gpt-oss-20b), or two GCDs of one MI250 card that share the model:
// both hold the dense weights, each holds half of every layer's experts
// (gpt-oss-120b, which does not fit on one GCD; which half: gpu_expert_order).
// Every GCD serves its own requests; inside a group the two GCDs only meet in
// the expert step, where each runs its experts for both GCDs' tokens. Groups
// never talk to each other (data parallel). On a GCD, up to `batch` requests
// run together and advance one position per step (static batching).
//
// Weights: bf16 for the dense matrices, MXFP4 for the experts, fp32 for norms,
// biases and sinks, exactly as stored in the checkpoint.
//
// Two paths:
// - exact (GPTOSS_EXACT=1): fp32 activations and KV cache, *_exact kernels; a
//   step reproduces hf_cpu_forward() bit for bit (tests/gpu_vs_cpu.cpp).
// - fast (default): bf16 inputs to every matrix multiply and a bf16 KV cache,
//   fp32 residual stream and accumulation (kernels/fast_*.hpp).
#pragma once

#include <stdlib.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "../kernels/attention.hpp"
#include "../kernels/fast_attention.hpp"
#include "../kernels/fast_gemm.hpp"
#include "../kernels/fast_moe_gemm.hpp"
#include "../kernels/fast_misc.hpp"
#include "../kernels/gemm.hpp"
#include "../kernels/misc.hpp"
#include "../kernels/moe.hpp"
#include "../kernels/moe_ep.hpp"
#include "ep_split_120b.hpp"
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

// The two signals a GCD sends its partner in every layer (ep_signal).
#define EP_ROUTED 0
#define EP_EXPERTS 1
#define EP_FLAG_STRIDE 64 // ints between the two flags: separate cache lines

struct GpuLayer {
  const uint16_t *w_qkv, *w_o, *w_router;
  const float *b_qkv, *b_o, *b_router, *sinks, *rms_attn, *rms_ffn;
  // this GCD's experts only: e_count of them, starting at e_begin; on the
  // fast path packed by moe_pack_mxfp4 when moe_mfma_ok
  const uint8_t *mlp1_blocks, *mlp1_scales, *mlp2_blocks, *mlp2_scales;
  const float *b_mlp1, *b_mlp2;
  // Experts split by a table (gpu_expert_order): the router writes expert
  // slots, not checkpoint numbers, to topk_e. slot_of: checkpoint expert ->
  // slot, phys_of: slot -> checkpoint expert. Null when slots and checkpoint
  // numbers are the same.
  const int *slot_of = nullptr, *phys_of = nullptr;
  float *kc, *vc;      // exact path KV cache (batch, slots, kv_dim)
  bf16_t *kc16;        // fast path K cache, chunk64 layout (fast_attention.hpp)
  int8_t *vq8;         // fast path V cache: int8 codes (vq_index_chunk64)
  float *vs;           //   and one scale per slot and head (vs_index_chunk64)
  int slots;           // positions kept: max_pos or the sliding window (fast path:
                       // rounded up to a multiple of 64)
  int window;          // 0 for full attention
};

struct GpuWorker {
  int device = 0;
  bool exact = false;
  hipStream_t stream;
  Config cfg;
  int batch = 0;   // requests per wave
  int max_pos = 0; // positions the KV cache holds
  int e_begin = 0, e_count = 0; // expert slots held: e_begin .. e_begin+e_count-1
  GpuWorker *peer = nullptr;    // the GCD holding the other experts, if split
  // Signals to the partner (ep_signal_kernel): flags[EP_ROUTED] reaches
  // sent[EP_ROUTED] when topk_e, topk_w and t of the layer may be read (and
  // the partner's t16 rows batch .. are written), flags[EP_EXPERTS] when Zg,
  // S and peer_row may. Fine-grained memory.
  int *flags = nullptr;
  int sent[2] = {0, 0}; // signals of each kind queued so far
  std::vector<GpuLayer> layers;
  const uint16_t *emb = nullptr, *unemb = nullptr;
  const float *rms_out = nullptr;
  float *cos_tab = nullptr, *sin_tab = nullptr; // (max_pos, head_dim / 2)
  int *tokens = nullptr, *next = nullptr;
  int *row_done = nullptr; // fast path: 1 for requests that already ended (no expert work)
  float *x, *t, *qkv, *att, *proj, *router, *topk_w, *Xg, *Yg, *Gg, *Zg, *logits;
  int *topk_e, *counts, *offsets, *row_pair, *own_row, *peer_row;
  bf16_t *t16, *att16, *Xg16, *Gg16; // fast path matrix-multiply inputs
  float *S = nullptr; // fast path, split experts: partial sums for the partner's tokens
  unsigned long long *amax_keys = nullptr; // fast path: one argmax key per row (logits unused)
  int *load_log = nullptr; // GPTOSS_EXPERT_LOAD: counts of every layer of the step
  std::vector<void *> weight_allocs, state_allocs;
};

// One GCD, or two GCDs that split the experts.
struct GpuGroup {
  std::vector<GpuWorker *> m;
  FILE *load_file = nullptr; // GPTOSS_EXPERT_LOAD
};

static std::vector<GpuWorker *> g_workers;
static std::vector<GpuGroup> g_groups;

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

// Which experts each GCD holds when two GCDs split them. Experts are handled
// by slot: the GCD of rank r holds slots r * E/2 .. (r+1) * E/2 - 1 of every
// layer, and the slot j of layer l is the checkpoint's expert
// gpu_expert_order(c, l)[j]. The table (ep_split_120b.hpp, made by
// tools/ep_split.py) gives each GCD of a pair a set of experts with about
// the same expected work, instead of the first and second half (the router
// picks some experts far more often than others). Null (slot = checkpoint
// number) for a model without a table or with GPTOSS_EP_SPLIT=0.
static const uint8_t *gpu_expert_order(const Config &c, int l) {
  static const int on = env_int("GPTOSS_EP_SPLIT", 1);
  if (!on || c.n_layers != EP_SPLIT_LAYERS || c.n_experts != EP_SPLIT_EXPERTS ||
      c.hidden_dim != EP_SPLIT_HIDDEN)
    return nullptr;
  return ep_split_120b[l];
}

// Rows of the grouped expert buffers: this GCD's pairs, plus its partner's.
static size_t gpu_expert_rows(const GpuWorker *w, int batch) {
  return (size_t)batch * w->cfg.experts_per_token * (w->e_count < w->cfg.n_experts ? 2 : 1);
}

// Weights come from the safetensors mapping (bf16, MXFP4) and from the fp32
// copies hf_build_transformer() made of the small tensors.
static void gpu_upload_weights(GpuWorker *w, const Transformer *t) {
  const Config &c = t->config;
  const TransformerWeights &tw = t->weights;
  const size_t H = c.hidden_dim, E = c.n_experts, I = c.intermediate_dim;
  const size_t QKV = (size_t)c.head_dim * (c.n_attn_heads + 2 * c.n_kv_heads);
  const size_t e0 = w->e_begin, ne = w->e_count;
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
  auto f32 = [&](const float *src, size_t n) {
    return (const float *)gpu_upload<float>(own, src, n);
  };
  // The checkpoint experts of this GCD's slots, in slot order, for layer l.
  std::vector<int> phys(ne);
  auto set_phys = [&](int l) {
    const uint8_t *order = ne < E ? gpu_expert_order(c, l) : nullptr;
    for (size_t j = 0; j < ne; j++)
      phys[j] = order ? order[e0 + j] : (int)(e0 + j);
  };
  // Copies `per` bytes of each of this GCD's experts from src (all experts,
  // `per` bytes each) to dst, one copy per run of consecutive experts.
  auto copy_experts = [&](uint8_t *dst, const uint8_t *src, size_t per) {
    for (size_t j = 0; j < ne;) {
      size_t n = 1;
      while (j + n < ne && phys[j + n] == phys[j] + (int)n)
        n++;
      HIP_CHECK(hipMemcpy(dst + j * per, src + (size_t)phys[j] * per, n * per,
                          hipMemcpyHostToDevice));
      j += n;
    }
  };
  auto expert_bytes = [&](const std::string &name, size_t per) {
    uint8_t *p = gpu_alloc<uint8_t>(own, ne * per);
    copy_experts(p, raw(name).data, per);
    return (const uint8_t *)p;
  };
  auto expert_f32 = [&](const float *src, size_t per) {
    std::vector<float> h(ne * per);
    for (size_t j = 0; j < ne; j++)
      memcpy(&h[j * per], src + (size_t)phys[j] * per, per * sizeof(float));
    return f32(h.data(), h.size());
  };
  // Expert weights: the fast path's matrix-core kernel reads them repacked
  // (moe_pack_mxfp4); the checkpoint bytes pass through a staging buffer.
  uint8_t *stage = nullptr;
  size_t stage_bytes = 0;
  auto experts = [&](const std::string &name, int N, int K, const uint8_t *&blocks,
                     const uint8_t *&scales) {
    const size_t gr = (size_t)N * (K / 32); // scale groups per expert
    if (w->exact || !moe_mfma_ok((int)ne, N, K)) {
      blocks = expert_bytes(name + ".blocks", gr * 16);
      scales = expert_bytes(name + ".scales", gr);
      return;
    }
    if (ne * gr * 17 > stage_bytes) {
      if (stage)
        HIP_CHECK(hipFree(stage));
      stage_bytes = ne * gr * 17;
      HIP_CHECK(hipMalloc(&stage, stage_bytes));
    }
    copy_experts(stage, raw(name + ".blocks").data, gr * 16);
    copy_experts(stage + ne * gr * 16, raw(name + ".scales").data, gr);
    uint8_t *pb = gpu_alloc<uint8_t>(own, ne * moe_packed_blocks_bytes(N, K));
    uint8_t *ps = gpu_alloc<uint8_t>(own, ne * moe_packed_scales_bytes(N, K));
    moe_pack_mxfp4(0, stage, stage + ne * gr * 16, (int)ne, N, K, pb, ps);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    blocks = pb;
    scales = ps;
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
    set_phys(l);
    experts(b + "mlp.mlp1_weight", (int)(2 * I), (int)H, g.mlp1_blocks, g.mlp1_scales);
    g.b_mlp1 = expert_f32(tw.b_mlp1 + l * E * 2 * I, 2 * I);
    experts(b + "mlp.mlp2_weight", (int)H, (int)I, g.mlp2_blocks, g.mlp2_scales);
    g.b_mlp2 = expert_f32(tw.b_mlp2 + l * E * H, H);
    g.slot_of = g.phys_of = nullptr;
    if (const uint8_t *order = ne < E ? gpu_expert_order(c, l) : nullptr) {
      std::vector<int> slot_of(E), phys_of(E);
      for (size_t j = 0; j < E; j++) {
        phys_of[j] = order[j];
        slot_of[order[j]] = (int)j;
      }
      g.slot_of = (const int *)gpu_upload<int>(own, slot_of.data(), E);
      g.phys_of = (const int *)gpu_upload<int>(own, phys_of.data(), E);
    }
  }
  if (stage)
    HIP_CHECK(hipFree(stage));
}

static size_t gpu_state_bytes(const GpuWorker *w, int batch, int max_pos) {
  const Config &c = w->cfg;
  const size_t B = batch, H = c.hidden_dim, K = c.experts_per_token;
  const size_t R = gpu_expert_rows(w, batch);
  const size_t QKV = (size_t)c.head_dim * (c.n_attn_heads + 2 * c.n_kv_heads);
  const size_t ATT = (size_t)c.head_dim * c.n_attn_heads;
  const size_t KVD = (size_t)c.head_dim * c.n_kv_heads;
  size_t kv_positions = 0;
  for (int l = 0; l < c.n_layers; l++) {
    int slots = (c.sliding_window > 0 && l % 2 == 0) ? std::min(c.sliding_window, max_pos)
                                                      : max_pos;
    kv_positions += w->exact ? slots : (slots + 63) / 64 * 64;
  }
  const size_t act = w->exact ? 4 : 2; // bytes per KV / matrix-input element
  const size_t kv_bytes = w->exact ? 2 * 4 * KVD : 2 * KVD + KVD + 4 * c.n_kv_heads; // per slot
  size_t bytes = B * kv_positions * kv_bytes               // KV cache
                 + B * (H + QKV + H + c.n_experts + K) * 4 // x qkv proj router topk_w
                 + B * (H + ATT) * act                     // t att
                 + R * (2 * c.intermediate_dim + H) * 4    // Yg Zg
                 + R * (H + c.intermediate_dim) * act      // Xg Gg
                 + (w->exact ? B * (size_t)c.vocab_size * 4 : 0) // logits
                 + 2 * (size_t)max_pos * c.head_dim / 2 * 4; // cos, sin
  if (!w->exact && w->e_count < c.n_experts)
    bytes += B * H * (4 + 2); // S, the partner's t16 rows
  return bytes + (B * K * 3 + R + w->e_count) * sizeof(int);
}

static void gpu_alloc_state(GpuWorker *w, int batch, int max_pos) {
  const Config &c = w->cfg;
  const size_t B = batch, H = c.hidden_dim, K = c.experts_per_token;
  const size_t R = gpu_expert_rows(w, batch);
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
    if (!w->exact)
      g.slots = (g.slots + 63) / 64 * 64;
    g.kc = g.vc = nullptr;
    g.kc16 = nullptr;
    g.vq8 = nullptr;
    g.vs = nullptr;
    if (w->exact) {
      g.kc = gpu_alloc<float>(own, B * g.slots * KVD);
      g.vc = gpu_alloc<float>(own, B * g.slots * KVD);
    } else {
      g.kc16 = gpu_alloc<bf16_t>(own, B * g.slots * KVD);
      g.vq8 = gpu_alloc<int8_t>(own, B * g.slots * KVD);
      g.vs = gpu_alloc<float>(own, B * g.slots * c.n_kv_heads);
    }
  }
  w->tokens = gpu_alloc<int>(own, B);
  w->next = gpu_alloc<int>(own, B);
  w->row_done = gpu_alloc<int>(own, B);
  w->x = gpu_alloc<float>(own, B * H);
  w->qkv = gpu_alloc<float>(own, B * QKV);
  w->proj = gpu_alloc<float>(own, B * H);
  w->router = gpu_alloc<float>(own, B * c.n_experts);
  w->topk_w = gpu_alloc<float>(own, B * K);
  w->topk_e = gpu_alloc<int>(own, B * K);
  w->counts = gpu_alloc<int>(own, w->e_count);
  w->offsets = gpu_alloc<int>(own, w->e_count + 1);
  w->row_pair = gpu_alloc<int>(own, R);
  w->own_row = gpu_alloc<int>(own, B * K);
  w->peer_row = gpu_alloc<int>(own, B * K);
  w->Yg = gpu_alloc<float>(own, R * 2 * c.intermediate_dim);
  w->Zg = gpu_alloc<float>(own, R * H);
  w->t = w->att = w->Xg = w->Gg = nullptr;
  w->t16 = w->att16 = w->Xg16 = w->Gg16 = nullptr;
  if (w->exact) {
    w->t = gpu_alloc<float>(own, B * H);
    w->att = gpu_alloc<float>(own, B * ATT);
    w->Xg = gpu_alloc<float>(own, R * H);
    w->Gg = gpu_alloc<float>(own, R * c.intermediate_dim);
  } else {
    // The gathering mlp1 reads its last 128-k step whole, so a row runs up
    // to 64 values into the next one, and the last row past the end of t16.
    // Those values meet zero weights; zeroing the buffer keeps a NaN from
    // appearing there. With a partner, rows batch .. hold the partner's rows:
    // its add_rmsnorm_bf16 writes them here, so that the expert multiply
    // gathers all its input rows from one table, in this GCD's memory.
    const size_t n16 = (w->e_count < c.n_experts ? 2 * B : B) * H + 64;
    w->t16 = gpu_alloc<bf16_t>(own, n16);
    HIP_CHECK(hipMemset(w->t16, 0, n16 * sizeof(bf16_t)));
    w->att16 = gpu_alloc<bf16_t>(own, B * ATT);
    w->Xg16 = gpu_alloc<bf16_t>(own, R * H);
    w->Gg16 = gpu_alloc<bf16_t>(own, R * c.intermediate_dim);
  }
  w->logits = nullptr;
  w->amax_keys = nullptr;
  if (w->exact) {
    w->logits = gpu_alloc<float>(own, B * c.vocab_size);
  } else {
    w->amax_keys = gpu_alloc<unsigned long long>(own, B);
    HIP_CHECK(hipMemset(w->amax_keys, 0, B * sizeof(unsigned long long)));
  }
  if (!w->exact && w->e_count < c.n_experts)
    w->S = gpu_alloc<float>(own, B * H);
  w->load_log = nullptr;
  if (getenv("GPTOSS_EXPERT_LOAD")) {
    w->load_log = gpu_alloc<int>(own, (size_t)c.n_layers * w->e_count);
    HIP_CHECK(hipMemset(w->load_log, 0, (size_t)c.n_layers * w->e_count * sizeof(int)));
  }

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

// Uploads the weights of experts e_begin .. e_begin+e_count-1; no run state yet.
static GpuWorker *gpu_new_worker(int device, const Transformer *t, bool exact,
                                 int e_begin, int e_count) {
  HIP_CHECK(hipSetDevice(device));
  GpuWorker *w = new GpuWorker();
  w->device = device;
  w->exact = exact;
  w->cfg = t->config;
  w->e_begin = e_begin;
  w->e_count = e_count;
  HIP_CHECK(hipStreamCreate(&w->stream));
  HIP_CHECK(hipExtMallocWithFlags((void **)&w->flags, 2 * EP_FLAG_STRIDE * sizeof(int),
                                  hipDeviceMallocFinegrained));
  HIP_CHECK(hipMemset(w->flags, 0, 2 * EP_FLAG_STRIDE * sizeof(int)));
  w->weight_allocs.push_back(w->flags);
  gpu_upload_weights(w, t);
  return w;
}

// Largest batch (halving from `batch`) whose run state fits in free memory.
static int gpu_fit_batch(const GpuWorker *w, int batch, int max_pos) {
  HIP_CHECK(hipSetDevice(w->device));
  size_t free_b = 0, total_b = 0;
  HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  const size_t reserve = (size_t)1 << 30;
  while (batch > 1 && gpu_state_bytes(w, batch, max_pos) + reserve > free_b)
    batch /= 2;
  return batch;
}

// A whole-model worker with its run state, for tests.
static GpuWorker *gpu_create_worker(int device, const Transformer *t, bool exact,
                                    int batch, int max_pos) {
  GpuWorker *w = gpu_new_worker(device, t, exact, 0, t->config.n_experts);
  gpu_alloc_state(w, gpu_fit_batch(w, batch, max_pos), max_pos);
  return w;
}

static void gpu_pair(GpuWorker *a, GpuWorker *b) {
  if (!a->exact && a->batch != b->batch) { // t16 holds batch rows of the partner
    fprintf(stderr, "gpu_pair: the fast path needs the same batch on both GCDs\n");
    exit(EXIT_FAILURE);
  }
  a->peer = b;
  b->peer = a;
  for (auto [x, y] : {std::make_pair(a, b), std::make_pair(b, a)}) {
    HIP_CHECK(hipSetDevice(x->device));
    hipError_t err = hipDeviceEnablePeerAccess(y->device, 0);
    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
      HIP_CHECK(err);
    (void)hipGetLastError();
  }
}

// ---- one decode step, in four parts per layer --------------------------------

static void step_embed(GpuWorker *w, int B) {
  if (B > 0)
    embed_kernel<<<B, 256, 0, w->stream>>>(w->emb, w->tokens, w->x, w->cfg.hidden_dim);
}

// x += attention(x) for this GCD's B rows.
static void layer_attention(GpuWorker *w, int l, int B, int pos) {
  if (B == 0)
    return;
  const Config &c = w->cfg;
  const int H = c.hidden_dim, D = c.head_dim;
  const int ATT = D * c.n_attn_heads, KVD = D * c.n_kv_heads, QKV = ATT + 2 * KVD;
  const int half = D / 2;
  GpuLayer &g = w->layers[l];
  hipStream_t s = w->stream;
  const float *cosv = w->cos_tab + (size_t)pos * half, *sinv = w->sin_tab + (size_t)pos * half;
  const int kv_slot = g.window > 0 ? pos % g.window : pos;
  if (w->exact) {
    rmsnorm_exact<<<B, 256, H * sizeof(float), s>>>(w->x, g.rms_attn, w->t, H);
    gemm_bf16_exact(s, w->t, B, g.w_qkv, g.b_qkv, w->qkv, QKV, H);
    rope_exact<<<B, 256, 0, s>>>(w->qkv, QKV, cosv, sinv, c.n_attn_heads + c.n_kv_heads, D);
    kv_write_kernel<<<B, 256, 0, s>>>(w->qkv, QKV, ATT, ATT + KVD, g.kc, g.vc, g.slots,
                                      kv_slot, KVD);
    const int n_pos = g.window > 0 ? std::min(pos + 1, g.window) : pos + 1;
    attention_exact<<<dim3(c.n_attn_heads, B), 64, (n_pos + 1) * sizeof(float), s>>>(
        w->qkv, QKV, g.kc, g.vc, g.slots, KVD, c.n_attn_heads / c.n_kv_heads, g.sinks,
        w->att, ATT, pos, g.window);
    gemm_bf16_exact(s, w->att, B, g.w_o, g.b_o, w->proj, H, ATT);
  } else {
    // later layers get t16 = RMSNorm(x) from the previous layer's combine
    if (l == 0)
      rmsnorm_bf16_kernel<<<B, 256, 0, s>>>(w->x, g.rms_attn, w->t16, H);
    gemm_bf16(s, w->t16, B, g.w_qkv, g.b_qkv, w->qkv, QKV, H);
    // RoPE: the keys here, the queries inside the attention kernel
    rope_kv_bf16(s, B, w->qkv, QKV, cosv, sinv, c.n_attn_heads, c.n_kv_heads, g.kc16, g.vq8,
                 g.vs, g.slots, kv_slot, /*rotate_q=*/false);
    attention_q8v_chunk64(s, w->qkv, QKV, g.kc16, g.vq8, g.vs, g.slots, KVD, g.sinks, w->att16,
                          ATT, B, pos, g.window, cosv, sinv);
    gemm_bf16(s, w->att16, B, g.w_o, g.b_o, w->proj, H, ATT);
    // residual add and the MoE input norm in one pass; with a partner, the
    // norm also goes to rows batch .. of the partner's t16
    GpuWorker *p = w->peer;
    add_rmsnorm_bf16(s, B, w->x, w->proj, g.rms_ffn, w->t16, H,
                     p ? p->t16 + (size_t)p->batch * H : nullptr);
    return;
  }
  add_inplace_kernel<<<((size_t)B * H + 255) / 256, 256, 0, s>>>(w->x, w->proj, (size_t)B * H);
}

// t = RMSNorm(x); each row's top-k experts and their weights.
static void layer_route(GpuWorker *w, int l, int B) {
  if (B == 0)
    return;
  const Config &c = w->cfg;
  const int H = c.hidden_dim, E = c.n_experts, K = c.experts_per_token;
  GpuLayer &g = w->layers[l];
  hipStream_t s = w->stream;
  if (w->exact) {
    rmsnorm_exact<<<B, 256, H * sizeof(float), s>>>(w->x, g.rms_ffn, w->t, H);
    gemm_bf16_exact(s, w->t, B, g.w_router, g.b_router, w->router, E, H);
  } else { // t16 = RMSNorm(x) from layer_attention
    gemm_bf16(s, w->t16, B, g.w_router, g.b_router, w->router, E, H);
  }
  if (E <= 256)
    router_topk_wave_kernel<<<(B + TOPK_WAVES - 1) / TOPK_WAVES, 64 * TOPK_WAVES, 0, s>>>(
        w->router, E, K, w->topk_w, w->topk_e, B, w->exact ? nullptr : w->row_done,
        g.slot_of);
  else // gpu_expert_order() has no table for more than 256 experts
    router_topk_exact<<<(B + 63) / 64, 64, 0, s>>>(w->router, E, K, w->topk_w, w->topk_e, B);
}

// This GCD's experts on every row (own and partner's) that chose them -> Zg.
static void layer_experts(GpuWorker *w, int l, int B, int B_peer) {
  const Config &c = w->cfg;
  const int H = c.hidden_dim, I = c.intermediate_dim, K = c.experts_per_token;
  const int R = (B + B_peer) * K;
  if (R == 0)
    return;
  GpuLayer &g = w->layers[l];
  hipStream_t s = w->stream;
  GpuWorker *p = w->peer;
  moe_group_ep(s, w->topk_e, B * K, p ? p->topk_e : nullptr, B_peer * K, w->batch * K,
               w->e_begin, w->e_count, w->counts, w->offsets, w->row_pair, w->own_row,
               w->peer_row);
  if (w->load_log)
    HIP_CHECK(hipMemcpyAsync(w->load_log + (size_t)l * w->e_count, w->counts,
                             w->e_count * sizeof(int), hipMemcpyDeviceToDevice, s));
  if (w->exact) {
    moe_gather_ep_kernel<float><<<R, 256, 0, s>>>(w->t, p ? p->t : nullptr, w->row_pair,
                                                  w->offsets, w->e_count, K, w->batch, w->Xg,
                                                  H);
    gemm_grouped_mxfp4_exact(s, w->Xg, w->offsets, B + B_peer, w->e_count, g.mlp1_blocks,
                             g.mlp1_scales, g.b_mlp1, w->Yg, 2 * I, H);
    swiglu_exact<<<((size_t)R * I + 255) / 256, 256, 0, s>>>(w->Yg, w->Gg, R, I, c.swiglu_limit);
    gemm_grouped_mxfp4_exact(s, w->Gg, w->offsets, B + B_peer, w->e_count, g.mlp2_blocks,
                             g.mlp2_scales, g.b_mlp2, w->Zg, H, I);
  } else {
    // Gather, mlp1 and SwiGLU in one kernel when it applies (same bits as
    // the three separate steps). Every input row is in t16: own tokens in
    // rows 0 .., the partner's (written there by the partner) in rows batch ..
    const bool fused = moe_mfma_ok(w->e_count, 2 * I, H);
    if (fused) {
      gemm_grouped_mxfp4_gather_swiglu(s, w->t16, w->row_pair, K, w->offsets, B + B_peer,
                                       w->e_count, g.mlp1_blocks, g.mlp1_scales,
                                       g.b_mlp1, w->Gg16, c.swiglu_limit, 2 * I, H);
    } else {
      moe_gather_ep_bf16(s, R, w->t16, w->t16 + (size_t)w->batch * H, w->row_pair,
                         w->offsets, w->e_count, K, w->batch, w->Xg16, H);
      gemm_grouped_mxfp4(s, w->Xg16, w->offsets, B + B_peer, w->e_count, g.mlp1_blocks,
                         g.mlp1_scales, g.b_mlp1, w->Yg, 2 * I, H);
      swiglu_bf16_kernel<<<((size_t)R * I + 255) / 256, 256, 0, s>>>(w->Yg, w->Gg16, R, I,
                                                                     c.swiglu_limit);
    }
    gemm_grouped_mxfp4(s, w->Gg16, w->offsets, B + B_peer, w->e_count, g.mlp2_blocks,
                       g.mlp2_scales, g.b_mlp2, w->Zg, H, I);
    if (p && B_peer > 0)
      moe_partial_ep(s, B_peer, w->S, w->Zg, w->peer_row, p->topk_w, p->topk_e, w->e_begin,
                     w->e_count, K, H);
  }
}

// x += weighted sum of each own row's expert outputs (some read from the partner).
static void layer_combine(GpuWorker *w, int l, int B) {
  if (B == 0)
    return;
  const Config &c = w->cfg;
  GpuWorker *p = w->peer;
  const float *Zp = p ? p->Zg : nullptr;
  const int *rows_p = p ? p->peer_row : nullptr;
  if (w->exact)
    moe_combine_ep_kernel<true><<<B, 256, 0, w->stream>>>(
        w->x, w->Zg, w->own_row, Zp, rows_p, w->topk_w, w->topk_e, w->e_begin, w->e_count,
        c.experts_per_token, c.hidden_dim, w->layers[l].phys_of);
  else {
    // the combine also writes t16 = RMSNorm(x) for the next layer (or the LM head)
    const float *norm_w = l + 1 < c.n_layers ? w->layers[l + 1].rms_attn : w->rms_out;
    if (p)
      moe_combine_partial_rmsnorm(w->stream, B, w->x, w->Zg, w->own_row, p->S, w->topk_w,
                                  w->topk_e, w->e_begin, w->e_count, c.experts_per_token,
                                  c.hidden_dim, norm_w, w->t16);
    else
      moe_combine_rmsnorm(w->stream, B, w->x, w->Zg, w->own_row, w->topk_w, w->topk_e,
                          c.experts_per_token, c.hidden_dim, norm_w, w->t16);
  }
}

// Final RMSNorm, LM head, greedy next token.
static void step_logits(GpuWorker *w, int B) {
  if (B == 0)
    return;
  const Config &c = w->cfg;
  const int H = c.hidden_dim, V = c.vocab_size;
  hipStream_t s = w->stream;
  if (w->exact) {
    rmsnorm_exact<<<B, 256, H * sizeof(float), s>>>(w->x, w->rms_out, w->t, H);
    gemm_bf16_exact(s, w->t, B, w->unemb, nullptr, w->logits, V, H);
    argmax_kernel<<<B, 256, 0, s>>>(w->logits, V, w->next);
  } else { // t16 = final RMSNorm(x) from the last layer's combine
    gemm_bf16_argmax(s, w->t16, B, w->unemb, nullptr, V, H, w->amax_keys, w->next);
  }
}

// This GCD's data of kind `kind` may now be read by its partner.
static void ep_signal(GpuWorker *w, int kind) {
  ep_signal_kernel<<<1, 1, 0, w->stream>>>(w->flags + kind * EP_FLAG_STRIDE, ++w->sent[kind]);
}

// The rest of this GCD's stream waits for the partner's last signal of `kind`
// queued so far.
static void ep_wait(GpuWorker *w, int kind) {
  const GpuWorker *p = w->peer;
  ep_wait_kernel<<<1, 1, 0, w->stream>>>(p->flags + kind * EP_FLAG_STRIDE, p->sent[kind]);
}

// One decode step for every GCD of a group at position `pos`: member i has
// B[i] rows, tokens in its `tokens`, greedy next tokens to its `next`.
// The partner reads a GCD's topk_e, topk_w and t after its EP_ROUTED signal,
// and its Zg, S and peer_row after EP_EXPERTS. A GCD overwrites those only in
// the next layer, after it has waited for its partner's next signal, which
// the partner sends after finishing these reads, so one copy of each buffer
// is enough. The same holds for the rows a GCD writes into the partner's t16
// (rows batch ..) on the fast path: written before EP_ROUTED, read by the
// partner's expert multiply, written again only after the partner's next
// EP_EXPERTS. The host queues every signal before the wait that refers to it.
static void gpu_step_group(GpuGroup &g, const int *B, int pos) {
  const int n = (int)g.m.size();
  for (int i = 0; i < n; i++) {
    HIP_CHECK(hipSetDevice(g.m[i]->device));
    step_embed(g.m[i], B[i]);
  }
  const int n_layers = g.m[0]->cfg.n_layers;
  for (int l = 0; l < n_layers; l++) {
    for (int i = 0; i < n; i++) {
      GpuWorker *w = g.m[i];
      HIP_CHECK(hipSetDevice(w->device));
      layer_attention(w, l, B[i], pos);
      layer_route(w, l, B[i]);
      if (n > 1)
        ep_signal(w, EP_ROUTED);
    }
    for (int i = 0; i < n; i++) {
      GpuWorker *w = g.m[i];
      HIP_CHECK(hipSetDevice(w->device));
      if (n > 1)
        ep_wait(w, EP_ROUTED);
      layer_experts(w, l, B[i], n > 1 ? B[1 - i] : 0);
      if (n > 1)
        ep_signal(w, EP_EXPERTS);
    }
    for (int i = 0; i < n; i++) {
      GpuWorker *w = g.m[i];
      HIP_CHECK(hipSetDevice(w->device));
      if (n > 1)
        ep_wait(w, EP_EXPERTS);
      layer_combine(w, l, B[i]);
    }
  }
  for (int i = 0; i < n; i++) {
    HIP_CHECK(hipSetDevice(g.m[i]->device));
    step_logits(g.m[i], B[i]);
  }
  HIP_CHECK(hipGetLastError());
}

// One decode step for a single whole-model worker (tests).
static void gpu_step(GpuWorker *w, int B, int pos) {
  GpuGroup g;
  g.m.push_back(w);
  gpu_step_group(g, &B, pos);
}

// ---- serving requests --------------------------------------------------------

// The requests one GCD serves in a wave, following simple_getp_generate() in
// getp_run.cpp for each of them.
struct WaveSlot {
  int first = 0, n = 0;
  std::vector<std::vector<int>> prompts;
  std::vector<int> cur, next, done;
  bool done_changed = false; // done must be copied to the GPU before the next step
};

static void wave_slot_init(WaveSlot &ws, Tokenizer *tok, Requests *req,
                           const Config &c, int first, int n) {
  ws.first = first;
  ws.n = n;
  ws.prompts.assign(n, {});
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
    ws.prompts[b] = std::move(buf);
  }
  ws.cur.resize(n);
  ws.next.resize(n);
  ws.done.assign(n, 0);
  for (int b = 0; b < n; b++)
    ws.cur[b] = ws.prompts[b][0];
}

// GPTOSS_EXPERT_LOAD=<file>: after every step, appends to <file>.<first GPU
// of the group> how many rows each expert got in each layer from the tokens
// of the whole group: int16, n_layers x n_experts, by checkpoint expert
// number. tools/ep_split.py reads these files.
static void gpu_write_load(GpuGroup &g, const char *path) {
  const Config &c = g.m[0]->cfg;
  if (!g.load_file) {
    const std::string name = std::string(path) + "." + std::to_string(g.m[0]->device);
    g.load_file = fopen(name.c_str(), "wb");
    if (!g.load_file) {
      fprintf(stderr, "GPTOSS_EXPERT_LOAD: cannot write %s\n", name.c_str());
      exit(EXIT_FAILURE);
    }
  }
  std::vector<int16_t> rec((size_t)c.n_layers * c.n_experts, 0);
  std::vector<int> h;
  for (GpuWorker *w : g.m) {
    h.resize((size_t)c.n_layers * w->e_count);
    HIP_CHECK(hipSetDevice(w->device));
    HIP_CHECK(hipMemcpy(h.data(), w->load_log, h.size() * sizeof(int), hipMemcpyDeviceToHost));
    for (int l = 0; l < c.n_layers; l++) {
      const uint8_t *order = w->e_count < c.n_experts ? gpu_expert_order(c, l) : nullptr;
      for (int j = 0; j < w->e_count; j++) {
        const int slot = w->e_begin + j;
        rec[(size_t)l * c.n_experts + (order ? order[slot] : slot)] =
            (int16_t)h[(size_t)l * w->e_count + j];
      }
    }
  }
  fwrite(rec.data(), sizeof(int16_t), rec.size(), g.load_file);
}

// Serves slots[i] on group member i, all advancing together. Returns the
// number of generated tokens.
static long long gpu_run_wave(GpuGroup &g, std::vector<WaveSlot> &slots,
                              Requests *req, int steps) {
  const int n = (int)g.m.size();
  std::vector<int> B(n);
  int active = 0;
  for (int i = 0; i < n; i++) {
    B[i] = slots[i].n;
    active += slots[i].n;
  }
  for (int i = 0; i < n; i++)
    if (B[i] > 0) {
      HIP_CHECK(hipSetDevice(g.m[i]->device));
      HIP_CHECK(hipMemset(g.m[i]->row_done, 0, B[i] * sizeof(int)));
    }
  long long generated = 0;
  int pos = 0;
  while (pos < steps && active > 0) {
    for (int i = 0; i < n; i++)
      if (B[i] > 0) {
        HIP_CHECK(hipSetDevice(g.m[i]->device));
        HIP_CHECK(hipMemcpyAsync(g.m[i]->tokens, slots[i].cur.data(), B[i] * sizeof(int),
                                 hipMemcpyHostToDevice, g.m[i]->stream));
        if (slots[i].done_changed) {
          HIP_CHECK(hipMemcpyAsync(g.m[i]->row_done, slots[i].done.data(), B[i] * sizeof(int),
                                   hipMemcpyHostToDevice, g.m[i]->stream));
          slots[i].done_changed = false;
        }
      }
    gpu_step_group(g, B.data(), pos);
    for (int i = 0; i < n; i++)
      if (B[i] > 0) {
        HIP_CHECK(hipSetDevice(g.m[i]->device));
        HIP_CHECK(hipMemcpyAsync(slots[i].next.data(), g.m[i]->next, B[i] * sizeof(int),
                                 hipMemcpyDeviceToHost, g.m[i]->stream));
      }
    for (int i = 0; i < n; i++) {
      HIP_CHECK(hipSetDevice(g.m[i]->device));
      HIP_CHECK(hipStreamSynchronize(g.m[i]->stream));
    }
    if (g.m[0]->load_log)
      gpu_write_load(g, getenv("GPTOSS_EXPERT_LOAD"));
    pos++;
    for (int i = 0; i < n; i++) {
      WaveSlot &ws = slots[i];
      for (int b = 0; b < ws.n; b++) {
        if (ws.done[b])
          continue;
        const int np = (int)ws.prompts[b].size();
        int *out = get_tok_gen_ptr(req, ws.first + b);
        int token;
        if (pos < np) {
          token = ws.prompts[b][pos];
        } else {
          token = ws.next[b];
          out[pos - np] = token;
        }
        if (token == 199999 || token == 200002) {
          ws.done[b] = 1;
          ws.done_changed = true;
          active--;
          if (pos - np + 1 >= 0)
            out[pos - np + 1] = -1;
          generated += pos - np + 1;
        }
        ws.cur[b] = token;
      }
    }
  }
  for (int i = 0; i < n; i++) {
    WaveSlot &ws = slots[i];
    for (int b = 0; b < ws.n; b++) {
      if (ws.done[b])
        continue;
      const int np = (int)ws.prompts[b].size();
      if (pos - np + 1 >= 0)
        get_tok_gen_ptr(req, ws.first + b)[pos - np + 1] = -1;
      generated += pos - np + 1;
    }
  }
  return generated;
}

// GCDs per group: GPTOSS_EP, or 2 when the weights would fill more than 60 %
// of one GCD's memory.
static int gpu_group_size(int n_dev) {
  int ep = env_int("GPTOSS_EP", 0);
  if (ep <= 0) {
    size_t free_b = 0, total_b = 0;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    ep = g_hf.st.total_bytes > total_b * 6 / 10 ? 2 : 1;
  }
  if (ep != 1 && ep != 2) {
    fprintf(stderr, "getp: GPTOSS_EP must be 1 or 2\n");
    exit(EXIT_FAILURE);
  }
  if (n_dev % ep != 0) {
    fprintf(stderr, "getp: %d GPUs cannot be split into groups of %d\n", n_dev, ep);
    exit(EXIT_FAILURE);
  }
  return ep;
}

static void gpu_warm_up(const Transformer *t) {
  int n_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&n_dev));
  const bool exact = env_int("GPTOSS_EXACT", 0) != 0;
  const int max_pos = std::min(env_int("GPTOSS_MAX_POS", GPU_DEFAULT_MAX_POS),
                               t->config.seq_len);
  const int ep = gpu_group_size(n_dev);
  const int E = t->config.n_experts;
  if (E % ep != 0) {
    fprintf(stderr, "getp: %d experts cannot be split %d ways\n", E, ep);
    exit(EXIT_FAILURE);
  }
  g_workers.assign(n_dev, nullptr);
  std::vector<std::thread> threads;
  for (int d = 0; d < n_dev; d++)
    threads.emplace_back([&, d] {
      const int rank = d % ep;
      g_workers[d] = gpu_new_worker(d, t, exact, rank * (E / ep), E / ep);
    });
  for (auto &th : threads)
    th.join();
  threads.clear();

  // the same batch on every GCD, as large as the fullest one allows
  int batch = env_int("GPTOSS_BATCH", GPU_DEFAULT_BATCH);
  for (GpuWorker *w : g_workers)
    batch = std::min(batch, gpu_fit_batch(w, batch, max_pos));
  for (int d = 0; d < n_dev; d++)
    threads.emplace_back([&, d] {
      HIP_CHECK(hipSetDevice(d));
      gpu_alloc_state(g_workers[d], batch, max_pos);
    });
  for (auto &th : threads)
    th.join();

  g_groups.assign(n_dev / ep, GpuGroup());
  for (int d = 0; d < n_dev; d++)
    g_groups[d / ep].m.push_back(g_workers[d]);
  if (ep == 2)
    for (GpuGroup &g : g_groups)
      gpu_pair(g.m[0], g.m[1]);
  printf("getp: %s path, %d GPU(s) in %d group(s) of %d, %d requests per GPU per wave, "
         "KV cache for %d positions\n",
         exact ? "exact" : "fast", n_dev, n_dev / ep, ep, batch, max_pos);
  fflush(stdout);
}

static long long gpu_inference(Tokenizer *tok, Requests *req) {
  const int n_groups = (int)g_groups.size();
  const int steps = req->max_seq_len;
  const int per_group = (req->num_reqs + n_groups - 1) / n_groups;
  std::vector<long long> generated(n_groups, 0);
  std::vector<std::thread> threads;
  for (int gi = 0; gi < n_groups; gi++) {
    threads.emplace_back([&, gi] {
      GpuGroup &g = g_groups[gi];
      const int n = (int)g.m.size();
      for (GpuWorker *w : g.m)
        if (steps > w->max_pos) { // KV cache too short for -n: grow it
          HIP_CHECK(hipSetDevice(w->device));
          const int batch = w->batch;
          gpu_free_state(w);
          gpu_alloc_state(w, batch, steps);
        }
      const int begin = std::min(gi * per_group, req->num_reqs);
      const int end = std::min(begin + per_group, req->num_reqs);
      const int per_wave = g.m[0]->batch * n;
      for (int first = begin; first < end; first += per_wave) {
        const int count = std::min(per_wave, end - first);
        std::vector<WaveSlot> slots(n);
        int f = first;
        for (int i = 0; i < n; i++) { // split the wave as evenly as possible
          const int ni = count / n + (i < count % n ? 1 : 0);
          wave_slot_init(slots[i], tok, req, g.m[i]->cfg, f, ni);
          f += ni;
        }
        generated[gi] += gpu_run_wave(g, slots, req, steps);
      }
    });
  }
  for (auto &th : threads)
    th.join();
  long long total = 0;
  for (long long x : generated)
    total += x;
  return total;
}

static void gpu_finish() {
  for (GpuGroup &g : g_groups)
    if (g.load_file)
      fclose(g.load_file);
  for (GpuWorker *w : g_workers) {
    HIP_CHECK(hipSetDevice(w->device));
    gpu_free_state(w);
    for (void *p : w->weight_allocs)
      HIP_CHECK(hipFree(p));
    HIP_CHECK(hipStreamDestroy(w->stream));
    delete w;
  }
  g_workers.clear();
  g_groups.clear();
}
