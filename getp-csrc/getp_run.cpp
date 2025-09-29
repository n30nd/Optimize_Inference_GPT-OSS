// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (e.g., MI250), BF16 weights/KV; FP32 accumulate.
// - Warm-up loads weights with chunked copy+convert to avoid OOM.
// - Runtime fixes: bound pos to seq_len; guard attention sink write.
//
// FlashAttention-like optimization (always enabled):
// - Simplified kernel focused on stability and correctness
// - Reduces global memory traffic by keeping scores in shared memory

#include "getp_eval.cpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cfloat>
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
#include <string>
#include <utility>

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

// ------------------------- GPU weights & run state -------------------------

typedef struct {
  hip_bfloat16 *token_embedding_table;
  hip_bfloat16 *out;
  hip_bfloat16 *out_packed16;
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
  uint8_t *w_mlp1_data;
  uint8_t *w_mlp1_scales;
  uint8_t *w_mlp2_data;
  uint8_t *w_mlp2_scales;
  hip_bfloat16 *b_mlp1;
  hip_bfloat16 *b_mlp2;
  // Packed BF16 weights for MFMA kernels (conversion happens inside kernels)
  hip_bfloat16 *w_qkv_packed16;   // [n_layers][Mpad * Kpad] BF16 tiles
  hip_bfloat16 *w_o_packed16;     // [n_layers][Mpad * Kpad] BF16 tiles
  hip_bfloat16 *w_router_packed16;// [n_layers][Mpad * Kpad] BF16 tiles
} GPUTransformerWeights;

// ------------------------- KV Cache Management -------------------------

enum LayerType {
  FULL_ATTENTION = 0,
  SLIDING_WINDOW = 1,
  WINDOW_ATTENTION = 2
};

struct KVCacheDeviceData {
  hip_bfloat16 *key_cache;
  hip_bfloat16 *value_cache;
  int *layer_offsets;
  int *layer_capacities;
  size_t kv_dim;
  int max_batch;
  int n_layers;
  int seq_len;
  int sliding_window_size;
  int window_attention_size;
  int tokens_per_slot;

  __device__ __forceinline__ LayerType get_layer_type(int layer) const {
    if (window_attention_size > 0 && (layer % 2 == 1)) {
      return WINDOW_ATTENTION;
    }
    if (sliding_window_size > 0 && (layer % 2 == 0)) {
      return SLIDING_WINDOW;
    }
    return FULL_ATTENTION;
  }

  __device__ __forceinline__ int layer_capacity(int layer) const {
    return layer_capacities[layer];
  }

  __device__ __forceinline__ hip_bfloat16 *get_key_token_ptr(int slot_id, int layer, int cache_pos) const {
    if (slot_id < 0 || slot_id >= max_batch) return nullptr;
    if (layer < 0 || layer >= n_layers) return nullptr;
    if (cache_pos < 0) return nullptr;
    int capacity = layer_capacities[layer];
    if (cache_pos >= capacity) return nullptr;
    size_t token_offset = ((size_t)slot_id * (size_t)tokens_per_slot +
                           (size_t)layer_offsets[layer] +
                           (size_t)cache_pos) * kv_dim;
    return key_cache + token_offset;
  }

  __device__ __forceinline__ hip_bfloat16 *get_value_token_ptr(int slot_id, int layer, int cache_pos) const {
    if (slot_id < 0 || slot_id >= max_batch) return nullptr;
    if (layer < 0 || layer >= n_layers) return nullptr;
    if (cache_pos < 0) return nullptr;
    int capacity = layer_capacities[layer];
    if (cache_pos >= capacity) return nullptr;
    size_t token_offset = ((size_t)slot_id * (size_t)tokens_per_slot +
                           (size_t)layer_offsets[layer] +
                           (size_t)cache_pos) * kv_dim;
    return value_cache + token_offset;
  }

  __device__ __forceinline__ int get_write_position(int slot_id, int layer, int pos) const {
    (void)slot_id;
    LayerType lt = get_layer_type(layer);
    int capacity = layer_capacities[layer];
    if ((lt == SLIDING_WINDOW || lt == WINDOW_ATTENTION) && capacity > 0) {
      return pos % capacity;
    }
    return pos;
  }

  __device__ __forceinline__ int get_cache_position(int slot_id, int layer,
                                                    int logical_pos, int current_pos) const {
    (void)slot_id;
    LayerType lt = get_layer_type(layer);
    int capacity = layer_capacities[layer];
    if ((lt == SLIDING_WINDOW || lt == WINDOW_ATTENTION) && capacity > 0) {
      if (current_pos < capacity) {
        return logical_pos;
      }
      int window_start = current_pos - capacity + 1;
      if (logical_pos < window_start) {
        return -1;
      }
      int offset_in_window = logical_pos - window_start;
      int cache_start_offset = window_start % capacity;
      return (cache_start_offset + offset_in_window) % capacity;
    }
    return logical_pos;
  }

  __device__ __forceinline__ void get_attention_range(int slot_id, int layer, int current_pos,
                                                      int *start_pos, int *end_pos) const {
    (void)slot_id;
    LayerType lt = get_layer_type(layer);
    if ((lt == SLIDING_WINDOW || lt == WINDOW_ATTENTION) && layer_capacities[layer] > 0) {
      int capacity = layer_capacities[layer];
      int start = current_pos - capacity + 1;
      if (start < 0) start = 0;
      *start_pos = start;
      *end_pos = current_pos;
    } else {
      *start_pos = 0;
      *end_pos = current_pos;
    }
  }
};

class KVCacheManager {
public:
  KVCacheManager(int max_batch, int n_layers, int seq_len, size_t kv_dim,
                 int sliding_window_size, int window_attention_size)
      : max_batch_(max_batch),
        n_layers_(n_layers),
        seq_len_(seq_len),
        kv_dim_(kv_dim),
        sliding_window_size_(sliding_window_size),
        window_attention_size_(window_attention_size),
        tokens_per_slot_(0),
        key_cache_(nullptr),
        value_cache_(nullptr),
        d_layer_offsets_(nullptr),
        d_layer_capacities_(nullptr),
        d_device_data_(nullptr) {
    if (max_batch_ <= 0 || n_layers_ <= 0 || kv_dim_ == 0) {
      fprintf(stderr, "KVCacheManager: invalid configuration.\n");
      exit(EXIT_FAILURE);
    }

    if (sliding_window_size_ < 0) sliding_window_size_ = 0;
    if (window_attention_size_ < 0) window_attention_size_ = 0;

    layer_offsets_.resize(n_layers_);
    layer_capacities_.resize(n_layers_);

    size_t running = 0;
    for (int l = 0; l < n_layers_; ++l) {
      LayerType lt = host_layer_type(l);
      int capacity = seq_len_;
      if (lt == SLIDING_WINDOW && sliding_window_size_ > 0) {
        capacity = std::min(sliding_window_size_, seq_len_);
      } else if (lt == WINDOW_ATTENTION && window_attention_size_ > 0) {
        capacity = std::min(window_attention_size_, seq_len_);
      }
      if (capacity <= 0) capacity = 1;
      if (running > (size_t)std::numeric_limits<int>::max()) {
        fprintf(stderr, "KVCacheManager: layer offset overflow.\n");
        exit(EXIT_FAILURE);
      }
      layer_offsets_[l] = static_cast<int>(running);
      layer_capacities_[l] = capacity;
      running += static_cast<size_t>(capacity);
    }

    if (running == 0) {
      fprintf(stderr, "KVCacheManager: zero-sized cache.\n");
      exit(EXIT_FAILURE);
    }
    if (running > (size_t)std::numeric_limits<int>::max()) {
      fprintf(stderr, "KVCacheManager: total token count overflow.\n");
      exit(EXIT_FAILURE);
    }

    tokens_per_slot_ = static_cast<int>(running);
    size_t total_tokens = (size_t)max_batch_ * (size_t)tokens_per_slot_;
    size_t bytes_per_token = kv_dim_ * sizeof(hip_bfloat16);
    size_t total_bytes = total_tokens * bytes_per_token;

    HIP_CHECK(hipMalloc(&key_cache_, total_bytes));
    HIP_CHECK(hipMalloc(&value_cache_, total_bytes));

    HIP_CHECK(hipMalloc(&d_layer_offsets_, n_layers_ * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_layer_capacities_, n_layers_ * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_layer_offsets_, layer_offsets_.data(), n_layers_ * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_layer_capacities_, layer_capacities_.data(), n_layers_ * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&d_device_data_, sizeof(KVCacheDeviceData)));
    KVCacheDeviceData device_data{};
    device_data.key_cache = key_cache_;
    device_data.value_cache = value_cache_;
    device_data.layer_offsets = d_layer_offsets_;
    device_data.layer_capacities = d_layer_capacities_;
    device_data.kv_dim = kv_dim_;
    device_data.max_batch = max_batch_;
    device_data.n_layers = n_layers_;
    device_data.seq_len = seq_len_;
    device_data.sliding_window_size = sliding_window_size_;
    device_data.window_attention_size = window_attention_size_;
    device_data.tokens_per_slot = tokens_per_slot_;
    HIP_CHECK(hipMemcpy(d_device_data_, &device_data, sizeof(KVCacheDeviceData), hipMemcpyHostToDevice));

    double total_gb = ((double)total_bytes * 2.0) / GB;
    double tokens_capacity = (double)tokens_per_slot_;
    printf("KVCacheManager (prealloc): tokens_per_slot=%d (%.2fK summed tokens per slot), %.2f GB\n",
           tokens_per_slot_, tokens_capacity / 1e3, total_gb);
    if (sliding_window_size_ > 0) {
      printf("  - Sliding window tokens on even layers: %d\n", std::min(sliding_window_size_, seq_len_));
    }
    if (window_attention_size_ > 0) {
      printf("  - Window attention tokens on odd layers: %d\n", std::min(window_attention_size_, seq_len_));
    }
  }

  ~KVCacheManager() {
    if (d_device_data_) HIP_CHECK(hipFree(d_device_data_));
    if (d_layer_capacities_) HIP_CHECK(hipFree(d_layer_capacities_));
    if (d_layer_offsets_) HIP_CHECK(hipFree(d_layer_offsets_));
    if (value_cache_) HIP_CHECK(hipFree(value_cache_));
    if (key_cache_) HIP_CHECK(hipFree(key_cache_));
  }

  KVCacheDeviceData *get_device_data() const { return d_device_data_; }

  void prepare_layer(int layer, const int *slot_ids, const int *positions, int batch) {
    if (batch <= 0) return;
    if (layer < 0 || layer >= n_layers_) return;
    int capacity = layer_capacities_[layer];
    LayerType lt = host_layer_type(layer);
    if (lt == FULL_ATTENTION) {
      for (int bi = 0; bi < batch; ++bi) {
        int sid = slot_ids[bi];
        if (sid < 0 || sid >= max_batch_) continue;
        int pos = positions[bi];
        if (pos < 0) pos = 0;
        if (pos >= capacity) {
          fprintf(stderr, "KVCacheManager: token position overflow (slot=%d layer=%d pos=%d cap=%d).\n",
                  sid, layer, pos, capacity);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  void reset_slot(int slot_id) {
    (void)slot_id;
    // No-op: cache entries are overwritten sequentially for each request.
  }

private:
  LayerType host_layer_type(int layer) const {
    if (window_attention_size_ > 0 && (layer % 2 == 1)) {
      return WINDOW_ATTENTION;
    }
    if (sliding_window_size_ > 0 && (layer % 2 == 0)) {
      return SLIDING_WINDOW;
    }
    return FULL_ATTENTION;
  }

  int max_batch_;
  int n_layers_;
  int seq_len_;
  size_t kv_dim_;
  int sliding_window_size_;
  int window_attention_size_;
  int tokens_per_slot_;
  hip_bfloat16 *key_cache_;
  hip_bfloat16 *value_cache_;
  std::vector<int> layer_offsets_;
  std::vector<int> layer_capacities_;
  int *d_layer_offsets_;
  int *d_layer_capacities_;
  KVCacheDeviceData *d_device_data_;
};

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
  KVCacheManager *kv_cache_manager;
  float *cos_vals;
  float *sin_vals;
  hip_bfloat16 *t_bf16;
  // FP32 outputs for sampling and host reads
  float *logits_fp32;        // final logits (host staging buffer)
  unsigned long long *argmax_packed;      // packed (val, idx) maxima per batch slot
  // Batched MoE staging (for grouped GEMMs across slots/experts)
  int   *sel_sids;        // [max_batch * experts_per_token]
  int   *sel_eids;        // [max_batch * experts_per_token]
  float *sel_w;           // [max_batch * experts_per_token]
  float *mlp1_out_all;    // [max_batch * experts_per_token][local_mlp1_out_dim]
  hip_bfloat16 *gate_up_all;     // [max_batch * experts_per_token][local_intermediate_dim]
  // MFMA packed buffers
  hip_bfloat16 *xpack_bf16; // Packed input buffer for MFMA [K_max x Bpad]
  // Grouped MoE metadata per launch
  int   *sel_block_cols;   // block -> base selection index (size >= selections)
  int   *sel_block_sizes;  // block -> number of valid columns in block (<=32)
  int   *sel_block_eids;   // block -> expert id
} GPURunState;


struct WeightCopyWorkspace {
  float *host_stage[2] = {nullptr, nullptr};
  float *device_stage[2] = {nullptr, nullptr};
  hipEvent_t copy_ready[2] = {nullptr, nullptr};
  hipEvent_t stage_free[2] = {nullptr, nullptr};
  hipStream_t copy_stream = nullptr;
  hipStream_t convert_stream = nullptr;
  hipStream_t quant_stream = nullptr;
  size_t capacity_elems = 0;
  std::vector<std::pair<std::string, size_t>> logs;
};

// Per-GPU device context
typedef struct {
  int device_id;
  int dp_rank;
  int local_q_heads;
  int local_kv_heads;
  int local_experts;
  int local_intermediate_dim;
  int local_mlp1_out_dim;
  int local_q_dim;
  int local_kv_dim;
  int local_qkv_dim;
  int logits_tile_offset;
  int logits_tile_count;
  int logits_row_offset;
  int logits_row_padded;
  int logits_row_actual;
  GPUTransformerWeights weights;
  GPURunState state;
  int max_batch; // number of concurrent slots per GPU for continuous batching
  hipStream_t batch_stream;              // shared stream for batched ops
  hipEvent_t batch_event;
  // Staging device buffers for batched indices and positions
  int *d_slot_ids;
  int *d_pos_list;
  int *d_token_ids;
  int *d_next_tokens;  // Persistent buffer for sampled next tokens
  int *d_batch_iota;   // Identity mapping for batch-ordered GEMM outputs
  WeightCopyWorkspace copy_ws;
} GPUContext;

static std::vector<GPUContext> gpu_contexts;
static thread_local GPUContext* t_ctx = nullptr;

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

// ------------------------- BF16 utils -------------------------

using float4v  = __attribute__((ext_vector_type(4)))  float;
using float16v = __attribute__((ext_vector_type(16))) float;
using short4v = __attribute__((ext_vector_type(4))) short;

__host__ __device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __host__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
}

__device__ __forceinline__ uint16_t bfloat16_to_bits(hip_bfloat16 bf) {
  union { hip_bfloat16 bf; uint16_t u; } u{};
  u.bf = bf;
  return u.u;
}

__host__ __device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
  union { hip_bfloat16 bf; uint16_t u; } u{};
  u.bf = float_to_bfloat16(f);
  return u.u;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.u = static_cast<uint32_t>(bits) << 16;
  return cvt.f;
}

__device__ __forceinline__ float4v load_bf16x4_as_float4v(const hip_bfloat16 *ptr) {
  const uint32_t *src32 = reinterpret_cast<const uint32_t*>(ptr);
  uint32_t packed0 = src32[0];
  uint32_t packed1 = src32[1];
  float4v val;
  val[0] = bf16_bits_to_float(static_cast<uint16_t>(packed0 & 0xFFFFu));
  val[1] = bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 16));
  val[2] = bf16_bits_to_float(static_cast<uint16_t>(packed1 & 0xFFFFu));
  val[3] = bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 16));
  return val;
}

__device__ __forceinline__ float4v make_float4v(float v) {
  float4v out = {v, v, v, v};
  return out;
}

// ------------------------- MXFP4 utils -------------------------

constexpr int MXFP4_BLOCK_SIZE = 32;
constexpr size_t MXFP4_TILE32X8_ROWS = 32;
constexpr size_t MXFP4_TILE32X8_COLS = 8;
constexpr size_t MXFP4_TILE32X8_ELEMS = MXFP4_TILE32X8_ROWS * MXFP4_TILE32X8_COLS;
constexpr size_t MXFP4_TILE32X8_SCALE_BYTES = MXFP4_TILE32X8_ROWS;
constexpr size_t MXFP4_TILE32X8_DATA_BYTES = MXFP4_TILE32X8_ELEMS / 2;
constexpr size_t MXFP4_TILE32X8_PACKET_BYTES = MXFP4_TILE32X8_SCALE_BYTES + MXFP4_TILE32X8_DATA_BYTES;
static_assert(MXFP4_TILE32X8_PACKET_BYTES == 160, "Tile packet must be 160 bytes.");
constexpr float MXFP4_LEVEL_ABS_MAX = 6.0f;
constexpr float MXFP4_LEVELS[16] = {
  +0.0f, +0.5f, +1.0f, +1.5f,
  +2.0f, +3.0f, +4.0f, +6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f,
  -2.0f, -3.0f, -4.0f, -6.0f
};
constexpr float MXFP4_THRESHOLDS[7] = {
  0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
};
constexpr uint16_t MXFP4_BASE_BF16[16] = {
  0x0000u, 0x3F00u, 0x3F80u, 0x3FC0u,
  0x4000u, 0x4040u, 0x4080u, 0x40C0u,
  0x8000u, 0xBF00u, 0xBF80u, 0xBFC0u,
  0xC000u, 0xC040u, 0xC080u, 0xC0C0u
};

__device__ __constant__ uint16_t MXFP4_DEQUANT_LUT_BF16[256][16];

__device__ __constant__ float MXFP4_DECODE_E8M0_LUT[256] = {
  5.87747175e-39f, 1.17549435e-38f, 2.35098870e-38f, 4.70197740e-38f, 9.40395481e-38f, 1.88079096e-37f, 3.76158192e-37f, 7.52316385e-37f,
  1.50463277e-36f, 3.00926554e-36f, 6.01853108e-36f, 1.20370622e-35f, 2.40741243e-35f, 4.81482486e-35f, 9.62964972e-35f, 1.92592994e-34f,
  3.85185989e-34f, 7.70371978e-34f, 1.54074396e-33f, 3.08148791e-33f, 6.16297582e-33f, 1.23259516e-32f, 2.46519033e-32f, 4.93038066e-32f,
  9.86076132e-32f, 1.97215226e-31f, 3.94430453e-31f, 7.88860905e-31f, 1.57772181e-30f, 3.15544362e-30f, 6.31088724e-30f, 1.26217745e-29f,
  2.52435490e-29f, 5.04870979e-29f, 1.00974196e-28f, 2.01948392e-28f, 4.03896783e-28f, 8.07793567e-28f, 1.61558713e-27f, 3.23117427e-27f,
  6.46234854e-27f, 1.29246971e-26f, 2.58493941e-26f, 5.16987883e-26f, 1.03397577e-25f, 2.06795153e-25f, 4.13590306e-25f, 8.27180613e-25f,
  1.65436123e-24f, 3.30872245e-24f, 6.61744490e-24f, 1.32348898e-23f, 2.64697796e-23f, 5.29395592e-23f, 1.05879118e-22f, 2.11758237e-22f,
  4.23516474e-22f, 8.47032947e-22f, 1.69406589e-21f, 3.38813179e-21f, 6.77626358e-21f, 1.35525272e-20f, 2.71050543e-20f, 5.42101086e-20f,
  1.08420217e-19f, 2.16840434e-19f, 4.33680869e-19f, 8.67361738e-19f, 1.73472348e-18f, 3.46944695e-18f, 6.93889390e-18f, 1.38777878e-17f,
  2.77555756e-17f, 5.55111512e-17f, 1.11022302e-16f, 2.22044605e-16f, 4.44089210e-16f, 8.88178420e-16f, 1.77635684e-15f, 3.55271368e-15f,
  7.10542736e-15f, 1.42108547e-14f, 2.84217094e-14f, 5.68434189e-14f, 1.13686838e-13f, 2.27373675e-13f, 4.54747351e-13f, 9.09494702e-13f,
  1.81898940e-12f, 3.63797881e-12f, 7.27595761e-12f, 1.45519152e-11f, 2.91038305e-11f, 5.82076609e-11f, 1.16415322e-10f, 2.32830644e-10f,
  4.65661287e-10f, 9.31322575e-10f, 1.86264515e-09f, 3.72529030e-09f, 7.45058060e-09f, 1.49011612e-08f, 2.98023224e-08f, 5.96046448e-08f,
  1.19209290e-07f, 2.38418579e-07f, 4.76837158e-07f, 9.53674316e-07f, 1.90734863e-06f, 3.81469727e-06f, 7.62939453e-06f, 1.52587891e-05f,
  3.05175781e-05f, 6.10351562e-05f, 1.22070312e-04f, 2.44140625e-04f, 4.88281250e-04f, 9.76562500e-04f, 1.95312500e-03f, 3.90625000e-03f,
  7.81250000e-03f, 1.56250000e-02f, 3.12500000e-02f, 6.25000000e-02f, 1.25000000e-01f, 2.50000000e-01f, 5.00000000e-01f, 1.00000000e+00f,
  2.00000000e+00f, 4.00000000e+00f, 8.00000000e+00f, 1.60000000e+01f, 3.20000000e+01f, 6.40000000e+01f, 1.28000000e+02f, 2.56000000e+02f,
  5.12000000e+02f, 1.02400000e+03f, 2.04800000e+03f, 4.09600000e+03f, 8.19200000e+03f, 1.63840000e+04f, 3.27680000e+04f, 6.55360000e+04f,
  1.31072000e+05f, 2.62144000e+05f, 5.24288000e+05f, 1.04857600e+06f, 2.09715200e+06f, 4.19430400e+06f, 8.38860800e+06f, 1.67772160e+07f,
  3.35544320e+07f, 6.71088640e+07f, 1.34217728e+08f, 2.68435456e+08f, 5.36870912e+08f, 1.07374182e+09f, 2.14748365e+09f, 4.29496730e+09f,
  8.58993459e+09f, 1.71798692e+10f, 3.43597384e+10f, 6.87194767e+10f, 1.37438953e+11f, 2.74877907e+11f, 5.49755814e+11f, 1.09951163e+12f,
  2.19902326e+12f, 4.39804651e+12f, 8.79609302e+12f, 1.75921860e+13f, 3.51843721e+13f, 7.03687442e+13f, 1.40737488e+14f, 2.81474977e+14f,
  5.62949953e+14f, 1.12589991e+15f, 2.25179981e+15f, 4.50359963e+15f, 9.00719925e+15f, 1.80143985e+16f, 3.60287970e+16f, 7.20575940e+16f,
  1.44115188e+17f, 2.88230376e+17f, 5.76460752e+17f, 1.15292150e+18f, 2.30584301e+18f, 4.61168602e+18f, 9.22337204e+18f, 1.84467441e+19f,
  3.68934881e+19f, 7.37869763e+19f, 1.47573953e+20f, 2.95147905e+20f, 5.90295810e+20f, 1.18059162e+21f, 2.36118324e+21f, 4.72236648e+21f,
  9.44473297e+21f, 1.88894659e+22f, 3.77789319e+22f, 7.55578637e+22f, 1.51115727e+23f, 3.02231455e+23f, 6.04462910e+23f, 1.20892582e+24f,
  2.41785164e+24f, 4.83570328e+24f, 9.67140656e+24f, 1.93428131e+25f, 3.86856262e+25f, 7.73712525e+25f, 1.54742505e+26f, 3.09485010e+26f,
  6.18970020e+26f, 1.23794004e+27f, 2.47588008e+27f, 4.95176016e+27f, 9.90352031e+27f, 1.98070406e+28f, 3.96140813e+28f, 7.92281625e+28f,
  1.58456325e+29f, 3.16912650e+29f, 6.33825300e+29f, 1.26765060e+30f, 2.53530120e+30f, 5.07060240e+30f, 1.01412048e+31f, 2.02824096e+31f,
  4.05648192e+31f, 8.11296384e+31f, 1.62259277e+32f, 3.24518554e+32f, 6.49037107e+32f, 1.29807421e+33f, 2.59614843e+33f, 5.19229686e+33f,
  1.03845937e+34f, 2.07691874e+34f, 4.15383749e+34f, 8.30767497e+34f, 1.66153499e+35f, 3.32306999e+35f, 6.64613998e+35f, 1.32922800e+36f,
  2.65845599e+36f, 5.31691198e+36f, 1.06338240e+37f, 2.12676479e+37f, 4.25352959e+37f, 8.50705917e+37f, 1.70141183e+38f, 3.40282347e+38f
};

__host__ __device__ __forceinline__ float decode_e8m0(uint8_t code) {
#if defined(__HIP_DEVICE_COMPILE__)
  return MXFP4_DECODE_E8M0_LUT[code];
#else
  int exp_unbiased = static_cast<int>(code) - 127;
  return ldexpf(1.0f, exp_unbiased);
#endif
}

__host__ __device__ inline uint8_t encode_e8m0(float scale) {
  if (!(scale > 0.0f) || !isfinite(scale)) {
    return static_cast<uint8_t>(127); // pow2 exponent 0 -> scale 1.0f
  }
  float log2_scale = log2f(scale);
  int exp_unbiased = static_cast<int>(ceilf(log2_scale));
  if (exp_unbiased < -127) exp_unbiased = -127;
  if (exp_unbiased > 128) exp_unbiased = 128;
  int encoded = exp_unbiased + 127;
  if (encoded < 0) encoded = 0;
  if (encoded > 255) encoded = 255;
  return static_cast<uint8_t>(encoded);
}

__host__ __device__ __forceinline__ float decode_mxfp4(uint8_t nibble) {
  return MXFP4_LEVELS[nibble & 0x0F];
}

__host__ __device__ inline uint8_t encode_mxfp4(float v) {
  if (!isfinite(v)) v = 0.0f;
  float a = fabsf(v);
  int idx = 0;
  while (idx < 7 && a > MXFP4_THRESHOLDS[idx]) {
    ++idx;
  }
  if (v < 0.0f) {
    return static_cast<uint8_t>(idx + 8);
  } else {
    return static_cast<uint8_t>(idx);
  }
}

__host__ __device__ __forceinline__ uint16_t mxfp4_dequant_to_bf16(uint8_t nibble, int scale_exp) {
  uint16_t base = MXFP4_BASE_BF16[nibble & 0x0F];
  if ((base & 0x7F80u) == 0u || scale_exp == 0) {
    return base;
  }
  int exp = (base >> 7) & 0xFF;
  int new_exp = exp + scale_exp;
  if (new_exp > 0 && new_exp < 0xFF) {
    return static_cast<uint16_t>((base & 0x807Fu) | (static_cast<uint16_t>(new_exp) << 7));
  }
  float scaled = ldexpf(MXFP4_LEVELS[nibble & 0x0F], scale_exp);
  return float_to_bf16_bits(scaled);
}

static void ensure_mxfp4_dequant_lut_initialized() {
  static std::once_flag build_once;
  static uint16_t host_lut[256][16];
  std::call_once(build_once, []() {
    for (int scale_code = 0; scale_code < 256; ++scale_code) {
      int scale_exp = scale_code - 127;
      for (int nib = 0; nib < 16; ++nib) {
        host_lut[scale_code][nib] = mxfp4_dequant_to_bf16(static_cast<uint8_t>(nib), scale_exp);
      }
    }
  });

  static std::mutex lut_mutex;
  static std::vector<char> initialized;
  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  std::lock_guard<std::mutex> guard(lut_mutex);
  if (device_id >= static_cast<int>(initialized.size())) {
    initialized.resize(device_id + 1, 0);
  }
  if (initialized[device_id]) {
    return;
  }
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(MXFP4_DEQUANT_LUT_BF16), host_lut, sizeof(host_lut)));
  initialized[device_id] = 1;
}

// ------------------------- Kernels -------------------------

// FP32 -> BF16 (device)
__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
}

// MFMA MLP1 directly from MXFP4-packed weights (no BF16 storage).
// Dequantizes per element on-the-fly using e8m0 scales and 4-bit levels.

#ifndef GETP_MLP1_BLOCK_ROWS
#define GETP_MLP1_BLOCK_ROWS 512
#endif

constexpr int MLP1_ROWS_PER_TILE = 32;
constexpr int MLP1_BLOCK_ROWS = GETP_MLP1_BLOCK_ROWS;
static_assert(MLP1_BLOCK_ROWS % MLP1_ROWS_PER_TILE == 0, "MLP1 block rows must be a multiple of 32.");
constexpr int MLP1_WAVES_PER_BLOCK = MLP1_BLOCK_ROWS / MLP1_ROWS_PER_TILE;
static_assert(MLP1_WAVES_PER_BLOCK >= 1, "MLP1 requires at least one wave per block.");
static_assert(MLP1_WAVES_PER_BLOCK <= 16, "MLP1 waves per block exceeds practical limits.");
constexpr int MLP1_BLOCK_THREADS = 64 * MLP1_WAVES_PER_BLOCK;
static_assert(MLP1_BLOCK_THREADS <= 1024, "MLP1 block thread count exceeds HIP limit.");
constexpr int MLP1_BLOCK_DIM_X = 32;
constexpr int MLP1_BLOCK_DIM_Y = 2 * MLP1_WAVES_PER_BLOCK;

__global__ __launch_bounds__(MLP1_BLOCK_THREADS, 4)
void mlp1_grouped_mfma_matvec_mxfp4_kernel(
    float* __restrict__ mlp1_out_all,        // [M selections][out_dim]
    const hip_bfloat16* __restrict__ t_bf16, // [B slots][hidden_dim] BF16
    const uint8_t* __restrict__ w_tiles,     // MXFP4 tiles with interleaved scales (32x8)
    const hip_bfloat16* __restrict__ b_mlp1, // [n_layers*n_experts][out_dim]
    const int* __restrict__ sel_sids,        // [M]
    const int* __restrict__ sel_block_cols,  // [num_blocks]
    const int* __restrict__ sel_block_sizes, // [num_blocks]
    const int* __restrict__ sel_block_eids,  // [num_blocks]
    int selections_M,
    int total_blocks,
    int hidden_dim, int out_dim,
    int layer, int n_experts)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int wave_id = ty >> 1;                          // 0..(waves per block - 1)
  const int lane_in_wave = ((ty & 1) << 5) + tx;        // 0..63 inside a wave
  const int lane_col = lane_in_wave & 31;               // logical column within 32 tile
  const int lane_wave = lane_in_wave >> 5;              // 0 or 1 (selects row quartet)

  const int tile_group = blockIdx.x;
  const int block_id = blockIdx.y;
  if (block_id >= total_blocks) return;

  const int col0 = sel_block_cols[block_id];
  const int valid_cols = sel_block_sizes[block_id];
  const int eid = sel_block_eids[block_id];
  if (valid_cols <= 0) return;

  const int base_row_group = tile_group * MLP1_BLOCK_ROWS;
  if (base_row_group >= out_dim) return;

  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ B = b_mlp1 ? (b_mlp1 + mat_index * (size_t)out_dim) : nullptr;

  const bool all_cols_active = (valid_cols == 32) && ((col0 + 31) < selections_M);

  __shared__ __align__(16) uint16_t sWbf16[MLP1_WAVES_PER_BLOCK * 32 * 8];
  __shared__ __align__(16) uint16_t sXbf16[8 * 32];
  __shared__ __align__(16) float sBias[MLP1_WAVES_PER_BLOCK * 32];
  __shared__ __align__(16) const hip_bfloat16* sXPtrs[32];
  __shared__ __align__(16) float* sOutPtrs[32];
  __shared__ __align__(16) uint8_t sScaleCodes[MLP1_WAVES_PER_BLOCK * 32];
  short4v* __restrict__ sXvec = reinterpret_cast<short4v*>(sXbf16);

  const int rows_remaining = out_dim - base_row_group;
  if (rows_remaining <= 0) return;
  const int row_tiles_available = (rows_remaining + 31) / 32;
  const int row_tiles = row_tiles_available < MLP1_WAVES_PER_BLOCK ? row_tiles_available : MLP1_WAVES_PER_BLOCK;
  const bool tile_active = wave_id < row_tiles;

  const int tile_m_global = tile_group * MLP1_WAVES_PER_BLOCK + wave_id;
  const int base_row = base_row_group + wave_id * 32;
  const bool full_row_tile = tile_active && ((base_row + 31) < out_dim);

  uint16_t* __restrict__ sWbf16_tile = sWbf16 + wave_id * 32 * 8;
  uint8_t* __restrict__ sScaleCodes_tile = sScaleCodes + wave_id * 32;
  float* __restrict__ sBias_tile = sBias + wave_id * 32;
  short4v* __restrict__ sWvec_tile = reinterpret_cast<short4v*>(sWbf16_tile);

  if (wave_id == 0 && lane_in_wave < 32) {
    const int msel = col0 + lane_in_wave;
    const bool active = (lane_in_wave < valid_cols) && (msel < selections_M);
    const hip_bfloat16* x_ptr = active ? (t_bf16 + (size_t)sel_sids[msel] * (size_t)hidden_dim) : nullptr;
    sXPtrs[lane_in_wave] = x_ptr;
    sOutPtrs[lane_in_wave] = active ? (mlp1_out_all + (size_t)msel * (size_t)out_dim) : nullptr;
  }

  if (tile_active && lane_in_wave < 32) {
    const int row = base_row + lane_in_wave;
    sBias_tile[lane_in_wave] = (B && row < out_dim) ? bfloat16_to_float(B[row]) : 0.f;
  }
  __syncthreads();

  float16v acc = {0.0f};

  const int K = hidden_dim;
  const int tiles_m = (out_dim + 31) / 32;
  const int tiles_k = (K + 7) / 8;
  const size_t tiles_per_row = (size_t)tiles_k;
  const size_t tiles_per_mat = (size_t)tiles_m * tiles_per_row;
  const size_t tile_packet_bytes = MXFP4_TILE32X8_PACKET_BYTES;
  const uint8_t* __restrict__ Wtiles = w_tiles + mat_index * tiles_per_mat * tile_packet_bytes;

  for (int kk = 0; kk < tiles_k; ++kk) {
    const bool full_k_tile = ((kk + 1) * 8) <= K;
    const bool fast_w_tile = full_row_tile && full_k_tile;
    const bool fast_x_tile = all_cols_active && full_k_tile;

    if (tile_active && lane_in_wave < 32) {
      const size_t tile_linear = (size_t)tile_m_global * tiles_per_row + (size_t)kk;
      const uint8_t* __restrict__ tile_base = Wtiles + tile_linear * tile_packet_bytes;
      uint8_t scale_code = static_cast<uint8_t>(0x7F);
      const int row = base_row + lane_in_wave;
      if (row < out_dim) {
        scale_code = tile_base[lane_in_wave];
      }
      sScaleCodes_tile[lane_in_wave] = scale_code;
    }
    __syncthreads();

    if (tile_active) {
      const size_t tile_linear = (size_t)tile_m_global * tiles_per_row + (size_t)kk;
      const size_t tile_base_byte = tile_linear * tile_packet_bytes + MXFP4_TILE32X8_SCALE_BYTES;

      int idx4 = lane_in_wave * 4;
      if (idx4 < 256) {
        int im = idx4 >> 3;            // tile row 0..31
        int ik_base = idx4 & 0x7;      // starting column within row (0 or 4)
        int row = base_row + im;
        int col_base = kk * 8 + ik_base;
        size_t byte_off = tile_base_byte + static_cast<size_t>(idx4 >> 1);
        uint16_t packed16 = *reinterpret_cast<const uint16_t*>(Wtiles + byte_off);
        const uint8_t scale_code = sScaleCodes_tile[im];
        const uint16_t* __restrict__ lut_row = MXFP4_DEQUANT_LUT_BF16[scale_code];
        short4v wvec = {0, 0, 0, 0};
        if (fast_w_tile) {
          #pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
            wvec[t4] = static_cast<short>(lut_row[nibble]);
          }
        } else {
          const bool row_valid = (row < out_dim);
          #pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            const int col = col_base + t4;
            if (row_valid && col < K) {
              const uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
              wvec[t4] = static_cast<short>(lut_row[nibble]);
            }
          }
        }
        sWvec_tile[im * 2 + (ik_base >> 2)] = wvec;
      }
    }

    if (wave_id == 0) {
      const int r = lane_in_wave % 8;            // K row within tile
      const int c4 = (lane_in_wave / 8) * 4;     // column block {0,4,...,28}
      const int k_row = kk * 8 + r;
      short4v xvec = {0, 0, 0, 0};
      const hip_bfloat16* x0 = sXPtrs[c4 + 0];
      const hip_bfloat16* x1 = sXPtrs[c4 + 1];
      const hip_bfloat16* x2 = sXPtrs[c4 + 2];
      const hip_bfloat16* x3 = sXPtrs[c4 + 3];
      if (fast_x_tile && x0 && x1 && x2 && x3) {
        xvec[0] = static_cast<short>(bfloat16_to_bits(x0[k_row]));
        xvec[1] = static_cast<short>(bfloat16_to_bits(x1[k_row]));
        xvec[2] = static_cast<short>(bfloat16_to_bits(x2[k_row]));
        xvec[3] = static_cast<short>(bfloat16_to_bits(x3[k_row]));
      } else {
        const bool row_valid = (k_row < K);
        if (row_valid && x0) xvec[0] = static_cast<short>(bfloat16_to_bits(x0[k_row]));
        if (row_valid && x1) xvec[1] = static_cast<short>(bfloat16_to_bits(x1[k_row]));
        if (row_valid && x2) xvec[2] = static_cast<short>(bfloat16_to_bits(x2[k_row]));
        if (row_valid && x3) xvec[3] = static_cast<short>(bfloat16_to_bits(x3[k_row]));
      }
      sXvec[r * 8 + (c4 >> 2)] = xvec;
    }

    __syncthreads();

    if (tile_active) {
      const int kbase = lane_wave * 4;
      short4v Areg;
      short4v Breg;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int kidx = kbase + j;
        uint16_t a_bits = sWbf16_tile[lane_col * 8 + kidx];
        uint16_t b_bits = sXbf16[kidx * 32 + lane_col];
        Areg[j] = static_cast<short>(a_bits);
        Breg[j] = static_cast<short>(b_bits);
      }
      acc = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(Areg, Breg, acc, 0, 0, 0);
    }

    __syncthreads();
  }

  float* out_vec = sOutPtrs[lane_col];
  if (tile_active && out_vec) {
    #pragma unroll
    for (int g = 0; g < 16; ++g) {
      const int group = g >> 2;
      const int g_mod = g & 3;
      const int row_in_tile = ((group << 3) & 31) + lane_wave * 4 + g_mod;
      const int row = base_row + row_in_tile;
      if (row >= out_dim) continue;
      float value = acc[g] + sBias_tile[row_in_tile];
      out_vec[row] = value;
    }
  }
}

__global__ void quantize_mxfp4_kernel(const float *tile, size_t tile_elems, size_t global_elem_offset,
                                      uint8_t *dst_data, uint8_t *dst_scales) {
  size_t block = blockIdx.x;
  size_t start = block * (size_t)MXFP4_BLOCK_SIZE;
  if (start >= tile_elems) return;

  size_t elems = tile_elems - start;
  if (elems > MXFP4_BLOCK_SIZE) elems = MXFP4_BLOCK_SIZE;

  const float *src = tile + start;
  float max_abs = 0.0f;
  for (size_t i = 0; i < elems; ++i) {
    float a = fabsf(src[i]);
    if (a > max_abs) max_abs = a;
  }

  float raw_scale = max_abs > 0.0f ? max_abs / MXFP4_LEVEL_ABS_MAX : 1.0f;
  uint8_t scale_code = encode_e8m0(raw_scale);
  float scale = decode_e8m0(scale_code);

  size_t global_start = global_elem_offset + start;
  size_t scale_index = global_start / MXFP4_BLOCK_SIZE;
  dst_scales[scale_index] = scale_code;

  for (size_t i = 0; i < elems; ++i) {
    float v = src[i] / scale;
    uint8_t code = encode_mxfp4(v);
    size_t global_elem = global_start + i;
    size_t byte_index = global_elem >> 1;
    bool even = ((global_elem & 1ULL) == 0ULL);
    uint8_t existing = dst_data[byte_index];
    if (even) {
      existing = (existing & 0xF0) | (code & 0x0F);
    } else {
      existing = (existing & 0x0F) | ((code & 0x0F) << 4);
    }
    dst_data[byte_index] = existing;
  }
}

// Helpers for MXFP4 tile packing
__device__ inline uint8_t mxfp4_decode_src_at(const uint8_t* __restrict__ src_data, size_t elem_index) {
  uint8_t packed = src_data[elem_index >> 1];
  return (elem_index & 1ULL) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xF);
}

__device__ inline void mxfp4_tile_map_rowcol(size_t elem_linear, size_t cols, size_t& row, size_t& col) {
  size_t tiles_per_row = (cols + 15) / 16;
  size_t tile_idx = elem_linear / 256;             // which 16x16 tile
  size_t in_tile = elem_linear - tile_idx * 256;   // 0..255
  size_t im = in_tile / 16;                        // 0..15
  size_t ik = in_tile - im * 16;                   // 0..15
  size_t tile_m = tile_idx / tiles_per_row;
  size_t tile_k = tile_idx - tile_m * tiles_per_row;
  row = tile_m * 16 + im;
  col = tile_k * 16 + ik;
}

// Repack MXFP4 row-major [rows x cols] into 16x16 tile-major order.
__global__ void pack_mxfp4_row_to_tile16_kernel(
    const uint8_t* __restrict__ src_data,
    uint8_t* __restrict__ dst_data,
    size_t rows, size_t cols) {
  size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elems = rows * cols;
  size_t total_bytes = (total_elems + 1) >> 1;
  if (byte_idx >= total_bytes) return;

  size_t elem0 = byte_idx * 2;
  size_t row0, col0; mxfp4_tile_map_rowcol(elem0, cols, row0, col0);
  uint8_t code0 = 0u;
  if (row0 < rows && col0 < cols) {
    size_t src_elem0 = row0 * cols + col0;
    code0 = mxfp4_decode_src_at(src_data, src_elem0);
  }

  size_t elem1 = elem0 + 1;
  uint8_t code1 = 0u;
  if (elem1 < total_elems) {
    size_t row1, col1; mxfp4_tile_map_rowcol(elem1, cols, row1, col1);
    if (row1 < rows && col1 < cols) {
      size_t src_elem1 = row1 * cols + col1;
      code1 = mxfp4_decode_src_at(src_data, src_elem1);
    }
  }

  dst_data[byte_idx] = (uint8_t)((code0 & 0x0F) | ((code1 & 0x0F) << 4));
}

// Reorder per-32-element scale codes into the same tile-major layout (16 codes per tile).
__global__ void pack_mxfp4_scales_row_to_tile16_kernel(
    const uint8_t* __restrict__ src_scales,
    uint8_t* __restrict__ dst_scales,
    size_t rows, size_t cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tiles_m = (rows + 15) / 16;
  size_t tiles_k = (cols + 15) / 16;
  size_t total_tiles = tiles_m * tiles_k;
  size_t total_entries = total_tiles * 16; // 16 rows per tile
  if (idx >= total_entries) return;

  size_t tile_idx = idx / 16;
  size_t row_in_tile = idx % 16;

  size_t tile_m = tile_idx / tiles_k;
  size_t tile_k = tile_idx - tile_m * tiles_k;

  size_t row = tile_m * 16 + row_in_tile;
  size_t col = tile_k * 16;

  size_t blocks_per_row = (cols + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  uint8_t code = static_cast<uint8_t>(127); // scale=1 default
  if (row < rows && col < cols) {
    size_t block_idx = row * blocks_per_row + (col / MXFP4_BLOCK_SIZE);
    code = src_scales[block_idx];
  }
  dst_scales[idx] = code;
}

// Map MXFP4 tile-major index (32x8 tiles) back to row-major coordinates.
__device__ inline void mxfp4_tile_map_rowcol_32x8(size_t elem_linear, size_t cols,
                                                  size_t& row, size_t& col) {
  size_t tiles_per_row = (cols + 7) / 8;
  size_t tile_idx = elem_linear / 256;           // which 32x8 tile
  size_t in_tile = elem_linear - tile_idx * 256; // 0..255
  size_t im = in_tile / 8;                       // 0..31
  size_t ik = in_tile - im * 8;                  // 0..7
  size_t tile_m = tile_idx / tiles_per_row;
  size_t tile_k = tile_idx - tile_m * tiles_per_row;
  row = tile_m * 32 + im;
  col = tile_k * 8 + ik;
}

// Repack MXFP4 row-major [rows x cols] into 32x8 tile-major order.
__global__ void pack_mxfp4_row_to_tile32x8_kernel(
    const uint8_t* __restrict__ src_data,
    uint8_t* __restrict__ dst_data,
    size_t rows, size_t cols) {
  size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elems = rows * cols;
  size_t total_bytes = (total_elems + 1) >> 1;
  if (byte_idx >= total_bytes) return;

  size_t elem0 = byte_idx * 2;
  size_t row0, col0;
  mxfp4_tile_map_rowcol_32x8(elem0, cols, row0, col0);
  uint8_t code0 = 0u;
  if (row0 < rows && col0 < cols) {
    size_t src_elem0 = row0 * cols + col0;
    code0 = mxfp4_decode_src_at(src_data, src_elem0);
  }

  size_t elem1 = elem0 + 1;
  uint8_t code1 = 0u;
  if (elem1 < total_elems) {
    size_t row1, col1;
    mxfp4_tile_map_rowcol_32x8(elem1, cols, row1, col1);
    if (row1 < rows && col1 < cols) {
      size_t src_elem1 = row1 * cols + col1;
      code1 = mxfp4_decode_src_at(src_data, src_elem1);
    }
  }

  dst_data[byte_idx] = static_cast<uint8_t>((code0 & 0x0F) | ((code1 & 0x0F) << 4));
}

// Reorder per-32-element scale codes into 32x8 tile-major layout (32 rows per tile).
__global__ void pack_mxfp4_scales_row_to_tile32x8_kernel(
    const uint8_t* __restrict__ src_scales,
    uint8_t* __restrict__ dst_scales,
    size_t rows, size_t cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tiles_m = (rows + 31) / 32;
  size_t tiles_k = (cols + 7) / 8;
  size_t total_tiles = tiles_m * tiles_k;
  size_t total_entries = total_tiles * 32; // 32 rows per tile
  if (idx >= total_entries) return;

  size_t tile_idx = idx / 32;
  size_t row_in_tile = idx % 32;

  size_t tile_m = tile_idx / tiles_k;
  size_t tile_k = tile_idx - tile_m * tiles_k;

  size_t row = tile_m * 32 + row_in_tile;
  size_t col = tile_k * 8;

  size_t blocks_per_row = (cols + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  uint8_t code = static_cast<uint8_t>(127);
  if (row < rows && col < cols) {
    size_t block_idx = row * blocks_per_row + (col / MXFP4_BLOCK_SIZE);
    code = src_scales[block_idx];
  }
  dst_scales[idx] = code;
}

__global__ void pack_mxfp4_row_to_tile32x8_interleaved_kernel(
    const uint8_t* __restrict__ src_data,
    uint8_t* __restrict__ dst_tiles,
    size_t rows, size_t cols,
    size_t tile_packet_bytes) {
  size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elems = rows * cols;
  size_t total_bytes = (total_elems + 1) >> 1;
  if (byte_idx >= total_bytes) return;

  size_t elem0 = byte_idx * 2;
  size_t row0, col0;
  mxfp4_tile_map_rowcol_32x8(elem0, cols, row0, col0);
  uint8_t code0 = 0u;
  if (row0 < rows && col0 < cols) {
    size_t src_elem0 = row0 * cols + col0;
    code0 = mxfp4_decode_src_at(src_data, src_elem0);
  }

  size_t elem1 = elem0 + 1;
  uint8_t code1 = 0u;
  if (elem1 < total_elems) {
    size_t row1, col1;
    mxfp4_tile_map_rowcol_32x8(elem1, cols, row1, col1);
    if (row1 < rows && col1 < cols) {
      size_t src_elem1 = row1 * cols + col1;
      code1 = mxfp4_decode_src_at(src_data, src_elem1);
    }
  }

  size_t tile_idx = elem0 / MXFP4_TILE32X8_ELEMS;
  size_t tile_elem0 = elem0 - tile_idx * MXFP4_TILE32X8_ELEMS;
  size_t tile_byte = tile_elem0 >> 1;
  size_t dst_offset = tile_idx * tile_packet_bytes + MXFP4_TILE32X8_SCALE_BYTES + tile_byte;
  dst_tiles[dst_offset] = static_cast<uint8_t>((code0 & 0x0F) | ((code1 & 0x0F) << 4));
}

__global__ void pack_mxfp4_scales_row_to_tile32x8_interleaved_kernel(
    const uint8_t* __restrict__ src_scales,
    uint8_t* __restrict__ dst_tiles,
    size_t rows, size_t cols,
    size_t tile_packet_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tiles_m = (rows + 31) / 32;
  size_t tiles_k = (cols + 7) / 8;
  size_t total_tiles = tiles_m * tiles_k;
  size_t total_entries = total_tiles * MXFP4_TILE32X8_SCALE_BYTES;
  if (idx >= total_entries) return;

  size_t tile_idx = idx / MXFP4_TILE32X8_SCALE_BYTES;
  size_t row_in_tile = idx % MXFP4_TILE32X8_SCALE_BYTES;

  size_t tile_m = tile_idx / tiles_k;
  size_t tile_k = tile_idx - tile_m * tiles_k;

  size_t row = tile_m * MXFP4_TILE32X8_ROWS + row_in_tile;
  size_t col = tile_k * MXFP4_TILE32X8_COLS;

  size_t blocks_per_row = (cols + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  uint8_t code = static_cast<uint8_t>(0x7F);
  if (row < rows && col < cols) {
    size_t block_idx = row * blocks_per_row + (col / MXFP4_BLOCK_SIZE);
    code = src_scales[block_idx];
  }

  dst_tiles[tile_idx * tile_packet_bytes + row_in_tile] = code;
}

// Grouped MLP2 matvec using MFMA and tile-packed MXFP4 weights.
// For each selection m: y = W[layer,eid] @ g[m] + b[layer,eid]; then atomicAdd into e_agg[sid] with weight sel_w[m].

#ifndef GETP_MLP2_BLOCK_ROWS
#define GETP_MLP2_BLOCK_ROWS 64
#endif

constexpr int MLP2_ROWS_PER_TILE = 32;
constexpr int MLP2_BLOCK_ROWS = GETP_MLP2_BLOCK_ROWS;
static_assert(MLP2_BLOCK_ROWS % MLP2_ROWS_PER_TILE == 0, "MLP2 block rows must be a multiple of 32.");
constexpr int MLP2_WAVES_PER_BLOCK = MLP2_BLOCK_ROWS / MLP2_ROWS_PER_TILE;
static_assert(MLP2_WAVES_PER_BLOCK >= 1, "MLP2 requires at least one wave per block.");
static_assert(MLP2_WAVES_PER_BLOCK <= 16, "MLP2 waves per block exceeds practical limits.");
constexpr int MLP2_BLOCK_THREADS = 64 * MLP2_WAVES_PER_BLOCK;
static_assert(MLP2_BLOCK_THREADS <= 1024, "MLP2 block thread count exceeds HIP limit.");
constexpr int MLP2_BLOCK_DIM_X = 32;
constexpr int MLP2_BLOCK_DIM_Y = 2 * MLP2_WAVES_PER_BLOCK;

__global__ __launch_bounds__(MLP2_BLOCK_THREADS, 4)
void mlp2_accum_grouped_mfma_matvec_mxfp4_kernel(
    float* __restrict__ e_agg,                // [slots, hidden_dim]
    const hip_bfloat16* __restrict__ gate_up_all, // [M, intermediate_dim] BF16
    const uint8_t* __restrict__ w_tiles,      // MXFP4 tiles with interleaved scales (32x8)
    const hip_bfloat16* __restrict__ b_mlp2,  // [n_layers*n_experts][hidden_dim]
    const int* __restrict__ sel_sids,         // [M]
    const float* __restrict__ sel_w,          // [M]
    const int* __restrict__ sel_block_cols,   // [num_blocks]
    const int* __restrict__ sel_block_sizes,  // [num_blocks]
    const int* __restrict__ sel_block_eids,   // [num_blocks]
    int selections_M,
    int total_blocks,
    int hidden_dim, int intermediate_dim,
    int layer, int n_experts,
    int eagg_stride) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int wave_id = ty >> 1;
  const int lane_in_wave = ((ty & 1) << 5) + tx;
  const int lane_col = lane_in_wave & 31;
  const int lane_wave = lane_in_wave >> 5;

  const int tile_group = blockIdx.x;
  const int block_id = blockIdx.y;
  if (block_id >= total_blocks) return;

  const int col0 = sel_block_cols[block_id];
  const int valid_cols = sel_block_sizes[block_id];
  const int eid = sel_block_eids[block_id];
  if (valid_cols <= 0) return;

  const int Mout = hidden_dim;
  const int Kdim = intermediate_dim;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ B = b_mlp2 ? (b_mlp2 + mat_index * (size_t)hidden_dim) : nullptr;

  const int base_row_group = tile_group * MLP2_BLOCK_ROWS;
  if (base_row_group >= Mout) return;

  const bool all_cols_active = (valid_cols == 32) && ((col0 + 31) < selections_M);

  __shared__ __align__(16) uint16_t sWbf16[MLP2_WAVES_PER_BLOCK * 32 * 8];
  __shared__ __align__(16) uint16_t sXbf16[8 * 32];
  __shared__ __align__(16) float sBias[MLP2_WAVES_PER_BLOCK * 32];
  __shared__ __align__(16) const hip_bfloat16* sGatePtrs[32];
  __shared__ __align__(16) int sSid[32];
  __shared__ __align__(16) float sSelW[32];
  __shared__ __align__(16) uint8_t sScaleCodes[MLP2_WAVES_PER_BLOCK * 32];
  short4v* __restrict__ sXvec = reinterpret_cast<short4v*>(sXbf16);

  const int rows_remaining = Mout - base_row_group;
  if (rows_remaining <= 0) return;
  const int row_tiles_available = (rows_remaining + 31) / 32;
  const int row_tiles = row_tiles_available < MLP2_WAVES_PER_BLOCK ? row_tiles_available : MLP2_WAVES_PER_BLOCK;
  const bool tile_active = wave_id < row_tiles;

  const int tile_m_global = tile_group * MLP2_WAVES_PER_BLOCK + wave_id;
  const int base_row = base_row_group + wave_id * 32;
  const bool full_row_tile = tile_active && ((base_row + 31) < Mout);

  uint16_t* __restrict__ sWbf16_tile = sWbf16 + wave_id * 32 * 8;
  uint8_t* __restrict__ sScaleCodes_tile = sScaleCodes + wave_id * 32;
  float* __restrict__ sBias_tile = sBias + wave_id * 32;
  short4v* __restrict__ sWvec_tile = reinterpret_cast<short4v*>(sWbf16_tile);

  if (wave_id == 0 && lane_in_wave < 32) {
    const int msel = col0 + lane_in_wave;
    const bool active = (lane_in_wave < valid_cols) && (msel < selections_M);
    sGatePtrs[lane_in_wave] = active ? (gate_up_all + (size_t)msel * (size_t)intermediate_dim) : nullptr;
    sSid[lane_in_wave] = active ? sel_sids[msel] : -1;
    sSelW[lane_in_wave] = active ? sel_w[msel] : 0.0f;
  }

  if (tile_active && lane_in_wave < 32) {
    const int row = base_row + lane_in_wave;
    sBias_tile[lane_in_wave] = (B && row < Mout) ? bfloat16_to_float(B[row]) : 0.f;
  }
  __syncthreads();

  float16v acc = {0.0f};

  const int tiles_m = (Mout + 31) / 32;
  const int tiles_k = (Kdim + 7) / 8;
  const size_t tiles_per_row = (size_t)tiles_k;
  const size_t tiles_per_mat = (size_t)tiles_m * tiles_per_row;
  const size_t tile_packet_bytes = MXFP4_TILE32X8_PACKET_BYTES;
  const uint8_t* __restrict__ Wtiles = w_tiles + mat_index * tiles_per_mat * tile_packet_bytes;

  for (int kk = 0; kk < tiles_k; ++kk) {
    const bool full_k_tile = ((kk + 1) * 8) <= Kdim;
    const bool fast_w_tile = full_row_tile && full_k_tile;
    const bool fast_x_tile = all_cols_active && full_k_tile;

    if (tile_active && lane_in_wave < 32) {
      const size_t tile_linear = (size_t)tile_m_global * tiles_per_row + (size_t)kk;
      const uint8_t* __restrict__ tile_base = Wtiles + tile_linear * tile_packet_bytes;
      const int row = base_row + lane_in_wave;
      uint8_t scale_code = static_cast<uint8_t>(0x7F);
      if (row < Mout) {
        scale_code = tile_base[lane_in_wave];
      }
      sScaleCodes_tile[lane_in_wave] = scale_code;
    }
    __syncthreads();

    if (tile_active) {
      const size_t tile_linear = (size_t)tile_m_global * tiles_per_row + (size_t)kk;
      const size_t tile_base_byte = tile_linear * tile_packet_bytes + MXFP4_TILE32X8_SCALE_BYTES;

      int idx4 = lane_in_wave * 4;
      if (idx4 < 256) {
        const int im = idx4 >> 3;
        const int ik_base = idx4 & 0x7;
        const int row = base_row + im;
        const int col_base = kk * 8 + ik_base;
        const size_t byte_off = tile_base_byte + static_cast<size_t>(idx4 >> 1);
        const uint16_t packed16 = *reinterpret_cast<const uint16_t*>(Wtiles + byte_off);
        const uint8_t scale_code = sScaleCodes_tile[im];
        const uint16_t* __restrict__ lut_row = MXFP4_DEQUANT_LUT_BF16[scale_code];
        short4v wvec = {0, 0, 0, 0};
        if (fast_w_tile) {
          #pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            const uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
            wvec[t4] = static_cast<short>(lut_row[nibble]);
          }
        } else {
          const bool row_valid = (row < Mout);
          #pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            const int col = col_base + t4;
            if (row_valid && col < Kdim) {
              const uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
              wvec[t4] = static_cast<short>(lut_row[nibble]);
            }
          }
        }
        sWvec_tile[im * 2 + (ik_base >> 2)] = wvec;
      }
    }

    if (wave_id == 0) {
      const int r = lane_in_wave % 8;
      const int c4 = (lane_in_wave / 8) * 4;
      const int k_row = kk * 8 + r;
      short4v xvec = {0, 0, 0, 0};
      const hip_bfloat16* x0 = sGatePtrs[c4 + 0];
      const hip_bfloat16* x1 = sGatePtrs[c4 + 1];
      const hip_bfloat16* x2 = sGatePtrs[c4 + 2];
      const hip_bfloat16* x3 = sGatePtrs[c4 + 3];
      if (fast_x_tile && x0 && x1 && x2 && x3) {
        xvec[0] = static_cast<short>(bfloat16_to_bits(x0[k_row]));
        xvec[1] = static_cast<short>(bfloat16_to_bits(x1[k_row]));
        xvec[2] = static_cast<short>(bfloat16_to_bits(x2[k_row]));
        xvec[3] = static_cast<short>(bfloat16_to_bits(x3[k_row]));
      } else {
        const bool row_valid = (k_row < Kdim);
        if (row_valid && x0) xvec[0] = static_cast<short>(bfloat16_to_bits(x0[k_row]));
        if (row_valid && x1) xvec[1] = static_cast<short>(bfloat16_to_bits(x1[k_row]));
        if (row_valid && x2) xvec[2] = static_cast<short>(bfloat16_to_bits(x2[k_row]));
        if (row_valid && x3) xvec[3] = static_cast<short>(bfloat16_to_bits(x3[k_row]));
      }
      sXvec[r * 8 + (c4 >> 2)] = xvec;
    }

    __syncthreads();

    if (tile_active) {
      const int kbase = lane_wave * 4;
      short4v Areg;
      short4v Breg;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kidx = kbase + j;
        const uint16_t a_bits = sWbf16_tile[(lane_col) * 8 + kidx];
        const uint16_t b_bits = sXbf16[kidx * 32 + lane_col];
        Areg[j] = static_cast<short>(a_bits);
        Breg[j] = static_cast<short>(b_bits);
      }
      acc = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(Areg, Breg, acc, 0, 0, 0);
    }

    __syncthreads();
  }

  if (tile_active && lane_col < valid_cols) {
    const int sid = sSid[lane_col];
    if (sid >= 0) {
      const float ew = sSelW[lane_col];
      float* out_slot = e_agg + (size_t)sid * (size_t)eagg_stride;
      #pragma unroll
      for (int g = 0; g < 16; ++g) {
        const int group = g >> 2;
        const int g_mod = g & 3;
        const int row_in_tile = ((group << 3) & 31) + lane_wave * 4 + g_mod;
        const int row = base_row + row_in_tile;
        if (row >= Mout) continue;
        float value = acc[g] + sBias_tile[row_in_tile];
        atomicAdd(&out_slot[row], ew * value);
      }
    }
  }
}

// ========================= FLOAT KERNELS =========================
// New kernels for float RunState buffers

// Batched RMSNorm (FP32): normalizes each slot in the batch using shared reductions.
__global__ __launch_bounds__(256) void rmsnorm_kernel_float_batched(
    float *__restrict__ out,
    const float *__restrict__ x,
    const hip_bfloat16 *__restrict__ weight,
    int hidden_dim,
    int batch,
    int stride,
    const int *slot_ids) {
  int b = blockIdx.y;
  if (b >= batch) {
    return;
  }

  int sid = slot_ids[b];
  const float *x_slot = x + (size_t)sid * (size_t)stride;
  float *out_slot = out + (size_t)sid * (size_t)stride;

  extern __shared__ float s_reduce[];

  float ss = 0.0f;
  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float v = x_slot[j];
    ss += v * v;
  }
  s_reduce[threadIdx.x] = ss;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + offset];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(s_reduce[0] / hidden_dim + 1e-5f);
  __syncthreads();

  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float w = bfloat16_to_float(weight[j]);
    out_slot[j] = x_slot[j] * inv_rms * w;
  }
}

// Pack weights từ BF16 row-major thành BF16 tiles 16x16 (no conversion)
__global__ void pack_bf16_rowmajor_to_bf16_tiles16_kernel(
    const hip_bfloat16 *src_bf16, int row_offset, int rows,
    int M, int K,
    hip_bfloat16 *dst_packed, int Mpad, int Kpad) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)rows * (size_t)K;
  if (idx >= total) return;

  int r = (int)(idx / K);
  int c = (int)(idx % K);
  int m = row_offset + r;

  int tile_m = m / 16;
  int tile_k = c / 16;
  int im = m % 16;
  int ik = c % 16;

  size_t tiles_per_row = (size_t)(Kpad + 15) / 16;
  size_t dst_index = ((size_t)tile_m * tiles_per_row + (size_t)tile_k) * (size_t)256 + (size_t)im * 16 + (size_t)ik;
  dst_packed[dst_index] = src_bf16[(size_t)r * (size_t)K + (size_t)c];
}

// Gather và pack input X từ multiple slots thành format BF16 cho MFMA
__global__ void gather_pack_x_fp32_to_bf16_mfma_kernel(const float *x, hip_bfloat16 *xpack,
                                                       int K, int B, int Bpad,
                                                       int in_stride, const int *slot_ids) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; // row in [0, K)
  int b = blockIdx.y; // batch index
  if (row < K && b < B) {
    int sid = slot_ids[b];
    const float *x_slot = x + (size_t)sid * (size_t)in_stride;
    xpack[(size_t)row * (size_t)Bpad + (size_t)b] = float_to_bfloat16(x_slot[row]);
  }
}

__global__ void gather_fp32_to_bf16_slots_kernel(const float *x, hip_bfloat16 *dst,
                                                 int K, int B,
                                                 int in_stride, const int *slot_ids) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (row < K && b < B) {
    int sid = slot_ids[b];
    const float *x_slot = x + (size_t)sid * (size_t)in_stride;
    hip_bfloat16 *dst_slot = dst + (size_t)sid * (size_t)in_stride;
    dst_slot[row] = float_to_bfloat16(x_slot[row]);
  }
}

// Mỗi block xử lý: 16 hàng (tile_m) × 16 cột batch (b_group)
// Grid.x = ceil(M/16), Grid.y = ceil(B/16)
__global__ __launch_bounds__(64)
void qkv_kernel(
    const hip_bfloat16* __restrict__ W_packed,    // [Mpad x Kpad] tile16 BF16
    const hip_bfloat16* __restrict__ Xpack,       // [K x Bpad] BF16 (cột = batch)
    float* __restrict__ dst,                      // scatter theo slot_ids
    const hip_bfloat16* __restrict__ bias,        // [M] hoặc nullptr
    int M, int K, int B, int Bpad,
    int out_stride,
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad)
{
  const int tx = threadIdx.x;     // cột trong tile (0..15) = cột batch
  const int ty = threadIdx.y;     // nhóm hàng (0..3)
  const int lane = ty * 16 + tx;  // 0..63

  const int tile_m  = blockIdx.x;          // tile dọc theo M
  const int b_group = blockIdx.y;          // nhóm 16 batch
  const int b_base  = b_group * 16;
  if (b_base >= B) return;

  const int cols_this = min(16, B - b_base);  // số cột hợp lệ trong nhóm này
  const int base_row = tile_m * 16;           // hàng cơ sở của tile M

  __shared__ __align__(16) uint16_t sW[2][16 * 16];
  __shared__ __align__(16) uint16_t sX[2][16 * 16];
  __shared__ int s_slot[16];
  __shared__ float s_bias_rows[16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  using short4v = __attribute__((ext_vector_type(4))) short;

  const bool has_bias = (bias != nullptr);
  const bool full_tile = (cols_this == 16);

  if (ty == 0) {
    const int col = tx;
    const int global_col = b_base + col;
    int slot_val = -1;
    if (col < cols_this && global_col < B) {
      slot_val = slot_ids[global_col];
    }
    s_slot[col] = slot_val;

    const int row = base_row + col;
    float bias_val = 0.0f;
    if (has_bias && row < M) {
      bias_val = bfloat16_to_float(bias[row]);
    }
    s_bias_rows[col] = bias_val;
  }
  __syncthreads();

  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int Ktiles = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)((Kpad + 15) / 16);
  const size_t tile_elems = 256;
  const hip_bfloat16* w_tile_row_base =
      W_packed + ((size_t)tile_m * tiles_per_row) * tile_elems;

  const size_t x_stride = (size_t)Bpad;
  const size_t b_base_sz = (size_t)b_base;

  {
    const int idx4 = lane * 4;
    if (idx4 < 256) {
      const hip_bfloat16* w_src = w_tile_row_base + idx4;
      const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
      uint32_t v0 = w_u32[0];
      uint32_t v1 = w_u32[1];
      uint16_t* dst_tile = &sW[0][0] + idx4;
      dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
      dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
      dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
      dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
    }

    const int row_in_tile = lane & 15;
    const int col_group   = lane >> 4;
    const int col_base    = col_group * 4;
    uint16_t vals[4] = {0, 0, 0, 0};
    const int k_row = row_in_tile;
    if (k_row < K) {
      const size_t row_offset = (size_t)k_row * x_stride + b_base_sz + (size_t)col_base;
      const hip_bfloat16* x_src = Xpack + row_offset;
      if (full_tile) {
        const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
        uint32_t xv0 = x_u32[0];
        uint32_t xv1 = x_u32[1];
        vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
        vals[1] = static_cast<uint16_t>(xv0 >> 16);
        vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
        vals[3] = static_cast<uint16_t>(xv1 >> 16);
      } else {
        const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
        if (col_base + 0 < cols_this) vals[0] = x_u16[0];
        if (col_base + 1 < cols_this) vals[1] = x_u16[1];
        if (col_base + 2 < cols_this) vals[2] = x_u16[2];
        if (col_base + 3 < cols_this) vals[3] = x_u16[3];
      }
    }
    uint16_t* dst_tile = &sX[0][row_in_tile * 16 + col_base];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst_tile[i] = vals[i];
    }
  }
  __syncthreads();

  int ping = 0;

  for (int kt = 0; kt < Ktiles; ++kt) {
    const int pong = ping ^ 1;

    if (kt + 1 < Ktiles) {
      const int idx4 = lane * 4;
      if (idx4 < 256) {
        const hip_bfloat16* w_src = w_tile_row_base + (size_t)(kt + 1) * tile_elems + idx4;
        const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
        uint32_t v0 = w_u32[0];
        uint32_t v1 = w_u32[1];
        uint16_t* dst_tile = &sW[pong][0] + idx4;
        dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
        dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
        dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
        dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
      }

      const int row_in_tile = lane & 15;
      const int col_group   = lane >> 4;
      const int col_base    = col_group * 4;
      uint16_t vals[4] = {0, 0, 0, 0};
      const int k_row = (kt + 1) * 16 + row_in_tile;
      if (k_row < K) {
        const size_t row_offset = (size_t)k_row * x_stride + b_base_sz + (size_t)col_base;
        const hip_bfloat16* x_src = Xpack + row_offset;
        if (full_tile) {
          const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
          uint32_t xv0 = x_u32[0];
          uint32_t xv1 = x_u32[1];
          vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
          vals[1] = static_cast<uint16_t>(xv0 >> 16);
          vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
          vals[3] = static_cast<uint16_t>(xv1 >> 16);
        } else {
          const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
          if (col_base + 0 < cols_this) vals[0] = x_u16[0];
          if (col_base + 1 < cols_this) vals[1] = x_u16[1];
          if (col_base + 2 < cols_this) vals[2] = x_u16[2];
          if (col_base + 3 < cols_this) vals[3] = x_u16[3];
        }
      }
      uint16_t* dst_tile = &sX[pong][row_in_tile * 16 + col_base];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        dst_tile[i] = vals[i];
      }
    }

    __syncthreads();

    short4v Areg;
    short4v Breg;
    const int kbase = ty * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kidx = kbase + j;
      Areg[j] = static_cast<short>(sW[ping][tx * 16 + kidx]);
      Breg[j] = static_cast<short>(sX[ping][kidx * 16 + tx]);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    ping = pong;
  }
  __syncthreads();

  const int r0 = base_row + ty * 4 + 0;
  const int r1 = base_row + ty * 4 + 1;
  const int r2 = base_row + ty * 4 + 2;
  const int r3 = base_row + ty * 4 + 3;

  if (tx < cols_this) {
    const int sid = s_slot[tx];
    if (sid >= 0) {
      float* __restrict__ out = dst + (size_t)sid * (size_t)out_stride;
      const int bias_base = ty * 4;
      if (r0 < M) out[r0] = acc[0] + s_bias_rows[bias_base + 0];
      if (r1 < M) out[r1] = acc[1] + s_bias_rows[bias_base + 1];
      if (r2 < M) out[r2] = acc[2] + s_bias_rows[bias_base + 2];
      if (r3 < M) out[r3] = acc[3] + s_bias_rows[bias_base + 3];
    }
  }
}

__global__ __launch_bounds__(64)
void o_kernel(
    const hip_bfloat16* __restrict__ W_packed,    // [Mpad x Kpad] tile16 BF16
    const hip_bfloat16* __restrict__ Xpack,       // [K x Bpad] BF16 (cột = batch)
    float* __restrict__ dst,                      // scatter theo slot_ids
    const hip_bfloat16* __restrict__ bias,        // [M] hoặc nullptr
    int M, int K, int B, int Bpad,
    int out_stride,
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad)
{
  const int tx = threadIdx.x;     // cột trong tile (0..15) = cột batch
  const int ty = threadIdx.y;     // nhóm hàng (0..3)
  const int lane = ty * 16 + tx;  // 0..63

  const int tile_m  = blockIdx.x;          // tile dọc theo M
  const int b_group = blockIdx.y;          // nhóm 16 batch
  const int b_base  = b_group * 16;
  if (b_base >= B) return;

  const int cols_this = min(16, B - b_base);  // số cột hợp lệ trong nhóm này
  const int base_row = tile_m * 16;           // hàng cơ sở của tile M

  __shared__ __align__(16) uint16_t sW[2][16 * 16];
  __shared__ __align__(16) uint16_t sX[2][16 * 16];
  __shared__ int s_slot[16];
  __shared__ float s_bias_rows[16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  using short4v = __attribute__((ext_vector_type(4))) short;

  const bool has_bias = (bias != nullptr);
  const bool full_tile = (cols_this == 16);

  if (ty == 0) {
    const int col = tx;
    const int global_col = b_base + col;
    int slot_val = -1;
    if (col < cols_this && global_col < B) {
      slot_val = slot_ids[global_col];
    }
    s_slot[col] = slot_val;

    const int row = base_row + col;
    float bias_val = 0.0f;
    if (has_bias && row < M) {
      bias_val = bfloat16_to_float(bias[row]);
    }
    s_bias_rows[col] = bias_val;
  }
  __syncthreads();

  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int Ktiles = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)((Kpad + 15) / 16);
  const size_t tile_elems = 256;
  const hip_bfloat16* w_tile_row_base =
      W_packed + ((size_t)tile_m * tiles_per_row) * tile_elems;

  const size_t x_stride = (size_t)Bpad;
  const size_t b_base_sz = (size_t)b_base;

  {
    const int idx4 = lane * 4;
    if (idx4 < 256) {
      const hip_bfloat16* w_src = w_tile_row_base + idx4;
      const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
      uint32_t v0 = w_u32[0];
      uint32_t v1 = w_u32[1];
      uint16_t* dst_tile = &sW[0][0] + idx4;
      dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
      dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
      dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
      dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
    }

    const int row_in_tile = lane & 15;
    const int col_group   = lane >> 4;
    const int col_base    = col_group * 4;
    uint16_t vals[4] = {0, 0, 0, 0};
    const int k_row = row_in_tile;
    if (k_row < K) {
      const size_t row_offset = (size_t)k_row * x_stride + b_base_sz + (size_t)col_base;
      const hip_bfloat16* x_src = Xpack + row_offset;
      if (full_tile) {
        const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
        uint32_t xv0 = x_u32[0];
        uint32_t xv1 = x_u32[1];
        vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
        vals[1] = static_cast<uint16_t>(xv0 >> 16);
        vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
        vals[3] = static_cast<uint16_t>(xv1 >> 16);
      } else {
        const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
        if (col_base + 0 < cols_this) vals[0] = x_u16[0];
        if (col_base + 1 < cols_this) vals[1] = x_u16[1];
        if (col_base + 2 < cols_this) vals[2] = x_u16[2];
        if (col_base + 3 < cols_this) vals[3] = x_u16[3];
      }
    }
    uint16_t* dst_tile = &sX[0][row_in_tile * 16 + col_base];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst_tile[i] = vals[i];
    }
  }
  __syncthreads();

  int ping = 0;

  for (int kt = 0; kt < Ktiles; ++kt) {
    const int pong = ping ^ 1;

    if (kt + 1 < Ktiles) {
      const int idx4 = lane * 4;
      if (idx4 < 256) {
        const hip_bfloat16* w_src = w_tile_row_base + (size_t)(kt + 1) * tile_elems + idx4;
        const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
        uint32_t v0 = w_u32[0];
        uint32_t v1 = w_u32[1];
        uint16_t* dst_tile = &sW[pong][0] + idx4;
        dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
        dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
        dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
        dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
      }

      const int row_in_tile = lane & 15;
      const int col_group   = lane >> 4;
      const int col_base    = col_group * 4;
      uint16_t vals[4] = {0, 0, 0, 0};
      const int k_row = (kt + 1) * 16 + row_in_tile;
      if (k_row < K) {
        const size_t row_offset = (size_t)k_row * x_stride + b_base_sz + (size_t)col_base;
        const hip_bfloat16* x_src = Xpack + row_offset;
        if (full_tile) {
          const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
          uint32_t xv0 = x_u32[0];
          uint32_t xv1 = x_u32[1];
          vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
          vals[1] = static_cast<uint16_t>(xv0 >> 16);
          vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
          vals[3] = static_cast<uint16_t>(xv1 >> 16);
        } else {
          const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
          if (col_base + 0 < cols_this) vals[0] = x_u16[0];
          if (col_base + 1 < cols_this) vals[1] = x_u16[1];
          if (col_base + 2 < cols_this) vals[2] = x_u16[2];
          if (col_base + 3 < cols_this) vals[3] = x_u16[3];
        }
      }
      uint16_t* dst_tile = &sX[pong][row_in_tile * 16 + col_base];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        dst_tile[i] = vals[i];
      }
    }

    __syncthreads();

    short4v Areg;
    short4v Breg;
    const int kbase = ty * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kidx = kbase + j;
      Areg[j] = static_cast<short>(sW[ping][tx * 16 + kidx]);
      Breg[j] = static_cast<short>(sX[ping][kidx * 16 + tx]);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    ping = pong;
  }
  __syncthreads();

  const int r0 = base_row + ty * 4 + 0;
  const int r1 = base_row + ty * 4 + 1;
  const int r2 = base_row + ty * 4 + 2;
  const int r3 = base_row + ty * 4 + 3;

  if (tx < cols_this) {
    const int sid = s_slot[tx];
    if (sid >= 0) {
      float* __restrict__ out = dst + (size_t)sid * (size_t)out_stride;
      const int bias_base = ty * 4;
      if (r0 < M) out[r0] = acc[0] + s_bias_rows[bias_base + 0];
      if (r1 < M) out[r1] = acc[1] + s_bias_rows[bias_base + 1];
      if (r2 < M) out[r2] = acc[2] + s_bias_rows[bias_base + 2];
      if (r3 < M) out[r3] = acc[3] + s_bias_rows[bias_base + 3];
    }
  }
}

__global__ __launch_bounds__(64)
void router_kernel(
    const hip_bfloat16* __restrict__ W_packed,    // [Mpad x Kpad] tile16 BF16
    const hip_bfloat16* __restrict__ Xpack,       // [K x Bpad] BF16 (cột = batch)
    float* __restrict__ dst,                      // scatter theo slot_ids
    const hip_bfloat16* __restrict__ bias,        // [M] hoặc nullptr
    int M, int K, int B, int Bpad,
    int out_stride,
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad)
{
  const int tx = threadIdx.x;     // cột trong tile (0..15) = cột batch
  const int ty = threadIdx.y;     // nhóm hàng (0..3)
  const int lane = ty * 16 + tx;  // 0..63

  const int tile_m  = blockIdx.x;          // tile dọc theo M
  const int b_group = blockIdx.y;          // nhóm 16 batch
  const int b_base  = b_group * 16;
  if (b_base >= B) return;

  const int cols_this = min(16, B - b_base);  // số cột hợp lệ trong nhóm này
  const int base_row = tile_m * 16;           // hàng cơ sở của tile M

  __shared__ __align__(16) uint16_t sW[2][16 * 16];
  __shared__ __align__(16) uint16_t sX[2][16 * 16];
  __shared__ int s_slot[16];
  __shared__ float s_bias_rows[16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  using short4v = __attribute__((ext_vector_type(4))) short;

  const bool has_bias = (bias != nullptr);
  const bool full_tile = (cols_this == 16);

  if (ty == 0) {
    const int col = tx;
    const int global_col = b_base + col;
    int slot_val = -1;
    if (col < cols_this && global_col < B) {
      slot_val = slot_ids[global_col];
    }
    s_slot[col] = slot_val;

    const int row = base_row + col;
    float bias_val = 0.0f;
    if (has_bias && row < M) {
      bias_val = bfloat16_to_float(bias[row]);
    }
    s_bias_rows[col] = bias_val;
  }
  __syncthreads();

  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int Ktiles = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)((Kpad + 15) / 16);
  const size_t tile_elems = 256;
  const hip_bfloat16* w_tile_row_base =
      W_packed + ((size_t)tile_m * tiles_per_row) * tile_elems;

  const size_t x_stride = (size_t)Bpad;
  const size_t b_base_sz = (size_t)b_base;

  {
    const int idx4 = lane * 4;
    if (idx4 < 256) {
      const hip_bfloat16* w_src = w_tile_row_base + idx4;
      const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
      uint32_t v0 = w_u32[0];
      uint32_t v1 = w_u32[1];
      uint16_t* dst_tile = &sW[0][0] + idx4;
      dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
      dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
      dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
      dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
    }

    const int row_in_tile = lane & 15;
    const int col_group   = lane >> 4;
    const int col_base    = col_group * 4;
    uint16_t vals[4] = {0, 0, 0, 0};
    const int k_row = row_in_tile;
    if (k_row < K) {
      const size_t row_offset = (size_t)k_row * x_stride + b_base_sz + (size_t)col_base;
      const hip_bfloat16* x_src = Xpack + row_offset;
      if (full_tile) {
        const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
        uint32_t xv0 = x_u32[0];
        uint32_t xv1 = x_u32[1];
        vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
        vals[1] = static_cast<uint16_t>(xv0 >> 16);
        vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
        vals[3] = static_cast<uint16_t>(xv1 >> 16);
      } else {
        const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
        if (col_base + 0 < cols_this) vals[0] = x_u16[0];
        if (col_base + 1 < cols_this) vals[1] = x_u16[1];
        if (col_base + 2 < cols_this) vals[2] = x_u16[2];
        if (col_base + 3 < cols_this) vals[3] = x_u16[3];
      }
    }
    uint16_t* dst_tile = &sX[0][row_in_tile * 16 + col_base];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst_tile[i] = vals[i];
    }
  }
  __syncthreads();

  int ping = 0;

  for (int kt = 0; kt < Ktiles; ++kt) {
    const int pong = ping ^ 1;

    if (kt + 1 < Ktiles) {
      const int idx4 = lane * 4;
      if (idx4 < 256) {
        const hip_bfloat16* w_src = w_tile_row_base + (size_t)(kt + 1) * tile_elems + idx4;
        const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
        uint32_t v0 = w_u32[0];
        uint32_t v1 = w_u32[1];
        uint16_t* dst_tile = &sW[pong][0] + idx4;
        dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
        dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
        dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
        dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
      }

      const int row_in_tile = lane & 15;
      const int col_group   = lane >> 4;
      const int col_base    = col_group * 4;
      uint16_t vals[4] = {0, 0, 0, 0};
      const int k_row = (kt + 1) * 16 + row_in_tile;
      if (k_row < K) {
        const size_t row_offset = (size_t)k_row * x_stride + b_base_sz + (size_t)col_base;
        const hip_bfloat16* x_src = Xpack + row_offset;
        if (full_tile) {
          const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
          uint32_t xv0 = x_u32[0];
          uint32_t xv1 = x_u32[1];
          vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
          vals[1] = static_cast<uint16_t>(xv0 >> 16);
          vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
          vals[3] = static_cast<uint16_t>(xv1 >> 16);
        } else {
          const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
          if (col_base + 0 < cols_this) vals[0] = x_u16[0];
          if (col_base + 1 < cols_this) vals[1] = x_u16[1];
          if (col_base + 2 < cols_this) vals[2] = x_u16[2];
          if (col_base + 3 < cols_this) vals[3] = x_u16[3];
        }
      }
      uint16_t* dst_tile = &sX[pong][row_in_tile * 16 + col_base];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        dst_tile[i] = vals[i];
      }
    }

    __syncthreads();

    short4v Areg;
    short4v Breg;
    const int kbase = ty * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kidx = kbase + j;
      Areg[j] = static_cast<short>(sW[ping][tx * 16 + kidx]);
      Breg[j] = static_cast<short>(sX[ping][kidx * 16 + tx]);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    ping = pong;
  }
  __syncthreads();

  const int r0 = base_row + ty * 4 + 0;
  const int r1 = base_row + ty * 4 + 1;
  const int r2 = base_row + ty * 4 + 2;
  const int r3 = base_row + ty * 4 + 3;

  if (tx < cols_this) {
    const int sid = s_slot[tx];
    if (sid >= 0) {
      float* __restrict__ out = dst + (size_t)sid * (size_t)out_stride;
      const int bias_base = ty * 4;
      if (r0 < M) out[r0] = acc[0] + s_bias_rows[bias_base + 0];
      if (r1 < M) out[r1] = acc[1] + s_bias_rows[bias_base + 1];
      if (r2 < M) out[r2] = acc[2] + s_bias_rows[bias_base + 2];
      if (r3 < M) out[r3] = acc[3] + s_bias_rows[bias_base + 3];
    }
  }
}

__device__ __forceinline__ unsigned int float_bits(float v) {
  union { float f; unsigned int u; } cvt{v};
  return cvt.u;
}

__device__ __forceinline__ float bits_float(unsigned int u) {
  union { unsigned int u; float f; } cvt{u};
  return cvt.f;
}

__device__ __forceinline__ unsigned long long pack_argmax(float val, int idx) {
  return (static_cast<unsigned long long>(float_bits(val)) << 32) |
         static_cast<unsigned long long>(static_cast<unsigned int>(idx));
}

__device__ __forceinline__ void unpack_argmax(unsigned long long packed,
                                              float &val_out, int &idx_out) {
  unsigned int vbits = static_cast<unsigned int>(packed >> 32);
  unsigned int ibits = static_cast<unsigned int>(packed & 0xFFFFFFFFull);
  val_out = bits_float(vbits);
  idx_out = static_cast<int>(ibits);
}

__device__ __forceinline__ void atomic_update_argmax(unsigned long long *addr,
                                                     float val, int idx) {
  if (idx < 0) return;
  unsigned long long new_pack = pack_argmax(val, idx);
  unsigned long long old = *addr;
  while (true) {
    float old_val;
    int old_idx;
    unpack_argmax(old, old_val, old_idx);
    bool replace = false;
    if (old_idx < 0) {
      replace = true;
    } else if (val > old_val) {
      replace = true;
    } else if (val == old_val && idx < old_idx) {
      replace = true;
    }
    if (!replace) break;
    unsigned long long prev = atomicCAS(addr, old, new_pack);
    if (prev == old) break;
    old = prev;
  }
}

constexpr int kMfmaTileSize = 16;
constexpr int kLogitsTileM = 64;  // Tunable block height (multiple of 16)
constexpr int kLogitsTileN = 64;  // Tunable block width (multiple of 16)
static_assert(kLogitsTileM % kMfmaTileSize == 0, "kLogitsTileM must be divisible by 16");
static_assert(kLogitsTileN % kMfmaTileSize == 0, "kLogitsTileN must be divisible by 16");
constexpr int kLogitsRowTilesPerBlock = kLogitsTileM / kMfmaTileSize;
constexpr int kLogitsGroupsPerBlock = kLogitsTileN / kMfmaTileSize;
constexpr int kLogitsThreadsPerBlock = 64 * kLogitsRowTilesPerBlock;
static_assert(kLogitsRowTilesPerBlock > 0, "kLogitsTileM must be positive");

__global__ __launch_bounds__(kLogitsThreadsPerBlock)
void logits_kernel(
    const hip_bfloat16* __restrict__ W_packed,
    const hip_bfloat16* __restrict__ Xpack,
    float* __restrict__ dst,
    const hip_bfloat16* __restrict__ bias,
    int M, int K, int B, int Bpad,
    int out_stride,
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad,
    int tile_offset,
    int tile_count,
    int row_offset,
    unsigned long long *max_packed,
    int vocab_limit)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int wave_idx = threadIdx.z;
  const int lane = ty * 16 + tx;

  const int tile_local_base = blockIdx.x * kLogitsRowTilesPerBlock;
  const int tile_global_base = tile_offset + tile_local_base;
  const int tile_limit = tile_offset + tile_count;
  const int tile_global = tile_global_base + wave_idx;
  const bool tile_active = (tile_global < tile_limit);
  const int group_base = blockIdx.y * kLogitsGroupsPerBlock;
  const int base_row = tile_global * 16;
  const int row_tile_offset = wave_idx * 16;

  __shared__ __align__(16) uint16_t sW[2][kLogitsRowTilesPerBlock][16 * 16];
  __shared__ __align__(16) uint16_t sX[2][kLogitsGroupsPerBlock][16 * 16];
  __shared__ int s_slot[kLogitsGroupsPerBlock * 16];
  __shared__ float s_bias_rows[kLogitsTileM];

  using float4v = __attribute__((ext_vector_type(4))) float;
  using short4v = __attribute__((ext_vector_type(4))) short;

  const bool has_bias = (bias != nullptr);

  int b_base_arr[kLogitsGroupsPerBlock];
  size_t b_base_sz_arr[kLogitsGroupsPerBlock];
  int cols_this_arr[kLogitsGroupsPerBlock];
  bool full_tile_arr[kLogitsGroupsPerBlock];
  bool active_group[kLogitsGroupsPerBlock];

  #pragma unroll
  for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
    const int b_group = group_base + g;
    const int b_base = b_group * 16;
    const bool active = (b_base < B);
    active_group[g] = active;
    b_base_arr[g] = b_base;
    b_base_sz_arr[g] = static_cast<size_t>(b_base);
    int cols = 0;
    bool full = false;
    if (active) {
      cols = min(16, B - b_base);
      full = (cols == 16);
    }
    cols_this_arr[g] = cols;
    full_tile_arr[g] = full;
  }

  bool any_group_active = false;
  #pragma unroll
  for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
    any_group_active |= active_group[g];
  }
  if (!any_group_active) return;

  if (wave_idx == 0 && ty == 0) {
    const int col = tx;
    #pragma unroll
    for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
      int slot_val = -1;
      if (active_group[g] && col < cols_this_arr[g]) {
        const int global_col = b_base_arr[g] + col;
        if (global_col < B) {
          slot_val = slot_ids[global_col];
        }
      }
      s_slot[g * 16 + col] = slot_val;
    }
  }

  if (ty == 0 && tile_active) {
    const int row = base_row + tx;
    float bias_val = 0.0f;
    if (has_bias && row < M) {
      bias_val = bfloat16_to_float(bias[row]);
    }
    s_bias_rows[row_tile_offset + tx] = bias_val;
  }
  __syncthreads();

  float4v accum[kLogitsGroupsPerBlock];
  #pragma unroll
  for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
    accum[g] = {0.f, 0.f, 0.f, 0.f};
  }

  const int Ktiles = (Kpad + 15) / 16;
  const size_t tiles_per_row = static_cast<size_t>((Kpad + 15) / 16);
  const size_t tile_elems = 256;
  const size_t tile_stride = tiles_per_row * tile_elems;
  const hip_bfloat16* w_tile_row_base =
      tile_active ? W_packed + (size_t)tile_global * tile_stride : nullptr;

  const size_t x_stride = static_cast<size_t>(Bpad);

  {
    const int idx4 = lane * 4;
    if (tile_active && idx4 < static_cast<int>(tile_elems)) {
      const hip_bfloat16* w_src = w_tile_row_base + idx4;
      const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
      uint32_t v0 = w_u32[0];
      uint32_t v1 = w_u32[1];
      uint16_t* dst_tile = &sW[0][wave_idx][0] + idx4;
      dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
      dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
      dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
      dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
    }

    if (wave_idx == 0) {
      const int row_in_tile = lane & 15;
      const int col_group = lane >> 4;
      const int col_base = col_group * 4;

      #pragma unroll
      for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
        uint16_t vals[4] = {0, 0, 0, 0};
        if (active_group[g]) {
          const int k_row = row_in_tile;
          if (k_row < K) {
            const size_t row_offset = static_cast<size_t>(k_row) * x_stride +
                                      b_base_sz_arr[g] + static_cast<size_t>(col_base);
            const hip_bfloat16* x_src = Xpack + row_offset;
            if (full_tile_arr[g]) {
              const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
              uint32_t xv0 = x_u32[0];
              uint32_t xv1 = x_u32[1];
              vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
              vals[1] = static_cast<uint16_t>(xv0 >> 16);
              vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
              vals[3] = static_cast<uint16_t>(xv1 >> 16);
            } else {
              const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
              if (col_base + 0 < cols_this_arr[g]) vals[0] = x_u16[0];
              if (col_base + 1 < cols_this_arr[g]) vals[1] = x_u16[1];
              if (col_base + 2 < cols_this_arr[g]) vals[2] = x_u16[2];
              if (col_base + 3 < cols_this_arr[g]) vals[3] = x_u16[3];
            }
          }
        }
        uint16_t* dst_tile = &sX[0][g][row_in_tile * 16 + col_base];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          dst_tile[i] = vals[i];
        }
      }
    }
  }
  __syncthreads();

  int ping = 0;

  for (int kt = 0; kt < Ktiles; ++kt) {
    const int pong = ping ^ 1;

    if (kt + 1 < Ktiles) {
      const int idx4 = lane * 4;
      if (tile_active && idx4 < static_cast<int>(tile_elems)) {
        const hip_bfloat16* w_src = w_tile_row_base +
                                    (size_t)(kt + 1) * tile_elems + idx4;
        const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_src);
        uint32_t v0 = w_u32[0];
        uint32_t v1 = w_u32[1];
        uint16_t* dst_tile = &sW[pong][wave_idx][0] + idx4;
        dst_tile[0] = static_cast<uint16_t>(v0 & 0xFFFFu);
        dst_tile[1] = static_cast<uint16_t>(v0 >> 16);
        dst_tile[2] = static_cast<uint16_t>(v1 & 0xFFFFu);
        dst_tile[3] = static_cast<uint16_t>(v1 >> 16);
      }

      if (wave_idx == 0) {
        const int row_in_tile = lane & 15;
        const int col_group = lane >> 4;
        const int col_base = col_group * 4;

        #pragma unroll
        for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
          uint16_t vals[4] = {0, 0, 0, 0};
          if (active_group[g]) {
            const int k_row = (kt + 1) * 16 + row_in_tile;
            if (k_row < K) {
              const size_t row_offset = static_cast<size_t>(k_row) * x_stride +
                                        b_base_sz_arr[g] + static_cast<size_t>(col_base);
              const hip_bfloat16* x_src = Xpack + row_offset;
              if (full_tile_arr[g]) {
                const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_src);
                uint32_t xv0 = x_u32[0];
                uint32_t xv1 = x_u32[1];
                vals[0] = static_cast<uint16_t>(xv0 & 0xFFFFu);
                vals[1] = static_cast<uint16_t>(xv0 >> 16);
                vals[2] = static_cast<uint16_t>(xv1 & 0xFFFFu);
                vals[3] = static_cast<uint16_t>(xv1 >> 16);
              } else {
                const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x_src);
                if (col_base + 0 < cols_this_arr[g]) vals[0] = x_u16[0];
                if (col_base + 1 < cols_this_arr[g]) vals[1] = x_u16[1];
                if (col_base + 2 < cols_this_arr[g]) vals[2] = x_u16[2];
                if (col_base + 3 < cols_this_arr[g]) vals[3] = x_u16[3];
              }
            }
          }
          uint16_t* dst_tile = &sX[pong][g][row_in_tile * 16 + col_base];
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            dst_tile[i] = vals[i];
          }
        }
      }
    }

    __syncthreads();

    if (tile_active && any_group_active) {
      short4v Areg;
      const int kbase = ty * 4;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kidx = kbase + j;
        Areg[j] = static_cast<short>(sW[ping][wave_idx][tx * 16 + kidx]);
      }

      #pragma unroll
      for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
        if (!active_group[g]) continue;
        short4v Breg;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int kidx = kbase + j;
          Breg[j] = static_cast<short>(sX[ping][g][kidx * 16 + tx]);
        }
        accum[g] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, accum[g], 0, 0, 0);
      }
    }

    ping = pong;
  }
  __syncthreads();

  const int r0_global = base_row + ty * 4 + 0;
  const int r1_global = base_row + ty * 4 + 1;
  const int r2_global = base_row + ty * 4 + 2;
  const int r3_global = base_row + ty * 4 + 3;
  const int r0_local = r0_global - row_offset;
  const int r1_local = r1_global - row_offset;
  const int r2_local = r2_global - row_offset;
  const int r3_local = r3_global - row_offset;

  const int bias_base = row_tile_offset + ty * 4;

  for (int g = 0; g < kLogitsGroupsPerBlock; ++g) {
    const bool active = tile_active && active_group[g] && cols_this_arr[g] > 0;
    float lane_best_val = -FLT_MAX;
    int lane_best_idx = -1;
    float4v out_vals = {0.f, 0.f, 0.f, 0.f};

    if (active) {
      float4v bias_vec;
      bias_vec[0] = s_bias_rows[bias_base + 0];
      bias_vec[1] = s_bias_rows[bias_base + 1];
      bias_vec[2] = s_bias_rows[bias_base + 2];
      bias_vec[3] = s_bias_rows[bias_base + 3];
      out_vals = accum[g] + bias_vec;

      if (r0_global >= 0 && r0_global < vocab_limit) {
        lane_best_val = out_vals[0];
        lane_best_idx = r0_global;
      }
      if (r1_global >= 0 && r1_global < vocab_limit) {
        const float cand_val = out_vals[1];
        const int cand_idx = r1_global;
        if (lane_best_idx < 0 || cand_val > lane_best_val ||
            (cand_val == lane_best_val && cand_idx < lane_best_idx)) {
          lane_best_val = cand_val;
          lane_best_idx = cand_idx;
        }
      }
      if (r2_global >= 0 && r2_global < vocab_limit) {
        const float cand_val = out_vals[2];
        const int cand_idx = r2_global;
        if (lane_best_idx < 0 || cand_val > lane_best_val ||
            (cand_val == lane_best_val && cand_idx < lane_best_idx)) {
          lane_best_val = cand_val;
          lane_best_idx = cand_idx;
        }
      }
      if (r3_global >= 0 && r3_global < vocab_limit) {
        const float cand_val = out_vals[3];
        const int cand_idx = r3_global;
        if (lane_best_idx < 0 || cand_val > lane_best_val ||
            (cand_val == lane_best_val && cand_idx < lane_best_idx)) {
          lane_best_val = cand_val;
          lane_best_idx = cand_idx;
        }
      }
    }

    if (active && tx < cols_this_arr[g]) {
      const int sid = s_slot[g * 16 + tx];
      const int global_col = b_base_arr[g] + tx;

      if (max_packed != nullptr && global_col < B) {
        float best_val_tile = lane_best_val;
        int best_idx_tile = lane_best_idx;

        const float peer_val16 = __shfl_down(best_val_tile, 16, 64);
        const int peer_idx16 = __shfl_down(best_idx_tile, 16, 64);
        if (ty == 0) {
          if (peer_idx16 >= 0 && (best_idx_tile < 0 || peer_val16 > best_val_tile ||
                                  (peer_val16 == best_val_tile && peer_idx16 < best_idx_tile))) {
            best_val_tile = peer_val16;
            best_idx_tile = peer_idx16;
          }
        }

        const float peer_val32 = __shfl_down(lane_best_val, 32, 64);
        const int peer_idx32 = __shfl_down(lane_best_idx, 32, 64);
        if (ty == 0) {
          if (peer_idx32 >= 0 && (best_idx_tile < 0 || peer_val32 > best_val_tile ||
                                  (peer_val32 == best_val_tile && peer_idx32 < best_idx_tile))) {
            best_val_tile = peer_val32;
            best_idx_tile = peer_idx32;
          }
        }

        const float peer_val48 = __shfl_down(lane_best_val, 48, 64);
        const int peer_idx48 = __shfl_down(lane_best_idx, 48, 64);
        if (ty == 0) {
          if (peer_idx48 >= 0 && (best_idx_tile < 0 || peer_val48 > best_val_tile ||
                                  (peer_val48 == best_val_tile && peer_idx48 < best_idx_tile))) {
            best_val_tile = peer_val48;
            best_idx_tile = peer_idx48;
          }
          if (best_idx_tile >= 0) {
            unsigned long long *slot_max = max_packed + global_col;
            atomic_update_argmax(slot_max, best_val_tile, best_idx_tile);
          }
        }
      }

      if (dst != nullptr && sid >= 0) {
        float* __restrict__ out = dst + (size_t)sid * (size_t)out_stride;
        if (r0_global < M && r0_local >= 0 && r0_local < out_stride)
          out[r0_local] = out_vals[0];
        if (r1_global < M && r1_local >= 0 && r1_local < out_stride)
          out[r1_local] = out_vals[1];
        if (r2_global < M && r2_local >= 0 && r2_local < out_stride)
          out[r2_local] = out_vals[2];
        if (r3_global < M && r3_local >= 0 && r3_local < out_stride)
          out[r3_local] = out_vals[3];
      }
    }
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

__global__ void add_selected_kernel_float(float *dst, const float *src, int size,
                                           int batch, const int *slot_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < size && b < batch) {
    int sid = slot_ids[b];
    size_t offset = (size_t)sid * (size_t)size + (size_t)i;
    dst[offset] += src[offset];
  }
}

__global__ void pack_selected_rows_kernel_float(const float *src, float *dst,
                                                int hidden_dim, int batch,
                                                int stride, const int *slot_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i < hidden_dim && b < batch) {
    int sid = slot_ids[b];
    float *dst_row = dst + (size_t)b * (size_t)hidden_dim;
    if (sid < 0) {
      dst_row[i] = 0.0f;
      return;
    }
    const float *src_row = src + (size_t)sid * (size_t)stride;
    dst_row[i] = src_row[i];
  }
}

__global__ void add_rows_kernel_float(float *dst, const float *src,
                                      int hidden_dim, int rows,
                                      int stride, const int *slot_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y;
  if (i < hidden_dim && r < rows) {
    int sid = slot_ids[r];
    if (sid < 0) return;
    float *dst_row = dst + (size_t)sid * (size_t)stride;
    const float *src_row = src + (size_t)r * (size_t)hidden_dim;
    dst_row[i] += src_row[i];
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

  constexpr int kMax = 8;
  if (k <= 0) return;

  int sid = slot_ids[b];
  const float *scores = router_scores + (size_t)sid * (size_t)score_stride;
  float *topv_s = topv + (size_t)sid * (size_t)topk_stride;
  int   *topi_s = topi + (size_t)sid * (size_t)topk_stride;

  if (k > kMax) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < k; ++i) { topv_s[i] = -INFINITY; topi_s[i] = -1; }
      for (int j = 0; j < n; ++j) {
        float v = scores[j];
        int pos = -1;
        for (int t = 0; t < k; ++t) { if (v > topv_s[t]) { pos = t; break; } }
        if (pos >= 0) {
          for (int t = k - 1; t > pos; --t) {
            topv_s[t] = topv_s[t - 1];
            topi_s[t] = topi_s[t - 1];
          }
          topv_s[pos] = v;
          topi_s[pos] = j;
        }
      }
      float maxv = topv_s[0];
      for (int i = 1; i < k; ++i) maxv = fmaxf(maxv, topv_s[i]);
      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        float ev = (topv_s[i] == -INFINITY) ? 0.0f : __expf(topv_s[i] - maxv);
        topv_s[i] = ev;
        sum += ev;
      }
      float inv = 1.0f / (sum + 1e-8f);
      for (int i = 0; i < k; ++i) topv_s[i] *= inv;
    }
    return;
  }

  int tid = threadIdx.x;
  float local_v[kMax];
  int local_i[kMax];
  for (int i = 0; i < k; ++i) {
    local_v[i] = -INFINITY;
    local_i[i] = -1;
  }

  for (int idx = tid; idx < n; idx += blockDim.x) {
    float v = scores[idx];
    int pos = -1;
    for (int t = 0; t < k; ++t) {
      if (v > local_v[t]) { pos = t; break; }
    }
    if (pos >= 0) {
      for (int t = k - 1; t > pos; --t) {
        local_v[t] = local_v[t - 1];
        local_i[t] = local_i[t - 1];
      }
      local_v[pos] = v;
      local_i[pos] = idx;
    }
  }

  extern __shared__ unsigned char smem[];
  float *shared_vals = reinterpret_cast<float*>(smem);
  int *shared_idx = reinterpret_cast<int*>(shared_vals + (size_t)blockDim.x * (size_t)k);

  for (int t = 0; t < k; ++t) {
    size_t write = (size_t)tid * (size_t)k + (size_t)t;
    if (write < (size_t)blockDim.x * (size_t)k) {
      shared_vals[write] = local_v[t];
      shared_idx[write] = local_i[t];
    }
  }
  __syncthreads();

  if (tid == 0) {
    for (int i = 0; i < k; ++i) {
      topv_s[i] = -INFINITY;
      topi_s[i] = -1;
    }
    int total = blockDim.x * k;
    for (int idx = 0; idx < total; ++idx) {
      float v = shared_vals[idx];
      int id = shared_idx[idx];
      if (id < 0) continue;
      int pos = -1;
      for (int t = 0; t < k; ++t) {
        if (v > topv_s[t]) { pos = t; break; }
      }
      if (pos >= 0) {
        for (int t = k - 1; t > pos; --t) {
          topv_s[t] = topv_s[t - 1];
          topi_s[t] = topi_s[t - 1];
        }
        topv_s[pos] = v;
        topi_s[pos] = id;
      }
    }

    float maxv = topv_s[0];
    for (int i = 1; i < k; ++i) {
      maxv = fmaxf(maxv, topv_s[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      float ev = (topv_s[i] == -INFINITY) ? 0.0f : __expf(topv_s[i] - maxv);
      topv_s[i] = ev;
      sum += ev;
    }
    float inv = 1.0f / (sum + 1e-8f);
    for (int i = 0; i < k; ++i) {
      topv_s[i] *= inv;
    }
  }
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

// Split + SwiGLU batched for all selections
__global__ void swiglu_grouped_kernel_bf16(hip_bfloat16 *gate_up_all, const float *mlp1_out_all,
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
    gate_up_all[(size_t)m * (size_t)intermediate_dim + (size_t)i] = float_to_bfloat16(v);
  }
}

// Block-wide reductions implemented with dynamic shared memory scratch.
__device__ inline float block_reduce_max(float val, float *shared) {
  int tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
    }
    __syncthreads();
  }
  float result = shared[0];
  __syncthreads();
  return result;
}

__device__ inline float block_reduce_sum(float val, float *shared) {
  int tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }
  float result = shared[0];
  __syncthreads();
  return result;
}

// ========= FlashAttention v2 (drop-in, tối ưu vi mô) =========
#ifndef FA_V2_BK
#define FA_V2_BK 64  // BK=64 thường hợp lý cho head_dim 64..128 (có thể 32/128 tùy profile)
#endif

// Gợi ý lượng shared mem để set khi launch (giữ nguyên chữ ký cũ)
static inline size_t flash_attention_v2_smem_bytes(int head_dim, int BK = FA_V2_BK) {
  size_t bytes = 0;
  // q_sh: [head_dim] float
  bytes += (size_t)head_dim * sizeof(float);
  // p_sh: [BK] float
  bytes += (size_t)BK * sizeof(float);
  // vptr_sh: [BK] hip_bfloat16*
  bytes += (size_t)BK * sizeof(hip_bfloat16*);
  // padding nhỏ
  bytes += 64;
  return bytes;
}

// Chỉ “quây” fast contract cho khu vực kernel này
#if defined(__clang__)
#pragma clang fp contract(fast)
#endif

__global__ __launch_bounds__(64)
void flash_attention_sliding(
    const float * __restrict__ all_q,
    KVCacheDeviceData * __restrict__ cache_data,
    const hip_bfloat16 * __restrict__ attn_sinks,
    float * __restrict__ all_output,
    int head_dim, int /*kv_dim_unused*/, int kv_mul,
    int n_heads, int seq_len, int layer,
    const int * __restrict__ pos_list,
    int batch,
    int q_stride,
    const int * __restrict__ slot_ids) {

  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (h >= n_heads || b >= batch) return;

  const int lane = threadIdx.x;          // 0..63 (wave64)
  const int sid  = slot_ids[b];
  int       pos  = pos_list[b];
  if (pos < 0) return;

  // KV-head base (GQA): mỗi kv-head “phục vụ” kv_mul q-heads
  const int head_base = (h / max(1, kv_mul)) * head_dim;

  const float * __restrict__ q_head =
      all_q + (size_t)sid * (size_t)q_stride + (size_t)h * (size_t)head_dim;
  float * __restrict__ out_head =
      all_output + (size_t)sid * (size_t)(n_heads * head_dim) + (size_t)h * (size_t)head_dim;

  // Range attention theo cache manager (full/sliding window)
  int start_pos = 0, end_pos = 0;
  cache_data->get_attention_range(sid, layer, pos, &start_pos, &end_pos);
  int att_len = end_pos - start_pos + 1;

  // -------------------- Dynamic shared memory --------------------
  extern __shared__ unsigned char smem_raw[];
  unsigned char* p = smem_raw;

  // q_sh: [head_dim] float (đÃ scale 1/sqrt(d) sẵn)
  float* q_sh = reinterpret_cast<float*>(p);
  p += (size_t)head_dim * sizeof(float);

  // p_sh: [BK] float (softmax weight block hiện tại)
  float* p_sh = reinterpret_cast<float*>(p);
  p += (size_t)FA_V2_BK * sizeof(float);

  // vptr_sh: [BK] hip_bfloat16* (V pointer cho từng token trong block)
  uintptr_t align = alignof(hip_bfloat16*);
  p = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(p) + align - 1) & ~(align - 1));
  hip_bfloat16** vptr_sh = reinterpret_cast<hip_bfloat16**>(p);
  // ----------------------------------------------------------------

  // Không có KV khả dụng → trả 0
  if (att_len <= 0) {
    for (int i = lane; i < head_dim; i += blockDim.x) out_head[i] = 0.f;
    return;
  }

  // Tiền-scale Q để giảm 1 nhân / score
  const float inv_sqrt_d = rsqrtf((float)head_dim);
  for (int i = lane; i < head_dim; i += blockDim.x) {
    q_sh[i] = q_head[i] * inv_sqrt_d;
  }
  __syncthreads();

  // Online softmax state
  float m = -INFINITY;  // running max
  float l = 0.f;        // running sum(exp(s - m))

  // Zero hoá output tích luỹ (y)
  for (int i = lane; i < head_dim; i += blockDim.x) {
    out_head[i] = 0.f;
  }

  // Duyệt theo block K/V dài BK
  for (int base = 0; base < att_len; base += FA_V2_BK) {
    const int valid = min(FA_V2_BK, att_len - base);

    // Mỗi thread ánh xạ 1 token trong block để tính score
    float s_val = -INFINITY;     // score cho token của thread
    hip_bfloat16* v_ptr_this = nullptr;

    if (lane < valid) {
      const int t = start_pos + base + lane;
      const int cp = (t <= pos) ? cache_data->get_cache_position(sid, layer, t, pos) : -1;
      if (cp >= 0) {
        const hip_bfloat16* k_base = cache_data->get_key_token_ptr(sid, layer, cp);
        hip_bfloat16* v_base = cache_data->get_value_token_ptr(sid, layer, cp);
        if (k_base) k_base += head_base;
        if (v_base) v_base += head_base;
        v_ptr_this = v_base;

        // Dot(Q, K_t): unroll theo 8 (2×bf16x4) + FMA
        float dot = 0.f;
        int k = 0;
        for (; k + 7 < head_dim; k += 8) {
          float4v q4a = {q_sh[k+0], q_sh[k+1], q_sh[k+2], q_sh[k+3]};
          float4v k4a = load_bf16x4_as_float4v(k_base + k);
          dot += q4a[0]*k4a[0] + q4a[1]*k4a[1] + q4a[2]*k4a[2] + q4a[3]*k4a[3];

          float4v q4b = {q_sh[k+4], q_sh[k+5], q_sh[k+6], q_sh[k+7]};
          float4v k4b = load_bf16x4_as_float4v(k_base + k + 4);
          dot += q4b[0]*k4b[0] + q4b[1]*k4b[1] + q4b[2]*k4b[2] + q4b[3]*k4b[3];
        }
        for (; k < head_dim; ++k) {
          dot = fmaf(q_sh[k], bfloat16_to_float(k_base[k]), dot);
        }
        s_val = dot;  // q đã scale → không nhân inv_sqrt_d nữa
      }
    }

    // Block-wide max(s_val) qua wave64
    float blk_max = s_val;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      blk_max = fmaxf(blk_max, __shfl_down(blk_max, off, 64));
    }
    blk_max = __shfl(blk_max, 0, 64);

    // Cập nhật m/l (online softmax)
    const float m_new    = fmaxf(m, blk_max);
    const float scale_old= __expf(m - m_new);

    // p_val = exp(s - m_new), lưu vào shared
    if (lane < valid) {
      const float p_val = (s_val == -INFINITY) ? 0.f : __expf(s_val - m_new);
      p_sh[lane]    = p_val;
      vptr_sh[lane] = v_ptr_this;
    }
    __syncthreads();

    // Tổng p trong block
    float sum_p = (lane < valid) ? p_sh[lane] : 0.f;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      sum_p += __shfl_down(sum_p, off, 64);
    }
    sum_p = __shfl(sum_p, 0, 64);

    // y = y*scale_old + Σ_j p_j * V_j  (đọc V theo i, coalesced)
    for (int i = lane; i < head_dim; i += blockDim.x) {
      float yi = out_head[i] * scale_old;
      #pragma unroll 4
      for (int j = 0; j < valid; ++j) {
        const float pw = p_sh[j];
        hip_bfloat16* vj = vptr_sh[j];
        if (pw > 0.f && vj) {
          yi = fmaf(pw, bfloat16_to_float(vj[i]), yi);
        }
      }
      out_head[i] = yi;
    }

    l = fmaf(l, scale_old, sum_p);
    m = m_new;
    __syncthreads();
  }

  // Attention sink: chỉ cộng vào mẫu số (không cộng giá trị); giữ ổn định online-softmax
  const int layer_cap = cache_data->layer_capacity(layer);
  const bool use_sink = ((pos + 1) < seq_len) && (att_len < layer_cap + 1);
  if (use_sink) {
    const size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    const float sink_val = bfloat16_to_float(attn_sinks[sink_idx]);
    const float m_new    = fmaxf(m, sink_val);
    const float scale_old= __expf(m - m_new);

    for (int i = lane; i < head_dim; i += blockDim.x) {
      out_head[i] *= scale_old;  // không cộng thêm gì vì sink không có V
    }
    l = fmaf(l, scale_old, __expf(sink_val - m_new));
    m = m_new;
    __syncthreads();
  }

  // Chuẩn hoá cuối: y /= (l + eps)
  const float inv_l = 1.f / (l + 1e-8f);
  for (int i = lane; i < head_dim; i += blockDim.x) {
    out_head[i] *= inv_l;
  }
}


__global__ __launch_bounds__(64)
void flash_attention_full(
    const float * __restrict__ all_q,
    KVCacheDeviceData * __restrict__ cache_data,
    const hip_bfloat16 * __restrict__ attn_sinks,
    float * __restrict__ all_output,
    int head_dim, int /*kv_dim_unused*/, int kv_mul,
    int n_heads, int seq_len, int layer,
    const int * __restrict__ pos_list,
    int batch,
    int q_stride,
    const int * __restrict__ slot_ids) {

  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (h >= n_heads || b >= batch) return;

  const int lane = threadIdx.x;          // 0..63 (wave64)
  const int sid  = slot_ids[b];
  int       pos  = pos_list[b];
  if (pos < 0) return;

  // KV-head base (GQA): mỗi kv-head “phục vụ” kv_mul q-heads
  const int head_base = (h / max(1, kv_mul)) * head_dim;

  const float * __restrict__ q_head =
      all_q + (size_t)sid * (size_t)q_stride + (size_t)h * (size_t)head_dim;
  float * __restrict__ out_head =
      all_output + (size_t)sid * (size_t)(n_heads * head_dim) + (size_t)h * (size_t)head_dim;

  // Range attention theo cache manager (full/sliding window)
  int start_pos = 0, end_pos = 0;
  cache_data->get_attention_range(sid, layer, pos, &start_pos, &end_pos);
  int att_len = end_pos - start_pos + 1;

  // -------------------- Dynamic shared memory --------------------
  extern __shared__ unsigned char smem_raw[];
  unsigned char* p = smem_raw;

  // q_sh: [head_dim] float (đÃ scale 1/sqrt(d) sẵn)
  float* q_sh = reinterpret_cast<float*>(p);
  p += (size_t)head_dim * sizeof(float);

  // p_sh: [BK] float (softmax weight block hiện tại)
  float* p_sh = reinterpret_cast<float*>(p);
  p += (size_t)FA_V2_BK * sizeof(float);

  // vptr_sh: [BK] hip_bfloat16* (V pointer cho từng token trong block)
  uintptr_t align = alignof(hip_bfloat16*);
  p = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(p) + align - 1) & ~(align - 1));
  hip_bfloat16** vptr_sh = reinterpret_cast<hip_bfloat16**>(p);
  // ----------------------------------------------------------------

  // Không có KV khả dụng → trả 0
  if (att_len <= 0) {
    for (int i = lane; i < head_dim; i += blockDim.x) out_head[i] = 0.f;
    return;
  }

  // Tiền-scale Q để giảm 1 nhân / score
  const float inv_sqrt_d = rsqrtf((float)head_dim);
  for (int i = lane; i < head_dim; i += blockDim.x) {
    q_sh[i] = q_head[i] * inv_sqrt_d;
  }
  __syncthreads();

  // Online softmax state
  float m = -INFINITY;  // running max
  float l = 0.f;        // running sum(exp(s - m))

  // Zero hoá output tích luỹ (y)
  for (int i = lane; i < head_dim; i += blockDim.x) {
    out_head[i] = 0.f;
  }

  // Duyệt theo block K/V dài BK
  for (int base = 0; base < att_len; base += FA_V2_BK) {
    const int valid = min(FA_V2_BK, att_len - base);

    // Mỗi thread ánh xạ 1 token trong block để tính score
    float s_val = -INFINITY;     // score cho token của thread
    hip_bfloat16* v_ptr_this = nullptr;

    if (lane < valid) {
      const int t = start_pos + base + lane;
      const int cp = (t <= pos) ? cache_data->get_cache_position(sid, layer, t, pos) : -1;
      if (cp >= 0) {
        const hip_bfloat16* k_base = cache_data->get_key_token_ptr(sid, layer, cp);
        hip_bfloat16* v_base = cache_data->get_value_token_ptr(sid, layer, cp);
        if (k_base) k_base += head_base;
        if (v_base) v_base += head_base;
        v_ptr_this = v_base;

        // Dot(Q, K_t): unroll theo 8 (2×bf16x4) + FMA
        float dot = 0.f;
        int k = 0;
        for (; k + 7 < head_dim; k += 8) {
          float4v q4a = {q_sh[k+0], q_sh[k+1], q_sh[k+2], q_sh[k+3]};
          float4v k4a = load_bf16x4_as_float4v(k_base + k);
          dot += q4a[0]*k4a[0] + q4a[1]*k4a[1] + q4a[2]*k4a[2] + q4a[3]*k4a[3];

          float4v q4b = {q_sh[k+4], q_sh[k+5], q_sh[k+6], q_sh[k+7]};
          float4v k4b = load_bf16x4_as_float4v(k_base + k + 4);
          dot += q4b[0]*k4b[0] + q4b[1]*k4b[1] + q4b[2]*k4b[2] + q4b[3]*k4b[3];
        }
        for (; k < head_dim; ++k) {
          dot = fmaf(q_sh[k], bfloat16_to_float(k_base[k]), dot);
        }
        s_val = dot;  // q đã scale → không nhân inv_sqrt_d nữa
      }
    }

    // Block-wide max(s_val) qua wave64
    float blk_max = s_val;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      blk_max = fmaxf(blk_max, __shfl_down(blk_max, off, 64));
    }
    blk_max = __shfl(blk_max, 0, 64);

    // Cập nhật m/l (online softmax)
    const float m_new    = fmaxf(m, blk_max);
    const float scale_old= __expf(m - m_new);

    // p_val = exp(s - m_new), lưu vào shared
    if (lane < valid) {
      const float p_val = (s_val == -INFINITY) ? 0.f : __expf(s_val - m_new);
      p_sh[lane]    = p_val;
      vptr_sh[lane] = v_ptr_this;
    }
    __syncthreads();

    // Tổng p trong block
    float sum_p = (lane < valid) ? p_sh[lane] : 0.f;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      sum_p += __shfl_down(sum_p, off, 64);
    }
    sum_p = __shfl(sum_p, 0, 64);

    // y = y*scale_old + Σ_j p_j * V_j  (đọc V theo i, coalesced)
    for (int i = lane; i < head_dim; i += blockDim.x) {
      float yi = out_head[i] * scale_old;
      #pragma unroll 4
      for (int j = 0; j < valid; ++j) {
        const float pw = p_sh[j];
        hip_bfloat16* vj = vptr_sh[j];
        if (pw > 0.f && vj) {
          yi = fmaf(pw, bfloat16_to_float(vj[i]), yi);
        }
      }
      out_head[i] = yi;
    }

    l = fmaf(l, scale_old, sum_p);
    m = m_new;
    __syncthreads();
  }

  // Attention sink: chỉ cộng vào mẫu số (không cộng giá trị); giữ ổn định online-softmax
  const int layer_cap = cache_data->layer_capacity(layer);
  const bool use_sink = ((pos + 1) < seq_len) && (att_len < layer_cap + 1);
  if (use_sink) {
    const size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    const float sink_val = bfloat16_to_float(attn_sinks[sink_idx]);
    const float m_new    = fmaxf(m, sink_val);
    const float scale_old= __expf(m - m_new);

    for (int i = lane; i < head_dim; i += blockDim.x) {
      out_head[i] *= scale_old;  // không cộng thêm gì vì sink không có V
    }
    l = fmaf(l, scale_old, __expf(sink_val - m_new));
    m = m_new;
    __syncthreads();
  }

  // Chuẩn hoá cuối: y /= (l + eps)
  const float inv_l = 1.f / (l + 1e-8f);
  for (int i = lane; i < head_dim; i += blockDim.x) {
    out_head[i] *= inv_l;
  }
}

#if defined(__clang__)
// (tuỳ bạn) tắt lại contract cho phần còn lại nếu muốn:
#pragma clang fp contract(off)
#endif





__global__ void init_argmax_packed_kernel(unsigned long long *packed, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    packed[idx] = pack_argmax(-FLT_MAX, -1);
}

__global__ void merge_argmax_candidates_kernel(const unsigned long long *packed,
                                               int *next_tokens, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    unsigned long long entry = packed[b];
    float best_val;
    int best_idx;
    unpack_argmax(entry, best_val, best_idx);
    next_tokens[b] = (best_idx >= 0) ? best_idx : 0;
}

__global__ void rmsnorm_final_batched_kernel(float *x, const hip_bfloat16 *weight,
                                            int hidden_dim, int batch,
                                            const int *slot_ids) {
    int b = blockIdx.y; // batch index; one block per slot (grid.x must be 1)
    if (b >= batch) return;

    int sid = slot_ids[b];
    float *x_slot = x + (size_t)sid * (size_t)hidden_dim;

    extern __shared__ float s_reduce[];

    // Compute sum of squares over the full hidden_dim with striding threads
    float ss = 0.0f;
    for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        float val = x_slot[j];
        ss += val * val;
    }
    s_reduce[threadIdx.x] = ss;
    __syncthreads();

    // In-block reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + offset];
        }
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(s_reduce[0] / hidden_dim + 1e-5f);
    __syncthreads();

    // Apply normalization and weight in-place
    for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        float w = bfloat16_to_float(weight[j]);
        x_slot[j] = x_slot[j] * inv_rms * w;
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
void gpu_malloc_run_state(GPUContext *ctx, Config *p, int max_batch);
void gpu_free_run_state(GPUContext *ctx);

// ------------------------- Chunked copy+convert of weights -------------------------

void gpu_copy_weights(GPUContext *ctx, TransformerWeights *cpu_w, Config *p) {
  GPUTransformerWeights *gpu_w = &ctx->weights;
  WeightCopyWorkspace &ws = ctx->copy_ws;
  ws.logs.clear();

  int head_dim         = p->head_dim;
  int n_layers         = p->n_layers;
  int hidden_dim       = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size       = p->vocab_size;
  int n_attn_heads     = p->n_attn_heads;
  int n_kv_heads       = p->n_kv_heads;
  int n_experts        = p->n_experts;

  auto to_gb = [](size_t bytes) { return (double)bytes / GB; };
  auto add_log = [&](const std::string &name, size_t bytes) {
    ws.logs.emplace_back(name, bytes);
  };

  auto wait_buffer = [&](int idx) {
    HIP_CHECK(hipEventSynchronize(ws.stage_free[idx]));
  };

  auto copy_fp32_to_bf16 = [&](const float *src, hip_bfloat16 *dst, size_t elems) {
    size_t done = 0;
    int buf = 0;
    const dim3 block_dim(256);
    while (done < elems) {
      wait_buffer(buf);
      size_t tile = std::min(ws.capacity_elems, elems - done);
      memcpy(ws.host_stage[buf], src + done, tile * sizeof(float));
      HIP_CHECK(hipMemcpyAsync(ws.device_stage[buf], ws.host_stage[buf], tile * sizeof(float),
                               hipMemcpyHostToDevice, ws.copy_stream));
      HIP_CHECK(hipEventRecord(ws.copy_ready[buf], ws.copy_stream));
      HIP_CHECK(hipStreamWaitEvent(ws.convert_stream, ws.copy_ready[buf], 0));
      int n = static_cast<int>(tile);
      dim3 grid_dim((n + block_dim.x - 1) / block_dim.x);
      convert_fp32_to_bf16_kernel<<<grid_dim, block_dim, 0, ws.convert_stream>>>(ws.device_stage[buf], dst + done, n);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipEventRecord(ws.stage_free[buf], ws.convert_stream));
      done += tile;
      buf ^= 1;
    }
  };

  auto copy_array = [&](const float *src, hip_bfloat16 **dst_ptr, size_t elems, const std::string &label) {
    HIP_CHECK(hipMalloc(dst_ptr, elems * sizeof(hip_bfloat16)));
    copy_fp32_to_bf16(src, *dst_ptr, elems);
    add_log(label, elems * sizeof(hip_bfloat16));
  };

  auto quantize_segment = [&](const float *src, size_t count, size_t elem_offset,
                              uint8_t *dst_data, uint8_t *dst_scales) {
    size_t done = 0;
    int buf = 0;
    while (done < count) {
      wait_buffer(buf);
      size_t tile = std::min(ws.capacity_elems, count - done);
      memcpy(ws.host_stage[buf], src + done, tile * sizeof(float));
      HIP_CHECK(hipMemcpyAsync(ws.device_stage[buf], ws.host_stage[buf], tile * sizeof(float),
                               hipMemcpyHostToDevice, ws.copy_stream));
      HIP_CHECK(hipEventRecord(ws.copy_ready[buf], ws.copy_stream));
      HIP_CHECK(hipStreamWaitEvent(ws.quant_stream, ws.copy_ready[buf], 0));
      size_t blocks = (tile + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
      quantize_mxfp4_kernel<<<blocks, 1, 0, ws.quant_stream>>>(ws.device_stage[buf], tile, elem_offset + done,
                                                              dst_data, dst_scales);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipEventRecord(ws.stage_free[buf], ws.quant_stream));
      done += tile;
      buf ^= 1;
    }
  };

  // Embedding & output (replicated)
  size_t embed_elems = (size_t)vocab_size * (size_t)hidden_dim;
  copy_array(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, embed_elems, "token_embedding");
  copy_array(cpu_w->out,                    &gpu_w->out,                    embed_elems, "output_proj");

  size_t rms_elems = (size_t)n_layers * (size_t)hidden_dim;
  copy_array(cpu_w->rms_attn_w, &gpu_w->rms_attn_w, rms_elems, "rms_attn");
  copy_array(cpu_w->rms_ffn_w,  &gpu_w->rms_ffn_w,  rms_elems, "rms_ffn");

  size_t rms_final = (size_t)hidden_dim;
  copy_array(cpu_w->rms_out_w, &gpu_w->rms_out_w, rms_final, "rms_out");

  // QKV weights (single GPU)
  size_t global_qkv_rows = head_dim * (n_attn_heads + 2 * n_kv_heads);
  size_t layer_qkv_elems = (size_t)hidden_dim * (size_t)ctx->local_qkv_dim;
  size_t total_qkv_elems = (size_t)n_layers * layer_qkv_elems;
  HIP_CHECK(hipMalloc(&gpu_w->w_qkv, total_qkv_elems * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&gpu_w->b_qkv, (size_t)n_layers * (size_t)ctx->local_qkv_dim * sizeof(hip_bfloat16)));

  for (int l = 0; l < n_layers; ++l) {
    const float *layer_src = cpu_w->w_qkv + (size_t)l * (size_t)hidden_dim * (size_t)global_qkv_rows;
    hip_bfloat16 *layer_dst = gpu_w->w_qkv + (size_t)l * layer_qkv_elems;
    copy_fp32_to_bf16(layer_src, layer_dst, layer_qkv_elems);

    const float *layer_bias_src = cpu_w->b_qkv + (size_t)l * (size_t)global_qkv_rows;
    hip_bfloat16 *layer_bias_dst = gpu_w->b_qkv + (size_t)l * (size_t)ctx->local_qkv_dim;
    copy_fp32_to_bf16(layer_bias_src, layer_bias_dst, ctx->local_qkv_dim);
  }
  add_log("w_qkv", total_qkv_elems * sizeof(hip_bfloat16));
  add_log("b_qkv", (size_t)n_layers * (size_t)ctx->local_qkv_dim * sizeof(hip_bfloat16));

  // W_o weights
  size_t global_o_cols = head_dim * n_attn_heads;
  size_t layer_o_elems = (size_t)hidden_dim * (size_t)ctx->local_q_dim;
  size_t total_o_elems = (size_t)n_layers * layer_o_elems;
  HIP_CHECK(hipMalloc(&gpu_w->w_o, total_o_elems * sizeof(hip_bfloat16)));

  for (int l = 0; l < n_layers; ++l) {
    const float *layer_src = cpu_w->w_o + (size_t)l * (size_t)hidden_dim * (size_t)global_o_cols;
    hip_bfloat16 *layer_dst = gpu_w->w_o + (size_t)l * layer_o_elems;
    copy_fp32_to_bf16(layer_src, layer_dst, layer_o_elems);
  }
  add_log("w_o", total_o_elems * sizeof(hip_bfloat16));

  copy_array(cpu_w->b_o, &gpu_w->b_o, (size_t)n_layers * (size_t)hidden_dim, "b_o");

  // Attention sinks
  size_t sinks_layer = (size_t)ctx->local_q_heads;
  HIP_CHECK(hipMalloc(&gpu_w->attn_sinks, (size_t)n_layers * sinks_layer * sizeof(hip_bfloat16)));
  for (int l = 0; l < n_layers; ++l) {
    const float *src = cpu_w->attn_sinks + (size_t)l * (size_t)n_attn_heads;
    hip_bfloat16 *dst = gpu_w->attn_sinks + (size_t)l * sinks_layer;
    copy_fp32_to_bf16(src, dst, sinks_layer);
  }
  add_log("attn_sinks", (size_t)n_layers * sinks_layer * sizeof(hip_bfloat16));

  // Router (replicated)
  size_t router_w_elems = (size_t)n_layers * (size_t)hidden_dim * (size_t)n_experts;
  copy_array(cpu_w->w_router, &gpu_w->w_router, router_w_elems, "w_router");
  size_t router_b_elems = (size_t)n_layers * (size_t)n_experts;
  copy_array(cpu_w->b_router, &gpu_w->b_router, router_b_elems, "b_router");

  // MoE weights
  int mlp1_rows_local = ctx->local_mlp1_out_dim;
  int mlp2_cols_local = ctx->local_intermediate_dim;
  int mlp1_row_start = 0;
  int mlp2_col_start = 0;

  size_t mlp1_local_elems = (size_t)n_layers * (size_t)ctx->local_experts * (size_t)mlp1_rows_local * (size_t)hidden_dim;
  size_t mlp1_blocks = (mlp1_local_elems + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  size_t mlp1_data_bytes = (mlp1_local_elems + 1) / 2;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp1_data), mlp1_data_bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp1_scales), mlp1_blocks * sizeof(uint8_t)));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp1_data, 0, mlp1_data_bytes, ws.quant_stream));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp1_scales, 0, mlp1_blocks * sizeof(uint8_t), ws.quant_stream));

  size_t mlp1_offset = 0;
  size_t per_expert_mlp1 = 2 * (size_t)intermediate_dim * (size_t)hidden_dim;
  size_t per_expert_mlp1_local = (size_t)mlp1_rows_local * (size_t)hidden_dim;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->w_mlp1 + (size_t)l * (size_t)n_experts * per_expert_mlp1;
    for (int e = 0; e < ctx->local_experts; ++e) {
      const float *expert_base = layer_base + (size_t)e * per_expert_mlp1 + (size_t)mlp1_row_start * (size_t)hidden_dim;
      quantize_segment(expert_base, per_expert_mlp1_local, mlp1_offset,
                       gpu_w->w_mlp1_data, gpu_w->w_mlp1_scales);
      mlp1_offset += per_expert_mlp1_local;
    }
  }
  // Build a tile-packed copy of MXFP4 nibbles for MLP1 to enable coalesced MFMA tiles.
  // Layout per (layer,eid): rows = out_dim (2*I), cols = hidden_dim.
  size_t out_dim = (size_t)mlp1_rows_local;
  size_t tiles_m = (out_dim + 31) / 32;
  size_t tiles_k = ((size_t)hidden_dim + 7) / 8;
  size_t tile_packets_per_mat = tiles_m * tiles_k;
  const size_t tile_packet_bytes = MXFP4_TILE32X8_PACKET_BYTES;
  size_t interleaved_bytes_per_mat = tile_packets_per_mat * tile_packet_bytes;
  size_t scale_blocks_per_mat = ((out_dim * (size_t)hidden_dim) + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  size_t total_mats = (size_t)n_layers * (size_t)ctx->local_experts;
  size_t total_interleaved_bytes = total_mats * interleaved_bytes_per_mat;
  uint8_t *w_mlp1_tiles_interleaved = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp1_tiles_interleaved), total_interleaved_bytes));
  HIP_CHECK(hipMemsetAsync(w_mlp1_tiles_interleaved, 0, total_interleaved_bytes, ws.quant_stream));

  // Pack each (layer,eid) matrix independently
  {
    dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < ctx->local_experts; ++e) {
        size_t mat_idx = (size_t)l * (size_t)ctx->local_experts + (size_t)e;
        size_t rows = out_dim;
        size_t cols = (size_t)hidden_dim;
        size_t elems = rows * cols;
        size_t bytes = (elems + 1) / 2;
        size_t src_offset_bytes = ((size_t)l * (size_t)ctx->local_experts + (size_t)e) * bytes;
        size_t src_scale_offset = mat_idx * scale_blocks_per_mat;
        size_t dst_tile_offset = mat_idx * interleaved_bytes_per_mat;
        dim3 grid((bytes + block.x - 1) / block.x);
        pack_mxfp4_row_to_tile32x8_interleaved_kernel<<<grid, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp1_data + src_offset_bytes,
          w_mlp1_tiles_interleaved + dst_tile_offset,
          rows, cols,
          tile_packet_bytes);
        HIP_CHECK(hipGetLastError());

        size_t scale_total_entries = tile_packets_per_mat * MXFP4_TILE32X8_SCALE_BYTES;
        dim3 grid_scale((scale_total_entries + block.x - 1) / block.x);
        pack_mxfp4_scales_row_to_tile32x8_interleaved_kernel<<<grid_scale, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp1_scales + src_scale_offset,
          w_mlp1_tiles_interleaved + dst_tile_offset,
          rows, cols,
          tile_packet_bytes);
        HIP_CHECK(hipGetLastError());
      }
    }
    hipEvent_t mlp1_pack_done;
    HIP_CHECK(hipEventCreateWithFlags(&mlp1_pack_done, hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(mlp1_pack_done, ws.quant_stream));
    HIP_CHECK(hipEventSynchronize(mlp1_pack_done));
    HIP_CHECK(hipEventDestroy(mlp1_pack_done));
  }
  // Reuse w_mlp1_data pointer to refer to tile-packed layout to keep memory flat.
  HIP_CHECK(hipFree(gpu_w->w_mlp1_data));
  HIP_CHECK(hipFree(gpu_w->w_mlp1_scales));
  gpu_w->w_mlp1_data = w_mlp1_tiles_interleaved;
  gpu_w->w_mlp1_scales = nullptr;
  // Associated scale codes now interleave with each 32x8 tile (32 bytes per tile preceding nibble data).

  add_log("w_mlp1", total_interleaved_bytes);

  size_t mlp1_bias_elems = (size_t)n_layers * (size_t)ctx->local_experts * (size_t)mlp1_rows_local;
  HIP_CHECK(hipMalloc(&gpu_w->b_mlp1, mlp1_bias_elems * sizeof(hip_bfloat16)));
  size_t mlp1_bias_offset = 0;
  size_t per_expert_bias = 2 * (size_t)intermediate_dim;
  size_t per_expert_bias_local = (size_t)mlp1_rows_local;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->b_mlp1 + (size_t)l * (size_t)n_experts * per_expert_bias;
    for (int e = 0; e < ctx->local_experts; ++e) {
      const float *bias_base = layer_base + (size_t)e * per_expert_bias + (size_t)mlp1_row_start;
      copy_fp32_to_bf16(bias_base, gpu_w->b_mlp1 + mlp1_bias_offset, per_expert_bias_local);
      mlp1_bias_offset += per_expert_bias_local;
    }
  }
  add_log("b_mlp1", mlp1_bias_elems * sizeof(hip_bfloat16));

  size_t mlp2_local_elems = (size_t)n_layers * (size_t)ctx->local_experts * (size_t)hidden_dim * (size_t)mlp2_cols_local;
  size_t mlp2_blocks = (mlp2_local_elems + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  size_t mlp2_data_bytes = (mlp2_local_elems + 1) / 2;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp2_data), mlp2_data_bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp2_scales), mlp2_blocks * sizeof(uint8_t)));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp2_data, 0, mlp2_data_bytes, ws.quant_stream));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp2_scales, 0, mlp2_blocks * sizeof(uint8_t), ws.quant_stream));

  size_t mlp2_offset = 0;
  size_t per_expert_mlp2 = (size_t)hidden_dim * (size_t)intermediate_dim;
  size_t per_expert_mlp2_local = (size_t)hidden_dim * (size_t)mlp2_cols_local;
  std::vector<float> mlp2_local_buffer(per_expert_mlp2_local);
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->w_mlp2 + (size_t)l * (size_t)n_experts * per_expert_mlp2;
    for (int e = 0; e < ctx->local_experts; ++e) {
      const float *expert_base = layer_base + (size_t)e * per_expert_mlp2;
      for (int r = 0; r < hidden_dim; ++r) {
        const float *row_src = expert_base + (size_t)r * (size_t)intermediate_dim + (size_t)mlp2_col_start;
        float *row_dst = mlp2_local_buffer.data() + (size_t)r * (size_t)mlp2_cols_local;
        memcpy(row_dst, row_src, (size_t)mlp2_cols_local * sizeof(float));
      }
      quantize_segment(mlp2_local_buffer.data(), per_expert_mlp2_local, mlp2_offset,
                       gpu_w->w_mlp2_data, gpu_w->w_mlp2_scales);
      mlp2_offset += per_expert_mlp2_local;
    }
  }
  (void)mlp2_data_bytes; // tile-packed layout logged after repack

  // Build a tile-packed copy of MXFP4 nibbles for MLP2 using interleaved scale+data packets (32x8 tiles).
  {
    size_t rows = (size_t)hidden_dim;
    size_t cols = (size_t)mlp2_cols_local;
    size_t tiles_m2 = (rows + 31) / 32;
    size_t tiles_k2 = (cols + 7) / 8;
    size_t tile_packets_per_mat = tiles_m2 * tiles_k2;
    const size_t tile_packet_bytes = MXFP4_TILE32X8_PACKET_BYTES;
    size_t interleaved_bytes_per_mat = tile_packets_per_mat * tile_packet_bytes;
    size_t scale_blocks_per_mat = ((rows * cols) + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    size_t total_mats = (size_t)n_layers * (size_t)ctx->local_experts;
    size_t total_interleaved_bytes = total_mats * interleaved_bytes_per_mat;
    uint8_t *w_mlp2_tiles_interleaved = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp2_tiles_interleaved), total_interleaved_bytes));
    HIP_CHECK(hipMemsetAsync(w_mlp2_tiles_interleaved, 0, total_interleaved_bytes, ws.quant_stream));

    dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < ctx->local_experts; ++e) {
        size_t mat_idx = (size_t)l * (size_t)ctx->local_experts + (size_t)e;
        size_t elems = rows * cols;
        size_t bytes = (elems + 1) / 2;
        size_t src_offset_bytes = ((size_t)l * (size_t)ctx->local_experts + (size_t)e) * bytes;
        size_t src_scale_offset = mat_idx * scale_blocks_per_mat;
        size_t dst_tile_offset = mat_idx * interleaved_bytes_per_mat;
        dim3 grid((unsigned int)((bytes + block.x - 1) / block.x));
        pack_mxfp4_row_to_tile32x8_interleaved_kernel<<<grid, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp2_data + src_offset_bytes,
          w_mlp2_tiles_interleaved + dst_tile_offset,
          rows, cols,
          tile_packet_bytes);
        HIP_CHECK(hipGetLastError());

        size_t scale_total_entries = tile_packets_per_mat * MXFP4_TILE32X8_SCALE_BYTES;
        dim3 grid_scale((unsigned int)((scale_total_entries + block.x - 1) / block.x));
        pack_mxfp4_scales_row_to_tile32x8_interleaved_kernel<<<grid_scale, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp2_scales + src_scale_offset,
          w_mlp2_tiles_interleaved + dst_tile_offset,
          rows, cols,
          tile_packet_bytes);
        HIP_CHECK(hipGetLastError());
      }
    }
    hipEvent_t mlp2_pack_done;
    HIP_CHECK(hipEventCreateWithFlags(&mlp2_pack_done, hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(mlp2_pack_done, ws.quant_stream));
    HIP_CHECK(hipEventSynchronize(mlp2_pack_done));
    HIP_CHECK(hipEventDestroy(mlp2_pack_done));
    HIP_CHECK(hipFree(gpu_w->w_mlp2_data));
    HIP_CHECK(hipFree(gpu_w->w_mlp2_scales));
    gpu_w->w_mlp2_data = w_mlp2_tiles_interleaved;
    gpu_w->w_mlp2_scales = nullptr;
    add_log("w_mlp2", total_interleaved_bytes);
  }

  size_t mlp2_bias_elems = (size_t)n_layers * (size_t)ctx->local_experts * (size_t)hidden_dim;
  HIP_CHECK(hipMalloc(&gpu_w->b_mlp2, mlp2_bias_elems * sizeof(hip_bfloat16)));
  size_t mlp2_bias_offset = 0;
  size_t per_expert_bias2 = (size_t)hidden_dim;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->b_mlp2 + (size_t)l * (size_t)n_experts * (size_t)hidden_dim;
    for (int e = 0; e < ctx->local_experts; ++e) {
      const float *bias_base = layer_base + (size_t)e * per_expert_bias2;
      copy_fp32_to_bf16(bias_base, gpu_w->b_mlp2 + mlp2_bias_offset, per_expert_bias2);
      mlp2_bias_offset += per_expert_bias2;
    }
  }
  add_log("b_mlp2", mlp2_bias_elems * sizeof(hip_bfloat16));

  hipEvent_t copy_done;
  hipEvent_t convert_done;
  hipEvent_t weights_ready;
  HIP_CHECK(hipEventCreateWithFlags(&copy_done, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&convert_done, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&weights_ready, hipEventDisableTiming));
  HIP_CHECK(hipEventRecord(copy_done, ws.copy_stream));
  HIP_CHECK(hipEventRecord(convert_done, ws.convert_stream));
  HIP_CHECK(hipStreamWaitEvent(ws.quant_stream, copy_done, 0));
  HIP_CHECK(hipStreamWaitEvent(ws.quant_stream, convert_done, 0));
  HIP_CHECK(hipEventRecord(weights_ready, ws.quant_stream));
  HIP_CHECK(hipEventSynchronize(weights_ready));
  HIP_CHECK(hipEventDestroy(weights_ready));
  HIP_CHECK(hipEventDestroy(convert_done));
  HIP_CHECK(hipEventDestroy(copy_done));

  // Create packed BF16 weights for MFMA kernels
  auto pack_weights_for_mfma = [&](const hip_bfloat16 *src, hip_bfloat16 **dst, int layers, int M, int K, const std::string &name) {
    int Mpad = ((M + 15) / 16) * 16;
    int Kpad = ((K + 15) / 16) * 16;
    size_t total_elems = (size_t)layers * (size_t)Mpad * (size_t)Kpad;
    HIP_CHECK(hipMalloc(dst, total_elems * sizeof(hip_bfloat16)));
    
    const dim3 block(256);
    for (int l = 0; l < layers; ++l) {
      const hip_bfloat16 *layer_src = src + (size_t)l * (size_t)M * (size_t)K;
      hip_bfloat16 *layer_dst = *dst + (size_t)l * (size_t)Mpad * (size_t)Kpad;
      
      size_t total = (size_t)M * (size_t)K;
      dim3 grid((total + block.x - 1) / block.x);
      pack_bf16_rowmajor_to_bf16_tiles16_kernel<<<grid, block, 0, ws.convert_stream>>>(
        layer_src, 0, M, M, K, layer_dst, Mpad, Kpad);
      HIP_CHECK(hipGetLastError());
    }
    hipEvent_t pack_done;
    HIP_CHECK(hipEventCreateWithFlags(&pack_done, hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(pack_done, ws.convert_stream));
    HIP_CHECK(hipEventSynchronize(pack_done));
    HIP_CHECK(hipEventDestroy(pack_done));
    add_log(name + "_packed16", total_elems * sizeof(hip_bfloat16));
  };

  // Pack QKV weights
  pack_weights_for_mfma(gpu_w->w_qkv, &gpu_w->w_qkv_packed16, n_layers, ctx->local_qkv_dim, hidden_dim, "w_qkv");
  
  // Pack W_o weights  
  pack_weights_for_mfma(gpu_w->w_o, &gpu_w->w_o_packed16, n_layers, hidden_dim, ctx->local_q_dim, "w_o");

  // Pack final output projection (replicated)
  pack_weights_for_mfma(gpu_w->out, &gpu_w->out_packed16, 1, vocab_size, hidden_dim, "out_final");
  
  // Pack router weights
  pack_weights_for_mfma(gpu_w->w_router, &gpu_w->w_router_packed16, n_layers, p->n_experts, hidden_dim, "w_router");

  if (!ws.logs.empty()) {
    printf("[Weights][GPU%d] copy summary:\n", ctx->device_id);
    for (const auto &entry : ws.logs) {
      printf("  %-22s %.3f GB\n", entry.first.c_str(), to_gb(entry.second));
    }
    fflush(stdout);
  }
}

void gpu_free_weights(GPUTransformerWeights *gpu_w);



// ------------------------- Forward -------------------------

// Batched attention path over all active slots; includes embedding, per-layer
// attention (RMSNorm->QKV->RoPE->FusedAttn->W_o+residual). MoE is batched & grouped across slots/experts.
// host_logits_out: optional, if not null and do_final=true, copies logits of each slot
// into host buffer packed as [batch, vocab].
// host_next_tokens_out: optional, if not null, returns next tokens sampled by argmax on device
void gpu_forward_batch_step(Transformer *transformer, GPUContext *gpu_ctx,
                            const int *h_tokens, const int *h_pos_list, const int *h_slot_ids,
                            int batch, int *host_next_tokens_out) {
  Config *p = &transformer->config;
  int hidden_dim = p->hidden_dim;
  int head_dim = p->head_dim;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;

  GPUContext *ctx = gpu_ctx;
  if (ctx == nullptr) {
    fprintf(stderr, "[GPU] Invalid GPU context for forward step.\n");
    return;
  }
  HIP_CHECK(hipSetDevice(ctx->device_id));

  auto record_batch_event = [&](GPUContext *ctx) {
    HIP_CHECK(hipSetDevice(ctx->device_id));
    HIP_CHECK(hipEventRecord(ctx->batch_event, ctx->batch_stream));
  };
  auto wait_batch_event_host = [&](GPUContext *ctx) {
    HIP_CHECK(hipEventSynchronize(ctx->batch_event));
  };

  // Stage indices to every GPU
  HIP_CHECK(hipMemcpyAsync(ctx->d_slot_ids,  h_slot_ids,  batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(ctx->d_pos_list,  h_pos_list,  batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(ctx->d_token_ids, h_tokens,    batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));

  // Embedding gather on each GPU
  auto launch_embed = [&](GPUContext *ctx) {
    struct Kernel {
      static __global__ void run(float *x, const hip_bfloat16 *emb, const int *tok,
                                 const int *slot_ids, int hidden_dim, int batch, int stride) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (i < hidden_dim && b < batch) {
          int sid = slot_ids[b];
          int t = tok[b];
          const hip_bfloat16 *row = emb + (size_t)t * (size_t)hidden_dim;
          float *dst = x + (size_t)sid * (size_t)stride;
          dst[i] = bfloat16_to_float(row[i]);
        }
      }
    };
    dim3 block(256);
    dim3 grid((hidden_dim + block.x - 1) / block.x, batch);
    Kernel::run<<<grid, block, 0, ctx->batch_stream>>>(ctx->state.x, ctx->weights.token_embedding_table,
                                                       ctx->d_token_ids, ctx->d_slot_ids,
                                                       hidden_dim, batch, hidden_dim);
  };

  launch_embed(ctx);

  // RoPE precompute per slot per GPU
  auto launch_rope = [&](GPUContext *ctx) {
    int d_half = head_dim / 2;
    dim3 block(256);
    dim3 grid((d_half + block.x - 1) / block.x, batch);
    struct Kernel {
      static __global__ void run(float *cos_out, float *sin_out,
                                 const int *pos_list, const int *slot_ids,
                                 int batch, float base, int head_dim,
                                 float scaling_factor, float initial_context_length,
                                 float ntk_beta, float ntk_alpha, int stride) {
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
          float *cos_dst = cos_out + (size_t)sid * (size_t)stride;
          float *sin_dst = sin_out + (size_t)sid * (size_t)stride;
          cos_dst[tid] = cosf(val) * concentration;
          sin_dst[tid] = sinf(val) * concentration;
        }
      }
    };
    Kernel::run<<<grid, block, 0, ctx->batch_stream>>>(ctx->state.cos_vals, ctx->state.sin_vals,
                                                       ctx->d_pos_list, ctx->d_slot_ids,
                                                       batch, p->rope_theta, head_dim,
                                                       p->rope_scaling_factor, p->initial_context_length,
                                                       32.0f, 1.0f, d_half);
  };

  launch_rope(ctx);

  struct ContiguousKVWriterV {
    static __global__ void run(const float *qkv, KVCacheDeviceData *cache_data, int layer,
                               int batch, int q_stride, int v_offset, int v_size,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < v_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        int write_pos = cache_data->get_write_position(sid, layer, pos);
        hip_bfloat16 *dst = cache_data->get_value_token_ptr(sid, layer, write_pos);
        if (dst == nullptr) return;
        const float *src = qkv + (size_t)sid * (size_t)q_stride + (size_t)v_offset;
        dst[i] = float_to_bfloat16(src[i]);
      }
    }
  };

  struct ContiguousKVWriterK {
    static __global__ void run(const float *qkv, KVCacheDeviceData *cache_data, int layer,
                               int batch, int q_stride, int k_offset, int k_size,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < k_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        int write_pos = cache_data->get_write_position(sid, layer, pos);
        hip_bfloat16 *dst = cache_data->get_key_token_ptr(sid, layer, write_pos);
        if (dst == nullptr) return;
        const float *src = qkv + (size_t)sid * (size_t)q_stride + (size_t)k_offset;
        dst[i] = float_to_bfloat16(src[i]);
      }
    }
  };

  for (int l = 0; l < p->n_layers; ++l) {
    HIP_CHECK(hipSetDevice(ctx->device_id));
    ctx->state.kv_cache_manager->prepare_layer(l, h_slot_ids, h_pos_list, batch);

    // RMSNorm (attention)
    if (batch > 0) {
      const hip_bfloat16 *rms_attn_w = ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
      dim3 block_norm(256), grid_norm(1, batch);
      size_t shared_mem_norm = block_norm.x * sizeof(float);
      rmsnorm_kernel_float_batched<<<grid_norm, block_norm, shared_mem_norm, ctx->batch_stream>>>(
        ctx->state.t,
        ctx->state.x,
        rms_attn_w,
        hidden_dim,
        batch,
        hidden_dim,
        ctx->d_slot_ids);
    }

    // QKV projection + bias using MFMA
    {
      int Bpad = ((batch + 15) / 16) * 16;
      dim3 block_pack(256);
      dim3 grid_pack((hidden_dim + block_pack.x - 1) / block_pack.x, batch);
      gather_pack_x_fp32_to_bf16_mfma_kernel<<<grid_pack, block_pack, 0, ctx->batch_stream>>>(
        ctx->state.t, ctx->state.xpack_bf16, hidden_dim, batch, Bpad, hidden_dim, ctx->d_slot_ids);

      const int Mpad_qkv = ((ctx->local_qkv_dim + 15) / 16) * 16;
      const int Kpad_qkv = ((hidden_dim + 15) / 16) * 16;
      const hip_bfloat16 *w_qkv_packed = ctx->weights.w_qkv_packed16 +
                                         (size_t)l * (size_t)Mpad_qkv * (size_t)Kpad_qkv;
      const hip_bfloat16 *b_qkv = ctx->weights.b_qkv + (size_t)l * (size_t)ctx->local_qkv_dim;
      const int group_y = (batch + 15) / 16;
      dim3 block_gemm(16, 4, 1);
      dim3 grid_gemm((Mpad_qkv + 15) / 16, group_y, 1);
      qkv_kernel<<<grid_gemm, block_gemm, 0, ctx->batch_stream>>>(
        w_qkv_packed, ctx->state.xpack_bf16, ctx->state.qkv, b_qkv,
        ctx->local_qkv_dim, hidden_dim, batch, Bpad, ctx->local_qkv_dim,
        ctx->d_slot_ids, Mpad_qkv, Kpad_qkv);
    }

    // Cache write + RoPE
    int q_stride = ctx->local_qkv_dim;
    int q_size = ctx->local_q_dim;
    int k_size = ctx->local_kv_dim;
    int v_size = ctx->local_kv_dim;

    dim3 block_v(256), grid_v((v_size + block_v.x - 1) / block_v.x, batch);
    ContiguousKVWriterV::run<<<grid_v, block_v, 0, ctx->batch_stream>>>(
      ctx->state.qkv,
      ctx->state.kv_cache_manager->get_device_data(), l,
      batch, q_stride, q_size + k_size, v_size,
      ctx->d_pos_list, ctx->d_slot_ids);

    dim3 grid_heads(ctx->local_q_heads, batch);
    apply_rotary_emb_kernel_float_batched<<<grid_heads, head_dim / 2, 0, ctx->batch_stream>>>(
      ctx->state.qkv, ctx->state.cos_vals, ctx->state.sin_vals,
      ctx->local_q_heads, head_dim, batch, q_stride, head_dim / 2, ctx->d_slot_ids);

    dim3 grid_heads_k(ctx->local_kv_heads, batch);
    apply_rotary_emb_kernel_float_batched<<<grid_heads_k, head_dim / 2, 0, ctx->batch_stream>>>(
      ctx->state.qkv + q_size, ctx->state.cos_vals, ctx->state.sin_vals,
      ctx->local_kv_heads, head_dim, batch, q_stride, head_dim / 2, ctx->d_slot_ids);

    dim3 block_k(256), grid_k((k_size + block_k.x - 1) / block_k.x, batch);
    ContiguousKVWriterK::run<<<grid_k, block_k, 0, ctx->batch_stream>>>(
      ctx->state.qkv,
      ctx->state.kv_cache_manager->get_device_data(), l,
      batch, q_stride, q_size, k_size,
      ctx->d_pos_list, ctx->d_slot_ids);

    // Flash attention
    // [MỚI]
      const int block_size = 64;                 // giữ 64 thread/block (wave64)
      dim3 block(block_size);
      dim3 grid(1, ctx->local_q_heads, batch);

      // BK = FA_V2_BK (mặc định 64). Bạn có thể chỉnh macro ở trên.
      size_t shared_mem_bytes = flash_attention_v2_smem_bytes(/*head_dim=*/head_dim, /*BK=*/FA_V2_BK);

      if ((l & 1) == 0) {
        flash_attention_sliding<<<grid, block, shared_mem_bytes, ctx->batch_stream>>>(
            ctx->state.qkv,
            ctx->state.kv_cache_manager->get_device_data(),
            ctx->weights.attn_sinks,
            ctx->state.tb,                  // output per-head
            head_dim,
            /*kv_dim_unused=*/0,            // tham số này không dùng trong v2 (giữ chỗ)
            kv_mul,
            ctx->local_q_heads,
            p->seq_len,
            l,
            ctx->d_pos_list,
            batch,
            ctx->local_qkv_dim,
            ctx->d_slot_ids);
      } else {
        flash_attention_full<<<grid, block, shared_mem_bytes, ctx->batch_stream>>>(
            ctx->state.qkv,
            ctx->state.kv_cache_manager->get_device_data(),
            ctx->weights.attn_sinks,
            ctx->state.tb,                  // output per-head
            head_dim,
            /*kv_dim_unused=*/0,            // tham số này không dùng trong v2 (giữ chỗ)
            kv_mul,
            ctx->local_q_heads,
            p->seq_len,
            l,
            ctx->d_pos_list,
            batch,
            ctx->local_qkv_dim,
            ctx->d_slot_ids);
      }

    // Output projection without bias using MFMA
    {
      int Bpad = ((batch + 15) / 16) * 16;
      dim3 block_pack(256);
      dim3 grid_pack((ctx->local_q_dim + block_pack.x - 1) / block_pack.x, batch);
      gather_pack_x_fp32_to_bf16_mfma_kernel<<<grid_pack, block_pack, 0, ctx->batch_stream>>>(
        ctx->state.tb, ctx->state.xpack_bf16, ctx->local_q_dim, batch, Bpad, ctx->local_q_dim, ctx->d_slot_ids);

      const int Mpad_o = ((hidden_dim + 15) / 16) * 16;
      const int Kpad_o = ((ctx->local_q_dim + 15) / 16) * 16;
      const hip_bfloat16 *w_o_packed = ctx->weights.w_o_packed16 +
                                       (size_t)l * (size_t)Mpad_o * (size_t)Kpad_o;
      const int group_y = (batch + 15) / 16;
      dim3 block_gemm(16, 4, 1);
      dim3 grid_gemm((Mpad_o + 15) / 16, group_y, 1);
      o_kernel<<<grid_gemm, block_gemm, 0, ctx->batch_stream>>>(
        w_o_packed, ctx->state.xpack_bf16, ctx->state.tb2, nullptr,
        hidden_dim, ctx->local_q_dim, batch, Bpad, hidden_dim,
        ctx->d_slot_ids, Mpad_o, Kpad_o);
    }

    // Add bias and residual connection
    {
      const hip_bfloat16 *b_o = ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      add_bias_kernel_float_batched<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.tb2, b_o, hidden_dim, batch, hidden_dim, ctx->d_slot_ids);
    }

    // Residual add with attention output
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      residual_connection_kernel_float_batched<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.x, ctx->state.tb2, hidden_dim, batch, hidden_dim, ctx->d_slot_ids);
    }

    // MoE RMSNorm per slot
    if (batch > 0) {
      const hip_bfloat16 *rms_ffn_w = ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
      dim3 block_norm(256), grid_norm(1, batch);
      size_t shared_mem_norm = block_norm.x * sizeof(float);
      rmsnorm_kernel_float_batched<<<grid_norm, block_norm, shared_mem_norm, ctx->batch_stream>>>(
        ctx->state.t,
        ctx->state.x,
        rms_ffn_w,
        hidden_dim,
        batch,
        hidden_dim,
        ctx->d_slot_ids);
    }

    // Router computation using MFMA
    {
      int Bpad = ((batch + 15) / 16) * 16;
      dim3 block_pack(256);
      dim3 grid_pack((hidden_dim + block_pack.x - 1) / block_pack.x, batch);
      gather_pack_x_fp32_to_bf16_mfma_kernel<<<grid_pack, block_pack, 0, ctx->batch_stream>>>(
        ctx->state.t, ctx->state.xpack_bf16, hidden_dim, batch, Bpad, hidden_dim, ctx->d_slot_ids);

      const int Mpad_router = ((p->n_experts + 15) / 16) * 16;
      const int Kpad_router = ((hidden_dim + 15) / 16) * 16;
      const hip_bfloat16 *w_router_packed = ctx->weights.w_router_packed16 +
                                            (size_t)l * (size_t)Mpad_router * (size_t)Kpad_router;
      const hip_bfloat16 *b_router = ctx->weights.b_router + (size_t)l * (size_t)p->n_experts;
      const int group_y = (batch + 15) / 16;
      dim3 block_gemm(16, 4, 1);
      dim3 grid_gemm((Mpad_router + 15) / 16, group_y, 1);
      router_kernel<<<grid_gemm, block_gemm, 0, ctx->batch_stream>>>(
        w_router_packed, ctx->state.xpack_bf16, ctx->state.router_score, b_router,
        p->n_experts, hidden_dim, batch, Bpad, p->n_experts,
        ctx->d_slot_ids, Mpad_router, Kpad_router);

      dim3 grid_topk(batch), block_topk(128);
      size_t shared_topk = (size_t)block_topk.x * (size_t)p->experts_per_token * (sizeof(float) + sizeof(int));
      topk_softmax_batched_kernel_float<<<grid_topk, block_topk, shared_topk, ctx->batch_stream>>>(
        ctx->state.router_score, p->n_experts, p->experts_per_token,
        ctx->state.topk_v, ctx->state.topk_i,
        p->n_experts, p->experts_per_token,
        ctx->d_slot_ids, batch);
    }

    // Zero MoE accumulation buffer
    {
      dim3 block_zero(256), grid_zero((hidden_dim + block_zero.x - 1) / block_zero.x, batch);
      zero_selected_slots_kernel_float<<<grid_zero, block_zero, 0, ctx->batch_stream>>>(
        ctx->state.e_agg, hidden_dim, ctx->d_slot_ids, batch);
    }

    // Gather top-k pairs and build grouped metadata on host
    const int M = batch * p->experts_per_token;
    int local_experts = ctx->local_experts;
    std::vector<int> host_sel_sids(M);
    std::vector<int> host_sel_eids(M);
    std::vector<float> host_sel_w(M);
    {
      dim3 block(256), grid((M + block.x - 1) / block.x);
      gather_topk_pairs_kernel<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.topk_v, ctx->state.topk_i,
        ctx->d_slot_ids, batch, p->experts_per_token,
        ctx->state.sel_w, ctx->state.sel_eids, ctx->state.sel_sids);
    }
    HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    HIP_CHECK(hipMemcpy(host_sel_sids.data(), ctx->state.sel_sids, M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_sel_eids.data(), ctx->state.sel_eids, M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_sel_w.data(), ctx->state.sel_w, M * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<int> grouped_sids;
    std::vector<int> grouped_eids;
    std::vector<float> grouped_w;
    grouped_sids.reserve(M);
    grouped_eids.reserve(M);
    grouped_w.reserve(M);
    for (int m = 0; m < M; ++m) {
      int eid = host_sel_eids[m];
      if (eid < 0 || eid >= local_experts) continue;
      grouped_sids.push_back(host_sel_sids[m]);
      grouped_eids.push_back(eid);
      grouped_w.push_back(host_sel_w[m]);
    }

    int M_local = static_cast<int>(grouped_sids.size());
    std::vector<int> block_cols;
    std::vector<int> block_sizes;
    std::vector<int> block_eids;
    if (M_local > 0) {
      std::vector<int> counts(local_experts, 0);
      for (int eid : grouped_eids) counts[eid]++;

      std::vector<int> prefix(local_experts + 1, 0);
      for (int e = 0; e < local_experts; ++e) {
        prefix[e + 1] = prefix[e] + counts[e];
      }

      std::vector<int> write_pos = prefix;
      std::vector<int> order(M_local);
      for (int i = 0; i < M_local; ++i) {
        int e = grouped_eids[i];
        int pos = write_pos[e]++;
        order[pos] = i;
      }

      std::vector<int> ordered_sids(M_local);
      std::vector<int> ordered_eids(M_local);
      std::vector<float> ordered_w(M_local);
      for (int dst = 0; dst < M_local; ++dst) {
        int src = order[dst];
        ordered_sids[dst] = grouped_sids[src];
        ordered_eids[dst] = grouped_eids[src];
        ordered_w[dst] = grouped_w[src];
      }

      grouped_sids.swap(ordered_sids);
      grouped_eids.swap(ordered_eids);
      grouped_w.swap(ordered_w);

      block_cols.reserve((M_local + 31) / 32);
      block_sizes.reserve((M_local + 31) / 32);
      block_eids.reserve((M_local + 31) / 32);

      for (int e = 0; e < local_experts; ++e) {
        int start = prefix[e];
        int end = prefix[e + 1];
        int remaining = end - start;
        int col = start;
        while (remaining > 0) {
          int chunk = remaining >= 32 ? 32 : remaining;
          block_cols.push_back(col);
          block_sizes.push_back(chunk);
          block_eids.push_back(e);
          col += chunk;
          remaining -= chunk;
        }
      }
    }

    int block_count = static_cast<int>(block_cols.size());
    if (M_local > 0) {
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_sids, grouped_sids.data(), M_local * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_eids, grouped_eids.data(), M_local * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_w, grouped_w.data(), M_local * sizeof(float), hipMemcpyHostToDevice, ctx->batch_stream));
    }
    if (block_count > 0) {
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_block_cols, block_cols.data(), block_count * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_block_sizes, block_sizes.data(), block_count * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_block_eids, block_eids.data(), block_count * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
    }

    if (M_local > 0 && block_count > 0) {
      if (batch > 0) {
        dim3 block_fp2bf16(256);
        dim3 grid_fp2bf16((hidden_dim + block_fp2bf16.x - 1) / block_fp2bf16.x, batch);
        gather_fp32_to_bf16_slots_kernel<<<grid_fp2bf16, block_fp2bf16, 0, ctx->batch_stream>>>(
          ctx->state.t,
          ctx->state.t_bf16,
          hidden_dim,
          batch,
          hidden_dim,
          ctx->d_slot_ids);
      }

      const int out_dim = ctx->local_mlp1_out_dim;
      dim3 block1(MLP1_BLOCK_DIM_X, MLP1_BLOCK_DIM_Y, 1);
      dim3 grid1((out_dim + MLP1_BLOCK_ROWS - 1) / MLP1_BLOCK_ROWS, block_count, 1);
      mlp1_grouped_mfma_matvec_mxfp4_kernel<<<grid1, block1, 0, ctx->batch_stream>>>(
        ctx->state.mlp1_out_all, ctx->state.t_bf16,
        ctx->weights.w_mlp1_data,
        ctx->weights.b_mlp1,
        ctx->state.sel_sids,
        ctx->state.sel_block_cols,
        ctx->state.sel_block_sizes,
        ctx->state.sel_block_eids,
        M_local, block_count,
        hidden_dim, out_dim,
        l, ctx->local_experts);

      dim3 block2(256), grid2((ctx->local_intermediate_dim + block2.x - 1) / block2.x, M_local);
      swiglu_grouped_kernel_bf16<<<grid2, block2, 0, ctx->batch_stream>>>(
        ctx->state.gate_up_all, ctx->state.mlp1_out_all,
        M_local, ctx->local_intermediate_dim, p->swiglu_limit);

      dim3 block3(MLP2_BLOCK_DIM_X, MLP2_BLOCK_DIM_Y, 1);
      dim3 grid3((hidden_dim + MLP2_BLOCK_ROWS - 1) / MLP2_BLOCK_ROWS, block_count, 1);
      mlp2_accum_grouped_mfma_matvec_mxfp4_kernel<<<grid3, block3, 0, ctx->batch_stream>>>(
        ctx->state.e_agg, ctx->state.gate_up_all,
        ctx->weights.w_mlp2_data,
        ctx->weights.b_mlp2,
        ctx->state.sel_sids, ctx->state.sel_w,
        ctx->state.sel_block_cols,
        ctx->state.sel_block_sizes,
        ctx->state.sel_block_eids,
        M_local, block_count,
        hidden_dim, ctx->local_intermediate_dim,
        l, ctx->local_experts, hidden_dim);
    }

    // Residual add with MoE output
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      residual_connection_kernel_float_batched<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.x, ctx->state.e_agg, hidden_dim, batch, hidden_dim, ctx->d_slot_ids);
    }
  }

  // Final RMSNorm + logits on a single GPU with greedy sampling
  if (host_next_tokens_out != nullptr) {
    HIP_CHECK(hipSetDevice(ctx->device_id));

    int vocab = p->vocab_size;
    int Bpad_logits = ((batch + 15) / 16) * 16;
    int Mpad_logits = ((vocab + 15) / 16) * 16;
    int Kpad_logits = ((hidden_dim + 15) / 16) * 16;
    int batch_tiles = (batch + 15) / 16;
    int grid_y = (batch_tiles + kLogitsGroupsPerBlock - 1) / kLogitsGroupsPerBlock;

    dim3 block_norm(128);
    dim3 grid_norm(1, batch);
    size_t shared_norm = block_norm.x * sizeof(float);
    dim3 block_pack_logits(256);
    dim3 grid_pack_logits((hidden_dim + block_pack_logits.x - 1) / block_pack_logits.x, batch);

    rmsnorm_final_batched_kernel<<<grid_norm, block_norm, shared_norm, ctx->batch_stream>>>(
      ctx->state.x, ctx->weights.rms_out_w,
      hidden_dim, batch, ctx->d_slot_ids);
    gather_pack_x_fp32_to_bf16_mfma_kernel<<<grid_pack_logits, block_pack_logits, 0, ctx->batch_stream>>>(
      ctx->state.x, ctx->state.xpack_bf16,
      hidden_dim, batch, Bpad_logits, hidden_dim, ctx->d_slot_ids);

    if (batch > 0) {
      int threads_init = 128;
      int blocks_init = (batch + threads_init - 1) / threads_init;
      init_argmax_packed_kernel<<<blocks_init, threads_init, 0, ctx->batch_stream>>>(
        ctx->state.argmax_packed, batch);
    }

    if (batch > 0) {
      if (ctx->logits_tile_count <= 0) {
        fprintf(stderr, "[GPU] Invalid logits tile assignment for single-GPU path.\n");
        exit(EXIT_FAILURE);
      }
      const hip_bfloat16 *w_out_packed = ctx->weights.out_packed16;
      dim3 block_gemm(16, 4, kLogitsRowTilesPerBlock);
      int grid_x = (ctx->logits_tile_count + kLogitsRowTilesPerBlock - 1) / kLogitsRowTilesPerBlock;
      dim3 grid_gemm(grid_x, grid_y, 1);
      logits_kernel<<<grid_gemm, block_gemm, 0, ctx->batch_stream>>>(
        w_out_packed, ctx->state.xpack_bf16, nullptr, nullptr,
        vocab, hidden_dim, batch, Bpad_logits, 0,
        ctx->d_batch_iota, Mpad_logits, Kpad_logits,
        ctx->logits_tile_offset, ctx->logits_tile_count, ctx->logits_row_offset,
        ctx->state.argmax_packed, vocab);

      dim3 block_merge(128);
      dim3 grid_merge((batch + block_merge.x - 1) / block_merge.x);
      merge_argmax_candidates_kernel<<<grid_merge, block_merge, 0, ctx->batch_stream>>>(
        ctx->state.argmax_packed,
        ctx->d_next_tokens,
        batch);
    }

    HIP_CHECK(hipMemcpyAsync(host_next_tokens_out, ctx->d_next_tokens,
                             batch * sizeof(int), hipMemcpyDeviceToHost, ctx->batch_stream));
    record_batch_event(ctx);
    wait_batch_event_host(ctx);
  }

  // // Keep follower activations aligned (optional)
  // if (ctxs.size() > 1) {
  //   size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
  //   for (size_t idx = 1; idx < ctxs.size(); ++idx) {
  //     GPUContext *peer = ctxs[idx];
  //     HIP_CHECK(hipMemcpyPeer(peer->state.x, peer->device_id,
  //                             leader->state.x, leader->device_id, bytes));
  //   }
  // }

}

// ------------------------- Warm-up & Finish -------------------------

void log_gpu_memory_usage(int device_id) {
  HIP_CHECK(hipSetDevice(device_id));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  
  size_t used_bytes = total_bytes - free_bytes;
  double used_gb = used_bytes / GB;
  double free_gb = free_bytes / GB;
  double total_gb = total_bytes / GB;
  
  printf("[GPU %d] Memory after warm-up: %.2f GB used, %.2f GB free, %.2f GB total (%.1f%% used)\n",
         device_id, used_gb, free_gb, total_gb, (used_bytes * 100.0) / total_bytes);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  (void)tokenizer;
  transformer->config.seq_len = 1024; // clamp to supported max seq len

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "No HIP devices found.\n");
    exit(EXIT_FAILURE);
  }

  Config *cfg = &transformer->config;
  int desired_dp = 1;
  if (const char *env_dp = getenv("GETP_DP_SIZE")) {
    int parsed = atoi(env_dp);
    if (parsed > 0) desired_dp = parsed;
  }
  int used_devices = std::min(device_count, desired_dp);
  if (used_devices <= 0) used_devices = 1;

  gpu_contexts.clear();
  gpu_contexts.resize(used_devices);

  int head_dim = cfg->head_dim;

  for (int d = 0; d < used_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    GPUContext &ctx = gpu_contexts[d];
    ctx.device_id = d;
    ctx.dp_rank = d;
    ctx.local_q_heads = cfg->n_attn_heads;
    ctx.local_kv_heads = cfg->n_kv_heads;
    ctx.local_experts = cfg->n_experts;
    ctx.local_intermediate_dim = cfg->intermediate_dim;
    ctx.local_mlp1_out_dim = 2 * ctx.local_intermediate_dim;
    ctx.local_q_dim = ctx.local_q_heads * head_dim;
    ctx.local_kv_dim = ctx.local_kv_heads * head_dim;
    ctx.local_qkv_dim = ctx.local_q_dim + 2 * ctx.local_kv_dim;

    ensure_mxfp4_dequant_lut_initialized();

    const char *env_mb = getenv("GETP_MAX_BATCH");
    ctx.max_batch = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 8;

    WeightCopyWorkspace &ws = ctx.copy_ws;
    ws.logs.clear();
    ws.logs.reserve(32);
    size_t stage_mb = 256;
    if (const char *env_stage = getenv("GETP_STAGE_MB")) {
      int parsed = atoi(env_stage);
      if (parsed > 0) stage_mb = (size_t)parsed;
    }
    ws.capacity_elems = (stage_mb * (size_t)1024 * (size_t)1024) / sizeof(float);
    if (ws.capacity_elems == 0) ws.capacity_elems = (size_t)65536;
    size_t stage_bytes = ws.capacity_elems * sizeof(float);
    HIP_CHECK(hipStreamCreateWithFlags(&ws.copy_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&ws.convert_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&ws.quant_stream, hipStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
      HIP_CHECK(hipHostMalloc((void**)&ws.host_stage[i], stage_bytes));
      HIP_CHECK(hipMalloc((void**)&ws.device_stage[i], stage_bytes));
      HIP_CHECK(hipEventCreateWithFlags(&ws.copy_ready[i], hipEventDisableTiming));
      HIP_CHECK(hipEventCreateWithFlags(&ws.stage_free[i], hipEventDisableTiming));
      HIP_CHECK(hipEventRecord(ws.stage_free[i], ws.quant_stream));
    }

    HIP_CHECK(hipStreamCreateWithFlags(&ctx.batch_stream, hipStreamNonBlocking));
    HIP_CHECK(hipEventCreateWithFlags(&ctx.batch_event, hipEventDisableTiming));

    HIP_CHECK(hipMalloc(&ctx.d_slot_ids,  ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_pos_list,  ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_token_ids, ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_next_tokens, ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_batch_iota, ctx.max_batch * sizeof(int)));
    std::vector<int> host_batch_iota(ctx.max_batch);
    for (int i = 0; i < ctx.max_batch; ++i) host_batch_iota[i] = i;
    HIP_CHECK(hipMemcpy(ctx.d_batch_iota, host_batch_iota.data(),
                       ctx.max_batch * sizeof(int), hipMemcpyHostToDevice));

    gpu_copy_weights(&ctx, &transformer->weights, cfg);
    gpu_malloc_run_state(&ctx, cfg, ctx.max_batch);
  }

  printf("[GPU] Warm-up done on %d device(s). Tensor parallel disabled.\n", used_devices);
  printf("[Attention] Using FlashAttention-like optimized kernel (always on)\n");

  for (int d = 0; d < used_devices; ++d) {
    log_gpu_memory_usage(d);
  }
}


void finish(Transformer *transformer, Tokenizer *tokenizer) {
  (void)transformer;
  (void)tokenizer;
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    if (gpu_ctx.batch_event) HIP_CHECK(hipEventDestroy(gpu_ctx.batch_event));
    gpu_ctx.batch_event = nullptr;
    if (gpu_ctx.batch_stream) HIP_CHECK(hipStreamDestroy(gpu_ctx.batch_stream));
    gpu_ctx.batch_stream = nullptr;
    if (gpu_ctx.d_slot_ids)    HIP_CHECK(hipFree(gpu_ctx.d_slot_ids));
    if (gpu_ctx.d_pos_list)    HIP_CHECK(hipFree(gpu_ctx.d_pos_list));
    if (gpu_ctx.d_token_ids)   HIP_CHECK(hipFree(gpu_ctx.d_token_ids));
    if (gpu_ctx.d_next_tokens) HIP_CHECK(hipFree(gpu_ctx.d_next_tokens));
    if (gpu_ctx.d_batch_iota)  HIP_CHECK(hipFree(gpu_ctx.d_batch_iota));
    gpu_ctx.d_slot_ids = gpu_ctx.d_pos_list = nullptr;
    gpu_ctx.d_token_ids = gpu_ctx.d_next_tokens = nullptr;
    gpu_ctx.d_batch_iota = nullptr;

    WeightCopyWorkspace &ws = gpu_ctx.copy_ws;
    for (int i = 0; i < 2; ++i) {
      if (ws.copy_ready[i]) HIP_CHECK(hipEventDestroy(ws.copy_ready[i]));
      if (ws.stage_free[i]) HIP_CHECK(hipEventDestroy(ws.stage_free[i]));
      if (ws.device_stage[i]) HIP_CHECK(hipFree(ws.device_stage[i]));
      if (ws.host_stage[i]) HIP_CHECK(hipHostFree(ws.host_stage[i]));
      ws.copy_ready[i] = ws.stage_free[i] = nullptr;
      ws.device_stage[i] = nullptr;
      ws.host_stage[i] = nullptr;
    }
    if (ws.copy_stream) HIP_CHECK(hipStreamDestroy(ws.copy_stream));
    if (ws.convert_stream) HIP_CHECK(hipStreamDestroy(ws.convert_stream));
    if (ws.quant_stream) HIP_CHECK(hipStreamDestroy(ws.quant_stream));
    ws.copy_stream = ws.convert_stream = ws.quant_stream = nullptr;
    ws.capacity_elems = 0;
    ws.logs.clear();

    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx);
  }
  gpu_contexts.clear();
  printf("[GPU] Cleanup done.");
}


// ------------------------- Driver (generate & inference) -------------------------

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  (void)sampler;
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  if (requests->num_reqs <= 0 || gpu_contexts.empty()) return 0;
  const int num_devices = static_cast<int>(gpu_contexts.size());

  std::atomic<int> next_idx{0};
  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      GPUContext *ctx = &gpu_contexts[t];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      t_ctx = ctx;

      std::vector<Slot> slots(ctx->max_batch);
      long long local_out = 0;

      auto reset_kv_slot = [&](int slot_id) {
        if (ctx->state.kv_cache_manager) {
          ctx->state.kv_cache_manager->reset_slot(slot_id);
        }
      };

      auto acquire_slot = [&](int sidx) -> bool {
        int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= requests->num_reqs) return false;
        reset_kv_slot(sidx);
        Slot &s = slots[sidx];
        s.active = true;
        s.req_idx = idx;
        const char *input_seq = get_str_req_ptr(requests, idx);
        s.out_ptr = get_tok_gen_ptr(requests, idx);
        int alloc = (int)strlen(input_seq) + 3;
        s.prompt.resize(alloc);
        encode(tokenizer, input_seq, -1, -1, s.prompt.data(), &s.num_prompt_tokens,
               transformer->config.initial_context_length);
        s.pos = 0;
        s.token = s.prompt[0];
        s.max_steps = std::min(requests->max_seq_len, transformer->config.seq_len);
        return true;
      };

      for (int i = 0; i < ctx->max_batch; ++i) {
        acquire_slot(i);
      }

      int active_cnt = 0;
      for (auto &s : slots) if (s.active) active_cnt++;

      while (active_cnt > 0) {
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

        std::vector<int> next_tokens_host(B);
        gpu_forward_batch_step(transformer, ctx, tokens.data(), poses.data(), slot_ids.data(), B, next_tokens_host.data());

        for (int bi = 0; bi < B; ++bi) {
          int sidx = slot_ids[bi];
          Slot &s = slots[sidx];

          s.pos++;
          int next;
          if (s.pos < s.num_prompt_tokens) {
            next = s.prompt[s.pos];
          } else {
            next = next_tokens_host[bi];
            int gen_index = s.pos - s.num_prompt_tokens;
            s.out_ptr[gen_index] = next;
          }

          bool finished = false;
          if (next == 199999 || next == 200002) finished = true;
          if (s.pos >= s.max_steps) finished = true;
          if (s.pos >= transformer->config.seq_len) finished = true;

          if (finished) {
            int gen_tokens = std::max(0, s.pos - s.num_prompt_tokens);
            s.out_ptr[gen_tokens] = -1;
            local_out += (long long)(gen_tokens + 1);
            s.active = false;
            s.req_idx = -1;
            s.prompt.clear();
            reset_kv_slot(sidx);
            active_cnt--;
            if (acquire_slot(sidx)) active_cnt++;
          } else {
            s.token = next;
          }
        }
      }

      total_out.fetch_add(local_out, std::memory_order_relaxed);
    });
  }

  for (auto &th : workers) th.join();
  return total_out.load(std::memory_order_relaxed);
}


#endif // GETP_RUN

void gpu_malloc_run_state(GPUContext *ctx, Config *p, int max_batch) {
  HIP_CHECK(hipSetDevice(ctx->device_id));
  GPURunState *s = &ctx->state;
  size_t B = (size_t)max_batch;
  int hidden_dim = p->hidden_dim;
  int local_q_dim = ctx->local_q_dim;
  int local_kv_dim = ctx->local_kv_dim;
  int local_qkv_dim = ctx->local_qkv_dim;
  int intermediate_local = ctx->local_intermediate_dim;
  int mlp1_out_local = ctx->local_mlp1_out_dim;

  int vocab = p->vocab_size;
  int Mpad_logits = ((vocab + 15) / 16) * 16;
  int total_tiles = (Mpad_logits > 0) ? (Mpad_logits / 16) : 0;

  ctx->logits_tile_offset = 0;
  ctx->logits_tile_count = total_tiles;
  ctx->logits_row_offset = 0;
  ctx->logits_row_padded = Mpad_logits;
  ctx->logits_row_actual = vocab;

  HIP_CHECK(hipMalloc(&s->x,   B * (size_t)hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->t,   B * (size_t)hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&s->t_bf16), B * (size_t)hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb,  B * (size_t)local_q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2, B * (size_t)hidden_dim * sizeof(float)));

  size_t router_elems = (size_t)p->n_experts;
  HIP_CHECK(hipMalloc(&s->router_score, B * router_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_v,       B * (size_t)p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_i,       B * (size_t)p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->mlp1_out, B * (size_t)mlp1_out_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate,     B * (size_t)intermediate_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up,       B * (size_t)intermediate_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up,  B * (size_t)intermediate_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->e_agg,    B * (size_t)hidden_dim * sizeof(float)));

  HIP_CHECK(hipMalloc(&s->qkv, B * (size_t)local_qkv_dim * sizeof(float)));

  int window_attention = 512;
  if (const char *env_window = getenv("GETP_WINDOW_ATTENTION")) {
    int parsed = atoi(env_window);
    if (parsed >= 0) {
      window_attention = parsed;
    }
  }
  if (window_attention > p->seq_len) {
    window_attention = p->seq_len;
  }

  s->kv_cache_manager = new KVCacheManager(max_batch, p->n_layers, p->seq_len,
                                           (size_t)local_kv_dim, p->sliding_window,
                                           window_attention);

  HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));

  size_t argmax_elems = (size_t)ctx->max_batch;
  HIP_CHECK(hipMalloc(&s->argmax_packed, argmax_elems * sizeof(unsigned long long)));

  HIP_CHECK(hipMalloc(&s->cos_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sin_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));

  size_t Mmax = (size_t)max_batch * (size_t)p->experts_per_token;
  HIP_CHECK(hipMalloc(&s->sel_sids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_eids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_w,        Mmax * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sel_block_cols,  Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_block_sizes, Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_block_eids,  Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax * (size_t)mlp1_out_local * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)intermediate_local * sizeof(hip_bfloat16)));

  // No tensor-parallel staging buffers when running on a single GPU.
  
  // MFMA packed buffers
  int Bpad = ((int)B + 15) / 16 * 16; // align to 16
  int K_max = std::max({hidden_dim, p->n_experts, local_qkv_dim});
  HIP_CHECK(hipMalloc(&s->xpack_bf16, (size_t)K_max * (size_t)Bpad * sizeof(hip_bfloat16)));
}

void gpu_free_run_state(GPUContext *ctx) {
  HIP_CHECK(hipSetDevice(ctx->device_id));
  GPURunState *s = &ctx->state;
  HIP_CHECK(hipFree(s->x));
  HIP_CHECK(hipFree(s->t));
  HIP_CHECK(hipFree(s->t_bf16));
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
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  if (s->logits_fp32) HIP_CHECK(hipFree(s->logits_fp32));
  if (s->argmax_packed) HIP_CHECK(hipFree(s->argmax_packed));
  HIP_CHECK(hipFree(s->sel_sids));
  HIP_CHECK(hipFree(s->sel_eids));
  HIP_CHECK(hipFree(s->sel_w));
  HIP_CHECK(hipFree(s->sel_block_cols));
  HIP_CHECK(hipFree(s->sel_block_sizes));
  HIP_CHECK(hipFree(s->sel_block_eids));
  HIP_CHECK(hipFree(s->mlp1_out_all));
  HIP_CHECK(hipFree(s->gate_up_all));
  HIP_CHECK(hipFree(s->xpack_bf16));
  if (s->kv_cache_manager) {
    delete s->kv_cache_manager;
    s->kv_cache_manager = nullptr;
  }
}

void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  HIP_CHECK(hipFree(gpu_w->token_embedding_table));
  HIP_CHECK(hipFree(gpu_w->out));
  if (gpu_w->out_packed16) HIP_CHECK(hipFree(gpu_w->out_packed16));
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
  HIP_CHECK(hipFree(gpu_w->w_mlp1_data));
  if (gpu_w->w_mlp1_scales) HIP_CHECK(hipFree(gpu_w->w_mlp1_scales));
  HIP_CHECK(hipFree(gpu_w->b_mlp1));
  HIP_CHECK(hipFree(gpu_w->w_mlp2_data));
  if (gpu_w->w_mlp2_scales) HIP_CHECK(hipFree(gpu_w->w_mlp2_scales));
  HIP_CHECK(hipFree(gpu_w->b_mlp2));
  // Free packed weights
  if (gpu_w->w_qkv_packed16) HIP_CHECK(hipFree(gpu_w->w_qkv_packed16));
  if (gpu_w->w_o_packed16) HIP_CHECK(hipFree(gpu_w->w_o_packed16));
  if (gpu_w->w_router_packed16) HIP_CHECK(hipFree(gpu_w->w_router_packed16));
}
