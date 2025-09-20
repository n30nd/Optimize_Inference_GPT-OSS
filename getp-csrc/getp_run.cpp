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

#ifndef MAX_EXPERT_STREAMS
#define MAX_EXPERT_STREAMS 4
#endif

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
  SLIDING_WINDOW = 1
};

struct KVCacheSlot {
  int page_table_offset;  // Base index into page table for this slot+layer
  int max_pages;          // Logical pages available for this layer
  LayerType layer_type;
};

struct KVCacheDeviceData {
  hip_bfloat16 *key_pages;
  hip_bfloat16 *value_pages;
  KVCacheSlot *slots;
  int *page_table;
  size_t kv_dim;
  int max_batch;
  int n_layers;
  int seq_len;
  int sliding_window_size;
  int page_size;
  int pages_per_slot;

  __device__ __forceinline__ LayerType get_layer_type(int layer) const {
    return (sliding_window_size > 0 && (layer % 2 == 0)) ? SLIDING_WINDOW : FULL_ATTENTION;
  }

  __device__ __forceinline__ int layer_capacity(int layer) const {
    LayerType lt = get_layer_type(layer);
    if (lt == SLIDING_WINDOW && sliding_window_size > 0) {
      return sliding_window_size;
    }
    return seq_len;
  }

  __device__ __forceinline__ int slot_table_base(int slot_id, int layer) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    return slot_id * pages_per_slot + slot.page_table_offset;
  }

  __device__ __forceinline__ hip_bfloat16 *get_key_token_ptr(int slot_id, int layer, int cache_pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.max_pages <= 0) return nullptr;
    int logical_page = cache_pos / page_size;
    if (logical_page >= slot.max_pages) return nullptr;
    int within_page = cache_pos % page_size;
    int table_index = slot_table_base(slot_id, layer) + logical_page;
    int physical_page = page_table[table_index];
    if (physical_page < 0) return nullptr;
    size_t token_offset = ((size_t)physical_page * (size_t)page_size + (size_t)within_page) * kv_dim;
    return key_pages + token_offset;
  }

  __device__ __forceinline__ hip_bfloat16 *get_value_token_ptr(int slot_id, int layer, int cache_pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.max_pages <= 0) return nullptr;
    int logical_page = cache_pos / page_size;
    if (logical_page >= slot.max_pages) return nullptr;
    int within_page = cache_pos % page_size;
    int table_index = slot_table_base(slot_id, layer) + logical_page;
    int physical_page = page_table[table_index];
    if (physical_page < 0) return nullptr;
    size_t token_offset = ((size_t)physical_page * (size_t)page_size + (size_t)within_page) * kv_dim;
    return value_pages + token_offset;
  }

  __device__ __forceinline__ int get_write_position(int slot_id, int layer, int pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.layer_type == SLIDING_WINDOW && sliding_window_size > 0) {
      return pos % sliding_window_size;
    }
    return pos;
  }

  __device__ __forceinline__ int get_cache_position(int slot_id, int layer,
                                                    int logical_pos, int current_pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.layer_type == SLIDING_WINDOW && sliding_window_size > 0) {
      if (current_pos < sliding_window_size) {
        return logical_pos;
      }
      int window_start = current_pos - sliding_window_size + 1;
      if (logical_pos < window_start) {
        return -1;
      }
      int offset_in_window = logical_pos - window_start;
      int cache_start_offset = window_start % sliding_window_size;
      return (cache_start_offset + offset_in_window) % sliding_window_size;
    }
    return logical_pos;
  }

  __device__ __forceinline__ void get_attention_range(int slot_id, int layer, int current_pos,
                                                      int *start_pos, int *end_pos) const {
    (void)slot_id;
    LayerType lt = get_layer_type(layer);
    if (lt == SLIDING_WINDOW && sliding_window_size > 0) {
      int start = current_pos - sliding_window_size + 1;
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
  KVCacheManager(int max_batch, int n_layers, int seq_len, size_t kv_dim, int sliding_window_size)
      : max_batch_(max_batch),
        n_layers_(n_layers),
        seq_len_(seq_len),
        kv_dim_(kv_dim),
        sliding_window_size_(sliding_window_size),
        page_size_(16),
        pages_per_slot_(0),
        total_logical_pages_(0),
        total_physical_pages_(0),
        key_pages_(nullptr),
        value_pages_(nullptr),
        page_table_(nullptr),
        d_slots_(nullptr),
        d_device_data_(nullptr) {
    const char *env_page = getenv("GETP_KV_PAGE_SIZE");
    if (env_page && atoi(env_page) > 0) {
      int candidate = atoi(env_page);
      if (candidate > 0 && (candidate % 4 == 0)) {
        page_size_ = candidate;
      }
    }

    // Build per-layer metadata (shared across slots)
    layer_offsets_.resize(n_layers_);
    slots_.resize(max_batch_);
    for (int b = 0; b < max_batch_; ++b) {
      slots_[b].resize(n_layers_);
    }

    int running_offset = 0;
    std::vector<int> layer_pages(n_layers_);
    for (int l = 0; l < n_layers_; ++l) {
      LayerType lt = host_layer_type(l);
      int capacity = (lt == SLIDING_WINDOW && sliding_window_size_ > 0)
                     ? sliding_window_size_ : seq_len_;
      int pages = (capacity + page_size_ - 1) / page_size_;
      if (pages < 1) pages = 1;
      layer_offsets_[l] = running_offset;
      layer_pages[l] = pages;
      running_offset += pages;
      for (int b = 0; b < max_batch_; ++b) {
        KVCacheSlot slot;
        slot.page_table_offset = layer_offsets_[l];
        slot.max_pages = pages;
        slot.layer_type = lt;
        slots_[b][l] = slot;
      }
    }
    pages_per_slot_ = running_offset;
    total_logical_pages_ = max_batch_ * pages_per_slot_;

    size_t total_entries = (size_t)max_batch_ * (size_t)pages_per_slot_;
    if (total_entries == 0) {
      fprintf(stderr, "KVCacheManager: zero-sized page table.\n");
      exit(EXIT_FAILURE);
    }

    // Determine physical page budget from available memory.
    size_t bytes_per_page = (size_t)page_size_ * kv_dim_ * sizeof(hip_bfloat16);
    size_t free_bytes = 0, total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    double usage_fraction = 0.80;
    const char *env_frac = getenv("GETP_KV_PAGE_FRACTION");
    if (env_frac && atof(env_frac) > 0.0) {
      double f = atof(env_frac);
      if (f > 0.0 && f <= 1.0) usage_fraction = f;
    }
    size_t budget_bytes = (size_t)(usage_fraction * (double)free_bytes);
    if (budget_bytes < bytes_per_page * 2) {
      fprintf(stderr, "KVCacheManager: insufficient free memory for paged cache.\n");
      exit(EXIT_FAILURE);
    }

    size_t max_pages_by_mem = budget_bytes / (bytes_per_page * 2);
    if (max_pages_by_mem > (size_t)total_logical_pages_) {
      max_pages_by_mem = (size_t)total_logical_pages_;
    }

    // Ensure minimal pages to cover sliding windows + first token for full layers
    size_t min_required_pages = 0;
    for (int l = 0; l < n_layers_; ++l) {
      LayerType lt = host_layer_type(l);
      if (lt == SLIDING_WINDOW && sliding_window_size_ > 0) {
        min_required_pages += (size_t)max_batch_ * (size_t)layer_pages[l];
      } else {
        min_required_pages += (size_t)max_batch_;
      }
    }
    if (max_pages_by_mem < min_required_pages) {
      // Attempt to allocate minimum pages; if impossible, fail explicitly.
      if (budget_bytes < min_required_pages * bytes_per_page * 2) {
        fprintf(stderr, "KVCacheManager: not enough memory for minimal KV allocation (need %.2f GB).\n",
                (min_required_pages * bytes_per_page * 2) / GB);
        exit(EXIT_FAILURE);
      }
      max_pages_by_mem = min_required_pages;
    }

    total_physical_pages_ = (int)max_pages_by_mem;
    if (total_physical_pages_ <= 0) {
      fprintf(stderr, "KVCacheManager: zero physical pages after budgeting.\n");
      exit(EXIT_FAILURE);
    }

    // Allocate page pools for keys/values and page table (managed memory for ease of updates).
    HIP_CHECK(hipMalloc(&key_pages_, (size_t)total_physical_pages_ * bytes_per_page));
    HIP_CHECK(hipMalloc(&value_pages_, (size_t)total_physical_pages_ * bytes_per_page));

    HIP_CHECK(hipMallocManaged(&page_table_, total_entries * sizeof(int), hipMemAttachGlobal));
    HIP_CHECK(hipMemset(page_table_, 0xFF, total_entries * sizeof(int))); // -1 marks free

    free_pages_.reserve(total_physical_pages_);
    for (int i = 0; i < total_physical_pages_; ++i) {
      free_pages_.push_back(i);
    }

    // Copy slot metadata to device
    size_t slots_bytes = (size_t)max_batch_ * (size_t)n_layers_ * sizeof(KVCacheSlot);
    HIP_CHECK(hipMalloc(&d_slots_, slots_bytes));
    std::vector<KVCacheSlot> flat_slots;
    flat_slots.reserve((size_t)max_batch_ * (size_t)n_layers_);
    for (int b = 0; b < max_batch_; ++b) {
      for (int l = 0; l < n_layers_; ++l) {
        flat_slots.push_back(slots_[b][l]);
      }
    }
    HIP_CHECK(hipMemcpy(d_slots_, flat_slots.data(), slots_bytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&d_device_data_, sizeof(KVCacheDeviceData)));
    KVCacheDeviceData device_data;
    device_data.key_pages = key_pages_;
    device_data.value_pages = value_pages_;
    device_data.slots = d_slots_;
    device_data.page_table = page_table_;
    device_data.kv_dim = kv_dim_;
    device_data.max_batch = max_batch_;
    device_data.n_layers = n_layers_;
    device_data.seq_len = seq_len_;
    device_data.sliding_window_size = sliding_window_size_;
    device_data.page_size = page_size_;
    device_data.pages_per_slot = pages_per_slot_;
    HIP_CHECK(hipMemcpy(d_device_data_, &device_data, sizeof(KVCacheDeviceData), hipMemcpyHostToDevice));

    double total_gb = ((double)total_physical_pages_ * (double)page_size_ * (double)kv_dim_ * sizeof(hip_bfloat16) * 2.0) / GB;
    double tokens_capacity = (double)total_physical_pages_ * (double)page_size_;
    printf("KVCacheManager (paged): %d physical pages, page_size=%d tokens (capacity %.2fM tokens, %.2f GB)\n",
           total_physical_pages_, page_size_, tokens_capacity / 1e6, total_gb);
    if (sliding_window_size_ > 0) {
      printf("  - Sliding window tokens per layer: %d\n", sliding_window_size_);
    }

    // Advise managed memory for all visible GPUs (best-effort)
    int device_count = 0;
    hipError_t dev_err = hipGetDeviceCount(&device_count);
    if (dev_err == hipSuccess) {
      for (int dev = 0; dev < device_count; ++dev) {
        hipError_t advise_err = hipMemAdvise(page_table_, total_entries * sizeof(int), hipMemAdviseSetAccessedBy, dev);
        if (advise_err != hipSuccess && advise_err != hipErrorNotSupported) {
          fprintf(stderr, "KVCacheManager: hipMemAdvise failed on device %d: %s\n",
                  dev, hipGetErrorString(advise_err));
        }
      }
    }
  }

  ~KVCacheManager() {
    if (d_device_data_) HIP_CHECK(hipFree(d_device_data_));
    if (d_slots_) HIP_CHECK(hipFree(d_slots_));
    if (page_table_) HIP_CHECK(hipFree(page_table_));
    if (key_pages_) HIP_CHECK(hipFree(key_pages_));
    if (value_pages_) HIP_CHECK(hipFree(value_pages_));
  }

  KVCacheDeviceData *get_device_data() const { return d_device_data_; }

  void prepare_layer(int layer, const int *slot_ids, const int *positions, int batch) {
    if (batch <= 0) return;
    for (int bi = 0; bi < batch; ++bi) {
      int sid = slot_ids[bi];
      if (sid < 0 || sid >= max_batch_) continue;
      int pos = positions[bi];
      if (pos < 0) pos = 0;
      const KVCacheSlot &slot_info = slots_[sid][layer];
      int write_pos = (slot_info.layer_type == SLIDING_WINDOW && sliding_window_size_ > 0)
                        ? (pos % sliding_window_size_) : pos;
      int logical_page = write_pos / page_size_;
      if (logical_page >= slot_info.max_pages) {
        fprintf(stderr, "KVCacheManager: token position overflow (slot=%d layer=%d pos=%d).\n",
                sid, layer, pos);
        exit(EXIT_FAILURE);
      }
      size_t table_index = slot_table_index(sid, layer, logical_page);
      if (page_table_[table_index] < 0) {
        int phys = allocate_page();
        if (phys < 0) {
          fprintf(stderr, "KVCacheManager: ran out of KV pages (slot=%d layer=%d pos=%d).\n",
                  sid, layer, pos);
          exit(EXIT_FAILURE);
        }
        page_table_[table_index] = phys;
      }
    }
  }

  void reset_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= max_batch_) return;
    size_t base = (size_t)slot_id * (size_t)pages_per_slot_;
    for (int offset = 0; offset < pages_per_slot_; ++offset) {
      int idx = page_table_[base + offset];
      if (idx >= 0) {
        free_pages_.push_back(idx);
        page_table_[base + offset] = -1;
      }
    }
  }

  int page_size() const { return page_size_; }

private:
  LayerType host_layer_type(int layer) const {
    return (sliding_window_size_ > 0 && (layer % 2 == 0)) ? SLIDING_WINDOW : FULL_ATTENTION;
  }

  size_t slot_table_index(int slot_id, int layer, int logical_page) const {
    const KVCacheSlot &slot = slots_[slot_id][layer];
    return (size_t)slot_id * (size_t)pages_per_slot_ + (size_t)slot.page_table_offset + (size_t)logical_page;
  }

  int allocate_page() {
    if (free_pages_.empty()) {
      return -1;
    }
    int page = free_pages_.back();
    free_pages_.pop_back();
    return page;
  }

  int max_batch_;
  int n_layers_;
  int seq_len_;
  size_t kv_dim_;
  int sliding_window_size_;
  int page_size_;
  int pages_per_slot_;
  int total_logical_pages_;
  int total_physical_pages_;
  hip_bfloat16 *key_pages_;
  hip_bfloat16 *value_pages_;
  int *page_table_;
  std::vector<int> free_pages_;
  std::vector<int> layer_offsets_;
  std::vector<std::vector<KVCacheSlot>> slots_;
  KVCacheSlot *d_slots_;
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
  float *logits;      // (unused for final FP32 logits, but kept)
  KVCacheManager *kv_cache_manager;
  float *cos_vals;
  float *sin_vals;
  hip_bfloat16 *t_bf16;
  // FP32 buffers for accumulation and precision-sensitive ops
  float *acc_buffer;         // x in FP32 for output matvec
  float *logits_fp32;        // final logits
  // Per-stream scratch buffers for expert parallelization
  float *mlp1_out_s;   // [MAX_EXPERT_STREAMS][2 * intermediate_dim]
  float *gate_s;       // [MAX_EXPERT_STREAMS][intermediate_dim]
  float *up_s;         // [MAX_EXPERT_STREAMS][intermediate_dim]
  float *gate_up_s;    // [MAX_EXPERT_STREAMS][intermediate_dim]
  float *tb2_s;        // [MAX_EXPERT_STREAMS][hidden_dim]
  // Batched MoE staging (for grouped GEMMs across slots/experts)
  int   *sel_sids;        // [max_batch * experts_per_token]
  int   *sel_eids;        // [max_batch * experts_per_token]
  float *sel_w;           // [max_batch * experts_per_token]
  float *mlp1_out_all;    // [max_batch * experts_per_token][2 * intermediate_dim]
  hip_bfloat16 *gate_up_all;     // [max_batch * experts_per_token][intermediate_dim]
  float *tb2_peer;        // cross-GPU accumulation buffers (leader)
  float *e_agg_peer;
  float *router_score_full;
  // MFMA packed buffers
  float *xpack_f32;       // Packed input buffer for MFMA [K_max x Bpad]
  // Grouped MoE metadata per launch
  int   *sel_block_cols;   // block -> base selection index (size >= selections)
  int   *sel_block_sizes;  // block -> number of valid columns in block (<=16)
  int   *sel_block_eids;   // block -> expert id
} GPURunState;

struct ParallelLayout {
  int tp_size = 1;
  int ep_size = 1;
  int dp_size = 1;
};

static ParallelLayout parallel_layout;

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
  int tp_rank;
  int tp_size;
  int dp_rank;
  int dp_size;
  int ep_rank;
  int ep_size;
  int head_offset_q;
  int head_offset_kv;
  int expert_offset;
  int local_q_heads;
  int local_kv_heads;
  int local_experts;
  int local_q_dim;
  int local_kv_dim;
  int local_qkv_dim;
  bool is_leader;
  GPUTransformerWeights weights;
  GPURunState state;
  hipStream_t expert_streams[MAX_EXPERT_STREAMS];
  int num_expert_streams;
  int max_batch; // number of concurrent slots per GPU for continuous batching
  std::vector<hipStream_t> slot_streams; // persistent stream per slot
  hipStream_t batch_stream;              // shared stream for batched ops
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

using float4v = __attribute__((ext_vector_type(4))) float;
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
__global__ __launch_bounds__(64)
void mlp1_grouped_mfma_matvec_mxfp4_kernel(
    float* __restrict__ mlp1_out_all,        // [M selections][out_dim]
    const hip_bfloat16* __restrict__ t_bf16, // [B slots][hidden_dim] BF16
    const uint8_t* __restrict__ w_data,      // MXFP4 nibbles (tile-packed 16x16)
    const uint8_t* __restrict__ w_scales,    // tile-packed (16 rows per tile) e8m0 codes
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
  const int tx = threadIdx.x;  // 0..15 (columns)
  const int ty = threadIdx.y;  // 0..3  (row groups)
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m  = blockIdx.x;  // along output rows (hidden_dim)
  const int block_id = blockIdx.y; // block covering up to 16 selections for one expert
  if (block_id >= total_blocks) return;

  const int col0 = sel_block_cols[block_id];
  const int valid_cols = sel_block_sizes[block_id];
  const int eid = sel_block_eids[block_id];
  if (valid_cols <= 0) return;

  const int base_row = tile_m * 16;

  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ B = b_mlp1 ? (b_mlp1 + mat_index * (size_t)out_dim) : nullptr;

  __shared__ __align__(16) uint16_t sWbf16[16 * 16];
  __shared__ __align__(16) uint16_t sXbf16[16 * 16];
  __shared__ __align__(16) float sBias[16];
  __shared__ __align__(16) const hip_bfloat16* sXPtrs[16];
  __shared__ __align__(16) float* sOutPtrs[16];

  if (lane < 16) {
    // Cache bias and per-selection base pointers once per block to reuse across all K tiles.
    const int row = base_row + lane;
    sBias[lane] = (B && row < out_dim) ? bfloat16_to_float(B[row]) : 0.f;
    const int col = lane;
    const int msel = col0 + col;
    const bool active = (col < valid_cols) && (msel < selections_M);
    const hip_bfloat16* x_ptr = active ? (t_bf16 + (size_t)sel_sids[msel] * (size_t)hidden_dim) : nullptr;
    sXPtrs[col] = x_ptr;
    sOutPtrs[col] = active ? (mlp1_out_all + (size_t)msel * (size_t)out_dim) : nullptr;
  }
  __syncthreads();

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int K = hidden_dim;
  const int tiles_m = (out_dim + 15) / 16;
  const int tiles_k = (K + 15) / 16;
  const size_t packed_mat_bytes = (size_t)tiles_m * (size_t)tiles_k * (size_t)128; // 128 bytes per 16x16 tile (256 nibbles)
  const uint8_t* __restrict__ Wpack = w_data + mat_index * packed_mat_bytes;
  const size_t tiles_per_row = (size_t)tiles_k;
  const size_t scales_per_mat = (size_t)tiles_m * (size_t)tiles_k * (size_t)16;
  const uint8_t* __restrict__ Wscale = w_scales + mat_index * scales_per_mat;

  for (int kk = 0; kk < tiles_k; ++kk) {
    const size_t tile_linear = (size_t)tile_m * tiles_per_row + (size_t)kk;
    const size_t tile_base_byte = tile_linear * (size_t)128;
    const size_t scale_base = tile_linear * (size_t)16;

    // Fill W tile [16x16] by dequantizing from MXFP4
    int idx4 = lane * 4;
    if (idx4 < 256) {
      int im = idx4 >> 4;            // tile row for this lane
      int ik_base = idx4 & 0xF;      // starting column within row
      int row = base_row + im;
      int col_base = kk * 16 + ik_base;
      int byte_off = tile_base_byte + (idx4 >> 1);
      uint16_t packed16 = *reinterpret_cast<const uint16_t*>(Wpack + byte_off);
      int lane_group = lane & ~3;
      int scale_code_int = 0x7F;
      if ((lane & 3) == 0 && row < out_dim) {
        scale_code_int = static_cast<int>(Wscale[scale_base + im]);
      }
      scale_code_int = __shfl(scale_code_int, lane_group, 4);
      uint8_t scale_code = static_cast<uint8_t>(scale_code_int & 0xFF);
      const uint16_t* __restrict__ lut_row = MXFP4_DEQUANT_LUT_BF16[scale_code];
      #pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        int ik = ik_base + t4;
        int col = col_base + t4;
        uint16_t bits = 0;
        if (row < out_dim && col < K) {
          uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
          bits = lut_row[nibble];
        }
        sWbf16[im * 16 + ik] = bits;
      }
    }

    // Build X tile [16x16] by gathering up to 16 distinct selections
    if (lane < 64) {
      const int r   = lane % 16;        // K row within tile
      const int c4  = (lane / 16) * 4;  // column block index {0,4,8,12}
      const int k_row = kk * 16 + r;
      uint16_t* dst = sXbf16 + r * 16 + c4;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        int col = c4 + jj;
        uint16_t bits = 0;
        const hip_bfloat16* x = sXPtrs[col];
        if (x && col < valid_cols && k_row < K) {
          hip_bfloat16 val = x[k_row];
          bits = bfloat16_to_bits(val);
        }
        dst[jj] = bits;
      }
    }

    __syncthreads();

    const int kbase = ty * 4;
    short4v Areg;
    short4v Breg;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int kidx = kbase + j;
      uint16_t a_bits = sWbf16[tx * 16 + kidx];
      uint16_t b_bits = sXbf16[kidx * 16 + tx];
      Areg[j] = static_cast<short>(a_bits);
      Breg[j] = static_cast<short>(b_bits);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    __syncthreads();
  }

  float* out_vec = sOutPtrs[tx];
  if (out_vec) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int rloc = ty * 4 + i;
      const int row = base_row + rloc;
      if (row < out_dim) {
        out_vec[row] = acc[i] + sBias[rloc];
      }
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

// Grouped MLP2 matvec using MFMA and tile-packed MXFP4 weights.
// For each selection m: y = W[layer,eid] @ g[m] + b[layer,eid]; then atomicAdd into e_agg[sid] with weight sel_w[m].
__global__ __launch_bounds__(64)
void mlp2_accum_grouped_mfma_matvec_mxfp4_kernel(
    float* __restrict__ e_agg,                // [slots, hidden_dim]
    const hip_bfloat16* __restrict__ gate_up_all, // [M, intermediate_dim] BF16
    const uint8_t* __restrict__ w_data,       // tile-packed MXFP4 nibbles (16x16)
    const uint8_t* __restrict__ w_scales,     // tile-packed (16 rows per tile) e8m0 codes
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

  const int tx = threadIdx.x;  // 0..15
  const int ty = threadIdx.y;  // 0..3
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m  = blockIdx.x;  // along output rows (hidden_dim)
  const int block_id = blockIdx.y; // block covering up to 16 selections for one expert
  if (block_id >= total_blocks) return;

  const int col0 = sel_block_cols[block_id];
  const int valid_cols = sel_block_sizes[block_id];
  const int eid = sel_block_eids[block_id];
  if (valid_cols <= 0) return;

  const int base_row = tile_m * 16;

  const int Mout = hidden_dim;
  const int Kdim = intermediate_dim;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ B = b_mlp2 ? (b_mlp2 + mat_index * (size_t)hidden_dim) : nullptr;

  __shared__ __align__(16) uint16_t sWbf16[16 * 16];
  __shared__ __align__(16) uint16_t sXbf16[16 * 16];
  __shared__ __align__(16) float sBias[16];

  if (lane < 16) {
    const int row = base_row + lane;
    sBias[lane] = (B && row < Mout) ? bfloat16_to_float(B[row]) : 0.f;
  }

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int tiles_m = (Mout + 15) / 16;
  const int tiles_k = (Kdim + 15) / 16;
  const size_t packed_mat_bytes = (size_t)tiles_m * (size_t)tiles_k * (size_t)128;
  const uint8_t* __restrict__ Wpack = w_data + mat_index * packed_mat_bytes;
  const size_t tiles_per_row = (size_t)tiles_k;
  const size_t scales_per_mat = (size_t)tiles_m * (size_t)tiles_k * (size_t)16;
  const uint8_t* __restrict__ Wscale = w_scales + mat_index * scales_per_mat;

  for (int kk = 0; kk < tiles_k; ++kk) {
    const size_t tile_linear = (size_t)tile_m * tiles_per_row + (size_t)kk;
    const size_t tile_base_byte = tile_linear * (size_t)128;
    const size_t scale_base = tile_linear * (size_t)16;

    int idx4 = lane * 4;
    if (idx4 < 256) {
      const int im = idx4 >> 4;
      const int ik_base = idx4 & 0xF;
      const int row = base_row + im;
      const int col_base = kk * 16 + ik_base;
      const int byte_off = tile_base_byte + (idx4 >> 1);
      const uint16_t packed16 = *reinterpret_cast<const uint16_t*>(Wpack + byte_off);
      int scale_code_int = 0x7F;
      const int lane_group = lane & ~3;
      if ((lane & 3) == 0 && row < Mout) {
        scale_code_int = static_cast<int>(Wscale[scale_base + im]);
      }
      scale_code_int = __shfl(scale_code_int, lane_group, 4);
      const uint8_t scale_code = static_cast<uint8_t>(scale_code_int & 0xFF);
      const uint16_t* __restrict__ lut_row = MXFP4_DEQUANT_LUT_BF16[scale_code];
      #pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        const int ik = ik_base + t4;
        const int col = col_base + t4;
        uint16_t bits = 0;
        if (row < Mout && col < Kdim) {
          const uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
          bits = lut_row[nibble];
        }
        sWbf16[im * 16 + ik] = bits;
      }
    }

    if (lane < 64) {
      const int r   = lane % 16;        // K row within tile
      const int c4  = (lane / 16) * 4;  // column block index {0,4,8,12}
      const int k_row = kk * 16 + r;
      uint16_t* dst = sXbf16 + r * 16 + c4;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int col = c4 + jj;
        uint16_t bits = 0;
        const int msel = col0 + col;
        if (col < valid_cols && msel < selections_M && k_row < Kdim) {
          const hip_bfloat16* x_ptr = gate_up_all + (size_t)msel * (size_t)intermediate_dim;
          bits = bfloat16_to_bits(x_ptr[k_row]);
        }
        dst[jj] = bits;
      }
    }

    __syncthreads();

    short4v Areg;
    short4v Breg;
    const int kbase = ty * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kidx = kbase + j;
      const uint16_t a_bits = sWbf16[tx * 16 + kidx];
      const uint16_t b_bits = sXbf16[kidx * 16 + tx];
      Areg[j] = static_cast<short>(a_bits);
      Breg[j] = static_cast<short>(b_bits);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    __syncthreads();
  }

  const int msel = col0 + tx;
  if (tx < valid_cols && msel < selections_M) {
    const int sid = sel_sids[msel];
    const float ew = sel_w[msel];
    float* out_slot = e_agg + (size_t)sid * (size_t)eagg_stride;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int rloc = ty * 4 + i;
      const int row = base_row + rloc;
      if (row < Mout) {
        float y = acc[i] + sBias[rloc];
        atomicAdd(&out_slot[row], ew * y);
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

// Gather và pack input X từ multiple slots thành format cho MFMA
__global__ void gather_pack_x_fp32_to_fp32_mfma_kernel(const float *x, float *xpack,
                                                       int K, int B, int Bpad,
                                                       int in_stride, const int *slot_ids) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; // row in [0, K)
  int b = blockIdx.y; // batch index
  if (row < K && b < B) {
    int sid = slot_ids[b];
    const float *x_slot = x + (size_t)sid * (size_t)in_stride;
    xpack[(size_t)row * (size_t)Bpad + (size_t)b] = x_slot[row];
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
void gemm16x16_mfma_f32_packedW_batched16_scatter_bias_kernel(
    const hip_bfloat16* __restrict__ W_packed,    // [Mpad x Kpad] tile16 BF16
    const float* __restrict__ Xpack,              // [K x Bpad] FP32 (cột = batch)
    float* __restrict__ dst,                      // scatter theo slot_ids
    const hip_bfloat16* __restrict__ bias,        // [M] hoặc nullptr
    int M, int K, int B, int Bpad,
    int out_stride,                               // = M
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad)
{
  // mapping 64 lanes: tx=col (0..15), ty=row groups (0..3); mỗi lane có 4 accumulator (4 hàng)
  const int tx = threadIdx.x;     // cột trong tile (0..15) = cột batch tương ứng
  const int ty = threadIdx.y;     // nhóm hàng (0..3)
  const int lane = ty * 16 + tx;  // 0..63

  const int tile_m  = blockIdx.x;          // tile dọc theo M
  const int b_group = blockIdx.y;          // nhóm 16 batch
  const int b_base  = b_group * 16;
  if (b_base >= B) return;

  // Số cột thực sự trong nhóm (nhóm cuối có thể < 16)
  const int cols_this = min(16, B - b_base);

  // Hàng cơ sở của tile M
  const int base_row = tile_m * 16;

  // Shared ping-pong
  __shared__ __align__(16) float sW[2][16 * 16];
  __shared__ __align__(16) float sX[2][16 * 16];
  __shared__ int s_slot[16];
  __shared__ float s_bias[16];

  using float4v = __attribute__((ext_vector_type(4))) float;

  // Helper: nạp 4 BF16 -> 4 FP32 (giống hàm bạn đang có, để inline tại đây cho tốc độ)
  auto bf16x4_to_f4 = [&](const hip_bfloat16* src) {
    // đọc 2x uint32 (mỗi uint32 chứa 2 BF16)
    const uint32_t *p32 = reinterpret_cast<const uint32_t*>(src);
    uint32_t p0 = p32[0];
    uint32_t p1 = p32[1];
    float4v v;
    // chuyển từng nửa-word sang float
    auto bits2f = [] __device__ (uint16_t bits) {
      union { uint32_t u; float f; } cvt;
      cvt.u = uint32_t(bits) << 16;
      return cvt.f;
    };
    v[0] = bits2f((uint16_t)(p0 & 0xFFFFu));
    v[1] = bits2f((uint16_t)(p0 >> 16));
    v[2] = bits2f((uint16_t)(p1 & 0xFFFFu));
    v[3] = bits2f((uint16_t)(p1 >> 16));
    return v;
  };

  const bool has_bias = (bias != nullptr);
  const bool full_tile = (cols_this == 16);

  // Chuẩn bị slot_id và bias tile cho block
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
    s_bias[col] = bias_val;
  }
  __syncthreads();

  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int Ktiles        = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)Kpad / 16;       // do Kpad bội 16
  const size_t tile_elems    = 256;                     // 16*16 BF16/FP32
  const hip_bfloat16* w_tile_row_base =
      W_packed + ((size_t)tile_m * tiles_per_row) * tile_elems;

  const size_t x_stride = static_cast<size_t>(Bpad);
  const size_t b_base_sz = static_cast<size_t>(b_base);

  // Preload Ô-0 (ping=0)
  {
    const int row_in_tile = lane & 15;      // 0..15
    const int col_group   = lane >> 4;      // 0..3 (4 nhóm cột cho vector write)

    // nạp W[16x16] tile k=0
    const hip_bfloat16* w_src0 = w_tile_row_base + 0 * tile_elems;
    reinterpret_cast<float4v*>(&sW[0][0])[lane] = bf16x4_to_f4(w_src0 + lane * 4);

    // nạp X[16x16]: lấy 16 hàng liên tiếp (k_row) × 16 cột batch (b_base..b_base+15)
    const int k_row0 = 0 * 16 + row_in_tile;
    float4v xv = {0.f, 0.f, 0.f, 0.f};
    if (k_row0 < K) {
      const int col_base = col_group * 4;
      const size_t row_offset = static_cast<size_t>(k_row0) * x_stride + b_base_sz + static_cast<size_t>(col_base);
      if (full_tile) {
        // Fast path: vector load when all 16 batch columns are valid
        xv = reinterpret_cast<const float4v*>(Xpack + row_offset)[0];
      } else {
        // Tail group: guard each column individually
        if (col_base + 0 < cols_this) xv[0] = Xpack[row_offset + 0];
        if (col_base + 1 < cols_this) xv[1] = Xpack[row_offset + 1];
        if (col_base + 2 < cols_this) xv[2] = Xpack[row_offset + 2];
        if (col_base + 3 < cols_this) xv[3] = Xpack[row_offset + 3];
      }
    }
    reinterpret_cast<float4v*>(&sX[0][row_in_tile * 16])[col_group] = xv;
  }
  __syncthreads();

  int ping = 0;

  // Vòng lặp trên K-tiles với prefetch ô tiếp theo vào pong
  for (int kt = 0; kt < Ktiles; ++kt)
  {
    const int pong = ping ^ 1;

    // Prefetch ô kế tiếp nếu có
    if (kt + 1 < Ktiles) {
      const int row_in_tile = lane & 15;
      const int col_group   = lane >> 4;

      // nạp W tile (kt+1) vào ping-pong
      const hip_bfloat16* w_src_next = w_tile_row_base + (size_t)(kt + 1) * tile_elems;
      reinterpret_cast<float4v*>(&sW[pong][0])[lane] = bf16x4_to_f4(w_src_next + lane * 4);

      // nạp X tile (kt+1)
      const int k_row = (kt + 1) * 16 + row_in_tile;
      float4v xv = {0.f, 0.f, 0.f, 0.f};
      if (k_row < K) {
        const int col_base = col_group * 4;
        const size_t row_offset = static_cast<size_t>(k_row) * x_stride + b_base_sz + static_cast<size_t>(col_base);
        if (full_tile) {
          // Fast path: vector load when all 16 batch columns are valid
          xv = reinterpret_cast<const float4v*>(Xpack + row_offset)[0];
        } else {
          // Tail group: guard each column individually
          if (col_base + 0 < cols_this) xv[0] = Xpack[row_offset + 0];
          if (col_base + 1 < cols_this) xv[1] = Xpack[row_offset + 1];
          if (col_base + 2 < cols_this) xv[2] = Xpack[row_offset + 2];
          if (col_base + 3 < cols_this) xv[3] = Xpack[row_offset + 3];
        }
      }
      reinterpret_cast<float4v*>(&sX[pong][row_in_tile * 16])[col_group] = xv;
    }

    // Tính trên ô hiện tại (ping)
    __syncthreads();
    // mỗi lane đọc theo “k4” = ty + 4*t
    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4;         // 0..15
      const float a_val = sW[ping][tx * 16 + k4]; // A(row=tx, col=k4)
      const float b_val = sX[ping][k4 * 16 + tx]; // B(row=k4,  col=tx)
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_val, acc, 0, 0, 0);
    }

    ping = pong;
  }
  __syncthreads();

  // Ghi ra 16 cột batch (không còn tx==0 nữa!)
  // Với MFMA 16x16x4f32: mỗi lane giữ 4 phần tử theo trục hàng của cột tx.
  // Các hàng lane này: r = base_row + (ty*4 + i), i=0..3
  const int r0 = base_row + ty * 4 + 0;
  const int r1 = base_row + ty * 4 + 1;
  const int r2 = base_row + ty * 4 + 2;
  const int r3 = base_row + ty * 4 + 3;

  // Nếu cột batch này hợp lệ thì scatter vào slot tương ứng
  if (tx < cols_this) {
    const int sid = s_slot[tx];
    if (sid >= 0) {
      float* __restrict__ out = dst + (size_t)sid * (size_t)out_stride;
      const int bias_base = ty * 4;
      if (r0 < M) out[r0] = acc[0] + s_bias[bias_base + 0];
      if (r1 < M) out[r1] = acc[1] + s_bias[bias_base + 1];
      if (r2 < M) out[r2] = acc[2] + s_bias[bias_base + 2];
      if (r3 < M) out[r3] = acc[3] + s_bias[bias_base + 3];
    }
  }
}

// Launcher tối ưu: gom batch theo nhóm 16 cột/khối.
// Wrapper function cho MFMA-based batched matmul
void launch_gemm16x16_mfma_f32_batched(
    const hip_bfloat16* W_packed,
    const float* X_packed,
    float* output,
    const hip_bfloat16* bias,
    int M, int K, int batch, int Bpad,
    int out_stride,
    const int* slot_ids,
    hipStream_t stream)
{
  const int Mpad = ((M + 15) / 16) * 16;
  const int Kpad = ((K + 15) / 16) * 16;

  // Dùng kernel tối ưu: 1 block xử lý 16 cột batch
  const int group_y = (batch + 15) / 16;
  dim3 block(16, 4, 1);                   // 64 threads / MFMA 16x16x4f32
  dim3 grid((Mpad + 15) / 16, group_y, 1);

  gemm16x16_mfma_f32_packedW_batched16_scatter_bias_kernel<<<grid, block, 0, stream>>>(
      W_packed, X_packed, output, bias,
      M, K, batch, Bpad, out_stride, slot_ids, Mpad, Kpad);
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
  if (threadIdx.x != 0) return;
  int sid = slot_ids[b];
  const float *scores = router_scores + (size_t)sid * (size_t)score_stride;
  float *topv_s = topv + (size_t)sid * (size_t)topk_stride;
  int   *topi_s = topi + (size_t)sid * (size_t)topk_stride;
  // init
  for (int i = 0; i < k; ++i) { topv_s[i] = -INFINITY; topi_s[i] = -1; }
  // top-k selection
  for (int j = 0; j < n; ++j) {
    float v = scores[j];
    int pos = -1;
    for (int t = 0; t < k; ++t) { if (v > topv_s[t]) { pos = t; break; } }
    if (pos >= 0) {
      for (int t = k - 1; t > pos; --t) { topv_s[t] = topv_s[t - 1]; topi_s[t] = topi_s[t - 1]; }
      topv_s[pos] = v; topi_s[pos] = j;
    }
  }
  // softmax over top-k
  float maxv = topv_s[0];
  for (int i = 1; i < k; ++i) if (topv_s[i] > maxv) maxv = topv_s[i];
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) { topv_s[i] = expf(topv_s[i] - maxv); sum += topv_s[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < k; ++i) topv_s[i] *= inv;
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


// Paged FlashAttention kernel that operates on KVCacheManager tiles (BF16 storage with sliding-window support).
__global__ __launch_bounds__(64)
void paged_flash_attention_kernel_batched(
    const float * __restrict__ all_q,
    KVCacheDeviceData * __restrict__ cache_data,
    const hip_bfloat16 * __restrict__ attn_sinks,
    float * __restrict__ all_output,
    int head_dim, int kv_dim, int kv_mul,
    int n_heads, int seq_len, int layer,
    const int * __restrict__ pos_list,
    int batch,
    int q_stride,
    const int * __restrict__ slot_ids) {

  int h = blockIdx.y;
  int b = blockIdx.z;
  if (h >= n_heads || b >= batch) return;

  int pos = pos_list[b];
  if (pos < 0) return;

  int sid = slot_ids[b];
  const float *q_head = all_q + (size_t)sid * (size_t)q_stride + (size_t)h * (size_t)head_dim;
  float *output = all_output + (size_t)sid * (size_t)(n_heads * head_dim) + (size_t)h * (size_t)head_dim;

  (void)kv_dim;

  int start_pos = 0;
  int end_pos = 0;
  cache_data->get_attention_range(sid, layer, pos, &start_pos, &end_pos);
  int att_len = end_pos - start_pos + 1;
  if (att_len <= 0) return;

  int layer_capacity = cache_data->layer_capacity(layer);

  extern __shared__ unsigned char shared_storage[];
  float *shared_att_paged = reinterpret_cast<float*>(shared_storage);
  size_t att_storage = (size_t)layer_capacity + 1;
  uintptr_t value_ptr_base = reinterpret_cast<uintptr_t>(shared_att_paged + att_storage);
  const uintptr_t value_ptr_align = alignof(hip_bfloat16*);
  value_ptr_base = (value_ptr_base + value_ptr_align - 1) & ~(value_ptr_align - 1);
  hip_bfloat16 **shared_value_ptr = reinterpret_cast<hip_bfloat16 **>(value_ptr_base);

  __shared__ int s_att_len;
  if (threadIdx.x == 0) s_att_len = att_len;
  __syncthreads();

  const int lane_id = threadIdx.x;
  const int tx = lane_id & 15;
  const int ty = (lane_id >> 4) & 3;
  const float inv_sqrt_d = rsqrtf((float)head_dim);
  const int head_base = (h / kv_mul) * head_dim;

  __shared__ short sQbf16[16];
  __shared__ int sColPos[16];
  __shared__ hip_bfloat16 *sKeyPtrs[16];

  for (int col0 = 0; col0 < att_len; col0 += 16) {
    const int col = col0 + tx;
    const bool col_in_range = (col < att_len);

    if (ty == 0 && tx < 16) {
      if (col_in_range) {
        const int t = start_pos + col;
        const int cp = (t <= pos) ? cache_data->get_cache_position(sid, layer, t, pos) : -1;
        sColPos[tx] = cp;
        hip_bfloat16 *k_base = nullptr;
        if (cp >= 0) {
          k_base = cache_data->get_key_token_ptr(sid, layer, cp);
          if (k_base != nullptr) {
            k_base += head_base;
          }
        }
        sKeyPtrs[tx] = k_base;
      } else {
        sColPos[tx] = -1;
        sKeyPtrs[tx] = nullptr;
      }
    }
    __syncthreads();

    float4v acc = {0.f, 0.f, 0.f, 0.f};
    const int Ktiles = (head_dim + 15) / 16;
    for (int kk = 0; kk < Ktiles; ++kk) {
      if (ty == 0 && tx < 16) {
        const int k_row = kk * 16 + tx;
        unsigned short qbits = 0u;
        if (k_row < head_dim) {
          union { hip_bfloat16 b; unsigned short u; } uq;
          uq.b = float_to_bfloat16(q_head[k_row]);
          qbits = uq.u;
        }
        sQbf16[tx] = (short)qbits;
      }
      __syncthreads();

      const int kbase = ty * 4;
      short4v Areg;
      short4v Breg;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kij = kbase + j;
        Areg[j] = (tx == kij) ? sQbf16[kij] : (short)0;
        short kval = 0;
        const int k_abs = kk * 16 + kij;
        const hip_bfloat16 *kb = sKeyPtrs[tx];
        if (kb != nullptr && k_abs < head_dim) {
          union { hip_bfloat16 b; unsigned short u; } uk;
          uk.b = kb[k_abs];
          kval = (short)uk.u;
        }
        Breg[j] = kval;
      }

      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);
      __syncthreads();
    }

    float partial = acc.x + acc.y + acc.z + acc.w;
    partial += __shfl_xor(partial, 16, 64);
    partial += __shfl_xor(partial, 32, 64);

    if (ty == 0 && col_in_range) {
      float score = (sColPos[tx] >= 0) ? (partial * inv_sqrt_d) : -INFINITY;
      shared_att_paged[col] = score;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && (pos + 1) < seq_len && s_att_len < layer_capacity + 1) {
    size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    shared_att_paged[s_att_len] = bfloat16_to_float(attn_sinks[sink_idx]);
    s_att_len++;
  }
  __syncthreads();
  const int len = s_att_len;

  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    local_max = fmaxf(local_max, shared_att_paged[i]);
  }

  const int W = 64;
  for (int offset = W / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down(local_max, offset, W);
    local_max = fmaxf(local_max, other);
  }

  __shared__ float warp_max[4];
  const int lane = threadIdx.x & (W - 1);
  const int wid = threadIdx.x / W;
  if (lane == 0) warp_max[wid] = local_max;
  __syncthreads();

  float max_val = -INFINITY;
  if (wid == 0) {
    int warps = blockDim.x / W;
    float val = (lane < warps) ? warp_max[lane] : -INFINITY;
    for (int offset = W / 2; offset > 0; offset >>= 1) {
      float other = __shfl_down(val, offset, W);
      val = fmaxf(val, other);
    }
    if (lane == 0) warp_max[0] = val;
  }
  __syncthreads();
  max_val = warp_max[0];

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    float v = shared_att_paged[i];
    v = (v == -INFINITY) ? 0.0f : expf(v - max_val);
    shared_att_paged[i] = v;
    local_sum += v;
  }

  for (int offset = W / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset, W);
  }

  __shared__ float warp_sum[4];
  if (lane == 0) warp_sum[wid] = local_sum;
  __syncthreads();

  float sum_val = 0.0f;
  if (wid == 0) {
    int warps = blockDim.x / W;
    float val = (lane < warps) ? warp_sum[lane] : 0.0f;
    for (int offset = W / 2; offset > 0; offset >>= 1) {
      val += __shfl_down(val, offset, W);
    }
    if (lane == 0) warp_sum[0] = val;
  }
  __syncthreads();
  sum_val = warp_sum[0];
  float inv_sum = 1.0f / (sum_val + 1e-8f);
  __syncthreads();

  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    shared_att_paged[i] *= inv_sum;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
    hip_bfloat16 *ptr = nullptr;
    if (idx < att_len) {
      const int t = start_pos + idx;
      const int cache_pos = cache_data->get_cache_position(sid, layer, t, pos);
      if (cache_pos >= 0) {
        hip_bfloat16 *base = cache_data->get_value_token_ptr(sid, layer, cache_pos);
        if (base != nullptr) {
          ptr = base + head_base;
        }
      }
    }
    shared_value_ptr[idx] = ptr;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float acc = 0.0f;
    for (int idx = 0; idx < len; ++idx) {
      const int t = start_pos + idx;
      const float p = shared_att_paged[idx];
      if (t <= pos && p > 0.0f) {
        const hip_bfloat16 *v_ptr = shared_value_ptr[idx];
        if (v_ptr != nullptr) {
          acc += p * bfloat16_to_float(v_ptr[i]);
        }
      }
    }
    output[i] = acc;
  }
}


__global__ void sample_argmax_kernel(const float *logits, int *next_tokens,
                                     int batch, int vocab_size) {
    int b = blockIdx.x;
    if (b >= batch) return;

    const float *logit = logits + (size_t)b * (size_t)vocab_size;
    int tid = threadIdx.x;

    float max_val = -FLT_MAX;
    int max_idx = -1;

    // Each thread scans a strided subset of the vocabulary.
    for (int idx = tid; idx < vocab_size; idx += blockDim.x) {
        float val = logit[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = idx;
        }
    }

    extern __shared__ unsigned char smem[];
    float *s_vals = reinterpret_cast<float*>(smem);
    int *s_indices = reinterpret_cast<int*>(s_vals + blockDim.x);

    s_vals[tid] = max_val;
    s_indices[tid] = max_idx;
    __syncthreads();

    // Block-wide reduction to find the maximum value and its index.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other_val = s_vals[tid + offset];
            int other_idx = s_indices[tid + offset];
            if (other_val > s_vals[tid] ||
                (other_val == s_vals[tid] && other_idx < s_indices[tid])) {
                s_vals[tid] = other_val;
                s_indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        next_tokens[b] = s_indices[0] >= 0 ? s_indices[0] : 0;
    }
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
void gpu_clone_weights(GPUContext *dst_ctx, GPUContext *src_ctx, Config *p);

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

  // QKV shard
  size_t global_qkv_rows = head_dim * (n_attn_heads + 2 * n_kv_heads);
  size_t layer_qkv_elems = (size_t)hidden_dim * (size_t)ctx->local_qkv_dim;
  size_t total_qkv_elems = (size_t)n_layers * layer_qkv_elems;
  HIP_CHECK(hipMalloc(&gpu_w->w_qkv, total_qkv_elems * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&gpu_w->b_qkv, (size_t)n_layers * (size_t)ctx->local_qkv_dim * sizeof(hip_bfloat16)));

  size_t q_rows_total = (size_t)n_attn_heads * (size_t)head_dim;
  size_t kv_rows_total = (size_t)n_kv_heads * (size_t)head_dim;

  for (int l = 0; l < n_layers; ++l) {
    const float *layer_src = cpu_w->w_qkv + (size_t)l * (size_t)hidden_dim * (size_t)global_qkv_rows;
    hip_bfloat16 *layer_dst = gpu_w->w_qkv + (size_t)l * layer_qkv_elems;

    size_t q_start_row = (size_t)ctx->head_offset_q * (size_t)head_dim;
    size_t q_rows = (size_t)ctx->local_q_dim;
    copy_fp32_to_bf16(layer_src + q_start_row * (size_t)hidden_dim,
                      layer_dst,
                      q_rows * (size_t)hidden_dim);

    size_t k_start_row = q_rows_total + (size_t)ctx->head_offset_kv * (size_t)head_dim;
    size_t kv_rows = (size_t)ctx->local_kv_dim;
    copy_fp32_to_bf16(layer_src + k_start_row * (size_t)hidden_dim,
                      layer_dst + q_rows * (size_t)hidden_dim,
                      kv_rows * (size_t)hidden_dim);

    size_t v_start_row = q_rows_total + kv_rows_total + (size_t)ctx->head_offset_kv * (size_t)head_dim;
    copy_fp32_to_bf16(layer_src + v_start_row * (size_t)hidden_dim,
                      layer_dst + (q_rows + kv_rows) * (size_t)hidden_dim,
                      kv_rows * (size_t)hidden_dim);

    const float *layer_bias_src = cpu_w->b_qkv + (size_t)l * (size_t)global_qkv_rows;
    hip_bfloat16 *layer_bias_dst = gpu_w->b_qkv + (size_t)l * (size_t)ctx->local_qkv_dim;
    copy_fp32_to_bf16(layer_bias_src + q_start_row, layer_bias_dst, q_rows);
    copy_fp32_to_bf16(layer_bias_src + k_start_row, layer_bias_dst + q_rows, kv_rows);
    copy_fp32_to_bf16(layer_bias_src + v_start_row, layer_bias_dst + q_rows + kv_rows, kv_rows);
  }
  add_log("w_qkv_shard", total_qkv_elems * sizeof(hip_bfloat16));
  add_log("b_qkv_shard", (size_t)n_layers * (size_t)ctx->local_qkv_dim * sizeof(hip_bfloat16));

  // W_o shard
  size_t global_o_cols = head_dim * n_attn_heads;
  size_t layer_o_elems = (size_t)hidden_dim * (size_t)ctx->local_q_dim;
  size_t total_o_elems = (size_t)n_layers * layer_o_elems;
  HIP_CHECK(hipMalloc(&gpu_w->w_o, total_o_elems * sizeof(hip_bfloat16)));

  std::vector<float> o_host(layer_o_elems);
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_src = cpu_w->w_o + (size_t)l * (size_t)hidden_dim * (size_t)global_o_cols;
    hip_bfloat16 *layer_dst = gpu_w->w_o + (size_t)l * layer_o_elems;
    for (int row = 0; row < hidden_dim; ++row) {
      const float *row_src = layer_src + (size_t)row * (size_t)global_o_cols + (size_t)ctx->head_offset_q * (size_t)head_dim;
      float *row_dst = o_host.data() + (size_t)row * (size_t)ctx->local_q_dim;
      memcpy(row_dst, row_src, (size_t)ctx->local_q_dim * sizeof(float));
    }
    copy_fp32_to_bf16(o_host.data(), layer_dst, layer_o_elems);
  }
  add_log("w_o_shard", total_o_elems * sizeof(hip_bfloat16));

  copy_array(cpu_w->b_o, &gpu_w->b_o, (size_t)n_layers * (size_t)hidden_dim, "b_o");

  // Attention sinks shard
  size_t sinks_layer = (size_t)ctx->local_q_heads;
  HIP_CHECK(hipMalloc(&gpu_w->attn_sinks, (size_t)n_layers * sinks_layer * sizeof(hip_bfloat16)));
  for (int l = 0; l < n_layers; ++l) {
    const float *src = cpu_w->attn_sinks + (size_t)l * (size_t)n_attn_heads + ctx->head_offset_q;
    hip_bfloat16 *dst = gpu_w->attn_sinks + (size_t)l * sinks_layer;
    copy_fp32_to_bf16(src, dst, sinks_layer);
  }
  add_log("attn_sinks_shard", (size_t)n_layers * sinks_layer * sizeof(hip_bfloat16));

  // Router (replicated)
  size_t router_w_elems = (size_t)n_layers * (size_t)hidden_dim * (size_t)n_experts;
  copy_array(cpu_w->w_router, &gpu_w->w_router, router_w_elems, "w_router");
  size_t router_b_elems = (size_t)n_layers * (size_t)n_experts;
  copy_array(cpu_w->b_router, &gpu_w->b_router, router_b_elems, "b_router");

  // MoE shards
  size_t mlp1_local_elems = (size_t)n_layers * (size_t)ctx->local_experts * 2 * (size_t)intermediate_dim * (size_t)hidden_dim;
  size_t mlp1_blocks = (mlp1_local_elems + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  size_t mlp1_data_bytes = (mlp1_local_elems + 1) / 2;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp1_data), mlp1_data_bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp1_scales), mlp1_blocks * sizeof(uint8_t)));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp1_data, 0, mlp1_data_bytes, ws.quant_stream));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp1_scales, 0, mlp1_blocks * sizeof(uint8_t), ws.quant_stream));

  size_t mlp1_offset = 0;
  size_t per_expert_mlp1 = 2 * (size_t)intermediate_dim * (size_t)hidden_dim;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->w_mlp1 + (size_t)l * (size_t)n_experts * per_expert_mlp1;
    const float *segment = layer_base + (size_t)ctx->expert_offset * per_expert_mlp1;
    size_t count = (size_t)ctx->local_experts * per_expert_mlp1;
    quantize_segment(segment, count, mlp1_offset, gpu_w->w_mlp1_data, gpu_w->w_mlp1_scales);
    mlp1_offset += count;
  }
  add_log("w_mlp1_shard", mlp1_data_bytes + mlp1_blocks * sizeof(uint8_t));

  // Build a tile-packed copy of MXFP4 nibbles for MLP1 to enable coalesced MFMA tiles.
  // Layout per (layer,eid): rows = out_dim (2*I), cols = hidden_dim.
  size_t out_dim = (size_t)2 * (size_t)intermediate_dim;
  size_t Mpad = ((out_dim + 15) / 16) * 16;
  size_t Kpad = ((size_t)hidden_dim + 15) / 16 * 16;
  size_t tiles_m = (out_dim + 15) / 16;
  size_t tiles_k = ((size_t)hidden_dim + 15) / 16;
  size_t packed_tiles_bytes_per_mat = ((Mpad * Kpad) + 1) / 2; // nibble-packed
  size_t scales_entries_per_mat = tiles_m * tiles_k * 16;       // 16 rows per tile
  size_t scale_blocks_per_mat = ((out_dim * (size_t)hidden_dim) + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  size_t total_mats = (size_t)n_layers * (size_t)ctx->local_experts;
  size_t total_packed_bytes = total_mats * packed_tiles_bytes_per_mat;
  size_t total_scale_entries = total_mats * scales_entries_per_mat;
  uint8_t *w_mlp1_data_packed = nullptr;
  uint8_t *w_mlp1_scales_packed = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp1_data_packed), total_packed_bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp1_scales_packed), total_scale_entries));
  HIP_CHECK(hipMemsetAsync(w_mlp1_data_packed, 0, total_packed_bytes, ws.quant_stream));
  HIP_CHECK(hipMemsetAsync(w_mlp1_scales_packed, 0x7F, total_scale_entries, ws.quant_stream));

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
        size_t dst_offset_bytes = mat_idx * packed_tiles_bytes_per_mat;
        size_t src_scale_offset = mat_idx * scale_blocks_per_mat;
        size_t dst_scale_offset = mat_idx * scales_entries_per_mat;
        dim3 grid((bytes + block.x - 1) / block.x);
        pack_mxfp4_row_to_tile16_kernel<<<grid, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp1_data + src_offset_bytes,
          w_mlp1_data_packed + dst_offset_bytes,
          rows, cols);
        HIP_CHECK(hipGetLastError());

        size_t scale_total_entries = tiles_m * tiles_k * 16;
        dim3 grid_scale((scale_total_entries + block.x - 1) / block.x);
        pack_mxfp4_scales_row_to_tile16_kernel<<<grid_scale, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp1_scales + src_scale_offset,
          w_mlp1_scales_packed + dst_scale_offset,
          rows, cols);
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
  gpu_w->w_mlp1_data = w_mlp1_data_packed;
  HIP_CHECK(hipFree(gpu_w->w_mlp1_scales));
  gpu_w->w_mlp1_scales = w_mlp1_scales_packed;
  // Associated scale codes now mirror the tile layout: 16 entries per tile (one per row).

  size_t mlp1_bias_elems = (size_t)n_layers * (size_t)ctx->local_experts * 2 * (size_t)intermediate_dim;
  HIP_CHECK(hipMalloc(&gpu_w->b_mlp1, mlp1_bias_elems * sizeof(hip_bfloat16)));
  size_t mlp1_bias_offset = 0;
  size_t per_expert_bias = 2 * (size_t)intermediate_dim;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->b_mlp1 + (size_t)l * (size_t)n_experts * per_expert_bias;
    const float *segment = layer_base + (size_t)ctx->expert_offset * per_expert_bias;
    copy_fp32_to_bf16(segment, gpu_w->b_mlp1 + mlp1_bias_offset, (size_t)ctx->local_experts * per_expert_bias);
    mlp1_bias_offset += (size_t)ctx->local_experts * per_expert_bias;
  }
  add_log("b_mlp1_shard", mlp1_bias_elems * sizeof(hip_bfloat16));

  size_t mlp2_local_elems = (size_t)n_layers * (size_t)ctx->local_experts * (size_t)hidden_dim * (size_t)intermediate_dim;
  size_t mlp2_blocks = (mlp2_local_elems + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  size_t mlp2_data_bytes = (mlp2_local_elems + 1) / 2;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp2_data), mlp2_data_bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp2_scales), mlp2_blocks * sizeof(uint8_t)));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp2_data, 0, mlp2_data_bytes, ws.quant_stream));
  HIP_CHECK(hipMemsetAsync(gpu_w->w_mlp2_scales, 0, mlp2_blocks * sizeof(uint8_t), ws.quant_stream));

  size_t mlp2_offset = 0;
  size_t per_expert_mlp2 = (size_t)hidden_dim * (size_t)intermediate_dim;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->w_mlp2 + (size_t)l * (size_t)n_experts * per_expert_mlp2;
    const float *segment = layer_base + (size_t)ctx->expert_offset * per_expert_mlp2;
    size_t count = (size_t)ctx->local_experts * per_expert_mlp2;
    quantize_segment(segment, count, mlp2_offset, gpu_w->w_mlp2_data, gpu_w->w_mlp2_scales);
    mlp2_offset += count;
  }
  add_log("w_mlp2_shard", mlp2_data_bytes + mlp2_blocks * sizeof(uint8_t));

  // Build a tile-packed copy of MXFP4 nibbles for MLP2 to enable coalesced MFMA tiles.
  // Layout per (layer,eid): rows = hidden_dim, cols = intermediate_dim.
  {
    size_t rows = (size_t)hidden_dim;
    size_t cols = (size_t)intermediate_dim;
    size_t Mpad2 = ((rows + 15) / 16) * 16;
    size_t Kpad2 = ((cols + 15) / 16) * 16;
    size_t tiles_m2 = (rows + 15) / 16;
    size_t tiles_k2 = (cols + 15) / 16;
    size_t packed_tiles_bytes_per_mat = ((Mpad2 * Kpad2) + 1) / 2; // nibble-packed
    size_t scales_entries_per_mat = tiles_m2 * tiles_k2 * 16;
    size_t scale_blocks_per_mat = ((rows * cols) + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    size_t total_mats = (size_t)n_layers * (size_t)ctx->local_experts;
    size_t total_packed_bytes = total_mats * packed_tiles_bytes_per_mat;
    size_t total_scale_entries = total_mats * scales_entries_per_mat;
    uint8_t *w_mlp2_data_packed = nullptr;
    uint8_t *w_mlp2_scales_packed = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp2_data_packed), total_packed_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp2_scales_packed), total_scale_entries));
    HIP_CHECK(hipMemsetAsync(w_mlp2_data_packed, 0, total_packed_bytes, ws.quant_stream));
    HIP_CHECK(hipMemsetAsync(w_mlp2_scales_packed, 0x7F, total_scale_entries, ws.quant_stream));

    dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < ctx->local_experts; ++e) {
        size_t mat_idx = (size_t)l * (size_t)ctx->local_experts + (size_t)e;
        size_t elems = rows * cols;
        size_t bytes = (elems + 1) / 2;
        size_t src_offset_bytes = ((size_t)l * (size_t)ctx->local_experts + (size_t)e) * bytes;
        size_t dst_offset_bytes = mat_idx * packed_tiles_bytes_per_mat;
        size_t src_scale_offset = mat_idx * scale_blocks_per_mat;
        size_t dst_scale_offset = mat_idx * scales_entries_per_mat;
        dim3 grid((unsigned int)((bytes + block.x - 1) / block.x));
        pack_mxfp4_row_to_tile16_kernel<<<grid, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp2_data + src_offset_bytes,
          w_mlp2_data_packed + dst_offset_bytes,
          rows, cols);
        HIP_CHECK(hipGetLastError());

        size_t scale_total_entries = tiles_m2 * tiles_k2 * 16;
        dim3 grid_scale((unsigned int)((scale_total_entries + block.x - 1) / block.x));
        pack_mxfp4_scales_row_to_tile16_kernel<<<grid_scale, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp2_scales + src_scale_offset,
          w_mlp2_scales_packed + dst_scale_offset,
          rows, cols);
        HIP_CHECK(hipGetLastError());
      }
    }
    hipEvent_t mlp2_pack_done;
    HIP_CHECK(hipEventCreateWithFlags(&mlp2_pack_done, hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(mlp2_pack_done, ws.quant_stream));
    HIP_CHECK(hipEventSynchronize(mlp2_pack_done));
    HIP_CHECK(hipEventDestroy(mlp2_pack_done));
    HIP_CHECK(hipFree(gpu_w->w_mlp2_data));
    gpu_w->w_mlp2_data = w_mlp2_data_packed;
    HIP_CHECK(hipFree(gpu_w->w_mlp2_scales));
    gpu_w->w_mlp2_scales = w_mlp2_scales_packed;
    // Scales are tile-packed alongside MXFP4 data (16 codes per tile/row slice).
  }

  size_t mlp2_bias_elems = (size_t)n_layers * (size_t)ctx->local_experts * (size_t)hidden_dim;
  HIP_CHECK(hipMalloc(&gpu_w->b_mlp2, mlp2_bias_elems * sizeof(hip_bfloat16)));
  size_t mlp2_bias_offset = 0;
  for (int l = 0; l < n_layers; ++l) {
    const float *layer_base = cpu_w->b_mlp2 + (size_t)l * (size_t)n_experts * (size_t)hidden_dim;
    const float *segment = layer_base + (size_t)ctx->expert_offset * (size_t)hidden_dim;
    copy_fp32_to_bf16(segment, gpu_w->b_mlp2 + mlp2_bias_offset, (size_t)ctx->local_experts * (size_t)hidden_dim);
    mlp2_bias_offset += (size_t)ctx->local_experts * (size_t)hidden_dim;
  }
  add_log("b_mlp2_shard", mlp2_bias_elems * sizeof(hip_bfloat16));

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
  
  // Pack router weights (only on leader)
  if (ctx->is_leader) {
    pack_weights_for_mfma(gpu_w->w_router, &gpu_w->w_router_packed16, n_layers, p->n_experts, hidden_dim, "w_router");
  } else {
    gpu_w->w_router_packed16 = nullptr;
  }

  if (!ws.logs.empty()) {
    printf("[Weights][GPU%d] copy summary:\n", ctx->device_id);
    for (const auto &entry : ws.logs) {
      printf("  %-22s %.3f GB\n", entry.first.c_str(), to_gb(entry.second));
    }
    fflush(stdout);
  }
}

void gpu_free_weights(GPUTransformerWeights *gpu_w);

void gpu_clone_weights(GPUContext *dst_ctx, GPUContext *src_ctx, Config *p) {
  HIP_CHECK(hipSetDevice(dst_ctx->device_id));
  GPUTransformerWeights *dst = &dst_ctx->weights;
  GPUTransformerWeights *src = &src_ctx->weights;
  WeightCopyWorkspace &ws = dst_ctx->copy_ws;
  hipStream_t stream = ws.copy_stream;
  ws.logs.clear();
  ws.logs.reserve(32);

  auto to_gb = [](size_t bytes) { return (double)bytes / GB; };
  const auto round16 = [](size_t v) -> size_t { return ((v + 15) / 16) * 16; };
  const auto tiles16 = [](size_t v) -> size_t { return (v + 15) / 16; };

  auto clone_array = [&](auto **dst_ptr, auto *src_ptr, size_t count, const char *label) {
    if (src_ptr == nullptr || count == 0) {
      *dst_ptr = nullptr;
      return;
    }
    size_t bytes = count * sizeof(*src_ptr);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(dst_ptr), bytes));
    hipError_t st = stream ?
      hipMemcpyPeerAsync(*dst_ptr, dst_ctx->device_id, src_ptr, src_ctx->device_id, bytes, stream) :
      hipMemcpyPeer(*dst_ptr, dst_ctx->device_id, src_ptr, src_ctx->device_id, bytes);
    HIP_CHECK(st);
    ws.logs.emplace_back(label, bytes);
  };

  int head_dim         = p->head_dim;
  int n_layers         = p->n_layers;
  int hidden_dim       = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size       = p->vocab_size;
  int n_experts        = p->n_experts;

  size_t embed_elems = (size_t)vocab_size * (size_t)hidden_dim;
  clone_array(&dst->token_embedding_table, src->token_embedding_table, embed_elems, "token_embedding");
  clone_array(&dst->out, src->out, embed_elems, "output_proj");

  size_t out_packed_elems = round16((size_t)vocab_size) * round16((size_t)hidden_dim);
  clone_array(&dst->out_packed16, src->out_packed16, out_packed_elems, "out_final_packed16");

  size_t rms_elems = (size_t)n_layers * (size_t)hidden_dim;
  clone_array(&dst->rms_attn_w, src->rms_attn_w, rms_elems, "rms_attn");
  clone_array(&dst->rms_ffn_w,  src->rms_ffn_w,  rms_elems, "rms_ffn");
  clone_array(&dst->rms_out_w,  src->rms_out_w,  (size_t)hidden_dim, "rms_out");

  size_t total_qkv_elems = (size_t)n_layers * (size_t)hidden_dim * (size_t)dst_ctx->local_qkv_dim;
  clone_array(&dst->w_qkv, src->w_qkv, total_qkv_elems, "w_qkv_shard");
  size_t b_qkv_elems = (size_t)n_layers * (size_t)dst_ctx->local_qkv_dim;
  clone_array(&dst->b_qkv, src->b_qkv, b_qkv_elems, "b_qkv_shard");

  size_t total_o_elems = (size_t)n_layers * (size_t)hidden_dim * (size_t)dst_ctx->local_q_dim;
  clone_array(&dst->w_o, src->w_o, total_o_elems, "w_o_shard");
  size_t b_o_elems = (size_t)n_layers * (size_t)hidden_dim;
  clone_array(&dst->b_o, src->b_o, b_o_elems, "b_o");

  size_t attn_sinks_elems = (size_t)n_layers * (size_t)dst_ctx->local_q_heads;
  clone_array(&dst->attn_sinks, src->attn_sinks, attn_sinks_elems, "attn_sinks_shard");

  size_t router_w_elems = (size_t)n_layers * (size_t)hidden_dim * (size_t)n_experts;
  clone_array(&dst->w_router, src->w_router, router_w_elems, "w_router");
  size_t router_b_elems = (size_t)n_layers * (size_t)n_experts;
  clone_array(&dst->b_router, src->b_router, router_b_elems, "b_router");

  size_t total_expert_mats = (size_t)n_layers * (size_t)dst_ctx->local_experts;
  size_t mlp1_rows = (size_t)2 * (size_t)intermediate_dim;
  size_t mlp1_cols = (size_t)hidden_dim;
  size_t mlp1_packed_bytes_per_mat = (round16(mlp1_rows) * round16(mlp1_cols) + 1) / 2;
  size_t mlp1_scales_per_mat = tiles16(mlp1_rows) * tiles16(mlp1_cols) * 16;
  size_t mlp1_total_bytes = total_expert_mats * mlp1_packed_bytes_per_mat;
  size_t mlp1_total_scales = total_expert_mats * mlp1_scales_per_mat;
  clone_array(&dst->w_mlp1_data, src->w_mlp1_data, mlp1_total_bytes, "w_mlp1_shard");
  clone_array(&dst->w_mlp1_scales, src->w_mlp1_scales, mlp1_total_scales, "w_mlp1_scales");
  size_t mlp1_bias_elems = total_expert_mats * mlp1_rows;
  clone_array(&dst->b_mlp1, src->b_mlp1, mlp1_bias_elems, "b_mlp1_shard");

  size_t mlp2_rows = (size_t)hidden_dim;
  size_t mlp2_cols = (size_t)intermediate_dim;
  size_t mlp2_packed_bytes_per_mat = (round16(mlp2_rows) * round16(mlp2_cols) + 1) / 2;
  size_t mlp2_scales_per_mat = tiles16(mlp2_rows) * tiles16(mlp2_cols) * 16;
  size_t mlp2_total_bytes = total_expert_mats * mlp2_packed_bytes_per_mat;
  size_t mlp2_total_scales = total_expert_mats * mlp2_scales_per_mat;
  clone_array(&dst->w_mlp2_data, src->w_mlp2_data, mlp2_total_bytes, "w_mlp2_shard");
  clone_array(&dst->w_mlp2_scales, src->w_mlp2_scales, mlp2_total_scales, "w_mlp2_scales");
  size_t mlp2_bias_elems = total_expert_mats * mlp2_rows;
  clone_array(&dst->b_mlp2, src->b_mlp2, mlp2_bias_elems, "b_mlp2_shard");

  size_t w_qkv_packed_elems = (size_t)n_layers * round16((size_t)dst_ctx->local_qkv_dim) * round16((size_t)hidden_dim);
  clone_array(&dst->w_qkv_packed16, src->w_qkv_packed16, w_qkv_packed_elems, "w_qkv_packed16");

  size_t w_o_packed_elems = (size_t)n_layers * round16((size_t)hidden_dim) * round16((size_t)dst_ctx->local_q_dim);
  clone_array(&dst->w_o_packed16, src->w_o_packed16, w_o_packed_elems, "w_o_packed16");

  if (dst_ctx->is_leader && src_ctx->is_leader) {
    size_t router_packed_elems = (size_t)n_layers * round16((size_t)n_experts) * round16((size_t)hidden_dim);
    clone_array(&dst->w_router_packed16, src->w_router_packed16, router_packed_elems, "w_router_packed16");
  } else {
    dst->w_router_packed16 = nullptr;
  }

  if (stream) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  if (!ws.logs.empty()) {
    printf("[Weights][GPU%d] cloned from GPU%d:\n", dst_ctx->device_id, src_ctx->device_id);
    for (const auto &entry : ws.logs) {
      printf("  %-22s %.3f GB\n", entry.first.c_str(), to_gb(entry.second));
    }
    fflush(stdout);
  }
}

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

  int dp_rank = gpu_ctx ? gpu_ctx->dp_rank : 0;
  std::vector<GPUContext*> ctxs;
  ctxs.reserve(parallel_layout.tp_size);
  for (int r = 0; r < parallel_layout.tp_size; ++r) {
    ctxs.push_back(&gpu_contexts[dp_rank * parallel_layout.tp_size + r]);
  }
  GPUContext *leader = ctxs[0];
  for (GPUContext *ctx : ctxs) {
    if (ctx->tp_rank == 0) {
      leader = ctx;
      break;
    }
  }

  std::vector<hipEvent_t> batch_events(ctxs.size(), nullptr);
  for (size_t idx = 0; idx < ctxs.size(); ++idx) {
    HIP_CHECK(hipSetDevice(ctxs[idx]->device_id));
    HIP_CHECK(hipEventCreateWithFlags(&batch_events[idx], hipEventDisableTiming));
  }
  auto record_batch_event = [&](size_t idx) {
    HIP_CHECK(hipSetDevice(ctxs[idx]->device_id));
    HIP_CHECK(hipEventRecord(batch_events[idx], ctxs[idx]->batch_stream));
  };
  auto wait_batch_event = [&](size_t idx) {
    HIP_CHECK(hipEventSynchronize(batch_events[idx]));
  };

  // Stage indices to every GPU
  for (GPUContext *ctx : ctxs) {
    HIP_CHECK(hipSetDevice(ctx->device_id));
    HIP_CHECK(hipMemcpyAsync(ctx->d_slot_ids,  h_slot_ids,  batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
    HIP_CHECK(hipMemcpyAsync(ctx->d_pos_list,  h_pos_list,  batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
    HIP_CHECK(hipMemcpyAsync(ctx->d_token_ids, h_tokens,    batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
  }

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

  for (GPUContext *ctx : ctxs) {
    HIP_CHECK(hipSetDevice(ctx->device_id));
    launch_embed(ctx);
  }

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

  for (GPUContext *ctx : ctxs) {
    HIP_CHECK(hipSetDevice(ctx->device_id));
    launch_rope(ctx);
  }

  struct PagedKVWriterV {
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

  struct PagedKVWriterK {
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
    for (GPUContext *ctx : ctxs) {
      ctx->state.kv_cache_manager->prepare_layer(l, h_slot_ids, h_pos_list, batch);
    }

    // RMSNorm (attention)
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      if (batch <= 0) {
        continue;
      }
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

    // QKV projection + bias per shard using MFMA
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      
      // Pack input X
      int Bpad = ((batch + 15) / 16) * 16;
      dim3 block_pack(256);
      dim3 grid_pack((hidden_dim + block_pack.x - 1) / block_pack.x, batch);
      gather_pack_x_fp32_to_fp32_mfma_kernel<<<grid_pack, block_pack, 0, ctx->batch_stream>>>(
        ctx->state.t, ctx->state.xpack_f32, hidden_dim, batch, Bpad, hidden_dim, ctx->d_slot_ids);
      
      // MFMA QKV 
      const int Mpad_qkv = ((ctx->local_qkv_dim + 15) / 16) * 16;
      const int Kpad_qkv = ((hidden_dim + 15) / 16) * 16;
      const hip_bfloat16 *w_qkv_packed = ctx->weights.w_qkv_packed16 +
                                         (size_t)l * (size_t)Mpad_qkv * (size_t)Kpad_qkv;
      const hip_bfloat16 *b_qkv = ctx->weights.b_qkv + (size_t)l * (size_t)ctx->local_qkv_dim;
      launch_gemm16x16_mfma_f32_batched(
        w_qkv_packed, ctx->state.xpack_f32, ctx->state.qkv, b_qkv,
        ctx->local_qkv_dim, hidden_dim, batch, Bpad, ctx->local_qkv_dim,
        ctx->d_slot_ids, ctx->batch_stream);
    }

    // Cache write + RoPE per shard
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      int q_stride = ctx->local_qkv_dim;
      int q_size = ctx->local_q_dim;
      int k_size = ctx->local_kv_dim;
      int v_size = ctx->local_kv_dim;

      dim3 block_v(256), grid_v((v_size + block_v.x - 1) / block_v.x, batch);
      PagedKVWriterV::run<<<grid_v, block_v, 0, ctx->batch_stream>>>(
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
      PagedKVWriterK::run<<<grid_k, block_k, 0, ctx->batch_stream>>>(
        ctx->state.qkv,
        ctx->state.kv_cache_manager->get_device_data(), l,
        batch, q_stride, q_size, k_size,
        ctx->d_pos_list, ctx->d_slot_ids);
    }

    // Flash attention per shard
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      const int block_size = 64;
      dim3 block(block_size);
      int max_attention_length = (p->sliding_window > 0 && (l % 2 == 0)) ? p->sliding_window : p->seq_len;
      size_t shared_mem_bytes = (size_t)(max_attention_length + 1) * sizeof(float);
      shared_mem_bytes += (size_t)(max_attention_length + 1) * sizeof(hip_bfloat16*);
      shared_mem_bytes += sizeof(hip_bfloat16*);
      dim3 grid(1, ctx->local_q_heads, batch);
      paged_flash_attention_kernel_batched<<<grid, block, shared_mem_bytes, ctx->batch_stream>>>(
        ctx->state.qkv,
        ctx->state.kv_cache_manager->get_device_data(),
        ctx->weights.attn_sinks,
        ctx->state.tb,
        head_dim,
        ctx->local_kv_dim,
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
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      
      // Pack input (tb from attention)
      int Bpad = ((batch + 15) / 16) * 16;
      dim3 block_pack(256);
      dim3 grid_pack((ctx->local_q_dim + block_pack.x - 1) / block_pack.x, batch);
      gather_pack_x_fp32_to_fp32_mfma_kernel<<<grid_pack, block_pack, 0, ctx->batch_stream>>>(
        ctx->state.tb, ctx->state.xpack_f32, ctx->local_q_dim, batch, Bpad, ctx->local_q_dim, ctx->d_slot_ids);
      
      // MFMA W_o
      const int Mpad_o = ((hidden_dim + 15) / 16) * 16;
      const int Kpad_o = ((ctx->local_q_dim + 15) / 16) * 16;
      const hip_bfloat16 *w_o_packed = ctx->weights.w_o_packed16 +
                                       (size_t)l * (size_t)Mpad_o * (size_t)Kpad_o;
      launch_gemm16x16_mfma_f32_batched(
        w_o_packed, ctx->state.xpack_f32, ctx->state.tb2, nullptr,  // no bias
        hidden_dim, ctx->local_q_dim, batch, Bpad, hidden_dim,
        ctx->d_slot_ids, ctx->batch_stream);
      record_batch_event(idx);
    }

    // Sum partial tb2 on leader and broadcast
    if (ctxs.size() > 1) {
      wait_batch_event(0);
      HIP_CHECK(hipSetDevice(leader->device_id));
      size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
      for (size_t idx = 1; idx < ctxs.size(); ++idx) {
        GPUContext *peer = ctxs[idx];
        wait_batch_event(idx);
        HIP_CHECK(hipMemcpyPeer(leader->state.tb2_peer, leader->device_id,
                                peer->state.tb2, peer->device_id, bytes));
        HIP_CHECK(hipSetDevice(leader->device_id));
        dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
        add_selected_kernel_float<<<grid, block, 0, leader->batch_stream>>>(
          leader->state.tb2, leader->state.tb2_peer, hidden_dim, batch, leader->d_slot_ids);
      }
      record_batch_event(0);
    }

    // Add bias once on leader and broadcast
    HIP_CHECK(hipSetDevice(leader->device_id));
    {
      const hip_bfloat16 *b_o = leader->weights.b_o + (size_t)l * (size_t)hidden_dim;
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      add_bias_kernel_float_batched<<<grid, block, 0, leader->batch_stream>>>(
        leader->state.tb2, b_o, hidden_dim, batch, hidden_dim, leader->d_slot_ids);
      record_batch_event(0);
      wait_batch_event(0);
    }

    if (ctxs.size() > 1) {
      wait_batch_event(0);
      size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
      for (size_t idx = 1; idx < ctxs.size(); ++idx) {
        GPUContext *peer = ctxs[idx];
        HIP_CHECK(hipMemcpyPeer(peer->state.tb2, peer->device_id,
                                leader->state.tb2, leader->device_id, bytes));
      }
    }

    // Residual add with attention output
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      residual_connection_kernel_float_batched<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.x, ctx->state.tb2, hidden_dim, batch, hidden_dim, ctx->d_slot_ids);
    }

    // MoE RMSNorm per slot
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      if (batch <= 0) {
        continue;
      }
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

    // Router computation on leader using MFMA
    HIP_CHECK(hipSetDevice(leader->device_id));
    {
      // Pack input
      int Bpad = ((batch + 15) / 16) * 16;
      dim3 block_pack(256);
      dim3 grid_pack((hidden_dim + block_pack.x - 1) / block_pack.x, batch);
      gather_pack_x_fp32_to_fp32_mfma_kernel<<<grid_pack, block_pack, 0, leader->batch_stream>>>(
        leader->state.t, leader->state.xpack_f32, hidden_dim, batch, Bpad, hidden_dim, leader->d_slot_ids);
      
      // MFMA router
      const int Mpad_router = ((p->n_experts + 15) / 16) * 16;
      const int Kpad_router = ((hidden_dim + 15) / 16) * 16;
      const hip_bfloat16 *w_router_packed = leader->weights.w_router_packed16 +
                                            (size_t)l * (size_t)Mpad_router * (size_t)Kpad_router;
      const hip_bfloat16 *b_router = leader->weights.b_router + (size_t)l * (size_t)p->n_experts;
      launch_gemm16x16_mfma_f32_batched(
        w_router_packed, leader->state.xpack_f32, leader->state.router_score, b_router,
        p->n_experts, hidden_dim, batch, Bpad, p->n_experts,
        leader->d_slot_ids, leader->batch_stream);
      
      dim3 grid_topk(batch), block_topk(1);
      topk_softmax_batched_kernel_float<<<grid_topk, block_topk, 0, leader->batch_stream>>>(
        leader->state.router_score, p->n_experts, p->experts_per_token,
        leader->state.topk_v, leader->state.topk_i,
        p->n_experts, p->experts_per_token,
        leader->d_slot_ids, batch);
    }

    // Zero MoE accumulation buffers on every shard
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      zero_selected_slots_kernel_float<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.e_agg, hidden_dim, ctx->d_slot_ids, batch);
    }

    // Gather global top-k pairs on leader
    int M = batch * p->experts_per_token;
    HIP_CHECK(hipSetDevice(leader->device_id));
    {
      dim3 block(256), grid((M + block.x - 1) / block.x);
      gather_topk_pairs_kernel<<<grid, block, 0, leader->batch_stream>>>(
        leader->state.topk_v, leader->state.topk_i,
        leader->d_slot_ids, batch, p->experts_per_token,
        leader->state.sel_w, leader->state.sel_eids, leader->state.sel_sids);
      record_batch_event(0);
      wait_batch_event(0);
    }

    std::vector<int> host_sel_sids(M);
    std::vector<int> host_sel_eids(M);
    std::vector<float> host_sel_w(M);
    HIP_CHECK(hipMemcpy(host_sel_sids.data(), leader->state.sel_sids, M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_sel_eids.data(), leader->state.sel_eids, M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_sel_w.data(), leader->state.sel_w, M * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<int> local_M(ctxs.size(), 0);
    std::vector<int> local_block_counts(ctxs.size(), 0);
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      std::vector<int> sel_sids_local;
      std::vector<int> sel_eids_local;
      std::vector<float> sel_w_local;
      sel_sids_local.reserve(M);
      sel_eids_local.reserve(M);
      sel_w_local.reserve(M);
      for (int m = 0; m < M; ++m) {
        int eid = host_sel_eids[m];
        if (eid >= ctx->expert_offset && eid < ctx->expert_offset + ctx->local_experts) {
          sel_sids_local.push_back(host_sel_sids[m]);
          sel_eids_local.push_back(eid - ctx->expert_offset);
          sel_w_local.push_back(host_sel_w[m]);
        }
      }
      int sel_count = (int)sel_sids_local.size();
      local_M[idx] = sel_count;
      HIP_CHECK(hipSetDevice(ctx->device_id));
      if (sel_count == 0) {
        continue;
      }

      int local_experts = ctx->local_experts;
      std::vector<int> counts(local_experts, 0);
      for (int eid_local : sel_eids_local) {
        counts[eid_local]++;
      }

      std::vector<int> prefix(local_experts + 1, 0);
      for (int e = 0; e < local_experts; ++e) {
        prefix[e + 1] = prefix[e] + counts[e];
      }

      std::vector<int> write_pos = prefix;
      std::vector<int> order(sel_count);
      for (int i = 0; i < sel_count; ++i) {
        int e = sel_eids_local[i];
        int pos = write_pos[e]++;
        order[pos] = i;
      }

      std::vector<int> grouped_sids(sel_count);
      std::vector<int> grouped_eids(sel_count);
      std::vector<float> grouped_w(sel_count);
      for (int dst = 0; dst < sel_count; ++dst) {
        int src = order[dst];
        grouped_sids[dst] = sel_sids_local[src];
        grouped_eids[dst] = sel_eids_local[src];
        grouped_w[dst] = sel_w_local[src];
      }

      std::vector<int> block_cols;
      std::vector<int> block_sizes;
      std::vector<int> block_eids;
      block_cols.reserve((sel_count + 15) / 16);
      block_sizes.reserve((sel_count + 15) / 16);
      block_eids.reserve((sel_count + 15) / 16);

      for (int e = 0; e < local_experts; ++e) {
        int start = prefix[e];
        int end = prefix[e + 1];
        int remaining = end - start;
        int col = start;
        while (remaining > 0) {
          int chunk = remaining >= 16 ? 16 : remaining;
          block_cols.push_back(col);
          block_sizes.push_back(chunk);
          block_eids.push_back(e);
          col += chunk;
          remaining -= chunk;
        }
      }

      local_block_counts[idx] = (int)block_cols.size();

      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_sids, grouped_sids.data(), sel_count * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_eids, grouped_eids.data(), sel_count * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      HIP_CHECK(hipMemcpyAsync(ctx->state.sel_w, grouped_w.data(), sel_count * sizeof(float), hipMemcpyHostToDevice, ctx->batch_stream));
      if (!block_cols.empty()) {
        HIP_CHECK(hipMemcpyAsync(ctx->state.sel_block_cols, block_cols.data(), block_cols.size() * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
        HIP_CHECK(hipMemcpyAsync(ctx->state.sel_block_sizes, block_sizes.data(), block_sizes.size() * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
        HIP_CHECK(hipMemcpyAsync(ctx->state.sel_block_eids, block_eids.data(), block_eids.size() * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
      }
    }

    // Grouped MoE compute per shard
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      int M_local = local_M[idx];
      int block_count = local_block_counts[idx];
      if (M_local == 0 || block_count == 0) continue;
      HIP_CHECK(hipSetDevice(ctx->device_id));
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
      // Launch MFMA matvec over 16-row tiles in output dimension
      const int out_dim = 2 * p->intermediate_dim;
      dim3 block1(16,4,1); // 64 threads per block for 16x16x4 MFMA
      dim3 grid1((out_dim + 15) / 16, block_count, 1);
      mlp1_grouped_mfma_matvec_mxfp4_kernel<<<grid1, block1, 0, ctx->batch_stream>>>(
        ctx->state.mlp1_out_all, ctx->state.t_bf16,
        ctx->weights.w_mlp1_data, ctx->weights.w_mlp1_scales,
        ctx->weights.b_mlp1,
        ctx->state.sel_sids,
        ctx->state.sel_block_cols,
        ctx->state.sel_block_sizes,
        ctx->state.sel_block_eids,
        M_local, block_count,
        hidden_dim, out_dim,
        l, ctx->local_experts);

      dim3 block2(256), grid2((p->intermediate_dim + block2.x - 1) / block2.x, M_local);
      swiglu_grouped_kernel_bf16<<<grid2, block2, 0, ctx->batch_stream>>>(
        ctx->state.gate_up_all, ctx->state.mlp1_out_all,
        M_local, p->intermediate_dim, p->swiglu_limit);

      // Launch MFMA MLP2 accumulation over 16-row tiles in hidden_dim
      dim3 block3(16,4,1);
      dim3 grid3((hidden_dim + 15) / 16, block_count, 1);
      mlp2_accum_grouped_mfma_matvec_mxfp4_kernel<<<grid3, block3, 0, ctx->batch_stream>>>(
        ctx->state.e_agg, ctx->state.gate_up_all,
        ctx->weights.w_mlp2_data, ctx->weights.w_mlp2_scales,
        ctx->weights.b_mlp2,
        ctx->state.sel_sids, ctx->state.sel_w,
        ctx->state.sel_block_cols,
        ctx->state.sel_block_sizes,
        ctx->state.sel_block_eids,
        M_local, block_count,
        hidden_dim, p->intermediate_dim,
        l, ctx->local_experts, hidden_dim);
      record_batch_event(idx);
    }

    // Sum e_agg on leader and broadcast
    if (ctxs.size() > 1) {
      wait_batch_event(0);
      HIP_CHECK(hipSetDevice(leader->device_id));
      size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
      for (size_t idx = 1; idx < ctxs.size(); ++idx) {
        GPUContext *peer = ctxs[idx];
        wait_batch_event(idx);
        HIP_CHECK(hipMemcpyPeer(leader->state.e_agg_peer, leader->device_id,
                                peer->state.e_agg, peer->device_id, bytes));
        HIP_CHECK(hipSetDevice(leader->device_id));
        dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
        add_selected_kernel_float<<<grid, block, 0, leader->batch_stream>>>(
          leader->state.e_agg, leader->state.e_agg_peer, hidden_dim, batch, leader->d_slot_ids);
      }
      record_batch_event(0);
      wait_batch_event(0);
      for (size_t idx = 1; idx < ctxs.size(); ++idx) {
        GPUContext *peer = ctxs[idx];
        HIP_CHECK(hipMemcpyPeer(peer->state.e_agg, peer->device_id,
                                leader->state.e_agg, leader->device_id, bytes));
      }
    }

    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      residual_connection_kernel_float_batched<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.x, ctx->state.e_agg, hidden_dim, batch, hidden_dim, ctx->d_slot_ids);
    }
  }

  // Final RMSNorm + logits on leader
  if (host_next_tokens_out != nullptr) {
    HIP_CHECK(hipSetDevice(leader->device_id));
    dim3 block_norm(128);
    dim3 grid_norm(1, batch);
    size_t shared_norm = block_norm.x * sizeof(float);
    rmsnorm_final_batched_kernel<<<grid_norm, block_norm, shared_norm, leader->batch_stream>>>(
      leader->state.x, leader->weights.rms_out_w,
      hidden_dim, batch, leader->d_slot_ids);

    int vocab = p->vocab_size;
    int Bpad_logits = ((batch + 15) / 16) * 16;
    dim3 block_pack_logits(256);
    dim3 grid_pack_logits((hidden_dim + block_pack_logits.x - 1) / block_pack_logits.x, batch);
    gather_pack_x_fp32_to_fp32_mfma_kernel<<<grid_pack_logits, block_pack_logits, 0, leader->batch_stream>>>(
      leader->state.x, leader->state.xpack_f32,
      hidden_dim, batch, Bpad_logits, hidden_dim, leader->d_slot_ids);

    const hip_bfloat16 *w_out_packed = leader->weights.out_packed16;
    launch_gemm16x16_mfma_f32_batched(
      w_out_packed, leader->state.xpack_f32, leader->state.logits_fp32, nullptr,
      vocab, hidden_dim, batch, Bpad_logits, vocab,
      leader->d_batch_iota, leader->batch_stream);

    if (batch > 0) {
      dim3 block_argmax(256);
      dim3 grid_argmax(batch);
      size_t shared_argmax = block_argmax.x * (sizeof(float) + sizeof(int));
      sample_argmax_kernel<<<grid_argmax, block_argmax, shared_argmax, leader->batch_stream>>>(
        leader->state.logits_fp32, leader->d_next_tokens, batch, p->vocab_size);
    }

    HIP_CHECK(hipMemcpyAsync(host_next_tokens_out, leader->d_next_tokens,
                             batch * sizeof(int), hipMemcpyDeviceToHost, leader->batch_stream));
    record_batch_event(0);
    wait_batch_event(0);
  }

  // Keep follower activations aligned (optional)
  if (ctxs.size() > 1) {
    size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
    for (size_t idx = 1; idx < ctxs.size(); ++idx) {
      GPUContext *peer = ctxs[idx];
      HIP_CHECK(hipMemcpyPeer(peer->state.x, peer->device_id,
                              leader->state.x, leader->device_id, bytes));
    }
  }

  for (size_t idx = 0; idx < batch_events.size(); ++idx) {
    if (batch_events[idx] != nullptr) {
      HIP_CHECK(hipSetDevice(ctxs[idx]->device_id));
      HIP_CHECK(hipEventDestroy(batch_events[idx]));
    }
  }
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
  transformer->config.seq_len = 1024; // override max seq len to 1024
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "No HIP devices found.\n");
    exit(EXIT_FAILURE);
  }

  Config *cfg = &transformer->config;

  int desired_tp = (device_count >= 2) ? 2 : 1;
  const char *env_tp = getenv("GETP_TP_SIZE");
  if (env_tp && atoi(env_tp) > 0) desired_tp = atoi(env_tp);
  if (desired_tp < 1) desired_tp = 1;
  if (desired_tp > device_count) desired_tp = device_count;

  int desired_ep = (device_count >= 2) ? 2 : 1;
  const char *env_ep = getenv("GETP_EP_SIZE");
  if (env_ep && atoi(env_ep) > 0) desired_ep = atoi(env_ep);
  if (desired_ep < 1) desired_ep = 1;
  if (desired_ep > device_count) desired_ep = device_count;

  int desired_dp = std::max(1, device_count / 2);
  const char *env_dp = getenv("GETP_DP_SIZE");
  if (env_dp && atoi(env_dp) > 0) desired_dp = atoi(env_dp);
  if (desired_dp < 1) desired_dp = 1;
  if (desired_dp * desired_tp > device_count) {
    desired_dp = device_count / desired_tp;
    if (desired_dp < 1) desired_dp = 1;
  }

  if (desired_tp != desired_ep) {
    fprintf(stderr, "[GPU] Only matching TP/EP sizes are supported (got TP=%d, EP=%d).\n",
            desired_tp, desired_ep);
    exit(EXIT_FAILURE);
  }

  parallel_layout.tp_size = desired_tp;
  parallel_layout.ep_size = desired_ep;
  parallel_layout.dp_size = desired_dp;
  int used_devices = parallel_layout.tp_size * parallel_layout.dp_size;

  if ((cfg->n_attn_heads % parallel_layout.tp_size) != 0 ||
      (cfg->n_kv_heads % parallel_layout.tp_size) != 0) {
    fprintf(stderr, "[GPU] Tensor parallel size %d incompatible with head counts (%d Q, %d KV).\n",
            parallel_layout.tp_size, cfg->n_attn_heads, cfg->n_kv_heads);
    exit(EXIT_FAILURE);
  }
  if ((cfg->n_experts % parallel_layout.ep_size) != 0) {
    fprintf(stderr, "[GPU] Expert parallel size %d incompatible with experts %d.\n",
            parallel_layout.ep_size, cfg->n_experts);
    exit(EXIT_FAILURE);
  }

  gpu_contexts.clear();
  gpu_contexts.resize(used_devices);

  int local_q_heads = cfg->n_attn_heads / parallel_layout.tp_size;
  int local_kv_heads = cfg->n_kv_heads / parallel_layout.tp_size;
  int local_experts = cfg->n_experts / parallel_layout.ep_size;
  int head_dim = cfg->head_dim;

  for (int d = 0; d < used_devices; ++d) {
    int device_id = d;
    HIP_CHECK(hipSetDevice(device_id));
    GPUContext &ctx = gpu_contexts[d];
    int dp_rank = d / parallel_layout.tp_size;
    int tp_rank = d % parallel_layout.tp_size;
    ctx.device_id = device_id;
    ctx.tp_rank = tp_rank;
    ctx.tp_size = parallel_layout.tp_size;
    ctx.dp_rank = dp_rank;
    ctx.dp_size = parallel_layout.dp_size;
    ctx.ep_rank = tp_rank;
    ctx.ep_size = parallel_layout.ep_size;
    ctx.head_offset_q = tp_rank * local_q_heads;
    ctx.head_offset_kv = tp_rank * local_kv_heads;
    ctx.expert_offset = tp_rank * local_experts;
    ctx.local_q_heads = local_q_heads;
    ctx.local_kv_heads = local_kv_heads;
    ctx.local_experts = local_experts;
    ctx.local_q_dim = ctx.local_q_heads * head_dim;
    ctx.local_kv_dim = ctx.local_kv_heads * head_dim;
    ctx.local_qkv_dim = ctx.local_q_dim + 2 * ctx.local_kv_dim;
    ctx.is_leader = (tp_rank == 0);

    ensure_mxfp4_dequant_lut_initialized();

    const char *env_mb = getenv("GETP_MAX_BATCH");
    int mb = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 8;
    ctx.max_batch = mb;

    WeightCopyWorkspace &ws = ctx.copy_ws;
    ws.logs.clear();
    ws.logs.reserve(32);
    if (ctx.dp_rank == 0) {
      size_t stage_mb = 256;
      const char *env_stage = getenv("GETP_STAGE_MB");
      if (env_stage && atoi(env_stage) > 0) stage_mb = (size_t)atoi(env_stage);
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
      }
      for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipEventRecord(ws.stage_free[i], ws.quant_stream));
      }
    } else {
      ws.capacity_elems = 0;
      HIP_CHECK(hipStreamCreateWithFlags(&ws.copy_stream, hipStreamNonBlocking));
      ws.convert_stream = nullptr;
      ws.quant_stream = nullptr;
      for (int i = 0; i < 2; ++i) {
        ws.host_stage[i] = nullptr;
        ws.device_stage[i] = nullptr;
        ws.copy_ready[i] = nullptr;
        ws.stage_free[i] = nullptr;
      }
    }

    if (ctx.dp_rank == 0) {
      gpu_copy_weights(&ctx, &transformer->weights, cfg);
    } else {
      GPUContext *src_ctx = &gpu_contexts[tp_rank];
      gpu_clone_weights(&ctx, src_ctx, cfg);
    }
    gpu_malloc_run_state(&ctx, cfg, ctx.max_batch);

    ctx.slot_streams.resize(ctx.max_batch);
    for (int s = 0; s < ctx.max_batch; ++s) {
      HIP_CHECK(hipStreamCreate(&ctx.slot_streams[s]));
    }
    HIP_CHECK(hipStreamCreate(&ctx.batch_stream));
    HIP_CHECK(hipMalloc(&ctx.d_slot_ids,  ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_pos_list,  ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_token_ids, ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_next_tokens, ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_batch_iota, ctx.max_batch * sizeof(int)));
    std::vector<int> host_batch_iota(ctx.max_batch);
    for (int i = 0; i < ctx.max_batch; ++i) host_batch_iota[i] = i;
    HIP_CHECK(hipMemcpy(ctx.d_batch_iota, host_batch_iota.data(),
                       ctx.max_batch * sizeof(int), hipMemcpyHostToDevice));

    ctx.num_expert_streams = MAX_EXPERT_STREAMS;
    for (int s = 0; s < ctx.num_expert_streams; ++s) {
      HIP_CHECK(hipStreamCreate(&ctx.expert_streams[s]));
    }

  }

  if (used_devices > 1) {
    for (int src = 0; src < used_devices; ++src) {
      HIP_CHECK(hipSetDevice(src));
      for (int dst = 0; dst < used_devices; ++dst) {
        if (src == dst) continue;
        int can_access = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src, dst));
        if (can_access) {
          hipError_t st = hipDeviceEnablePeerAccess(dst, 0);
          if (st != hipErrorPeerAccessAlreadyEnabled) {
            HIP_CHECK(st);
          }
        }
      }
    }
  }

  printf("[GPU] Warm-up done on %d device(s). Layout TP=%d, DP=%d, EP=%d.\n",
         used_devices, parallel_layout.tp_size, parallel_layout.dp_size, parallel_layout.ep_size);
  printf("[Attention] Using FlashAttention-like optimized kernel (always on)\n");
  
  // Log memory usage for each GPU device after warm-up
  for (int d = 0; d < used_devices; ++d) {
    log_gpu_memory_usage(d);
  }
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    // Destroy per-slot streams
    for (auto &st : gpu_ctx.slot_streams) {
      if (st != nullptr) HIP_CHECK(hipStreamDestroy(st));
    }
    gpu_ctx.slot_streams.clear();
    if (gpu_ctx.batch_stream) HIP_CHECK(hipStreamDestroy(gpu_ctx.batch_stream));
    if (gpu_ctx.d_slot_ids)    HIP_CHECK(hipFree(gpu_ctx.d_slot_ids));
    if (gpu_ctx.d_pos_list)    HIP_CHECK(hipFree(gpu_ctx.d_pos_list));
    if (gpu_ctx.d_token_ids)   HIP_CHECK(hipFree(gpu_ctx.d_token_ids));
    if (gpu_ctx.d_next_tokens) HIP_CHECK(hipFree(gpu_ctx.d_next_tokens));
    if (gpu_ctx.d_batch_iota)  HIP_CHECK(hipFree(gpu_ctx.d_batch_iota));
    WeightCopyWorkspace &ws = gpu_ctx.copy_ws;
    for (int i = 0; i < 2; ++i) {
      if (ws.copy_ready[i]) HIP_CHECK(hipEventDestroy(ws.copy_ready[i]));
      if (ws.stage_free[i]) HIP_CHECK(hipEventDestroy(ws.stage_free[i]));
      if (ws.device_stage[i]) HIP_CHECK(hipFree(ws.device_stage[i]));
      if (ws.host_stage[i]) HIP_CHECK(hipHostFree(ws.host_stage[i]));
    }
    if (ws.copy_stream) HIP_CHECK(hipStreamDestroy(ws.copy_stream));
    if (ws.convert_stream) HIP_CHECK(hipStreamDestroy(ws.convert_stream));
    if (ws.quant_stream) HIP_CHECK(hipStreamDestroy(ws.quant_stream));
    ws.copy_stream = ws.convert_stream = ws.quant_stream = nullptr;
    ws.capacity_elems = 0;
    ws.logs.clear();
    for (int s = 0; s < gpu_ctx.num_expert_streams; ++s) {
      if (gpu_ctx.expert_streams[s] != nullptr) HIP_CHECK(hipStreamDestroy(gpu_ctx.expert_streams[s]));
    }
    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx);
  }
  gpu_contexts.clear();
  printf("[GPU] Cleanup done.\n");
}

// ------------------------- Driver (generate & inference) -------------------------

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  if (requests->num_reqs <= 0 || gpu_contexts.empty()) return 0;
  const int num_devices = parallel_layout.dp_size;
  // Global request index for all workers (lock-free producer)
  std::atomic<int> next_idx{0};

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  // Continuous batching per GPU worker
  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      int dp_rank = t;
      GPUContext *ctx = nullptr;
      std::vector<GPUContext*> group_ctxs;
      group_ctxs.reserve(parallel_layout.tp_size);
      for (GPUContext &gctx : gpu_contexts) {
        if (gctx.dp_rank == dp_rank) {
          group_ctxs.push_back(&gctx);
          if (gctx.tp_rank == 0) ctx = &gctx;
        }
      }
      if (ctx == nullptr) {
        fprintf(stderr, "[GPU] Missing leader for DP rank %d.\n", dp_rank);
        return;
      }

      HIP_CHECK(hipSetDevice(ctx->device_id));
      t_ctx = ctx;

      std::vector<Slot> slots(ctx->max_batch);
      long long local_out = 0;

      auto reset_kv_slot = [&](int slot_id) {
        for (GPUContext *gctx : group_ctxs) {
          if (gctx->state.kv_cache_manager) {
            gctx->state.kv_cache_manager->reset_slot(slot_id);
          }
        }
      };

      auto acquire_slot = [&](int sidx) {
        int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= requests->num_reqs) return false;
        reset_kv_slot(sidx);
        Slot &s = slots[sidx];
        s.active = true;
        s.req_idx = idx;
        const char *input_seq = get_str_req_ptr(requests, idx);
        s.out_ptr = get_tok_gen_ptr(requests, idx);
        // Encode prompt
        int alloc = (int)strlen(input_seq) + 3;
        s.prompt.resize(alloc);
        encode(tokenizer, input_seq, -1, -1, s.prompt.data(), &s.num_prompt_tokens,
               transformer->config.initial_context_length);
        // Initialize state
        s.pos = 0;
        s.token = s.prompt[0];
        s.max_steps = std::min(requests->max_seq_len, transformer->config.seq_len);
        return true;
      };

      // Prime initial batch
      for (int i = 0; i < ctx->max_batch; ++i) {
        acquire_slot(i);
      }

      int active_cnt = 0;
      for (auto &s : slots) if (s.active) active_cnt++;

      while (active_cnt > 0) {
        // Build batch of current active slots
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

        // Run batched forward for all active slots and get next tokens
        std::vector<int> next_tokens_host(B);
        gpu_forward_batch_step(transformer, ctx, tokens.data(), poses.data(), slot_ids.data(), B, next_tokens_host.data());

        // Advance all slots with sampling and book-keeping
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
          if (s.pos >= s.max_steps) finished = true; // bound pos to seq_len
          if (s.pos >= transformer->config.seq_len) finished = true;

          if (finished) {
            int gen_tokens = std::max(0, s.pos - s.num_prompt_tokens);
            s.out_ptr[gen_tokens] = -1;
            local_out += (long long)(gen_tokens + 1); // include sentinel
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

  HIP_CHECK(hipMalloc(&s->x,   B * (size_t)hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->t,   B * (size_t)hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&s->t_bf16), B * (size_t)hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb,  B * (size_t)local_q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2, B * (size_t)hidden_dim * sizeof(float)));

  size_t router_elems = ctx->is_leader ? (size_t)p->n_experts : (size_t)ctx->local_experts;
  HIP_CHECK(hipMalloc(&s->router_score, B * router_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_v,       B * (size_t)p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_i,       B * (size_t)p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->mlp1_out, B * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate,     B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up,       B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up,  B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->e_agg,    B * (size_t)hidden_dim * sizeof(float)));

  HIP_CHECK(hipMalloc(&s->qkv, B * (size_t)local_qkv_dim * sizeof(float)));

  s->kv_cache_manager = new KVCacheManager(max_batch, p->n_layers, p->seq_len,
                                           (size_t)local_kv_dim, p->sliding_window);

  if (ctx->is_leader) {
    HIP_CHECK(hipMalloc(&s->logits,      B * (size_t)p->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));
  } else {
    s->logits = nullptr;
    s->logits_fp32 = nullptr;
  }

  HIP_CHECK(hipMalloc(&s->cos_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sin_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));

  s->acc_buffer = nullptr;

  HIP_CHECK(hipMalloc(&s->mlp1_out_s, MAX_EXPERT_STREAMS * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_s,     MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up_s,       MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_s,  MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2_s,      MAX_EXPERT_STREAMS * (size_t)hidden_dim * sizeof(float)));

  size_t Mmax = (size_t)max_batch * (size_t)p->experts_per_token;
  HIP_CHECK(hipMalloc(&s->sel_sids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_eids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_w,        Mmax * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sel_block_cols,  Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_block_sizes, Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_block_eids,  Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)p->intermediate_dim * sizeof(hip_bfloat16)));

  s->tb2_peer = nullptr;
  s->e_agg_peer = nullptr;
  if (ctx->is_leader) {
    HIP_CHECK(hipMalloc(&s->tb2_peer, B * (size_t)hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->e_agg_peer, B * (size_t)hidden_dim * sizeof(float)));
    s->router_score_full = s->router_score;
  } else {
    s->router_score_full = nullptr;
  }
  
  // MFMA packed buffers
  int Bpad = ((int)B + 15) / 16 * 16; // align to 16
  int K_max = std::max({hidden_dim, p->n_experts, local_qkv_dim});
  HIP_CHECK(hipMalloc(&s->xpack_f32, (size_t)K_max * (size_t)Bpad * sizeof(float)));
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
  if (s->logits) HIP_CHECK(hipFree(s->logits));
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  if (s->acc_buffer) HIP_CHECK(hipFree(s->acc_buffer));
  if (s->logits_fp32) HIP_CHECK(hipFree(s->logits_fp32));
  HIP_CHECK(hipFree(s->mlp1_out_s));
  HIP_CHECK(hipFree(s->gate_s));
  HIP_CHECK(hipFree(s->up_s));
  HIP_CHECK(hipFree(s->gate_up_s));
  HIP_CHECK(hipFree(s->tb2_s));
  HIP_CHECK(hipFree(s->sel_sids));
  HIP_CHECK(hipFree(s->sel_eids));
  HIP_CHECK(hipFree(s->sel_w));
  HIP_CHECK(hipFree(s->sel_block_cols));
  HIP_CHECK(hipFree(s->sel_block_sizes));
  HIP_CHECK(hipFree(s->sel_block_eids));
  HIP_CHECK(hipFree(s->mlp1_out_all));
  HIP_CHECK(hipFree(s->gate_up_all));
  if (s->tb2_peer) HIP_CHECK(hipFree(s->tb2_peer));
  if (s->e_agg_peer) HIP_CHECK(hipFree(s->e_agg_peer));
  HIP_CHECK(hipFree(s->xpack_f32));
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
  HIP_CHECK(hipFree(gpu_w->w_mlp1_scales));
  HIP_CHECK(hipFree(gpu_w->b_mlp1));
  HIP_CHECK(hipFree(gpu_w->w_mlp2_data));
  HIP_CHECK(hipFree(gpu_w->w_mlp2_scales));
  HIP_CHECK(hipFree(gpu_w->b_mlp2));
  // Free packed weights
  if (gpu_w->w_qkv_packed16) HIP_CHECK(hipFree(gpu_w->w_qkv_packed16));
  if (gpu_w->w_o_packed16) HIP_CHECK(hipFree(gpu_w->w_o_packed16));
  if (gpu_w->w_router_packed16) HIP_CHECK(hipFree(gpu_w->w_router_packed16));
}
