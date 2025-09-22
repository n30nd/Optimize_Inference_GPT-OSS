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
#include <limits>
#include <limits.h>
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
#ifndef GETP_RUN
#define GETP_RUN

#ifndef GB
#define GB (1024.0*1024.0*1024.0)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global BLOCK_B: configurable via environment variable GETP_MOE_BLOCK.
// Falls back to 16 if not set or invalid.
static int read_BLOCK_B_env() {
  const char* s = getenv("GETP_MOE_BLOCK");
  if (!s || !*s) return 16;
  int v = atoi(s);
  if (v <= 0) return 16;
  return v;
}
static const int BLOCK_B = read_BLOCK_B_env();

#define HIP_CHECK(cmd) \
  do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// Removed expert stream support; batch_stream only is used for inference


// 128-bit alias cho load/stor 16B
using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;
using u32x2 = __attribute__((ext_vector_type(2))) unsigned int;

// ------------------------- Utility: GPU Memory Logging -------------------------
static inline void log_gpu_mem(int device_id, const char* tag) {
  HIP_CHECK(hipSetDevice(device_id));
  size_t free_b = 0, total_b = 0;
  hipError_t st = hipMemGetInfo(&free_b, &total_b);
  if (st != hipSuccess) {
    fprintf(stderr, "[GPU#%d][%s] hipMemGetInfo failed: %s\n", device_id, tag, hipGetErrorString(st));
    return;
  }
  size_t used_b = total_b - free_b;
  double used_g = (double)used_b / GB;
  double free_g = (double)free_b / GB;
  double total_g = (double)total_b / GB;
  printf("[GPU#%d][%s] Memory used %.2f GB | free %.2f GB | total %.2f GB\n", device_id, tag, used_g, free_g, total_g);
  fflush(stdout);
}


// ------------------------- KV Cache Management -------------------------

enum LayerType {
  FULL_ATTENTION = 0,
  SLIDING_WINDOW = 1
};

struct KVCacheSlot {
  hip_bfloat16 *key_ptr;
  hip_bfloat16 *value_ptr;
  size_t allocated_tokens;  // Number of tokens allocated for this slot
  int current_pos;          // Current position in the cache
  LayerType layer_type;     // Type of layer this slot serves
};

// Device-side KV cache management data structure
struct KVCacheDeviceData {
  hip_bfloat16 *key_cache_base;
  hip_bfloat16 *value_cache_base;
  KVCacheSlot *slots;  // Device pointer to slots data [max_batch][n_layers]
  size_t kv_dim;
  int max_batch;
  int n_layers;
  int seq_len;
  int sliding_window_size;
  
  __device__ LayerType get_layer_type(int layer) const {
    // Even layers use sliding window, odd layers use full attention
    return (sliding_window_size > 0 && (layer % 2 == 0)) ? SLIDING_WINDOW : FULL_ATTENTION;
  }
  
  __device__ void get_cache_ptrs(int slot_id, int layer, hip_bfloat16 **key_ptr, hip_bfloat16 **value_ptr, size_t *cache_size) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    *key_ptr = slot.key_ptr;
    *value_ptr = slot.value_ptr;
    *cache_size = slot.allocated_tokens * kv_dim;
  }
  
  __device__ int get_write_position(int slot_id, int layer, int pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.layer_type == SLIDING_WINDOW) {
      return pos % sliding_window_size;
    }
    return pos;
  }
  
  __device__ int get_cache_position(int slot_id, int layer, int logical_pos, int current_pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.layer_type == SLIDING_WINDOW) {
      // For sliding window, we need to map logical position to cache position
      // maintaining the sliding window invariant
      if (current_pos < sliding_window_size) {
        // Still filling the initial window
        return logical_pos;
      } else {
        // Window is full, map to circular buffer position
        // The cache contains positions [current_pos - sliding_window_size + 1, current_pos]
        // mapped to cache positions [0, sliding_window_size - 1]
        int window_start = current_pos - sliding_window_size + 1;
        if (logical_pos < window_start) {
          // Position is outside the sliding window, should not happen in correct usage
          return -1;
        }
        int offset_in_window = logical_pos - window_start;
        int cache_start_offset = window_start % sliding_window_size;
        return (cache_start_offset + offset_in_window) % sliding_window_size;
      }
    }
    return logical_pos;
  }
  
  __device__ void get_attention_range(int slot_id, int layer, int current_pos, int *start_pos, int *end_pos) const {
    const KVCacheSlot &slot = slots[slot_id * n_layers + layer];
    if (slot.layer_type == SLIDING_WINDOW) {
      *start_pos = max(0, current_pos - sliding_window_size + 1);
      *end_pos = current_pos;
    } else {
      *start_pos = 0;
      *end_pos = current_pos;
    }
  }
};

class KVCacheManager {
private:
  hip_bfloat16 *key_cache_base;
  hip_bfloat16 *value_cache_base;
  std::vector<std::vector<KVCacheSlot>> slots; // [max_batch][n_layers]
  KVCacheSlot *d_slots; // Device copy of slots data
  KVCacheDeviceData *d_device_data; // Device copy of device data structure
  size_t kv_dim;
  int max_batch;
  int n_layers;
  int seq_len;
  int sliding_window_size;
  size_t total_allocated_elements;

public:
  KVCacheManager(int max_batch, int n_layers, int seq_len, size_t kv_dim, int sliding_window_size)
    : max_batch(max_batch), n_layers(n_layers), seq_len(seq_len), kv_dim(kv_dim), 
      sliding_window_size(sliding_window_size), total_allocated_elements(0), d_slots(nullptr), d_device_data(nullptr) {
    
    slots.resize(max_batch);
    for (int batch_idx = 0; batch_idx < max_batch; batch_idx++) {
      slots[batch_idx].resize(n_layers);
    }
    
    // Calculate total memory needed
    size_t total_elements = 0;
    for (int batch_idx = 0; batch_idx < max_batch; batch_idx++) {
      for (int layer = 0; layer < n_layers; layer++) {
        LayerType layer_type = get_layer_type(layer);
        size_t layer_tokens = (layer_type == SLIDING_WINDOW) ? sliding_window_size : seq_len;
        total_elements += layer_tokens * kv_dim;
      }
    }
    
    // Allocate base memory
  HIP_CHECK(hipMalloc(&key_cache_base, total_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&value_cache_base, total_elements * sizeof(hip_bfloat16)));
    
  // Setup slot pointers
  hip_bfloat16 *key_offset = key_cache_base;
  hip_bfloat16 *value_offset = value_cache_base;
    
    for (int batch_idx = 0; batch_idx < max_batch; batch_idx++) {
      for (int layer = 0; layer < n_layers; layer++) {
        LayerType layer_type = get_layer_type(layer);
        size_t layer_tokens = (layer_type == SLIDING_WINDOW) ? sliding_window_size : seq_len;
        size_t layer_size = layer_tokens * kv_dim;
        
        slots[batch_idx][layer].key_ptr = key_offset;
        slots[batch_idx][layer].value_ptr = value_offset;
        slots[batch_idx][layer].allocated_tokens = layer_tokens;
        slots[batch_idx][layer].current_pos = 0;
        slots[batch_idx][layer].layer_type = layer_type;
        
        key_offset += layer_size;
        value_offset += layer_size;
      }
    }
    
    // Copy slots data to device
    size_t slots_size = max_batch * n_layers * sizeof(KVCacheSlot);
    HIP_CHECK(hipMalloc(&d_slots, slots_size));
    
    // Flatten slots for device copy
    std::vector<KVCacheSlot> flat_slots;
    for (int batch_idx = 0; batch_idx < max_batch; batch_idx++) {
      for (int layer = 0; layer < n_layers; layer++) {
        flat_slots.push_back(slots[batch_idx][layer]);
      }
    }
    HIP_CHECK(hipMemcpy(d_slots, flat_slots.data(), slots_size, hipMemcpyHostToDevice));
    
    // Create and copy device data structure
    HIP_CHECK(hipMalloc(&d_device_data, sizeof(KVCacheDeviceData)));
  KVCacheDeviceData device_data;
    device_data.key_cache_base = key_cache_base;
    device_data.value_cache_base = value_cache_base;
    device_data.slots = d_slots;
    device_data.kv_dim = kv_dim;
    device_data.max_batch = max_batch;
    device_data.n_layers = n_layers;
    device_data.seq_len = seq_len;
    device_data.sliding_window_size = sliding_window_size;
    HIP_CHECK(hipMemcpy(d_device_data, &device_data, sizeof(KVCacheDeviceData), hipMemcpyHostToDevice));
    
    total_allocated_elements = total_elements;
    
  printf("KVCacheManager allocated: %.2f GB (BF16, optimized per-layer)\n", 
       (float)(2 * total_elements * sizeof(hip_bfloat16)) / (1024.0f * 1024.0f * 1024.0f));
    printf("  - Full attention layers: %d tokens per slot\n", seq_len);
    printf("  - Sliding window layers: %d tokens per slot\n", sliding_window_size);
  }
  
  ~KVCacheManager() {
    if (key_cache_base) HIP_CHECK(hipFree(key_cache_base));
    if (value_cache_base) HIP_CHECK(hipFree(value_cache_base));
  }
  
  LayerType get_layer_type(int layer) const {
    // Even layers use sliding window, odd layers use full attention
    // This matches the existing logic: (l % 2 == 0) for sliding window
    return (sliding_window_size > 0 && (layer % 2 == 0)) ? SLIDING_WINDOW : FULL_ATTENTION;
  }
  
  // Get device data pointer for kernels
  KVCacheDeviceData* get_device_data() const { return d_device_data; }
  
  // Get cache pointers for a specific slot and layer
  void get_cache_ptrs(int slot_id, int layer, hip_bfloat16 **key_ptr, hip_bfloat16 **value_ptr, size_t *cache_size) const {
    const KVCacheSlot &slot = slots[slot_id][layer];
    *key_ptr = slot.key_ptr;
    *value_ptr = slot.value_ptr;
    *cache_size = slot.allocated_tokens * kv_dim;
  }
  
  // Get the effective cache position for writing (handles sliding window wraparound)
  int get_write_position(int slot_id, int layer, int pos) const {
    const KVCacheSlot &slot = slots[slot_id][layer];
    if (slot.layer_type == SLIDING_WINDOW) {
      return pos % sliding_window_size;
    }
    return pos;
  }
  
  // Get the cache position for reading a specific logical position
  int get_cache_position(int slot_id, int layer, int logical_pos, int current_pos) const {
    const KVCacheSlot &slot = slots[slot_id][layer];
    if (slot.layer_type == SLIDING_WINDOW) {
      // For sliding window, we need to map logical position to cache position
      // maintaining the sliding window invariant
      if (current_pos < sliding_window_size) {
        // Still filling the initial window
        return logical_pos;
      } else {
        // Window is full, map to circular buffer position
        // The cache contains positions [current_pos - sliding_window_size + 1, current_pos]
        // mapped to cache positions [0, sliding_window_size - 1]
        int window_start = current_pos - sliding_window_size + 1;
        if (logical_pos < window_start) {
          // Position is outside the sliding window, should not happen in correct usage
          return -1;
        }
        int offset_in_window = logical_pos - window_start;
        int cache_start_offset = window_start % sliding_window_size;
        return (cache_start_offset + offset_in_window) % sliding_window_size;
      }
    }
    return logical_pos;
  }
  
  // Get the range of valid positions for attention computation
  void get_attention_range(int slot_id, int layer, int current_pos, int *start_pos, int *end_pos) const {
    const KVCacheSlot &slot = slots[slot_id][layer];
    if (slot.layer_type == SLIDING_WINDOW) {
      *start_pos = std::max(0, current_pos - sliding_window_size + 1);
      *end_pos = current_pos;
    } else {
      *start_pos = 0;
      *end_pos = current_pos;
    }
  }
  
  // Legacy compatibility: get base pointers (for existing code)
  hip_bfloat16 *get_key_cache_base() const { return key_cache_base; }
  hip_bfloat16 *get_value_cache_base() const { return value_cache_base; }
  
  // Get the block size for a specific slot (for existing KV access patterns)
  size_t get_kv_block_size(int slot_id) const {
    size_t total_size = 0;
    for (int layer = 0; layer < n_layers; layer++) {
      total_size += slots[slot_id][layer].allocated_tokens * kv_dim;
    }
    return total_size;
  }
  
  // Get layer offset within a slot's cache block
  size_t get_layer_offset(int slot_id, int layer) const {
    size_t offset = 0;
    for (int l = 0; l < layer; l++) {
      offset += slots[slot_id][l].allocated_tokens * kv_dim;
    }
    return offset;
  }
  
  size_t get_total_allocated_elements() const { return total_allocated_elements; }
};

// ------------------------- GPU weights & run state -------------------------

typedef struct {
  hip_bfloat16 *token_embedding_table;
  hip_bfloat16 *out;
  hip_bfloat16 *w_out_packed16;  // Packed 16x16 tiles for final classifier
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
  uint8_t      *w_mlp1_data;
  uint8_t      *w_mlp1_scales;
  hip_bfloat16 *w_mlp2;
  hip_bfloat16 *b_mlp1;
  hip_bfloat16 *b_mlp2;
  // Packed (16x16 tiling) BF16 weights for GEMM core (convert to FP32 per tile in shared memory)
  hip_bfloat16 *w_qkv_packed16;   // [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_o_packed16;     // [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_router_packed16;// [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_mlp2_packed16;  // NEW: MLP2 packed 16x16 (H x I)

} GPUTransformerWeights;

typedef struct {
  float *x;
  float *t;
  float *tb;
  float *tb2;
  float *router_score;
  float *topk_v;
  int *topk_i;
  float *e_agg;
  float *qkv;
  // Removed legacy 'logits' buffer; we use logits_fp32 only
  float *key_cache;   // Legacy pointer for backward compatibility
  float *value_cache; // Legacy pointer for backward compatibility
  float *mask;
  float *cos_vals;
  float *sin_vals;
  // FP32 buffers for accumulation and precision-sensitive ops
  float *acc_buffer;         // x in FP32 for output matvec
  float *logits_fp32;        // final logits
  // Packed input buffer for MLP1 (BF16), size: K_max x Bpad_mlp1
  hip_bfloat16 *xpack_bf16;

  // Batched MoE staging (for grouped GEMMs across slots/experts)
  int   *sel_sids;        // [max_batch * experts_per_token]
  int   *sel_eids;        // [max_batch * experts_per_token]
  float *sel_w;           // [max_batch * experts_per_token]
  float *mlp1_out_all;    // [max_batch * experts_per_token][2 * intermediate_dim]
  hip_bfloat16 *gate_up_all; // BF16 MoE activations packed row-major as [intermediate_dim][Bpad_total]
  int   *eid_counts;         // [n_experts]
  int   *eid_cursors;        // [n_experts] for scatter
  // --- New: Block-aligned MoE staging for MLP1 ---
  int   *sorted_indices;     // [M_pad]  index m gốc (hoặc -1 cho pad)
  int   *sorted_sids;        // [M_pad]  sid theo thứ tự đã sắp xếp + đệm
  int   *sorted_eids;        // [M_pad]  eid theo thứ tự đã sắp xếp + đệm
  float *sorted_w;           // [M_pad]  trọng số top-k; 0.0f cho pad
  int   *padded_offsets;     // [n_experts + 1] prefix-sum sau khi align block
  int   *block_expert_ids;   // [M_pad / BLOCK_B] expert id cho mỗi block  (BLOCK_B=16)
  int   *d_col_ids;          // [M_pad]  iota: 0..M_pad-1 dùng để scatter output theo cột
  
  // New: Paged Attention KV Cache Manager
  KVCacheManager *kv_cache_manager;
} GPURunState;

// Per-GPU device context
typedef struct {
  int device_id;
  GPUTransformerWeights weights;
  GPURunState state;
  int max_batch; // number of concurrent slots per GPU for continuous batching
  hipStream_t batch_stream;              // shared stream for batched ops
  // Staging device buffers for batched indices and positions
  int *d_slot_ids;
  int *d_pos_list;
  int *d_token_ids;
  int *d_next_tokens;  // Persistent buffer for sampled next tokens
  int *d_batch_iota;   // Identity mapping [0,1,2,...] for batch-ordered outputs
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

__host__ __device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __host__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
}
__host__ __device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
  union { hip_bfloat16 bf; uint16_t u; } u{};
  u.bf = float_to_bfloat16(f);
  return u.u;
}
__host__ __device__ __forceinline__ int round_up16(int x) { return ((x + 15) / 16) * 16; }

// (removed) pack_bf16x2 helper no longer needed after switching to short4 MFMA operands

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
    return static_cast<uint8_t>(127);
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

__global__ void quantize_mxfp4_kernel(const float *tile, size_t tile_elems, size_t global_elem_offset,
                                      uint8_t *dst_data, uint8_t *dst_scales) {
  size_t block = blockIdx.x;
  size_t start = block * static_cast<size_t>(MXFP4_BLOCK_SIZE);
  if (start >= tile_elems) return;

  size_t elems = tile_elems - start;
  if (elems > static_cast<size_t>(MXFP4_BLOCK_SIZE)) elems = static_cast<size_t>(MXFP4_BLOCK_SIZE);

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
  size_t scale_index = global_start / static_cast<size_t>(MXFP4_BLOCK_SIZE);
  dst_scales[scale_index] = scale_code;

  for (size_t i = 0; i < elems; ++i) {
    float v = src[i] / scale;
    uint8_t code = encode_mxfp4(v);
    size_t global_elem = global_start + i;
    size_t byte_index = global_elem >> 1;
    bool even = ((global_elem & 1ULL) == 0ULL);
    uint8_t existing = dst_data[byte_index];
    if (even) {
      existing = (existing & 0xF0u) | (code & 0x0Fu);
    } else {
      existing = (existing & 0x0Fu) | ((code & 0x0Fu) << 4);
    }
    dst_data[byte_index] = existing;
  }
}

__device__ inline uint8_t mxfp4_decode_src_at(const uint8_t* __restrict__ src_data, size_t elem_index) {
  uint8_t packed = src_data[elem_index >> 1];
  return (elem_index & 1ULL) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0F);
}

__device__ inline void mxfp4_tile_map_rowcol(size_t elem_linear, size_t cols, size_t& row, size_t& col) {
  size_t tiles_per_row = (cols + 15) / 16;
  size_t tile_idx = elem_linear / 256;
  size_t in_tile = elem_linear - tile_idx * 256;
  size_t im = in_tile / 16;
  size_t ik = in_tile - im * 16;
  size_t tile_m = tile_idx / tiles_per_row;
  size_t tile_k = tile_idx - tile_m * tiles_per_row;
  row = tile_m * 16 + im;
  col = tile_k * 16 + ik;
}

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

  dst_data[byte_idx] = static_cast<uint8_t>((code0 & 0x0F) | ((code1 & 0x0F) << 4));
}

__global__ void pack_mxfp4_scales_row_to_tile16_kernel(
    const uint8_t* __restrict__ src_scales,
    uint8_t* __restrict__ dst_scales,
    size_t rows, size_t cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tiles_m = (rows + 15) / 16;
  size_t tiles_k = (cols + 15) / 16;
  size_t total_tiles = tiles_m * tiles_k;
  size_t total_entries = total_tiles * 16;
  if (idx >= total_entries) return;

  size_t tile_idx = idx / 16;
  size_t row_in_tile = idx % 16;

  size_t tile_m = tile_idx / tiles_k;
  size_t tile_k = tile_idx - tile_m * tiles_k;

  size_t row = tile_m * 16 + row_in_tile;
  size_t col = tile_k * 16;

  size_t blocks_per_row = (cols + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
  uint8_t code = static_cast<uint8_t>(127);
  if (row < rows && col < cols) {
    size_t block_idx = row * blocks_per_row + (col / MXFP4_BLOCK_SIZE);
    code = src_scales[block_idx];
  }
  dst_scales[idx] = code;
}

// Pack X: gather FP32 inputs per-slot to [K x Bpad] in BF16
__global__ void gather_pack_x_fp32_to_bf16_kernel(const float *x, hip_bfloat16 *xpack,
                                                  int K, int B, int Bpad,
                                                  int in_stride, const int *slot_ids) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)K * (size_t)Bpad;
  if (idx >= total) return;
  int k = (int)(idx / Bpad);
  int b = (int)(idx % Bpad);
  if (b < B) {
    int sid = slot_ids[b];
    float v = x[(size_t)sid * (size_t)in_stride + (size_t)k];
    xpack[(size_t)k * (size_t)Bpad + (size_t)b] = float_to_bfloat16(v);
  } else {
    xpack[(size_t)k * (size_t)Bpad + (size_t)b] = hip_bfloat16(0.0f);
  }
}


// Pack X: gather FP32 inputs per-slot and pack to [K x Bpad] (FP32)
__global__ void gather_pack_x_fp32_to_fp32_kernel(const float *x, float *xpack,
                                                  int K, int B, int Bpad,
                                                  int in_stride, const int *slot_ids) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)K * (size_t)Bpad;
  if (idx >= total) return;
  int k = (int)(idx / Bpad);
  int b = (int)(idx % Bpad);
  if (b < B) {
    int sid = slot_ids[b];
    float v = x[(size_t)sid * (size_t)in_stride + (size_t)k];
    xpack[(size_t)k * (size_t)Bpad + (size_t)b] = v;
  } else {
    xpack[(size_t)k * (size_t)Bpad + (size_t)b] = 0.0f;
  }
}


// Pack X with runtime B from device pointer: gather FP32 inputs and store to BF16 [K x Bpad]
// Blive = *total_post_pad_ptr; for b >= Blive, write zeros. slot_ids is either d_slot_ids or sorted_sids.
__global__ void gather_pack_x_fp32_to_bf16_kernel_Bptr(const float *x, hip_bfloat16 *xpack,
                                                       int K, int Bpad,
                                                       int in_stride, const int *slot_ids,
                                                       const int *total_post_pad_ptr) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)K * (size_t)Bpad;
  if (idx >= total) return;
  int k = (int)(idx / Bpad);
  int b = (int)(idx % Bpad);
  const int Blive = *total_post_pad_ptr;
  if (b < Blive) {
    int sid = slot_ids[b];
    float v = x[(size_t)sid * (size_t)in_stride + (size_t)k];
    xpack[(size_t)k * (size_t)Bpad + (size_t)b] = float_to_bfloat16(v);
  } else {
    xpack[(size_t)k * (size_t)Bpad + (size_t)b] = hip_bfloat16(0.0f);
  }
}




// ========================= FLOAT KERNELS =========================



// ------------------------- New BF16 GEMM path (packed 16x16) -------------------------



// GEMM MFMA FP32: W_packed [Mpad x Kpad] (tile 16x16 contiguous), Xpack BF16 [K x Bpad] (row-major K, stride=Bpad)
// Compute uses BF16×BF16→FP32 MFMA. Accumulate FP32, add BF16 bias, and scatter to per-slot outputs via slot_ids.
// Each block computes one 16x16 tile with 64 threads (dim3 block(16,4,1)).
__global__ __launch_bounds__(64)
void gemm16x16_mfma_f32_packedW_batched_scatter_bias(
    const hip_bfloat16* __restrict__ W_packed,
    const hip_bfloat16* __restrict__ Xpack,
    float* __restrict__ dst,
    const hip_bfloat16* __restrict__ bias,  // nullable
    int M, int K, int B, int Bpad,
    int out_stride,
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad)
{
  // Thread mapping matches AMD 16x16x4f32 MFMA sample: blockDim(16,4,1)
  const int tx = threadIdx.x;  // 0..15 -> column within tile (N)
  const int ty = threadIdx.y;  // 0..3  -> selects k within 4-wide micro-tile and output row group

  // Compute lane id for shared loads grouping
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;     // along M
  const int tile_n = blockIdx.y;     // along B
  const int base_row = tile_m * 16;
  const int base_col = tile_n * 16;

  const int Ktiles = (Kpad + 15) / 16;
  const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);

  __shared__ __align__(16) hip_bfloat16 sWb[2][16 * 16];
  __shared__ __align__(16) hip_bfloat16 sXb[2][16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  // Preload first tile into buffer 0
  int curr = 0;
  {
    const int kk = 0;
    if (lane < 32) {
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W_packed + w_tile_base;
      reinterpret_cast<u32x4*>(sWb[curr])[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
    }
    if (lane < 32) {
      const int r   = lane % 16;
      const int h   = lane / 16;  // 0 or 1
      const int c8  = h * 8;      // columns 0..7 or 8..15
      const int k_row = kk * 16 + r;
      const bool in_k = (k_row < K);
      hip_bfloat16* dst = &sXb[curr][0] + r * 16 + c8;
      if (in_k) {
        const hip_bfloat16* src = Xpack + (size_t)k_row * (size_t)Bpad + (size_t)base_col + (size_t)c8;
        reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
      } else {
        reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
      }
    }
    __syncthreads();
  }

  for (int kk = 0; kk < Ktiles; ++kk) {
    const int next = curr ^ 1;
    // Prefetch next tile
    if (kk + 1 < Ktiles) {
      if (lane < 32) {
        const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)(kk + 1)) * (size_t)256;
        const hip_bfloat16* w_src = W_packed + w_tile_base;
        reinterpret_cast<u32x4*>(sWb[next])[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
      }
      if (lane < 32) {
        const int r   = lane % 16;
        const int h   = lane / 16;  // 0 or 1
        const int c8  = h * 8;      // columns 0..7 or 8..15
        const int k_row = (kk + 1) * 16 + r;
        const bool in_k = (k_row < K);
        hip_bfloat16* dst = &sXb[next][0] + r * 16 + c8;
        if (in_k) {
          const hip_bfloat16* src = Xpack + (size_t)k_row * (size_t)Bpad + (size_t)base_col + (size_t)c8;
          reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
        } else {
          reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
        }
      }
    }

    // Compute on current tile
    {
      const int kbase = ty * 4;  // 4 BF16s per operand per lane
      using short4v = __attribute__((ext_vector_type(4))) short;
      union UB { hip_bfloat16 b; unsigned short u; } u0, u1, u2, u3;
      short4v Areg, Breg;
      u0.b = sWb[curr][tx * 16 + (kbase + 0)]; u1.b = sWb[curr][tx * 16 + (kbase + 1)];
      u2.b = sWb[curr][tx * 16 + (kbase + 2)]; u3.b = sWb[curr][tx * 16 + (kbase + 3)];
      Areg.x = (short)u0.u; Areg.y = (short)u1.u; Areg.z = (short)u2.u; Areg.w = (short)u3.u;
      u0.b = sXb[curr][(kbase + 0) * 16 + tx]; u1.b = sXb[curr][(kbase + 1) * 16 + tx];
      u2.b = sXb[curr][(kbase + 2) * 16 + tx]; u3.b = sXb[curr][(kbase + 3) * 16 + tx];
      Breg.x = (short)u0.u; Breg.y = (short)u1.u; Breg.z = (short)u2.u; Breg.w = (short)u3.u;
      dmn = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, dmn, 0, 0, 0);
    }

    __syncthreads();
    curr = next;
  }

  // Epilogue: add bias and scatter store
  const int col = base_col + tx;
  if (col < B) {
    const int sid = slot_ids[col];
    float* out_s = dst + (size_t)sid * (size_t)out_stride;
    // Each thread writes 4 rows: row = base_row + (ty*4 + i)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = base_row + (ty * 4 + i);
      if (row < M) {
        const float biasv = bias ? bfloat16_to_float(bias[row]) : 0.f;
        out_s[row] = dmn[i] + biasv;
      }
    }
  }
}

// Pack and convert FP32 row-major [M x K] to BF16 16x16 tiles with padding
__global__ void pack_convert_fp32_to_bf16_tiles16_kernel(
    const float *src_fp32, int row_offset, int rows,
    int M, int K,
    hip_bfloat16 *dst_packed, int Mpad, int Kpad) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)rows * (size_t)K;
  if (idx >= total) return;
  int r = (int)(idx / K);
  int c = (int)(idx % K);
  int m = row_offset + r;
  // Compute tile index and within-tile offset
  int tile_m = m / 16;
  int tile_k = c / 16;
  int im = m % 16;
  int ik = c % 16;
  size_t tiles_per_row = (size_t)(Kpad + 15) / 16;
  size_t dst_index = ((size_t)tile_m * tiles_per_row + (size_t)tile_k) * (size_t)256 + (size_t)im * 16 + (size_t)ik;
  dst_packed[dst_index] = float_to_bfloat16(src_fp32[(size_t)r * (size_t)K + (size_t)c]);
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


// Batched RMSNorm (FP32): out = weight * (x / rms(x)) for each slot in batch.
// Optimized for AMD MI250: one block per slot (grid.y = batch),
// threads stride across hidden_dim for coalesced access; in-block reduction in shared memory.
__global__ __launch_bounds__(256) void rmsnorm_kernel_float_batched(
    float *__restrict__ out,                 // [slots, hidden_dim]
    float *__restrict__ x,                  // [slots, hidden_dim] (may be updated if residual != nullptr)
    const hip_bfloat16 *__restrict__ weight, // [hidden_dim]
    const float *__restrict__ residual,     // optional [slots, hidden_dim]; if not null: x += residual (in-place)
    int hidden_dim,
    int batch,
    int stride,                 // per-slot stride == hidden_dim
    const int *slot_ids) {

  int b = blockIdx.y; // slot index within the batch
  if (b >= batch) return;

  int sid = slot_ids[b];
  float *x_slot = x + (size_t)sid * (size_t)stride;
  float *out_slot = out + (size_t)sid * (size_t)stride;
  const float *r_slot = residual ? (residual + (size_t)sid * (size_t)stride) : nullptr;

  extern __shared__ float s_reduce[]; // size = blockDim.x

  // 1) Compute sum of squares across hidden_dim with thread striding.
  //    If residual provided, add it in-place (x_slot[j] += r_slot[j]).
  float ss = 0.0f;
  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float v = x_slot[j];
    if (r_slot) {
      v += r_slot[j];
      x_slot[j] = v; // in-place residual update
    }
    ss += v * v;
  }
  s_reduce[threadIdx.x] = ss;
  __syncthreads();

  // 2) In-block parallel reduction
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + offset];
    }
    __syncthreads();
  }

  // 3) Compute inverse RMS and write normalized, scaled output
  float inv_rms = rsqrtf(s_reduce[0] / hidden_dim + 1e-5f);
  __syncthreads();

  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float w = bfloat16_to_float(weight[j]);
    out_slot[j] = x_slot[j] * inv_rms * w;
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

// Build histogram of expert IDs for M selections
__global__ void histogram_eid_counts_kernel(const int *sel_eids, int M, int *eid_counts, int n_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M) {
    int e = sel_eids[idx];
    if (e >= 0 && e < n_experts) {
      atomicAdd(&eid_counts[e], 1);
    }
  }
}

// Fused kernel: per-slot TopK + softmax + direct gather to sel_* + histogram (eid_counts)
// One block handles one slot (only thread 0 used for simplicity; k is small)
__global__ void router_fused_topk_gather_hist_kernel(
    const float *router_scores, // [max_batch, n_experts] via sid*score_stride
    int n_experts,              // total experts (N)
    int k,                      // experts_per_token (Top-K)
    const int *slot_ids,        // [batch]
    int batch,
    int score_stride,           // stride for router_scores per slot
    float *sel_w,               // [batch*k]
    int *sel_eids,              // [batch*k]
    int *sel_sids,              // [batch*k]
    int *eid_counts)            // [n_experts], must be zeroed before launch
{
  int b = blockIdx.x;
  if (b >= batch) return;
  if (threadIdx.x != 0) return;

  int sid = slot_ids[b];
  const float *scores = router_scores + (size_t)sid * (size_t)score_stride;

  extern __shared__ unsigned char smem[];
  float *topv = reinterpret_cast<float*>(smem);          // length = k
  int   *topi = reinterpret_cast<int*>(topv + k);        // length = k

  // init
  for (int i = 0; i < k; ++i) { topv[i] = -INFINITY; topi[i] = -1; }
  // top-k (simple insertion loop like baseline)
  for (int j = 0; j < n_experts; ++j) {
    float v = scores[j];
    int pos = -1;
    for (int t = 0; t < k; ++t) { if (v > topv[t]) { pos = t; break; } }
    if (pos >= 0) {
      for (int t = k - 1; t > pos; --t) { topv[t] = topv[t - 1]; topi[t] = topi[t - 1]; }
      topv[pos] = v; topi[pos] = j;
    }
  }
  // softmax over top-k
  float maxv = topv[0];
  for (int i = 1; i < k; ++i) if (topv[i] > maxv) maxv = topv[i];
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) { topv[i] = expf(topv[i] - maxv); sum += topv[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < k; ++i) topv[i] *= inv;

  // write flattened selections and update histogram
  int base = b * k;
  for (int t = 0; t < k; ++t) {
    int out = base + t;
    int e = topi[t];
    sel_w[out]    = topv[t];
    sel_eids[out] = e;
    sel_sids[out] = sid;
    if (e >= 0 && e < n_experts) {
      atomicAdd(&eid_counts[e], 1);
    }
  }
}

// Exclusive prefix sum for small arrays (single-thread)
__global__ void exclusive_prefix_sum_small_kernel(const int *counts, int *offsets, int n_experts) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < n_experts; ++i) {
      offsets[i] = sum;
      sum += counts[i];
    }
    offsets[n_experts] = sum;
  }
}


// Optimized vectorized bfloat16 to float conversion
__device__ __forceinline__ float4 bfloat16_to_float4(const hip_bfloat16 *bf16_ptr) {
  float4 result;
  result.x = bfloat16_to_float(bf16_ptr[0]);
  result.y = bfloat16_to_float(bf16_ptr[1]);
  result.z = bfloat16_to_float(bf16_ptr[2]);
  result.w = bfloat16_to_float(bf16_ptr[3]);
  return result;
}


// Batched FlashAttention-like kernel (simplified for reliability)
__global__ void flash_attention_kernel_optimized_batched(
    const float *all_q,           // [B, n_heads*head_dim]
    const float *key_cache,       // [B, layers*seq_len*kv_dim]
    const float *value_cache,     // [B, layers*seq_len*kv_dim]
    const hip_bfloat16 *attn_sinks, // [layers * n_heads]
    float *all_output,            // [B, n_heads*head_dim]
    const float *mask,            // Sliding window mask (optional)
    int head_dim, int loff, int kv_dim, int kv_mul,
    int n_heads, int seq_len, int layer,
    const int *pos_list,          // [B]
    bool can_add_sink,
    int batch,
    int q_stride,
    int kv_block_size,
    const int *slot_ids,
    int tile_size) {

  int h = blockIdx.y;   // head index
  int b = blockIdx.z;   // batch index
  if (h >= n_heads || b >= batch) return;

  int pos = pos_list[b];
  int sid = slot_ids[b];
  const float *q_head = all_q + (size_t)sid * (size_t)q_stride + (size_t)h * (size_t)head_dim;
  float *output = all_output + (size_t)sid * (size_t)(n_heads * head_dim) + (size_t)h * (size_t)head_dim;

  const float *key_cache_base   = key_cache   + (size_t)sid * (size_t)kv_block_size;
  const float *value_cache_base = value_cache + (size_t)sid * (size_t)kv_block_size;

  // Simple shared memory for attention scores
  extern __shared__ float shared_att_batch[];
  
  // Phase 1: Compute attention scores
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float *k = key_cache_base + (size_t)loff + (size_t)t * (size_t)kv_dim + (size_t)(h / kv_mul) * (size_t)head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += q_head[i] * k[i];
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      score += mask[(size_t)pos * (size_t)seq_len + (size_t)t];
    }
    shared_att_batch[t] = score;
  }
  __syncthreads();

  // Phase 2: Add attention sink
  if (threadIdx.x == 0 && can_add_sink && (pos + 1) < seq_len) {
    size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    shared_att_batch[pos + 1] = bfloat16_to_float(attn_sinks[sink_idx]);
  }
  __syncthreads();

  // Phase 3: Softmax
  if (threadIdx.x == 0) {
    int len = can_add_sink ? (pos + 2) : (pos + 1);
    if (len > 0) {
      float max_val = shared_att_batch[0];
      for (int i = 1; i < len; i++) max_val = fmaxf(max_val, shared_att_batch[i]);
      float sum_val = 0.0f;
      for (int i = 0; i < len; i++) { 
        float v = expf(shared_att_batch[i] - max_val); 
        shared_att_batch[i] = v; 
        sum_val += v; 
      }
      float inv = 1.0f / (sum_val + 1e-8f); // Add epsilon for stability
      for (int i = 0; i < len; i++) shared_att_batch[i] *= inv;
    }
  }
  __syncthreads();

  // Phase 4: Weighted sum with values
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float *v = value_cache_base + (size_t)loff + (size_t)t * (size_t)kv_dim + (size_t)(h / kv_mul) * (size_t)head_dim;
      sum += shared_att_batch[t] * v[i];
    }
    output[i] = sum;
  }
}

// New paged attention kernel that works with KVCacheManager
// Optimized: use MFMA BF16xBF16->FP32 for Q·K^T score computation in 16-token tiles.
// Each block uses 64 threads (16x4) to feed 16x16x16bf16 MFMA and reduce rows → 1x16 scores.
__global__ __launch_bounds__(64)
void paged_flash_attention_kernel_batched(
    const float * __restrict__ all_q,           // [B, n_heads*head_dim]
    KVCacheDeviceData * __restrict__ cache_data, // Paged cache manager device data
    const hip_bfloat16 * __restrict__ attn_sinks, // [layers * n_heads]
    float * __restrict__ all_output,            // [B, n_heads*head_dim]
    int head_dim, int kv_dim, int kv_mul,
    int n_heads, int seq_len, int layer,
    const int * __restrict__ pos_list,          // [B]
    int batch,
    int q_stride,
    const int * __restrict__ slot_ids) {

  int h = blockIdx.y;   // head index
  int b = blockIdx.z;   // batch index
  if (h >= n_heads || b >= batch) return;

  int pos = pos_list[b];
  int sid = slot_ids[b];
  const float *q_head = all_q + (size_t)sid * (size_t)q_stride + (size_t)h * (size_t)head_dim;
  float *output = all_output + (size_t)sid * (size_t)(n_heads * head_dim) + (size_t)h * (size_t)head_dim;

  // Get cache pointers for this slot and layer
  hip_bfloat16 *key_cache_layer, *value_cache_layer;
  size_t cache_size;
  cache_data->get_cache_ptrs(sid, layer, &key_cache_layer, &value_cache_layer, &cache_size);
  
  // Get attention range for this layer type
  int start_pos, end_pos;
  cache_data->get_attention_range(sid, layer, pos, &start_pos, &end_pos);

  // Simple shared memory for attention scores (allocate enough for sliding window)
  extern __shared__ float shared_att_paged[];
  __shared__ int s_att_len;
  int att_len = end_pos - start_pos + 1;
  // Capacity (tokens) for this layer's cache (without sink). We allocate +1 slot in shared mem via host launch.
  int layer_capacity = (cache_data->get_layer_type(layer) == SLIDING_WINDOW) ? cache_data->sliding_window_size : seq_len;
  if (threadIdx.x == 0) s_att_len = att_len;
  __syncthreads();
  
  // Phase 1: Compute attention scores using MFMA BF16×BF16→FP32 in 16-token tiles.
  // Map 64 threads as (tx: 0..15 columns, ty: 0..3 row-groups) like other MFMA kernels.
  const int lane_id = threadIdx.x;         // 0..63
  const int tx   = lane_id & 15;           // column within 16-token tile
  const int ty   = (lane_id >> 4) & 3;     // row group (4 rows per lane)
  const float inv_sqrt_d = rsqrtf((float)head_dim);
  const int head_base = (h / kv_mul) * head_dim;

  // Shared helpers for MFMA tiling
  using float4v = __attribute__((ext_vector_type(4))) float;
  using short4v = __attribute__((ext_vector_type(4))) short;
  __shared__ short sQbf16[16];       // Q tile (16 bf16 packed to i16)
  __shared__ int   sColPos[16];      // cache positions for 16 tokens in this tile (-1 if invalid)
  __shared__ float sColSum[16];      // accumulate partial sums across rows for each column

  // Iterate over attention columns (positions) in tiles of 16
  for (int col0 = 0; col0 < att_len; col0 += 16) {
    const int col = col0 + tx;
    const bool col_in_range = (col < att_len);

    // Compute cache position for each column once
    if (ty == 0 && tx < 16) {
      if (col_in_range) {
        const int t = start_pos + col;
        const int cp = (t <= pos) ? cache_data->get_cache_position(sid, layer, t, pos) : -1;
        sColPos[tx] = cp;
      } else {
        sColPos[tx] = -1;
      }
      sColSum[tx] = 0.0f;
    }
    __syncthreads();

    // Accumulator for this 16-column tile
    float4v acc = {0.f, 0.f, 0.f, 0.f};

    // Walk K dimension in tiles of 16
    const int Ktiles = (head_dim + 15) / 16;
    for (int kk = 0; kk < Ktiles; ++kk) {
      // Preload 16 Q scalars for this K tile, convert to bf16 once
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

      // Build A and B operands for MFMA from 4 k-rows handled by this lane
      const int kbase = ty * 4;  // rows within 16
      short4v Areg, Breg;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kij = kbase + j;                    // 0..15
        // A tile is diagonal with Q[kij] on (kij,kij); encode directly without shared memory writes.
        Areg[j] = (tx == kij) ? sQbf16[kij] : (short)0;
        // Breg pulls K at column 'tx' (one position) and row kij of this K tile, if valid
        short kval = 0;
        const int cp = sColPos[tx];
        const int k_abs = kk * 16 + kij;
        if (cp >= 0 && k_abs < head_dim) {
          const hip_bfloat16 *kb = key_cache_layer + (size_t)cp * (size_t)kv_dim + (size_t)head_base + (size_t)k_abs;
          union { hip_bfloat16 b; unsigned short u; } uk;
          uk.b = *kb;
          kval = (short)uk.u;
        }
        Breg[j] = kval;
      }

      // MFMA accumulate 16x16x16 (bf16) into FP32 registers
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);
      __syncthreads();
    }

    // Reduce 16 rows → 1 value per column: sum 4 rows in this lane, then shfl-xor across ty groups.
    float partial = acc.x + acc.y + acc.z + acc.w;
    // Sum across lanes with same tx but different ty: toggle bit 4 (16) and 5 (32)
    partial += __shfl_xor(partial, 16, 64);
    partial += __shfl_xor(partial, 32, 64);

    // Write normalized scores to shared buffer (one lane per column)
    if (ty == 0 && col_in_range) {
      float score = (sColPos[tx] >= 0) ? (partial * inv_sqrt_d) : -INFINITY;
      shared_att_paged[col] = score;
    }
    __syncthreads();
  }

  // Phase 2: Add attention sink (match baseline behavior: add for all layers if space and within seq_len)
  // Baseline always injects sink at position pos+1 (even for sliding-window masked layers) and includes it in softmax
  // but excludes it from value accumulation, effectively reserving probability mass for the sink. We replicate that.
  if (threadIdx.x == 0 && (pos + 1) < seq_len) {
    size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    // We allocated shared memory for (layer_capacity + 1) scores; only append if space.
    if (s_att_len < layer_capacity + 1) {
      shared_att_paged[s_att_len] = bfloat16_to_float(attn_sinks[sink_idx]);
      s_att_len++;
    }
  }
  __syncthreads();
  const int len = s_att_len;

  // Phase 3: Softmax (parallel across threads)
  // Compute max
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    local_max = fmaxf(local_max, shared_att_paged[i]);
  }
  // Reduce max across block using shuffles then shared memory scalars in last warp
  // Warp reduce
  const int W = 64;
  for (int offset = W/2; offset > 0; offset >>= 1) {
    float other = __shfl_down(local_max, offset, W);
    local_max = fmaxf(local_max, other);
  }
  // Write warp results to shared, then have warp 0 reduce them
  __shared__ float warp_max[4];
  const int lane = threadIdx.x & (W-1);
  const int wid  = threadIdx.x / W; // 64-thread wavefronts
  if (lane == 0) warp_max[wid] = local_max;
  __syncthreads();
  float max_val = -INFINITY;
  if (wid == 0) {
    int warps = blockDim.x / W;
    float val = (lane < warps) ? warp_max[lane] : -INFINITY;
    for (int offset = W/2; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_down(val, offset, W));
    if (lane == 0) warp_max[0] = val;
  }
  __syncthreads();
  max_val = warp_max[0];

  // Compute exp and sum
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    float v = shared_att_paged[i];
    v = (v == -INFINITY) ? 0.0f : expf(v - max_val);
    shared_att_paged[i] = v;
    local_sum += v;
  }
  // Warp reduce sum
  for (int offset = W/2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset, W);
  }
  __shared__ float warp_sum[4];
  if (lane == 0) warp_sum[wid] = local_sum;
  __syncthreads();
  float sum_val = 0.0f;
  if (wid == 0) {
    int warps = blockDim.x / W;
    float val = (lane < warps) ? warp_sum[lane] : 0.0f;
    for (int offset = W/2; offset > 0; offset >>= 1) val += __shfl_down(val, offset, W);
    if (lane == 0) warp_sum[0] = val;
  }
  __syncthreads();
  sum_val = warp_sum[0];
  const float inv_sum = 1.0f / (sum_val + 1e-8f);
  __syncthreads();
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    shared_att_paged[i] *= inv_sum;
  }
  __syncthreads();

  // Phase 4: Weighted sum with values (threads stride across dims)
  {
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      float acc = 0.0f;
      for (int idx = 0; idx < len; idx++) {
        const int t = start_pos + idx;
        const float p = shared_att_paged[idx];
        if (t <= pos && p > 0.f) {
          const int cache_pos = cache_data->get_cache_position(sid, layer, t, pos);
          if (cache_pos >= 0) {
            const hip_bfloat16 *v_bf16 = value_cache_layer + (size_t)cache_pos * (size_t)kv_dim + (size_t)head_base;
            acc += p * bfloat16_to_float(v_bf16[i]);
          }
        }
      }
      output[i] = acc;
    }
  }
}


#define WARP_SIZE 64  // AMD wavefront size
#define BLOCK_SIZE 256
#define UNROLL_FACTOR 8

// Warp-level reduction for finding max
__device__ __forceinline__ void warp_reduce_max(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down(val, offset);
        int other_idx = __shfl_down(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Block-level reduction using shared memory
__device__ __forceinline__ void block_reduce_max(float& val, int& idx, float* shmem_val, int* shmem_idx) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // First, do warp-level reduction
    warp_reduce_max(val, idx);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shmem_val[wid] = val;
        shmem_idx[wid] = idx;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (wid == 0) {
        val = (lane < num_warps) ? shmem_val[lane] : -INFINITY;
        idx = (lane < num_warps) ? shmem_idx[lane] : 0;
        warp_reduce_max(val, idx);
    }
}

// Optimized kernel using block-wide parallel reduction
__global__ void sample_argmax_kernel_optimized(
    const float* __restrict__ logits, 
    int* __restrict__ next_tokens, 
    int batch, 
    int vocab_size) {
    
    extern __shared__ char shared_mem[];
    float* shmem_val = (float*)shared_mem;
    int* shmem_idx = (int*)(shmem_val + (BLOCK_SIZE / WARP_SIZE));
    
    const int b = blockIdx.x;
    if (b >= batch) return;
    
    const float* logit = logits + b * vocab_size;
    
    float max_val = -INFINITY;
    int max_idx = 0;
    
    // Each thread processes multiple elements with unrolling
    const int tid = threadIdx.x;
    const int stride = BLOCK_SIZE;
    
    // Main loop with unrolling for better ILP
    int i = tid;
    
    // Process multiple elements per thread with explicit unrolling
    if (vocab_size - i >= UNROLL_FACTOR * stride) {
        for (; i < vocab_size - (UNROLL_FACTOR-1) * stride; i += UNROLL_FACTOR * stride) {
            float vals[UNROLL_FACTOR];
            int indices[UNROLL_FACTOR];
            
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                int idx = i + u * stride;
                vals[u] = logit[idx];
                indices[u] = idx;
            }
            
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                if (vals[u] > max_val) {
                    max_val = vals[u];
                    max_idx = indices[u];
                }
            }
        }
    }
    
    // Handle remaining elements
    for (; i < vocab_size; i += stride) {
        float val = logit[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    // Block-wide reduction
    block_reduce_max(max_val, max_idx, shmem_val, shmem_idx);
    
    // Write result
    if (threadIdx.x == 0) {
        next_tokens[b] = max_idx;
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


// ------------------------- MoE kernels -------------------------
__global__ void build_padded_offsets_kernel(const int* __restrict__ counts,
                                            int* __restrict__ padded_offsets,
                                            int n_experts, int block_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int sum = 0;
  padded_offsets[0] = 0;
  for (int e = 0; e < n_experts; ++e) {
    int c = counts[e];
    int padc = ((c + block_size - 1) / block_size) * block_size;
    padded_offsets[e] = sum;
    sum += padc;
  }
  padded_offsets[n_experts] = sum; // total_post_pad
}

__global__ void scatter_sorted_real_kernel(const int* __restrict__ sel_sids_in,
                                           const int* __restrict__ sel_eids_in,
                                           const float* __restrict__ sel_w_in,
                                           int M, const int* __restrict__ padded_offsets,
                                           int* __restrict__ eid_cursors, // zeroed
                                           int* __restrict__ sorted_indices,
                                           int* __restrict__ sorted_sids,
                                           int* __restrict__ sorted_eids,
                                           float* __restrict__ sorted_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  int e = sel_eids_in[idx];
  int pos = atomicAdd(&eid_cursors[e], 1);                // 0..counts[e)-1
  int out = padded_offsets[e] + pos;
  sorted_indices[out] = idx;
  sorted_sids[out]    = sel_sids_in[idx];
  sorted_eids[out]    = e;
  sorted_w[out]       = sel_w_in[idx];
}

__global__ void fill_padding_and_blocks_kernel(const int* __restrict__ counts,
                                               const int* __restrict__ padded_offsets,
                                               int n_experts, int block_size,
                                               int* __restrict__ sorted_indices,
                                               int* __restrict__ sorted_sids,
                                               int* __restrict__ sorted_eids,
                                               float* __restrict__ sorted_w,
                                               int* __restrict__ block_expert_ids) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_experts) return;

  int start = padded_offsets[e];
  int c     = counts[e];
  int padc  = ((c + block_size - 1) / block_size) * block_size;

  // Fill padding slots
  for (int i = c; i < padc; ++i) {
    int o = start + i;
    sorted_indices[o] = -1;
    sorted_sids[o] = 0;
    sorted_eids[o] = e;
    sorted_w[o] = 0.0f;  // đánh dấu pad
  }
  // Set block->expert id
  for (int b = 0; b < padc / block_size; ++b) {
    block_expert_ids[(start / block_size) + b] = e;
  }
}

__global__ void iota_kernel(int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = i;
}

// Fused GEMM (W_mlp1 * Xpack) + bias + SwiGLU + scatter to gate_up_all.
// - W_data/W_scales: MXFP4 tile-packed 16x16 blocks dequantized on-the-fly, Mout = 2*I
// - Xpack:    FP32 [K x Bpad], ta xử lý Be cột liên tiếp (đã align BLOCK_B)
// - gate_up_out: BF16 [B_global x I] (ở đây ta scatter theo "column id" toàn cục)
// - slot_ids: ánh xạ cột local (0..Be-1) -> column id toàn cục (d_col_ids + c0)
// Mỗi block: 64 threads (16x4) tính 1 tile 16x16 (M x N).
// Grouped variant: handle all experts in a single launch by selecting the expert per 16-column block.
// Uses block_expert_ids[blockIdx.y] to pick the expert and computes fused (GEMM + bias + SwiGLU) then scatters.
__global__ __launch_bounds__(64)
void gemm16x16_mlp1_fused_swiglu_scatter_grouped(
    const uint8_t* __restrict__ W_data_all,          // MXFP4 tile-packed nibbles [n_layers*n_experts][(Mpad*Kpad)/2]
    const uint8_t* __restrict__ W_scales_all,        // MXFP4 scales tile-packed [n_layers*n_experts][tiles_m*tiles_k*16]
    const hip_bfloat16* __restrict__ Xpack,          // [K x Bpad_total] BF16, columns are globally sorted/padded
    hip_bfloat16*       __restrict__ gate_up_out,    // [I x Bpad_total] BF16 row-major (row stride = Bpad_total)
    const hip_bfloat16* __restrict__ bias_all,       // [n_layers*n_experts][Mout]
    // Logical sizes
    int Mout, int K, int Btotal, int Bpad_total,
    int out_stride_cols,                              // leading dimension across columns (Bpad_total)
    const int* __restrict__ d_col_ids,               // [Btotal] global column ids (iota)
    // Mapping from 16-col block -> expert id
    const int* __restrict__ block_expert_ids,        // [Bpad_total/BLOCK_B]
    int block_size,                                  // BLOCK_B (must be multiple of 16)
    int n_experts, int layer,
    int Mpad, int Kpad,
    float swiglu_limit,
    const int* __restrict__ total_post_pad_ptr)
{
  const int tx = threadIdx.x;   // 0..15 (N within tile)
  const int ty = threadIdx.y;   // 0..3
  const int lane = ty*16 + tx;  // 0..63

  const int tile_m  = blockIdx.x;     // along M (rows)
  const int tile_n  = blockIdx.y;     // along N (cols), each tile is a 16-col block
  const int row0    = tile_m * 16;
  const int col0    = tile_n * 16;

  // Early cull tiles beyond live padded width computed on device
  const int Btotal_live = *total_post_pad_ptr;
  const int Bpad_live   = ((Btotal_live + 15) / 16) * 16;
  if (col0 >= Bpad_live) return;     // skip fully padded tiles

  // Determine expert for this 16-col tile via BLOCK_B mapping
  int blocks_per_b = block_size / 16; // tiles per BLOCK_B
  if (blocks_per_b < 1) blocks_per_b = 1;
  const int block_idx = tile_n / blocks_per_b;
  const int eid = block_expert_ids[block_idx];

  // Resolve per-(layer,eid) bases
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const size_t packed_mat_bytes = ((size_t)Mpad * (size_t)Kpad + 1) / 2;
  const int tiles_m = (Mpad + 15) / 16;
  const int tiles_k = (Kpad + 15) / 16;
  const size_t scales_per_mat = (size_t)tiles_m * (size_t)tiles_k * (size_t)16;
  const uint8_t* __restrict__ W_data = W_data_all + mat_index * packed_mat_bytes;
  const uint8_t* __restrict__ W_scales = W_scales_all + mat_index * scales_per_mat;
  const hip_bfloat16* __restrict__ B = bias_all ? (bias_all + mat_index * (size_t)Mout) : nullptr;

  // Double-buffered LDS tiles (ping-pong) to reduce synchronization overhead
  __shared__ __align__(16) hip_bfloat16 sWb[2][16*16];
  __shared__ __align__(16) hip_bfloat16 sX [2][16*16];
  __shared__ __align__(16) float        sBias[16];  // cache bias for this 16-row tile

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f,0.f,0.f,0.f};

  const int Ktiles = (Kpad + 15)/16;
  const size_t tilesPerRow = (size_t)tiles_k;
  // Cache bias for this tile once if present
  if (B) {
    if (lane < 16) {
      const int r = row0 + lane;
      sBias[lane] = (r < Mout) ? bfloat16_to_float(B[r]) : 0.f;
    }
    __syncthreads();
  }

  // Preload first K tile into buffer 0
  int curr = 0;
  if (Ktiles > 0) {
    const int kk = 0;
    if (lane < 64) {
      int idx4 = lane * 4;
      if (idx4 < 256) {
        int im = idx4 >> 4;
        int ik_base = idx4 & 0x0F;
        int row = row0 + im;
        int col_base = kk * 16 + ik_base;
        const size_t tile_linear = (size_t)tile_m * tilesPerRow + (size_t)kk;
        const size_t tile_base_byte = tile_linear * (size_t)128;
        const size_t scale_base = tile_linear * (size_t)16;
        uint16_t packed16 = *reinterpret_cast<const uint16_t*>(W_data + tile_base_byte + (idx4 >> 1));
        int lane_group = lane & ~3;
        int scale_code_int = 0x7F;
        if ((lane & 3) == 0) {
          scale_code_int = static_cast<int>(W_scales[scale_base + im]);
        }
        scale_code_int = __shfl(scale_code_int, lane_group, 4);
        uint8_t scale_code = static_cast<uint8_t>(scale_code_int & 0xFF);
        const uint16_t* __restrict__ lut_row = MXFP4_DEQUANT_LUT_BF16[scale_code];
        #pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
          int ik = ik_base + t4;
          int col = col_base + t4;
          uint16_t bits = 0;
          if (row < Mout && col < K) {
            uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
            bits = lut_row[nibble];
          }
          hip_bfloat16 val;
          *reinterpret_cast<uint16_t*>(&val) = bits;
          sWb[curr][im * 16 + ik] = val;
        }
      }
    }
    if (lane < 32) {
      const int r   = lane % 16;
      const int h   = lane / 16;   // 0 or 1
      const int c8  = h * 8;       // columns 0..7 or 8..15
      const int k_row = kk*16 + r;
      hip_bfloat16* dst = &sX[curr][0] + r*16 + c8;
      const bool in_k = (k_row < K);
      if (in_k) {
        const hip_bfloat16* src = Xpack + (size_t)k_row * (size_t)Bpad_total + (size_t)col0 + (size_t)c8;
        // Copy 8 bf16 values (16B) per lane
        reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
      } else {
        reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
      }
    }
    __syncthreads();
  }

  for (int kk = 0; kk < Ktiles; ++kk) {
    // Prefetch next into alternate buffer (if any)
    int next = curr ^ 1;
    if (kk + 1 < Ktiles) {
      if (lane < 64) {
        int idx4 = lane * 4;
        if (idx4 < 256) {
          int im = idx4 >> 4;
          int ik_base = idx4 & 0x0F;
          int row = row0 + im;
          int col_base = (kk + 1) * 16 + ik_base;
          const size_t tile_linear = (size_t)tile_m * tilesPerRow + (size_t)(kk + 1);
          const size_t tile_base_byte = tile_linear * (size_t)128;
          const size_t scale_base = tile_linear * (size_t)16;
          uint16_t packed16 = *reinterpret_cast<const uint16_t*>(W_data + tile_base_byte + (idx4 >> 1));
          int lane_group = lane & ~3;
          int scale_code_int = 0x7F;
          if ((lane & 3) == 0) {
            scale_code_int = static_cast<int>(W_scales[scale_base + im]);
          }
          scale_code_int = __shfl(scale_code_int, lane_group, 4);
          uint8_t scale_code = static_cast<uint8_t>(scale_code_int & 0xFF);
          const uint16_t* __restrict__ lut_row = MXFP4_DEQUANT_LUT_BF16[scale_code];
          #pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            int ik = ik_base + t4;
            int col = col_base + t4;
            uint16_t bits = 0;
            if (row < Mout && col < K) {
              uint8_t nibble = static_cast<uint8_t>((packed16 >> (t4 * 4)) & 0xF);
              bits = lut_row[nibble];
            }
            hip_bfloat16 val;
            *reinterpret_cast<uint16_t*>(&val) = bits;
            sWb[next][im * 16 + ik] = val;
          }
        }
      }
      if (lane < 32) {
        const int r   = lane % 16;
        const int h   = lane / 16;   // 0 or 1
        const int c8  = h * 8;       // columns 0..7 or 8..15
        const int k_row = (kk + 1)*16 + r;
        hip_bfloat16* dst = &sX[next][0] + r*16 + c8;
        const bool in_k = (k_row < K);
        if (in_k) {
          const hip_bfloat16* src = Xpack + (size_t)k_row * (size_t)Bpad_total + (size_t)col0 + (size_t)c8;
          reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
        } else {
          reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
        }
      }
    }

    // Compute on current tile
    const int kbase = ty * 4;  // 4 BF16s per operand per lane
    using short4v = __attribute__((ext_vector_type(4))) short;
    union UB { hip_bfloat16 b; unsigned short u; } u0, u1, u2, u3;
    short4v Areg, Breg;
    u0.b = sWb[curr][tx*16 + kbase + 0]; u1.b = sWb[curr][tx*16 + kbase + 1];
    u2.b = sWb[curr][tx*16 + kbase + 2]; u3.b = sWb[curr][tx*16 + kbase + 3];
    Areg.x = (short)u0.u; Areg.y = (short)u1.u; Areg.z = (short)u2.u; Areg.w = (short)u3.u;
    u0.b = sX[curr][(kbase + 0)*16 + tx]; u1.b = sX[curr][(kbase + 1)*16 + tx];
    u2.b = sX[curr][(kbase + 2)*16 + tx]; u3.b = sX[curr][(kbase + 3)*16 + tx];
    Breg.x = (short)u0.u; Breg.y = (short)u1.u; Breg.z = (short)u2.u; Breg.w = (short)u3.u;
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    __syncthreads();
    curr = next;
  }

  // Epilogue fused: (2I) → I (SwiGLU) and scatter by global column id; guard with Btotal
  const int col = col0 + tx;
  if (col >= Btotal_live) return;

  const float alpha = 1.702f;
  const int   I     = Mout >> 1;

  const int col_id = d_col_ids[col];

  const int row_stride = out_stride_cols;

  #pragma unroll
  for (int iter = 0; iter < 2; ++iter) {     // even/odd pairs within this thread
    const int i  = iter * 2;
    const int r  = row0 + (ty*4 + i);
    const int rp = r + 1;
    if (r < Mout && rp < Mout) {
      float gate = (iter == 0) ? acc.x : acc.z;
      float up   = (iter == 0) ? acc.y : acc.w;
      if (B) {
        const int rloc = (ty*4 + i);
        gate += sBias[rloc];
        up   += sBias[rloc + 1];
      }
      gate = fminf(swiglu_limit, gate);
      up   = fminf(swiglu_limit, fmaxf(-swiglu_limit, up));
      gate *= (1.f / (1.f + __expf(-alpha * gate)));
      const int row_half = r >> 1;
      if (row_half < I) {
        float fused = gate * (up + 1.f);
        hip_bfloat16* row_base = gate_up_out + (size_t)row_half * (size_t)row_stride;
        row_base[col_id] = float_to_bfloat16(fused);
      }
    }
  }
}


// Grouped MLP2 matvec using MFMA and packed BF16 weights with accumulation into e_agg.
// For each selection m (sid,eid), compute y = W_mlp2[layer,eid] @ gate_up_all[m] + b_mlp2[layer,eid]
// then atomically accumulate e_agg[sid] += sel_w[m] * y.
// W is packed as 16x16 BF16 tiles (Mpad x Kpad). Accumulate in FP32.
// Optimized: each block processes 16 contiguous selections (all from the same expert),
// filling the 16 MFMA columns with distinct vectors. Accumulates directly into x (residual fused).
__global__ __launch_bounds__(64)
void gemm16x16_mlp2_grouped_mfma_matvec_residual_kernel(
    float* __restrict__ x,                    // [slots][hidden_dim] — destination (residual fused)
    const hip_bfloat16* __restrict__ gate_up_all, // BF16 inputs packed row-major [intermediate_dim][Bpad_total]
    const hip_bfloat16* __restrict__ w_packed_mlp2,
    const hip_bfloat16* __restrict__ b_mlp2,
    const int*   __restrict__ sorted_sids,
    const int*   __restrict__ sorted_eids,
    const float* __restrict__ sorted_w,
    const int*   __restrict__ sorted_indices,
    int Btotal,
    int hidden_dim, int intermediate_dim,
    int gate_stride,
    int layer, int n_experts,
    int x_stride,
    int Mpad, int Kpad,
    const int* __restrict__ total_post_pad_ptr)
{
  // Each block computes a 16x16 tile: 16 rows (M) x 16 selections (N) for the same expert.
  // Y-dimension of the grid is over blocks of 16 contiguous selections grouped by expert.
  const int tx = threadIdx.x;  // 0..15 (columns)
  const int ty = threadIdx.y;  // 0..3  (row groups)
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;         // along hidden_dim (rows)
  const int block_b = blockIdx.y;        // block of 16 selections
  const int col0 = block_b * 16;
  const int Btotal_live = *total_post_pad_ptr;
  const int Bpad_live   = ((Btotal_live + 15) / 16) * 16;
  (void)Btotal;
  if (col0 >= Bpad_live) return;
  const int base_row = tile_m * 16;

  // All 16 columns in this block share the same expert due to prior grouping/padding.
  const int eid = sorted_eids[col0];

  // Pointers per (layer,eid)
  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W = w_packed_mlp2 + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = b_mlp2 + mat_index * (size_t)hidden_dim;

  __shared__ __align__(16) hip_bfloat16 sWb[2][16 * 16];
  __shared__ __align__(16) hip_bfloat16 sX [2][16 * 16];
  __shared__ __align__(16) float        sBias[16];   // cache bias for this 16-row tile

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int K = intermediate_dim;
  const int Ktiles = (Kpad + 15) / 16;
  const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);

  // Cache bias for this tile once if present
  if (B) {
    if (lane < 16) {
      const int r = base_row + lane;
      sBias[lane] = (r < hidden_dim) ? bfloat16_to_float(B[r]) : 0.f;
    }
    __syncthreads();
  }

  // Preload first tile (if any) into buffer 0
  int curr = 0;
  if (Ktiles > 0) {
    const int kk = 0;
    if (lane < 32) {
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W + w_tile_base;
      reinterpret_cast<u32x4*>(sWb[curr])[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
    }
    if (lane < 32) {
      const int r   = lane & 15;
      const int h   = lane >> 4;      // 0 or 1
      const int c8  = h * 8;          // columns 0..7 or 8..15 within tile
      const int k_row = kk * 16 + r;
      hip_bfloat16* dst = &sX[curr][r * 16 + c8];
      if (k_row < K) {
        const int col_base = col0 + c8;
        const size_t row_offset = (size_t)k_row * (size_t)gate_stride + (size_t)col_base;
        const int remaining = Btotal_live - col_base;
        if (remaining >= 8) {
          const hip_bfloat16* src = gate_up_all + row_offset;
          reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
        } else {
          alignas(16) hip_bfloat16 tmp[8];
          #pragma unroll
          for (int t = 0; t < 8; ++t) {
            const int c = col_base + t;
            tmp[t] = (c < Btotal_live) ? gate_up_all[row_offset + t] : hip_bfloat16(0.0f);
          }
          reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(tmp)[0];
        }
      } else {
        reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
      }
    }
    __syncthreads();
  }

  for (int kk = 0; kk < Ktiles; ++kk) {
    int next = curr ^ 1;
    if (kk + 1 < Ktiles) {
      if (lane < 32) {
        const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)(kk + 1)) * (size_t)256;
        const hip_bfloat16* w_src = W + w_tile_base;
        reinterpret_cast<u32x4*>(sWb[next])[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
      }
      if (lane < 32) {
        const int r   = lane & 15;
        const int h   = lane >> 4;
        const int c8  = h * 8;
        const int k_row = (kk + 1) * 16 + r;
        hip_bfloat16* dst = &sX[next][r * 16 + c8];
        if (k_row < K) {
          const int col_base = col0 + c8;
          const size_t row_offset = (size_t)k_row * (size_t)gate_stride + (size_t)col_base;
          const int remaining = Btotal_live - col_base;
          if (remaining >= 8) {
            const hip_bfloat16* src = gate_up_all + row_offset;
            reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
          } else {
            alignas(16) hip_bfloat16 tmp[8];
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
              const int c = col_base + t;
              tmp[t] = (c < Btotal_live) ? gate_up_all[row_offset + t] : hip_bfloat16(0.0f);
            }
            reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(tmp)[0];
          }
        } else {
          reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
        }
      }
    }

    // Build MFMA operands from BF16 tiles and accumulate
    const int kbase = ty * 4;
    using short4v = __attribute__((ext_vector_type(4))) short;
    union BF16Bits { hip_bfloat16 b; unsigned short u; } a0, a1, a2, a3, b0, b1, b2, b3;
    a0.b = sWb[curr][tx*16 + kbase + 0];
    a1.b = sWb[curr][tx*16 + kbase + 1];
    a2.b = sWb[curr][tx*16 + kbase + 2];
    a3.b = sWb[curr][tx*16 + kbase + 3];
    b0.b = sX[curr][(kbase + 0) * 16 + tx];
    b1.b = sX[curr][(kbase + 1) * 16 + tx];
    b2.b = sX[curr][(kbase + 2) * 16 + tx];
    b3.b = sX[curr][(kbase + 3) * 16 + tx];
    short4v Areg = {(short)a0.u, (short)a1.u, (short)a2.u, (short)a3.u};
    short4v Breg = {(short)b0.u, (short)b1.u, (short)b2.u, (short)b3.u};
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Areg, Breg, acc, 0, 0, 0);

    __syncthreads();
    curr = next;
  }

  // Helper to access accumulator lanes
  auto comp4 = [&] __device__ (int idx) -> float {
    return (idx==0)?acc.x : (idx==1)?acc.y : (idx==2)?acc.z : acc.w;
  };

  // Each thread contributes its 4 rows for its column (selection)
  const int msel = col0 + tx;  // selection index for this column
  if (msel < Btotal_live) {
    const int valid_idx = sorted_indices[msel];
    const float ew = sorted_w[msel];
    if (valid_idx >= 0 && ew != 0.f) {
      const int sid = sorted_sids[msel];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int rloc = (ty * 4 + i);
        const int row = base_row + rloc;
        if (row < hidden_dim) {
          float y = comp4(i) + (B ? sBias[rloc] : 0.f);
          atomicAdd(&x[(size_t)sid * (size_t)x_stride + (size_t)row], ew * y);
        }
      }
    }
  }
}





// ------------------------- Alloc/Free run state -------------------------
void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch);
void gpu_free_run_state(GPURunState *s);


// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
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
  int head_dim         = p->head_dim;
  int hidden_dim       = p->hidden_dim;
  int kv_dim           = p->head_dim * p->n_kv_heads;
  int kv_mul           = p->n_attn_heads / p->n_kv_heads;

  // Copy indices to device staging
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_slot_ids,  h_slot_ids,  batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_pos_list,  h_pos_list,  batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_token_ids, h_tokens,    batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));

  // 1) Embedding gather: x_slot[sid] = Emb[token]
  // Kernel: for b in batch and i in hidden_dim, copy embedding row
  // Implement simple gather using a kernel
  auto launch_embed = [&]() {
    struct Kernel {
      static __global__ void run(float *x, const hip_bfloat16 *emb, const int *tok, const int *slot_ids,
                                 int hidden_dim, int batch, int x_stride) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (i < hidden_dim && b < batch) {
          int sid = slot_ids[b];
          int t = tok[b];
          const hip_bfloat16 *row = emb + (size_t)t * (size_t)hidden_dim;
          float *x_s = x + (size_t)sid * (size_t)x_stride;
          x_s[i] = bfloat16_to_float(row[i]);
        }
      }
    };
    dim3 block(256);
    dim3 grid((hidden_dim + block.x - 1) / block.x, batch);
    Kernel::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.x, gpu_ctx->weights.token_embedding_table,
                                                           gpu_ctx->d_token_ids, gpu_ctx->d_slot_ids,
                                                           hidden_dim, batch, hidden_dim);
  };
  launch_embed();

  // 2) RoPE precompute per slot
  {
    int d_half = head_dim / 2;
    dim3 block(256);
    dim3 grid((d_half + block.x - 1) / block.x, batch);
    // Batched cos/sin compute kernel
    struct KernelCS {
      static __global__ void run(float *cos_out, float *sin_out, const int *pos_list, const int *slot_ids,
                                 int batch, float base, int head_dim, float scaling_factor,
                                 float initial_context_length, float ntk_beta, float ntk_alpha, int cs_stride) {
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
          float *cos_b = cos_out + (size_t)sid * (size_t)cs_stride;
          float *sin_b = sin_out + (size_t)sid * (size_t)cs_stride;
          cos_b[tid] = cosf(val) * concentration;
          sin_b[tid] = sinf(val) * concentration;
        }
      }
    };
    KernelCS::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
                                                             gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids,
                                                             batch, p->rope_theta, head_dim, p->rope_scaling_factor,
                                                             p->initial_context_length, 32.0f, 1.0f, d_half);
  }

  // Precompute constants for caches
  const size_t kv_block_size  = (size_t)p->n_layers * (size_t)p->seq_len * (size_t)kv_dim;
  const int q_size = head_dim * p->n_attn_heads;
  const int k_size = head_dim * p->n_kv_heads;
  const int v_size = head_dim * p->n_kv_heads;

  // Helper kernels to write caches from qkv buffers
  struct KVWriterV {
    static __global__ void run(const float *qkv, float *value_cache, int batch, int qkv_stride,
                               int v_offset, int v_size, int kv_dim, int loff,
                               size_t kv_block_size,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < v_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        const float *v_src = qkv + (size_t)sid * (size_t)qkv_stride + (size_t)v_offset;
        float *v_dst = value_cache + (size_t)sid * kv_block_size + (size_t)loff + (size_t)pos * (size_t)kv_dim;
        v_dst[i] = v_src[i];
      }
    }
  };
  
  struct KVWriterK {
    static __global__ void run(const float *qkv, float *key_cache, int batch, int qkv_stride,
                               int k_offset, int k_size, int kv_dim, int loff,
                               size_t kv_block_size,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < k_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        const float *k_src = qkv + (size_t)sid * (size_t)qkv_stride + (size_t)k_offset;
        float *k_dst = key_cache + (size_t)sid * kv_block_size + (size_t)loff + (size_t)pos * (size_t)kv_dim;
        k_dst[i] = k_src[i];
      }
    }
  };

  // New paged attention KV writers that handle sliding window caches
  struct PagedKVWriterV {
    static __global__ void run(const float *qkv, KVCacheDeviceData *cache_data, int layer,
                               int batch, int qkv_stride, int v_offset, int v_size, int kv_dim,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < v_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        
        // Get the write position (handles sliding window wraparound)
        int write_pos = cache_data->get_write_position(sid, layer, pos);
        
        // Get cache pointers for this slot and layer
  hip_bfloat16 *key_ptr, *value_ptr;
        size_t cache_size;
        cache_data->get_cache_ptrs(sid, layer, &key_ptr, &value_ptr, &cache_size);
        
        const float *v_src = qkv + (size_t)sid * (size_t)qkv_stride + (size_t)v_offset;
  hip_bfloat16 *v_dst = value_ptr + (size_t)write_pos * (size_t)kv_dim;
  v_dst[i] = float_to_bfloat16(v_src[i]);
      }
    }
  };
  
  struct PagedKVWriterK {
    static __global__ void run(const float *qkv, KVCacheDeviceData *cache_data, int layer,
                               int batch, int qkv_stride, int k_offset, int k_size, int kv_dim,
                               const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < k_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        
        // Get the write position (handles sliding window wraparound)
        int write_pos = cache_data->get_write_position(sid, layer, pos);
        
        // Get cache pointers for this slot and layer
  hip_bfloat16 *key_ptr, *value_ptr;
        size_t cache_size;
        cache_data->get_cache_ptrs(sid, layer, &key_ptr, &value_ptr, &cache_size);
        
        const float *k_src = qkv + (size_t)sid * (size_t)qkv_stride + (size_t)k_offset;
  hip_bfloat16 *k_dst = key_ptr + (size_t)write_pos * (size_t)kv_dim;
  k_dst[i] = float_to_bfloat16(k_src[i]);
      }
    }
  };

  // 3) Loop over layers with batched Attention and W_o
  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (batched across slots)
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    {
      dim3 block_norm(256), grid_norm(1, batch);
      size_t shared_mem_norm = block_norm.x * sizeof(float);
      rmsnorm_kernel_float_batched<<<grid_norm, block_norm, shared_mem_norm, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.x, rms_attn_w, /*residual=*/nullptr,
        hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }

    // QKV projection (batched) via packed BF16 GEMM
    int loff = l * p->seq_len * kv_dim;
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l *
                                ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; // M
    int Kqkv = hidden_dim;                                             // K
    int Bpad = ((batch + 15) / 16) * 16;
    int Mpad = ((qkv_out_dim + 15) / 16) * 16;
    int Kpad = ((Kqkv + 15) / 16) * 16;
    // Gather/pack inputs t -> xpack_bf16
    {
      size_t total = (size_t)Kqkv * (size_t)Bpad;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_bf16_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.xpack_bf16, Kqkv, batch, Bpad, hidden_dim, gpu_ctx->d_slot_ids);
    }
    // GEMM W_qkv_packed x Xpack -> qkv (scattered per slot) + bias
    {
      const hip_bfloat16 *w_qkv_packed = gpu_ctx->weights.w_qkv_packed16 + (size_t)l * (size_t)Mpad * (size_t)Kpad;
      dim3 grid((unsigned)(Mpad + 15) / 16, (unsigned)(Bpad + 15) / 16);
      dim3 block(16, 4, 1);
      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        w_qkv_packed, gpu_ctx->state.xpack_bf16, gpu_ctx->state.qkv, b_qkv,
        qkv_out_dim, Kqkv, batch, Bpad, qkv_out_dim, gpu_ctx->d_slot_ids, Mpad, Kpad);
    }

    // Write V to cache (before RoPE) - Use paged cache manager
    {
      dim3 block(256), grid((v_size + block.x - 1) / block.x, batch);
      // Use the new paged KV writer that handles different layer types
      PagedKVWriterV::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.qkv, gpu_ctx->state.kv_cache_manager->get_device_data(), l,
        batch, qkv_out_dim, q_size + k_size, v_size, kv_dim, 
        gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
    }

    // Apply RoPE to Q and K (batched)
    {
      dim3 grid_heads(p->n_attn_heads, batch);
      apply_rotary_emb_kernel_float_batched<<<grid_heads, head_dim/2, 0, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.qkv, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
      p->n_attn_heads, head_dim, batch, qkv_out_dim, head_dim/2, gpu_ctx->d_slot_ids);
    }
    {
      dim3 grid_heads_k(p->n_kv_heads, batch);
      apply_rotary_emb_kernel_float_batched<<<grid_heads_k, head_dim/2, 0, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.qkv + q_size, gpu_ctx->state.cos_vals, gpu_ctx->state.sin_vals,
      p->n_kv_heads, head_dim, batch, qkv_out_dim, head_dim/2, gpu_ctx->d_slot_ids);
    }

    // Write K (after RoPE) - Use paged cache manager
    {
      dim3 block(256), grid((k_size + block.x - 1) / block.x, batch);
      // Use the new paged KV writer that handles different layer types
      PagedKVWriterK::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.qkv, gpu_ctx->state.kv_cache_manager->get_device_data(), l,
        batch, qkv_out_dim, q_size, k_size, kv_dim,
        gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
    }

    // Attention computation (FlashAttention-like, always enabled) -> tb
    {
      dim3 block(64);
      dim3 grid(1, p->n_attn_heads, batch);
      // Use shared memory for attention scores: use sliding window size for paged attention
      int max_attention_length = (p->sliding_window > 0 && (l % 2 == 0)) ? p->sliding_window : p->seq_len;
      // Allocate +1 element to store optional attention sink (baseline behavior) even for sliding-window layers.
      size_t shared_mem_bytes = (size_t)(max_attention_length + 1) * sizeof(float);
      
      // Use the new paged attention kernel that handles different layer types
      // Sliding window is handled by cache compaction (no explicit mask)
      paged_flash_attention_kernel_batched<<<grid, block, shared_mem_bytes, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.qkv, gpu_ctx->state.kv_cache_manager->get_device_data(), gpu_ctx->weights.attn_sinks,
        gpu_ctx->state.tb, head_dim, kv_dim, kv_mul,
        p->n_attn_heads, p->seq_len, l, gpu_ctx->d_pos_list, batch,
        qkv_out_dim, gpu_ctx->d_slot_ids);
    }

    // Output projection W_o (batched) via packed BF16 GEMM: tb2 = tb @ W_o + b_o
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    {
      int Mwo = hidden_dim;
      int Kow = head_dim * p->n_attn_heads;
      int Bpad2 = ((batch + 15) / 16) * 16;
      int Mpad2 = ((Mwo + 15) / 16) * 16;
      int Kpad2 = ((Kow + 15) / 16) * 16;
      // Gather/pack tb -> xpack_bf16
      size_t total = (size_t)Kow * (size_t)Bpad2;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_bf16_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.tb, gpu_ctx->state.xpack_bf16, Kow, batch, Bpad2, Kow, gpu_ctx->d_slot_ids);
      const hip_bfloat16 *w_o_packed = gpu_ctx->weights.w_o_packed16 + (size_t)l * (size_t)Mpad2 * (size_t)Kpad2;
      dim3 blockG(16, 4, 1), gridG((unsigned int)((Mpad2 + 15) / 16), (unsigned int)((Bpad2 + 15) / 16));

      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
        w_o_packed, gpu_ctx->state.xpack_bf16, gpu_ctx->state.tb2, b_o,
        Mwo, Kow, batch, Bpad2, Mwo, gpu_ctx->d_slot_ids, Mpad2, Kpad2);
    }

    // Residual add and FFN RMSNorm fused: t = RMSNorm(x += tb2)
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    {
      dim3 block_norm(256), grid_norm(1, batch);
      size_t shared_mem_norm = block_norm.x * sizeof(float);
      rmsnorm_kernel_float_batched<<<grid_norm, block_norm, shared_mem_norm, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.x, rms_ffn_w, /*residual=*/gpu_ctx->state.tb2,
        hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }

    // -------- MoE batched & grouped across slots and experts --------
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)p->n_experts;
    int total_post_pad = 0;
    // 2) Router matmul + bias (batched across slots) via packed BF16 GEMM
    {
      int Mr = p->n_experts;
      int Kr = hidden_dim;
      int Bpadr = ((batch + 15) / 16) * 16;
      int Mpadr = ((Mr + 15) / 16) * 16;
      int Kpadr = ((Kr + 15) / 16) * 16;
      // Gather/pack t -> xpack_bf16
      size_t total = (size_t)Kr * (size_t)Bpadr;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_bf16_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.xpack_bf16, Kr, batch, Bpadr, hidden_dim, gpu_ctx->d_slot_ids);
      const hip_bfloat16 *w_router_packed = gpu_ctx->weights.w_router_packed16 + (size_t)l * (size_t)Mpadr * (size_t)Kpadr;
      dim3 blockG(16, 4, 1), gridG((unsigned)(Mpadr + 15) / 16, (unsigned)(Bpadr + 15) / 16);
      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
        w_router_packed, gpu_ctx->state.xpack_bf16, gpu_ctx->state.router_score, b_router,
        Mr, Kr, batch, Bpadr, Mr, gpu_ctx->d_slot_ids, Mpadr, Kpadr);
    }
    // 3) Fused Router: Top-K + softmax + gather + histogram
    {
      // Ensure eid_counts is zeroed before atomic adds in fused kernel
      HIP_CHECK(hipMemsetAsync(gpu_ctx->state.eid_counts, 0, (size_t)p->n_experts * sizeof(int), gpu_ctx->batch_stream));
      dim3 grid(batch), block(1);
      size_t shared_bytes = (size_t)p->experts_per_token * (sizeof(float) + sizeof(int));
      router_fused_topk_gather_hist_kernel<<<grid, block, shared_bytes, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.router_score,
        p->n_experts,
        p->experts_per_token,
        gpu_ctx->d_slot_ids,
        batch,
        p->n_experts,
        gpu_ctx->state.sel_w,
        gpu_ctx->state.sel_eids,
        gpu_ctx->state.sel_sids,
        gpu_ctx->state.eid_counts);
    }
    // 4) Zero e_agg for selected slots
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      zero_selected_slots_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.e_agg, hidden_dim, gpu_ctx->d_slot_ids, batch);
    }
    // 5) Flatten selections are already populated by fused kernel
    int M = batch * p->experts_per_token;
    // 6) Group selections by expert (eid) into grouped arrays
    {
      // Zero cursors only (counts were built by fused kernel)
      HIP_CHECK(hipMemsetAsync(gpu_ctx->state.eid_cursors, 0, (size_t)p->n_experts * sizeof(int), gpu_ctx->batch_stream));
      // 6') Xây prefix-sum đã đệm theo BLOCK_B (global)
      build_padded_offsets_kernel<<<1,1,0,gpu_ctx->batch_stream>>>(
          gpu_ctx->state.eid_counts,
          gpu_ctx->state.padded_offsets,
          p->n_experts, BLOCK_B);

      // Bỏ đồng bộ D2H cho total_post_pad: không cần đọc về host nữa.
      // Thay vào đó, mọi launch phía sau sẽ dùng upper bound theo Mmax_pad.

      // Reset cursors để scatter
      HIP_CHECK(hipMemsetAsync(gpu_ctx->state.eid_cursors, 0,
                              (size_t)p->n_experts * sizeof(int),
                              gpu_ctx->batch_stream));

      // Không cần memset toàn bộ Mmax_pad; các kernel sau sẽ tự guard bằng Btotal live đọc từ device.

      // 6'') Scatter các selection thật theo expert vào vùng đã sort
      {
        int threads = 256;
        int blocks  = (M + threads - 1) / threads;
        scatter_sorted_real_kernel<<<blocks, threads, 0, gpu_ctx->batch_stream>>>(
            gpu_ctx->state.sel_sids, gpu_ctx->state.sel_eids, gpu_ctx->state.sel_w,
            M,
            gpu_ctx->state.padded_offsets,
            gpu_ctx->state.eid_cursors,
            gpu_ctx->state.sorted_indices,
            gpu_ctx->state.sorted_sids,
            gpu_ctx->state.sorted_eids,
            gpu_ctx->state.sorted_w);
      }

      // 6''') Điền padding + block→expert id
      {
        int threads = 128;
        int blocks  = (p->n_experts + threads - 1) / threads;
        fill_padding_and_blocks_kernel<<<blocks, threads, 0, gpu_ctx->batch_stream>>>(
            gpu_ctx->state.eid_counts,
            gpu_ctx->state.padded_offsets,
            p->n_experts, BLOCK_B,
            gpu_ctx->state.sorted_indices,
            gpu_ctx->state.sorted_sids,
            gpu_ctx->state.sorted_eids,
            gpu_ctx->state.sorted_w,
            gpu_ctx->state.block_expert_ids);
      }

      // 6'''') Chuẩn bị iota cột toàn cục: fill theo Mmax_pad (kích thước thật của d_col_ids)
      {
        const size_t Mmax      = (size_t)gpu_ctx->max_batch * (size_t)p->experts_per_token;
        const size_t Mmax_pad  = Mmax + (size_t)p->n_experts * (size_t)(BLOCK_B - 1);
        int threads = 256;
        int blocks  = ((int)Mmax_pad + threads - 1) / threads;
        iota_kernel<<<blocks, threads, 0, gpu_ctx->batch_stream>>>(
            gpu_ctx->state.d_col_ids, (int)Mmax_pad);
      }

    }
    {
        // 7') Pack X cho MLP1 theo thứ tự đã sort+pad
    int Kmlp1 = p->hidden_dim;
    // Dùng upper bound cố định theo worst-case đã alloc
    const size_t Mmax      = (size_t)gpu_ctx->max_batch * (size_t)p->experts_per_token;
    const size_t Mmax_pad  = Mmax + (size_t)p->n_experts * (size_t)(BLOCK_B - 1);
    int Bpad_total = round_up16((int)Mmax_pad); // Bpad dùng khi pack (toàn cục)
    int Bmlp1 = (int)Mmax_pad;                  // số cột thực dùng cho guard và gather; phần đuôi pad điền 0
    {
      size_t total = (size_t)Kmlp1 * (size_t)Bpad_total;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_bf16_kernel_Bptr<<<grid, block, 0, gpu_ctx->batch_stream>>>(
          gpu_ctx->state.t,            // nguồn: x đã RMSNorm-FFN per-slot
          gpu_ctx->state.xpack_bf16,   // đích:  [K x Bpad_total] BF16
          Kmlp1,
          Bpad_total,
          p->hidden_dim,
          gpu_ctx->state.sorted_sids,  // cột b → sid = sorted_sids[b]
          gpu_ctx->state.padded_offsets + p->n_experts // runtime Btotal
      );
    }
        // 7'') GEMM MLP1 per-expert (fused epilogue SwiGLU → ghi thẳng gate_up_all)
    {
      const int Mout       = 2 * p->intermediate_dim;
      const int I         = p->intermediate_dim;
      const int Mpad_mlp1 = round_up16(Mout);
      const int Kpad_mlp1 = round_up16(p->hidden_dim);

      // Launch a single grouped kernel across all experts/columns using block→expert mapping.
      {
        dim3 blockG(16, 4, 1);
        dim3 gridG((unsigned)((Mpad_mlp1 + 15) / 16), (unsigned)((Bpad_total + 15) / 16));
        gemm16x16_mlp1_fused_swiglu_scatter_grouped<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
            /*W_data*/  gpu_ctx->weights.w_mlp1_data,
            /*W_scale*/ gpu_ctx->weights.w_mlp1_scales,
            /*Xpack*/   gpu_ctx->state.xpack_bf16,
            /*dst*/     gpu_ctx->state.gate_up_all,
            /*bias_all*/gpu_ctx->weights.b_mlp1,
            /*sizes*/   Mout, p->hidden_dim, Bmlp1, Bpad_total,
            /*stride*/  Bpad_total,
            /*col_ids*/ gpu_ctx->state.d_col_ids,
            /*blk→eid*/ gpu_ctx->state.block_expert_ids,
            /*BLOCK_B*/ BLOCK_B,
            /*nexp,l*/  p->n_experts, l,
            /*pads*/    Mpad_mlp1, Kpad_mlp1,
            /*limit*/   p->swiglu_limit,
            /*Btotal* */gpu_ctx->state.padded_offsets + p->n_experts);
      }
    }
        
    }

    // 8) Grouped MLP2 via MFMA matvec (packed) with in-kernel accumulation
    {
      int Mpad_mlp2 = ((hidden_dim + 15) / 16) * 16;
      int Kpad_mlp2 = ((p->intermediate_dim + 15) / 16) * 16;
      dim3 blockG2(16, 4, 1);
      // Dùng upper bound cho lưới; guard nội bộ với Btotal = Mmax_pad
      const size_t Mmax      = (size_t)gpu_ctx->max_batch * (size_t)p->experts_per_token;
      const size_t Mmax_pad  = Mmax + (size_t)p->n_experts * (size_t)(BLOCK_B - 1);
      const int Bpad_total   = round_up16((int)Mmax_pad);
      int Btotal = (int)Mmax_pad;
      // Launch one block per 16 contiguous selections (grouped by expert)
      dim3 gridG2((unsigned int)((Mpad_mlp2 + 15) / 16), (unsigned int)((Bpad_total + 15) / 16));
      gemm16x16_mlp2_grouped_mfma_matvec_residual_kernel<<<gridG2, blockG2, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.x,
        gpu_ctx->state.gate_up_all,
        gpu_ctx->weights.w_mlp2_packed16,
        gpu_ctx->weights.b_mlp2,
        gpu_ctx->state.sorted_sids,
        gpu_ctx->state.sorted_eids,
        gpu_ctx->state.sorted_w,
        gpu_ctx->state.sorted_indices,
        Btotal, hidden_dim, p->intermediate_dim,
        Bpad_total,
        l, p->n_experts, hidden_dim,
        Mpad_mlp2, Kpad_mlp2,
        gpu_ctx->state.padded_offsets + p->n_experts);
    }

    // 9) Residual fused in step 8 (mlp2 kernel accumulates directly into x)
  }

  // Fully batched final RMSNorm and logits computation
  if (host_next_tokens_out != nullptr) {
    // Batched RMSNorm across all slots with proper shared memory
    dim3 block_norm(128);  // Reduced block size for better occupancy
    dim3 grid_norm(1, batch); // one block per slot; threads stride over hidden_dim
    size_t shared_mem_norm = block_norm.x * sizeof(float);
    rmsnorm_final_batched_kernel<<<grid_norm, block_norm, shared_mem_norm, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.x, gpu_ctx->weights.rms_out_w,
      hidden_dim, batch, gpu_ctx->d_slot_ids);

    // Batched logits via packed BF16 GEMM: logits = W_out @ X (batched)
    {
      int Mout = p->vocab_size;
      int Kout = hidden_dim;
      int Bpad3 = ((batch + 15) / 16) * 16;
      int Mpad3 = ((Mout + 15) / 16) * 16;
      int Kpad3 = ((Kout + 15) / 16) * 16;
      // Gather/pack x -> xpack_bf16 in batch order using slot_ids
      size_t total = (size_t)Kout * (size_t)Bpad3;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_bf16_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.x, gpu_ctx->state.xpack_bf16, Kout, batch, Bpad3, hidden_dim, gpu_ctx->d_slot_ids);
      const hip_bfloat16 *w_out_packed = gpu_ctx->weights.w_out_packed16;
      dim3 blockG(16, 4, 1), gridG((unsigned int)((Mpad3 + 15) / 16), (unsigned int)((Bpad3 + 15) / 16));
      // Use identity mapping for output rows (b -> row b)
      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
        w_out_packed, gpu_ctx->state.xpack_bf16, gpu_ctx->state.logits_fp32, /*bias*/nullptr,
        Mout, Kout, batch, Bpad3, Mout, gpu_ctx->d_batch_iota, Mpad3, Kpad3);
    }

    // Launch optimized argmax kernel: one block per batch element
    dim3 block_argmax(256); // must match BLOCK_SIZE used in kernel
    dim3 grid_argmax(batch);
    const size_t num_warps = (size_t)(BLOCK_SIZE / WARP_SIZE);
    size_t smem_argmax = num_warps * (sizeof(float) + sizeof(int));
    sample_argmax_kernel_optimized<<<grid_argmax, block_argmax, smem_argmax, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.logits_fp32, gpu_ctx->d_next_tokens, batch, p->vocab_size);
    HIP_CHECK(hipMemcpyAsync(host_next_tokens_out, gpu_ctx->d_next_tokens, batch * sizeof(int), hipMemcpyDeviceToHost, gpu_ctx->batch_stream));
    HIP_CHECK(hipStreamSynchronize(gpu_ctx->batch_stream));
  }
}

// ------------------------- Warm-up & Finish -------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  transformer->config.seq_len = 1024; // override max seq len to 1024
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "No HIP devices found.\n");
    exit(EXIT_FAILURE);
  }
  // FlashAttention is always enabled; no env toggles required
  
  // Limit number of devices used via env GETP_MAX_DEVICES (optional)
  int max_devices = device_count;
  if (const char* env_nd = getenv("GETP_MAX_DEVICES")) {
    int nd = atoi(env_nd);
    if (nd > 0 && nd < max_devices) max_devices = nd;
  }
  gpu_contexts.clear();
  gpu_contexts.resize(max_devices);

  for (int d = 0; d < max_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    gpu_contexts[d].device_id = d;
    // Configure per-GPU max batch from env, default 512
    const char *env_mb = getenv("GETP_MAX_BATCH");
    int mb = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 512;
    gpu_contexts[d].max_batch = mb;
    log_gpu_mem(d, "[GPU] Before memory allocation on device:");
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    log_gpu_mem(d, "[GPU] After weight allocation on device:");
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config, gpu_contexts[d].max_batch);
    log_gpu_mem(d, "[GPU] After run state allocation on device:");
    // Create batch stream and staging device buffers
    HIP_CHECK(hipStreamCreate(&gpu_contexts[d].batch_stream));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_slot_ids,  gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_pos_list,  gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_token_ids, gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_next_tokens, gpu_contexts[d].max_batch * sizeof(int)));
    // Identity mapping buffer for batch-ordered outputs (0..B-1)
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_batch_iota, gpu_contexts[d].max_batch * sizeof(int)));
    {
      dim3 threads(256);
      dim3 blocks((unsigned int)((gpu_contexts[d].max_batch + threads.x - 1) / threads.x));
      iota_kernel<<<blocks, threads>>>(gpu_contexts[d].d_batch_iota, gpu_contexts[d].max_batch);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
    }
    // Sliding window mask is not used in paged attention path; no mask allocation/copy
  }
  
  printf("[GPU] Warm-up done on %d device(s). Packed BF16 tiles for QKV/W_O/Router/MLP1/MLP2/OUT loaded.\n", max_devices);
  printf("[Attention] Using FlashAttention-like optimized kernel (always on)\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    if (gpu_ctx.batch_stream) HIP_CHECK(hipStreamDestroy(gpu_ctx.batch_stream));
    if (gpu_ctx.d_slot_ids)    HIP_CHECK(hipFree(gpu_ctx.d_slot_ids));
    if (gpu_ctx.d_pos_list)    HIP_CHECK(hipFree(gpu_ctx.d_pos_list));
    if (gpu_ctx.d_token_ids)   HIP_CHECK(hipFree(gpu_ctx.d_token_ids));
    if (gpu_ctx.d_next_tokens) HIP_CHECK(hipFree(gpu_ctx.d_next_tokens));
    if (gpu_ctx.d_batch_iota)  HIP_CHECK(hipFree(gpu_ctx.d_batch_iota));
    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx.state);
  }
  gpu_contexts.clear();
  printf("[GPU] Cleanup done.\n");
}

// ------------------------- Driver (generate & inference) -------------------------



long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  const int num_devices = (int)gpu_contexts.size();
  if (requests->num_reqs <= 0 || num_devices <= 0) return 0;
  // Global request index for all workers (lock-free producer)
  std::atomic<int> next_idx{0};

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);
  
  // Continuous batching per GPU worker
  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[t].device_id));
      GPUContext *ctx = &gpu_contexts[t];
      t_ctx = ctx;

      std::vector<Slot> slots(ctx->max_batch);
      long long local_out = 0;

      auto acquire_slot = [&](int sidx) {
        int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= requests->num_reqs) return false;
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




void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch) {
  int kv_dim = p->head_dim * p->n_kv_heads;
  size_t B = (size_t)max_batch;

  HIP_CHECK(hipMalloc(&s->x,   B * (size_t)p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->t,   B * (size_t)p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb,  B * (size_t)p->head_dim * (size_t)p->n_attn_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2, B * (size_t)p->hidden_dim * sizeof(float)));

  HIP_CHECK(hipMalloc(&s->router_score,      B * (size_t)p->n_experts * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_v,            B * (size_t)p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_i,            B * (size_t)p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->e_agg,    B * (size_t)p->hidden_dim * sizeof(float)));

  // (commented out unused separate q/k/v buffers; fused qkv used only)
  HIP_CHECK(hipMalloc(&s->qkv, B * (size_t)p->head_dim * (size_t)(p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float)));

  // Initialize new KV Cache Manager with paged attention support
  s->kv_cache_manager = new KVCacheManager(max_batch, p->n_layers, p->seq_len, kv_dim, p->sliding_window);
  
  // Removed legacy 'logits' buffer

  // Mask not needed for paged attention path; keep null to avoid large alloc
  s->mask = NULL;

  HIP_CHECK(hipMalloc(&s->cos_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sin_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));

  // Note: acc_buffer no longer needed since we use float state directly
  s->acc_buffer = nullptr;
  HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));

  // Grouped MoE staging buffers
  size_t Mmax = (size_t)max_batch * (size_t)p->experts_per_token;
  HIP_CHECK(hipMalloc(&s->sel_sids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_eids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_w,        Mmax * sizeof(float)));

  // Expert-grouped arrays and counters
  HIP_CHECK(hipMalloc(&s->eid_counts,   (size_t)p->n_experts * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->eid_cursors,  (size_t)p->n_experts * sizeof(int)));

  
  // size_t Mmax      = (size_t)max_batch * (size_t)p->experts_per_token;               // B tối đa của selections
  size_t Mmax_pad  = Mmax + (size_t)p->n_experts * (size_t)(BLOCK_B - 1);            // trừ hao pad tối đa
  int Bpad_mlp1    = (int)round_up16((int)Mmax_pad);                                  // cho pack MLP1

    size_t Kmax_xpack = std::max({
      (size_t)p->hidden_dim,
      (size_t)p->head_dim * (size_t)p->n_attn_heads,
      (size_t)p->intermediate_dim    // NEW
  });
  
  HIP_CHECK(hipMalloc(&s->xpack_bf16, Kmax_xpack * (size_t)Bpad_mlp1 * sizeof(hip_bfloat16)));
  // HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  // HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)p->intermediate_dim * sizeof(hip_bfloat16)));

  // mlp1_out_all is unused in fused path; skip allocation to save memory
  s->mlp1_out_all = nullptr;
  HIP_CHECK(hipMalloc(&s->gate_up_all,  (size_t)Bpad_mlp1 * (size_t)p->intermediate_dim * sizeof(hip_bfloat16)));

  // --- New: block-aligned staging buffers ---
  HIP_CHECK(hipMalloc(&s->sorted_indices,   Mmax_pad * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sorted_sids,      Mmax_pad * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sorted_eids,      Mmax_pad * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sorted_w,         Mmax_pad * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->padded_offsets,  ((size_t)p->n_experts + 1) * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->block_expert_ids, ((Mmax_pad + BLOCK_B - 1) / BLOCK_B) * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->d_col_ids,        Mmax_pad * sizeof(int)));

  size_t Mout = 2 * (size_t)p->intermediate_dim;
}


void gpu_free_run_state(GPURunState *s) {
  HIP_CHECK(hipFree(s->x));
  HIP_CHECK(hipFree(s->t));
  HIP_CHECK(hipFree(s->tb));
  HIP_CHECK(hipFree(s->tb2));
  HIP_CHECK(hipFree(s->router_score));
  HIP_CHECK(hipFree(s->topk_v));
  HIP_CHECK(hipFree(s->topk_i));
  HIP_CHECK(hipFree(s->e_agg));
  HIP_CHECK(hipFree(s->qkv));
  
  // Clean up KV Cache Manager (also frees key_cache and value_cache)
  if (s->kv_cache_manager) {
    delete s->kv_cache_manager;
    s->kv_cache_manager = nullptr;
    s->key_cache = nullptr;    // These were pointing to the cache manager's memory
    s->value_cache = nullptr;  // These were pointing to the cache manager's memory
  } else {
    // Fallback for legacy code path
    if (s->key_cache) HIP_CHECK(hipFree(s->key_cache));
    if (s->value_cache) HIP_CHECK(hipFree(s->value_cache));
  }
  
  // No legacy logits or mask buffers to free
  if (s->mask) HIP_CHECK(hipFree(s->mask));
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  if (s->acc_buffer) HIP_CHECK(hipFree(s->acc_buffer));  // acc_buffer may be nullptr
  HIP_CHECK(hipFree(s->logits_fp32));
  HIP_CHECK(hipFree(s->sel_sids));
  HIP_CHECK(hipFree(s->sel_eids));
  HIP_CHECK(hipFree(s->sel_w));
  if (s->mlp1_out_all) HIP_CHECK(hipFree(s->mlp1_out_all));
  HIP_CHECK(hipFree(s->gate_up_all));
  HIP_CHECK(hipFree(s->eid_counts));
  HIP_CHECK(hipFree(s->eid_cursors));
  HIP_CHECK(hipFree(s->xpack_bf16));
  HIP_CHECK(hipFree(s->sorted_indices));
  HIP_CHECK(hipFree(s->sorted_sids));
  HIP_CHECK(hipFree(s->sorted_eids));
  HIP_CHECK(hipFree(s->sorted_w));
  HIP_CHECK(hipFree(s->padded_offsets));
  HIP_CHECK(hipFree(s->block_expert_ids));
  HIP_CHECK(hipFree(s->d_col_ids));
}

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int dev = 0; HIP_CHECK(hipGetDevice(&dev));
  int head_dim      = p->head_dim;
  int n_layers      = p->n_layers;
  int n_experts     = p->n_experts;
  int hidden_dim    = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size    = p->vocab_size;
  int n_attn_heads  = p->n_attn_heads;
  int n_kv_heads    = p->n_kv_heads;

  // Ensure unused un-packed pointers are null (we use packed versions for these)
  gpu_w->w_qkv = nullptr;
  gpu_w->w_o = nullptr;
  gpu_w->w_router = nullptr;
  gpu_w->w_mlp1_data = nullptr;
  gpu_w->w_mlp1_scales = nullptr;

  // Shared staging resources for weight loads (reused across helpers)
  constexpr size_t STAGING_BYTES_TARGET = (size_t)512 * 1024 * 1024; // 512 MB
  size_t staging_elems = std::max((size_t)1, STAGING_BYTES_TARGET / sizeof(float));
  size_t largest_matrix_k = std::max({(size_t)hidden_dim,
                                      (size_t)intermediate_dim,
                                      (size_t)head_dim * (size_t)n_attn_heads,
                                      (size_t)head_dim * (size_t)std::max(1, n_kv_heads),
                                      (size_t)n_experts});
  if (staging_elems < largest_matrix_k) staging_elems = largest_matrix_k;
  const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
  if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;

  hipStream_t load_stream;
  hipError_t stream_status = hipStreamCreateWithFlags(&load_stream, hipStreamNonBlocking);
  if (stream_status != hipSuccess) {
    hipGetLastError();
    HIP_CHECK(hipStreamCreate(&load_stream));
  }

  float *d_stage_fp32 = nullptr;
  HIP_CHECK(hipMalloc(&d_stage_fp32, staging_elems * sizeof(float)));

  auto maybe_pin_host = [&](const void *ptr, size_t elem_count, size_t bytes_per_elem) {
    size_t bytes = elem_count * bytes_per_elem;
    if (bytes < ((size_t)4 << 20)) return hipErrorInvalidValue; // signal no pin attempt for small buffers
    return hipHostRegister(const_cast<void*>(ptr), bytes, 0);
  };

  auto cpu_convert_fp32_to_bf16 = [](const float *src, hip_bfloat16 *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = hip_bfloat16(src[i]);
    }
  };

  // Helper: host FP32 -> device staging FP32 (tile) -> device BF16 dest
  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    const size_t bf16_size = count * sizeof(hip_bfloat16);
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));

    // For small buffers, convert on CPU to avoid GPU kernel/stream overheads
    const size_t CPU_CONVERT_THRESHOLD = 4096; // elements
    if (count <= CPU_CONVERT_THRESHOLD) {
      std::vector<hip_bfloat16> tmp(count);
      cpu_convert_fp32_to_bf16(cpu_data, tmp.data(), count);
      HIP_CHECK(hipMemcpy(*gpu_data, tmp.data(), bf16_size, hipMemcpyHostToDevice));
      return;
    }

    hipError_t pin_ret = maybe_pin_host(cpu_data, count, sizeof(float));
    if (pin_ret != hipSuccess && pin_ret != hipErrorInvalidValue) {
      hipGetLastError(); // clear sticky error state if registration failed
    }

    size_t done = 0;
    const dim3 block(256);
    while (done < count) {
      size_t tile = std::min(staging_elems, count - done);
      if (tile > INT_MAX_ELEMS) tile = INT_MAX_ELEMS;
      size_t tile_bytes_fp32 = tile * sizeof(float);

      HIP_CHECK(hipMemcpyAsync(d_stage_fp32, cpu_data + done, tile_bytes_fp32,
                               hipMemcpyHostToDevice, load_stream));

      int n = static_cast<int>(tile);
      dim3 grid((n + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block, 0, load_stream>>>(d_stage_fp32, *gpu_data + done, n);
      HIP_CHECK(hipGetLastError());

      done += tile;
    }

    HIP_CHECK(hipStreamSynchronize(load_stream));
    if (pin_ret == hipSuccess) {
      HIP_CHECK(hipHostUnregister(const_cast<float*>(cpu_data)));
    }
  };

  // Helper: pack FP32 row-major weights to BF16 16x16 tiles directly on device with streaming
  auto round_up16 = [](int x) { return ((x + 15) / 16) * 16; };
  auto pack_weights_fp32_rowmajor_to_bf16_tiles16 = [&](const float *cpu_data,
                                                        hip_bfloat16 **gpu_packed,
                                                        int layers, int M, int K) {
    int Mpad = round_up16(M);
    int Kpad = round_up16(K);
    size_t per_layer_elems = (size_t)Mpad * (size_t)Kpad;
    HIP_CHECK(hipMalloc(gpu_packed, (size_t)layers * per_layer_elems * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemset(*gpu_packed, 0, (size_t)layers * per_layer_elems * sizeof(hip_bfloat16)));

    size_t total_f32 = (size_t)layers * (size_t)M * (size_t)K;
    hipError_t pin_ret = maybe_pin_host(cpu_data, total_f32, sizeof(float));
    if (pin_ret != hipSuccess && pin_ret != hipErrorInvalidValue) {
      hipGetLastError();
    }

    size_t max_rows = staging_elems / (size_t)K;
    if (max_rows == 0) {
      max_rows = 1;
    }

    const dim3 block(256);
    for (int l = 0; l < layers; ++l) {
      const float *layer_src = cpu_data + (size_t)l * (size_t)M * (size_t)K;
      hip_bfloat16 *layer_dst = *gpu_packed + (size_t)l * (size_t)per_layer_elems;
      for (int row_off = 0; row_off < M; row_off += (int)max_rows) {
        int rows = std::min((int)max_rows, M - row_off);
        size_t total = (size_t)rows * (size_t)K;
        size_t tile_bytes = total * sizeof(float);
        HIP_CHECK(hipMemcpyAsync(d_stage_fp32, layer_src + (size_t)row_off * (size_t)K,
                                 tile_bytes, hipMemcpyHostToDevice, load_stream));
        dim3 grid((unsigned int)((total + block.x - 1) / block.x));
        pack_convert_fp32_to_bf16_tiles16_kernel<<<grid, block, 0, load_stream>>>(
            d_stage_fp32, row_off, rows, M, K, layer_dst, Mpad, Kpad);
        HIP_CHECK(hipGetLastError());
      }
    }
    HIP_CHECK(hipStreamSynchronize(load_stream));

    if (pin_ret == hipSuccess) {
      HIP_CHECK(hipHostUnregister(const_cast<float*>(cpu_data)));
    }
  };


  // Counts as size_t to avoid overflow
  size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
  printf("[GPU#%d] Copy embed/out...\n", dev);
  copy_and_convert(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, token_embed_count);
  copy_and_convert(cpu_w->out,                    &gpu_w->out,                    token_embed_count);
  // Pack final classifier (out) into 16x16 BF16 tiles for MFMA GEMM path
  printf("[GPU#%d] Pack OUT (vocab x hidden) into 16x16 BF16 tiles...\n", dev);
  {
    int M_out = vocab_size;
    int K_out = hidden_dim;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(cpu_w->out, &gpu_w->w_out_packed16, 1, M_out, K_out);
  }

  size_t rms_count = (size_t)n_layers * (size_t)hidden_dim;
  printf("[GPU#%d] Copy RMS weights...\n", dev);
  copy_and_convert(cpu_w->rms_attn_w, &gpu_w->rms_attn_w, rms_count);
  copy_and_convert(cpu_w->rms_ffn_w,  &gpu_w->rms_ffn_w,  rms_count);

  size_t rms_out_count = (size_t)hidden_dim;
  copy_and_convert(cpu_w->rms_out_w,  &gpu_w->rms_out_w,  rms_out_count);

  // Pack selected projection weights (QKV, W_O, Router) into 16x16 BF16 tiles
  printf("[GPU#%d] Pack QKV...\n", dev);
  int M_qkv = head_dim * n_attn_heads + 2 * head_dim * n_kv_heads;
  int K_qkv = hidden_dim;
  pack_weights_fp32_rowmajor_to_bf16_tiles16(cpu_w->w_qkv, &gpu_w->w_qkv_packed16, n_layers, M_qkv, K_qkv);

  size_t qkv_b_count = (size_t)n_layers *
                       ((size_t)head_dim * (size_t)n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
  copy_and_convert(cpu_w->b_qkv, &gpu_w->b_qkv, qkv_b_count);

  printf("[GPU#%d] Pack W_O...\n", dev);
  int M_o = hidden_dim;
  int K_o = head_dim * n_attn_heads;
  pack_weights_fp32_rowmajor_to_bf16_tiles16(cpu_w->w_o, &gpu_w->w_o_packed16, n_layers, M_o, K_o);

  size_t o_b_count = (size_t)n_layers * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_o, &gpu_w->b_o, o_b_count);

  printf("[GPU#%d] Copy attn sinks...\n", dev);
  size_t attn_sinks_count = (size_t)n_layers * (size_t)n_attn_heads;
  copy_and_convert(cpu_w->attn_sinks, &gpu_w->attn_sinks, attn_sinks_count);

  printf("[GPU#%d] Pack Router...\n", dev);
  int M_router = n_experts;
  int K_router = hidden_dim;
  pack_weights_fp32_rowmajor_to_bf16_tiles16(cpu_w->w_router, &gpu_w->w_router_packed16, n_layers, M_router, K_router);

  size_t router_b_count = (size_t)n_layers * (size_t)n_experts;
  copy_and_convert(cpu_w->b_router, &gpu_w->b_router, router_b_count);

  printf("[GPU#%d] Quantize MLP1 (layers=%d, experts=%d) to MXFP4...\n", dev, n_layers, n_experts);
  ensure_mxfp4_dequant_lut_initialized();
  int M_mlp1 = 2 * intermediate_dim;
  int K_mlp1 = hidden_dim;
  size_t mats_mlp1 = (size_t)n_layers * (size_t)n_experts;
  size_t elems_per_mat_mlp1 = (size_t)M_mlp1 * (size_t)K_mlp1;
  size_t total_elems_mlp1 = mats_mlp1 * elems_per_mat_mlp1;
  size_t row_bytes_per_mat = (elems_per_mat_mlp1 + 1) / 2;
  size_t scale_blocks_per_mat = (elems_per_mat_mlp1 + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;

  uint8_t *mlp1_data_row = nullptr;
  uint8_t *mlp1_scales_row = nullptr;
  size_t total_row_bytes = mats_mlp1 * row_bytes_per_mat;
  size_t total_scale_blocks = mats_mlp1 * scale_blocks_per_mat;
  HIP_CHECK(hipMalloc(&mlp1_data_row, total_row_bytes));
  HIP_CHECK(hipMalloc(&mlp1_scales_row, total_scale_blocks * sizeof(uint8_t)));
  HIP_CHECK(hipMemset(mlp1_data_row, 0, total_row_bytes));
  HIP_CHECK(hipMemset(mlp1_scales_row, 0, total_scale_blocks * sizeof(uint8_t)));

  size_t total_f32_mlp1 = total_elems_mlp1;
  hipError_t pin_ret_mlp1 = maybe_pin_host(cpu_w->w_mlp1, total_f32_mlp1, sizeof(float));
  if (pin_ret_mlp1 != hipSuccess && pin_ret_mlp1 != hipErrorInvalidValue) {
    hipGetLastError();
  }
  size_t done_mlp1 = 0;
  while (done_mlp1 < total_elems_mlp1) {
    size_t tile = std::min(staging_elems, total_elems_mlp1 - done_mlp1);
    size_t tile_bytes = tile * sizeof(float);
    HIP_CHECK(hipMemcpyAsync(d_stage_fp32, cpu_w->w_mlp1 + done_mlp1,
                             tile_bytes, hipMemcpyHostToDevice, load_stream));
    size_t blocks = (tile + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    quantize_mxfp4_kernel<<<blocks, 1, 0, load_stream>>>(d_stage_fp32, tile, done_mlp1,
                                                         mlp1_data_row, mlp1_scales_row);
    HIP_CHECK(hipGetLastError());
    done_mlp1 += tile;
  }
  HIP_CHECK(hipStreamSynchronize(load_stream));
  if (pin_ret_mlp1 == hipSuccess) {
    HIP_CHECK(hipHostUnregister(const_cast<float*>(cpu_w->w_mlp1)));
  }

  int Mpad_mlp1 = round_up16(M_mlp1);
  int Kpad_mlp1 = round_up16(K_mlp1);
  size_t packed_bytes_per_mat = ((size_t)Mpad_mlp1 * (size_t)Kpad_mlp1 + 1) / 2;
  size_t tiles_m = (M_mlp1 + 15) / 16;
  size_t tiles_k = (K_mlp1 + 15) / 16;
  size_t scales_entries_per_mat = tiles_m * tiles_k * 16;
  size_t total_packed_bytes = mats_mlp1 * packed_bytes_per_mat;
  size_t total_scale_entries = mats_mlp1 * scales_entries_per_mat;

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp1_data), total_packed_bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpu_w->w_mlp1_scales), total_scale_entries * sizeof(uint8_t)));
  HIP_CHECK(hipMemset(gpu_w->w_mlp1_data, 0, total_packed_bytes));
  HIP_CHECK(hipMemset(gpu_w->w_mlp1_scales, 0x7F, total_scale_entries * sizeof(uint8_t)));

  {
    const dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < n_experts; ++e) {
        size_t mat_idx = (size_t)l * (size_t)n_experts + (size_t)e;
        const uint8_t *src_data = mlp1_data_row + mat_idx * row_bytes_per_mat;
        uint8_t *dst_data = gpu_w->w_mlp1_data + mat_idx * packed_bytes_per_mat;
        size_t bytes = row_bytes_per_mat;
        dim3 grid((unsigned int)((bytes + block.x - 1) / block.x));
        pack_mxfp4_row_to_tile16_kernel<<<grid, block, 0, load_stream>>>(src_data, dst_data,
                                                                         (size_t)M_mlp1, (size_t)K_mlp1);
        HIP_CHECK(hipGetLastError());

        const uint8_t *src_scale = mlp1_scales_row + mat_idx * scale_blocks_per_mat;
        uint8_t *dst_scale = gpu_w->w_mlp1_scales + mat_idx * scales_entries_per_mat;
        size_t scale_entries = scales_entries_per_mat;
        dim3 grid_scale((unsigned int)((scale_entries + block.x - 1) / block.x));
        pack_mxfp4_scales_row_to_tile16_kernel<<<grid_scale, block, 0, load_stream>>>(
            src_scale, dst_scale, (size_t)M_mlp1, (size_t)K_mlp1);
        HIP_CHECK(hipGetLastError());
      }
    }
    HIP_CHECK(hipStreamSynchronize(load_stream));
  }

  HIP_CHECK(hipFree(mlp1_data_row));
  HIP_CHECK(hipFree(mlp1_scales_row));

  size_t mlp1_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->b_mlp1, &gpu_w->b_mlp1, mlp1_b_count);

  // size_t mlp2_w_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim * (size_t)intermediate_dim;
  // copy_and_convert(cpu_w->w_mlp2, &gpu_w->w_mlp2, mlp2_w_count);
  printf("[GPU#%d] Pack MLP2 (layers=%d, experts=%d)...\n", dev, n_layers, n_experts);
  {
    int M_mlp2 = hidden_dim;          // H
    int K_mlp2 = intermediate_dim;    // I

    // giống nhánh pack MLP1 ngay phía trên, chỉ đổi kích thước M/K và nguồn
    size_t total_f32 = (size_t)n_layers * (size_t)n_experts * (size_t)M_mlp2 * (size_t)K_mlp2;
    hipError_t pin_ret_mlp2 = maybe_pin_host(cpu_w->w_mlp2,
        total_f32, sizeof(float));
    if (pin_ret_mlp2 != hipSuccess && pin_ret_mlp2 != hipErrorInvalidValue) {
      hipGetLastError();
    }

    int Mpad = round_up16(M_mlp2);
    int Kpad = round_up16(K_mlp2);
    size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
    HIP_CHECK(hipMalloc(&gpu_w->w_mlp2_packed16,
        (size_t)n_layers * (size_t)n_experts * per_mat_elems * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemset(gpu_w->w_mlp2_packed16, 0,
        (size_t)n_layers * (size_t)n_experts * per_mat_elems * sizeof(hip_bfloat16)));

    size_t max_rows = staging_elems / (size_t)K_mlp2;
    if (max_rows == 0) max_rows = 1;
    const dim3 block(256);

    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < n_experts; ++e) {
        int idx = l * n_experts + e;
        const float *src = cpu_w->w_mlp2
            + (size_t)l * (size_t)n_experts * (size_t)M_mlp2 * (size_t)K_mlp2
            + (size_t)e * (size_t)M_mlp2 * (size_t)K_mlp2;
        hip_bfloat16 *dst = gpu_w->w_mlp2_packed16 + (size_t)idx * per_mat_elems;

        for (int row_off = 0; row_off < M_mlp2; row_off += (int)max_rows) {
          int rows = std::min((int)max_rows, M_mlp2 - row_off);
          size_t total = (size_t)rows * (size_t)K_mlp2;
          size_t tile_bytes = total * sizeof(float);
          HIP_CHECK(hipMemcpyAsync(d_stage_fp32, src + (size_t)row_off * (size_t)K_mlp2,
                                   tile_bytes, hipMemcpyHostToDevice, load_stream));
          dim3 grid((unsigned)((total + block.x - 1) / block.x));
          pack_convert_fp32_to_bf16_tiles16_kernel<<<grid, block, 0, load_stream>>>(
              d_stage_fp32, row_off, rows, M_mlp2, K_mlp2, dst, Mpad, Kpad);
          HIP_CHECK(hipGetLastError());
        }
      }
    }

    HIP_CHECK(hipStreamSynchronize(load_stream));
    if (pin_ret_mlp2 == hipSuccess) {
      HIP_CHECK(hipHostUnregister(const_cast<float*>(cpu_w->w_mlp2)));
    }

    // (tùy chọn) bỏ con trỏ w_mlp2 cũ, vì không còn dùng đường dot-product:
    gpu_w->w_mlp2 = nullptr;
  }

  size_t mlp2_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_mlp2, &gpu_w->b_mlp2, mlp2_b_count);

  HIP_CHECK(hipStreamSynchronize(load_stream));
  HIP_CHECK(hipFree(d_stage_fp32));
  HIP_CHECK(hipStreamDestroy(load_stream));
}

void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  if (gpu_w->token_embedding_table) HIP_CHECK(hipFree(gpu_w->token_embedding_table));
  if (gpu_w->out)                   HIP_CHECK(hipFree(gpu_w->out));
  if (gpu_w->w_out_packed16)        HIP_CHECK(hipFree(gpu_w->w_out_packed16));
  if (gpu_w->rms_attn_w)            HIP_CHECK(hipFree(gpu_w->rms_attn_w));
  if (gpu_w->rms_ffn_w)             HIP_CHECK(hipFree(gpu_w->rms_ffn_w));
  if (gpu_w->rms_out_w)             HIP_CHECK(hipFree(gpu_w->rms_out_w));
  if (gpu_w->w_qkv)                 HIP_CHECK(hipFree(gpu_w->w_qkv));
  if (gpu_w->b_qkv)                 HIP_CHECK(hipFree(gpu_w->b_qkv));
  if (gpu_w->w_o)                   HIP_CHECK(hipFree(gpu_w->w_o));
  if (gpu_w->b_o)                   HIP_CHECK(hipFree(gpu_w->b_o));
  if (gpu_w->attn_sinks)            HIP_CHECK(hipFree(gpu_w->attn_sinks));
  if (gpu_w->w_router)              HIP_CHECK(hipFree(gpu_w->w_router));
  if (gpu_w->b_router)              HIP_CHECK(hipFree(gpu_w->b_router));
  if (gpu_w->w_mlp1_data)           HIP_CHECK(hipFree(gpu_w->w_mlp1_data));
  if (gpu_w->w_mlp1_scales)         HIP_CHECK(hipFree(gpu_w->w_mlp1_scales));
  if (gpu_w->b_mlp1)                HIP_CHECK(hipFree(gpu_w->b_mlp1));
  if (gpu_w->w_mlp2)                HIP_CHECK(hipFree(gpu_w->w_mlp2));
  if (gpu_w->w_mlp2_packed16)       HIP_CHECK(hipFree(gpu_w->w_mlp2_packed16));
  if (gpu_w->b_mlp2)                HIP_CHECK(hipFree(gpu_w->b_mlp2));
  if (gpu_w->w_qkv_packed16)        HIP_CHECK(hipFree(gpu_w->w_qkv_packed16));
  if (gpu_w->w_o_packed16)          HIP_CHECK(hipFree(gpu_w->w_o_packed16));
  if (gpu_w->w_router_packed16)     HIP_CHECK(hipFree(gpu_w->w_router_packed16));
}
