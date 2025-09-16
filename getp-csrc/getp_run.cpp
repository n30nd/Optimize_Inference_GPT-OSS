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


static constexpr int HOST_COPY_STREAMS = 3;
static constexpr size_t HOST_COPY_DEFAULT_BYTES = 32ull * 1024 * 1024;

struct HostCopyPipeline {
  bool initialized;
  size_t buffer_elems;
  int device_id;
  hipStream_t streams[HOST_COPY_STREAMS];
  hip_bfloat16* host_buffers[HOST_COPY_STREAMS];
};

static HostCopyPipeline g_host_copy_pipeline = {false, 0, -1, {0,0,0}, {nullptr, nullptr, nullptr}};

static void destroy_host_copy_pipeline();
static inline void sync_host_copy_streams();

static inline hip_bfloat16 float_to_bf16_host(float v) {
  return hip_bfloat16(v);
}

static inline int round_up16_host(int x) {
  return ((x + 15) / 16) * 16;
}

static void ensure_host_copy_pipeline(size_t min_elems) {
  size_t default_elems = HOST_COPY_DEFAULT_BYTES / sizeof(hip_bfloat16);
  if (default_elems == 0) default_elems = 1;
  size_t target_elems = std::max(min_elems, default_elems);
  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));
  bool need_reinit = !g_host_copy_pipeline.initialized ||
                     g_host_copy_pipeline.device_id != current_device ||
                     g_host_copy_pipeline.buffer_elems < target_elems;
  if (!need_reinit) {
    return;
  }
  if (g_host_copy_pipeline.initialized) {
    sync_host_copy_streams();
    destroy_host_copy_pipeline();
  }
  HIP_CHECK(hipSetDevice(current_device));
  for (int i = 0; i < HOST_COPY_STREAMS; ++i) {
    HIP_CHECK(hipStreamCreateWithFlags(&g_host_copy_pipeline.streams[i], hipStreamNonBlocking));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&g_host_copy_pipeline.host_buffers[i]),
                            target_elems * sizeof(hip_bfloat16), hipHostMallocDefault));
  }
  g_host_copy_pipeline.buffer_elems = target_elems;
  g_host_copy_pipeline.device_id = current_device;
  g_host_copy_pipeline.initialized = true;
}

static void destroy_host_copy_pipeline() {
  if (!g_host_copy_pipeline.initialized) return;
  int prev_device = -1;
  HIP_CHECK(hipGetDevice(&prev_device));
  int pipeline_device = g_host_copy_pipeline.device_id;
  if (pipeline_device >= 0 && prev_device != pipeline_device) {
    HIP_CHECK(hipSetDevice(pipeline_device));
  }
  for (int i = 0; i < HOST_COPY_STREAMS; ++i) {
    if (g_host_copy_pipeline.host_buffers[i]) {
      HIP_CHECK(hipHostFree(g_host_copy_pipeline.host_buffers[i]));
      g_host_copy_pipeline.host_buffers[i] = nullptr;
    }
    if (g_host_copy_pipeline.streams[i]) {
      HIP_CHECK(hipStreamDestroy(g_host_copy_pipeline.streams[i]));
      g_host_copy_pipeline.streams[i] = nullptr;
    }
  }
  g_host_copy_pipeline.buffer_elems = 0;
  g_host_copy_pipeline.device_id = -1;
  g_host_copy_pipeline.initialized = false;
  if (prev_device >= 0) {
    HIP_CHECK(hipSetDevice(prev_device));
  }
}

static inline void sync_host_copy_streams() {
  if (!g_host_copy_pipeline.initialized) return;
  int prev_device = -1;
  HIP_CHECK(hipGetDevice(&prev_device));
  if (g_host_copy_pipeline.device_id >= 0 && prev_device != g_host_copy_pipeline.device_id) {
    HIP_CHECK(hipSetDevice(g_host_copy_pipeline.device_id));
  }
  for (int i = 0; i < HOST_COPY_STREAMS; ++i) {
    HIP_CHECK(hipStreamSynchronize(g_host_copy_pipeline.streams[i]));
  }
  if (prev_device >= 0 && prev_device != g_host_copy_pipeline.device_id) {
    HIP_CHECK(hipSetDevice(prev_device));
  }
}

static void pack_tile_row_cpu(const float *layer_src, int M, int K, int tile_m,
                              int tiles_per_row, hip_bfloat16 *dst_row) {
  const int base_row = tile_m * 16;
  for (int tile_k = 0; tile_k < tiles_per_row; ++tile_k) {
    for (int im = 0; im < 16; ++im) {
      const int row = base_row + im;
      const float *row_ptr = (row < M) ? (layer_src + (size_t)row * (size_t)K) : nullptr;
      for (int ik = 0; ik < 16; ++ik) {
        const int col = tile_k * 16 + ik;
        const size_t dst_index = (size_t)tile_k * 256 + (size_t)im * 16 + (size_t)ik;
        float val = 0.0f;
        if (row_ptr && col < K) val = row_ptr[col];
        dst_row[dst_index] = float_to_bf16_host(val);
      }
    }
  }
}

static void pack_matrices_cpu_to_device(const float *src, int matrices, int M, int K,
                                        size_t src_stride, hip_bfloat16 *dst_device) {
  if (matrices <= 0) return;
  int Mpad = round_up16_host(M);
  int Kpad = round_up16_host(K);
  int tiles_per_row = (Kpad + 15) / 16;
  int tile_rows = (Mpad + 15) / 16;
  size_t tile_row_elems = (size_t)tiles_per_row * 256;
  ensure_host_copy_pipeline(tile_row_elems);
  size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;

  for (int mat = 0; mat < matrices; ++mat) {
    const float *mat_src = src + (size_t)mat * src_stride;
    hip_bfloat16 *mat_dst = dst_device + (size_t)mat * per_mat_elems;
    for (int tile_m = 0; tile_m < tile_rows; ++tile_m) {
      int buf_idx = tile_m % HOST_COPY_STREAMS;
      hipStream_t stream = g_host_copy_pipeline.streams[buf_idx];
      hip_bfloat16 *host_buf = g_host_copy_pipeline.host_buffers[buf_idx];
      HIP_CHECK(hipStreamSynchronize(stream));
      pack_tile_row_cpu(mat_src, M, K, tile_m, tiles_per_row, host_buf);
      hip_bfloat16 *dst_row = mat_dst + (size_t)tile_m * tile_row_elems;
      HIP_CHECK(hipMemcpyAsync(dst_row, host_buf, tile_row_elems * sizeof(hip_bfloat16),
                               hipMemcpyHostToDevice, stream));
    }
    sync_host_copy_streams();
  }
}

static void convert_array_fp32_to_bf16_device(const float *src, size_t count, hip_bfloat16 *dst_device) {
  if (count == 0) return;
  ensure_host_copy_pipeline(1);
  size_t offset = 0;
  int stream_idx = 0;
  while (offset < count) {
    hipStream_t stream = g_host_copy_pipeline.streams[stream_idx];
    hip_bfloat16 *host_buf = g_host_copy_pipeline.host_buffers[stream_idx];
    HIP_CHECK(hipStreamSynchronize(stream));
    size_t chunk = std::min(g_host_copy_pipeline.buffer_elems, count - offset);
    for (size_t i = 0; i < chunk; ++i) {
      host_buf[i] = float_to_bf16_host(src[offset + i]);
    }
    HIP_CHECK(hipMemcpyAsync(dst_device + offset, host_buf, chunk * sizeof(hip_bfloat16),
                             hipMemcpyHostToDevice, stream));
    offset += chunk;
    stream_idx = (stream_idx + 1) % HOST_COPY_STREAMS;
  }
  sync_host_copy_streams();
}

static void allocate_and_convert_array(const float *src, size_t count, hip_bfloat16 **dst_device) {
  if (count == 0) {
    *dst_device = nullptr;
    return;
  }
  HIP_CHECK(hipMalloc(dst_device, count * sizeof(hip_bfloat16)));
  convert_array_fp32_to_bf16_device(src, count, *dst_device);
}


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

static void enable_peer_access_if_possible(int src_device, int dst_device) {
  if (src_device == dst_device) return;
  int can_access = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src_device, dst_device));
  if (!can_access) {
    fprintf(stderr, "[GPU#%d] Peer access to GPU#%d not available.\n", src_device, dst_device);
    return;
  }

  int prev_device = -1;
  HIP_CHECK(hipGetDevice(&prev_device));
  HIP_CHECK(hipSetDevice(src_device));
  hipError_t err = hipDeviceEnablePeerAccess(dst_device, 0);
  if (err == hipErrorPeerAccessAlreadyEnabled) {
    (void)hipGetLastError();
  } else if (err != hipSuccess) {
    fprintf(stderr, "[GPU#%d] Failed to enable peer access to GPU#%d: %s\n",
            src_device, dst_device, hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  if (prev_device >= 0) {
    HIP_CHECK(hipSetDevice(prev_device));
  }
}

static inline void clone_peer_bf16_array(hip_bfloat16 **dst, hip_bfloat16 *src,
                                         size_t count, int dst_device, int src_device,
                                         hipStream_t stream) {
  if (!src || count == 0) {
    *dst = nullptr;
    return;
  }
  HIP_CHECK(hipMalloc(dst, count * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpyPeerAsync(*dst, dst_device, src, src_device,
                               count * sizeof(hip_bfloat16), stream));
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
  int layer_offset;
  
  __device__ LayerType get_layer_type(int layer) const {
    int global_layer = layer_offset + layer;
    return (sliding_window_size > 0 && (global_layer % 2 == 0)) ? SLIDING_WINDOW : FULL_ATTENTION;
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
  int layer_offset;

public:
  KVCacheManager(int max_batch, int n_layers, int seq_len, size_t kv_dim, int sliding_window_size, int layer_offset)
    : max_batch(max_batch), n_layers(n_layers), seq_len(seq_len), kv_dim(kv_dim), 
      sliding_window_size(sliding_window_size), total_allocated_elements(0), d_slots(nullptr),
      d_device_data(nullptr), layer_offset(layer_offset) {
    
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
    device_data.layer_offset = layer_offset;
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
  hip_bfloat16 *w_mlp1;
  hip_bfloat16 *w_mlp2;
  hip_bfloat16 *b_mlp1;
  hip_bfloat16 *b_mlp2;
  // Packed (16x16 tiling) BF16 weights for GEMM core (convert to FP32 per tile in shared memory)
  hip_bfloat16 *w_qkv_packed16;   // [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_o_packed16;     // [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_router_packed16;// [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_mlp1_packed16;  // [n_layers*n_experts][Mpad * Kpad]
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
  float *gate_up_all;     // [max_batch * experts_per_token][intermediate_dim]
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

struct PipelineStageInfo {
  int group_id = 0;
  int stage_id = 0;
  int total_stages = 1;
  int layer_start = 0;
  int layer_end = 0; // exclusive
  bool is_first() const { return stage_id == 0; }
  bool is_last() const { return stage_id == total_stages - 1; }
  int layer_count() const { return layer_end - layer_start; }
};

// Per-GPU device context
typedef struct {
  int device_id;
  GPUTransformerWeights weights;
  GPURunState state;
  int max_batch; // number of concurrent slots per GPU for continuous batching
  hipStream_t batch_stream;              // shared stream for batched ops
  hipStream_t comm_stream;               // dedicated stream for inter-stage transfers
  // Staging device buffers for batched indices and positions
  int *d_slot_ids;
  int *d_pos_list;
  int *d_token_ids;
  int *d_next_tokens;  // Persistent buffer for sampled next tokens
  int *d_batch_iota;   // Identity mapping [0,1,2,...] for batch-ordered outputs
  PipelineStageInfo stage;
} GPUContext;

static std::vector<GPUContext> gpu_contexts;
static thread_local GPUContext* t_ctx = nullptr;

struct PipelineGroup {
  int group_id = 0;
  std::vector<GPUContext*> stages;
  int max_batch = 0;
};

static std::vector<PipelineGroup> pipeline_groups;

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

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __host__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
}
__host__ __device__ __forceinline__ int round_up16(int x) { return ((x + 15) / 16) * 16; }

// (removed) pack_bf16x2 helper no longer needed after switching to short4 MFMA operands

// ------------------------- Kernels -------------------------

// FP32 -> BF16 (device)
__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
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
// - W_packed: BF16 packed 16x16 tiles (Mpad x Kpad), Mout = 2*I
// - Xpack:    FP32 [K x Bpad], ta xử lý Be cột liên tiếp (đã align BLOCK_B)
// - gate_up_out: FP32 [B_global x I] (ở đây ta scatter theo "column id" toàn cục)
// - slot_ids: ánh xạ cột local (0..Be-1) -> column id toàn cục (d_col_ids + c0)
// Mỗi block: 64 threads (16x4) tính 1 tile 16x16 (M x N).
// Grouped variant: handle all experts in a single launch by selecting the expert per 16-column block.
// Uses block_expert_ids[blockIdx.y] to pick the expert and computes fused (GEMM + bias + SwiGLU) then scatters.
__global__ __launch_bounds__(64)
void gemm16x16_mlp1_fused_swiglu_scatter_grouped(
    const hip_bfloat16* __restrict__ W_packed_all,   // [n_layers*n_experts][Mpad*Kpad]
    const hip_bfloat16* __restrict__ Xpack,          // [K x Bpad_total] BF16, columns are globally sorted/padded
    float*              __restrict__ gate_up_out,    // [Btotal x I] FP32, scatter by global column id
    const hip_bfloat16* __restrict__ bias_all,       // [n_layers*n_experts][Mout]
    // Logical sizes
    int Mout, int K, int Btotal, int Bpad_total,
    int out_stride,                                   // I = Mout/2
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
  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W_packed = W_packed_all + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = bias_all ? (bias_all + mat_index * (size_t)Mout) : nullptr;

  // Double-buffered LDS tiles (ping-pong) to reduce synchronization overhead
  __shared__ __align__(16) hip_bfloat16 sWb[2][16*16];
  __shared__ __align__(16) hip_bfloat16 sX [2][16*16];
  __shared__ __align__(16) float        sBias[16];  // cache bias for this 16-row tile

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f,0.f,0.f,0.f};

  const int Ktiles = (Kpad + 15)/16;
  const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
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
  {
    const int kk = 0;
    if (lane < 32) {
      const size_t w_tile = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* src = W_packed + w_tile;
      reinterpret_cast<u32x4*>(sWb[curr])[lane] = reinterpret_cast<const u32x4*>(src)[lane];
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
      if (lane < 32) {
        const size_t w_tile = ((size_t)tile_m * tilesPerRow + (size_t)(kk + 1)) * (size_t)256;
        const hip_bfloat16* src = W_packed + w_tile;
        reinterpret_cast<u32x4*>(sWb[next])[lane] = reinterpret_cast<const u32x4*>(src)[lane];
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
  float* out_base  = gate_up_out + (size_t)col_id * (size_t)out_stride;

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
        out_base[row_half] = gate * (up + 1.f);
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
    const float* __restrict__ gate_up_all,    // [Btotal][intermediate_dim]
    const hip_bfloat16* __restrict__ w_packed_mlp2,
    const hip_bfloat16* __restrict__ b_mlp2,
    const int*   __restrict__ sorted_sids,
    const int*   __restrict__ sorted_eids,
    const float* __restrict__ sorted_w,
    const int*   __restrict__ sorted_indices,
    int Btotal,
    int hidden_dim, int intermediate_dim,
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
  if (col0 >= Bpad_live) return;
  const int base_row = tile_m * 16;

  // All 16 columns in this block share the same expert due to prior grouping/padding.
  const int eid = sorted_eids[col0];

  // Pointers per (layer,eid)
  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W = w_packed_mlp2 + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = b_mlp2 + mat_index * (size_t)hidden_dim;

  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float        sX[16 * 16];
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

  for (int kk = 0; kk < Ktiles; ++kk) {
    // Load W tile (shared across the 16 selections in this block)
    if (lane < 32) {
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W + w_tile_base;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
    }

    // Load 16 columns of X for this kk tile: gather one scalar from each selection
    if (lane < 64) {
      const int r   = lane % 16;           // K row within tile
      const int c4  = (lane / 16) * 4;     // column group: 0,4,8,12
      const int k_row = kk * 16 + r;       // absolute K index
      float* dst_vec = sX + r * 16 + c4;

      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
      if (k_row < K) {
        // Gather from four selections (columns)
        const int col_0 = col0 + (c4 + 0);
        const int col_1 = col0 + (c4 + 1);
        const int col_2 = col0 + (c4 + 2);
        const int col_3 = col0 + (c4 + 3);

        if (col_0 < Btotal_live) v0 = gate_up_all[(size_t)col_0 * (size_t)intermediate_dim + (size_t)k_row];
        if (col_1 < Btotal_live) v1 = gate_up_all[(size_t)col_1 * (size_t)intermediate_dim + (size_t)k_row];
        if (col_2 < Btotal_live) v2 = gate_up_all[(size_t)col_2 * (size_t)intermediate_dim + (size_t)k_row];
        if (col_3 < Btotal_live) v3 = gate_up_all[(size_t)col_3 * (size_t)intermediate_dim + (size_t)k_row];
      }
      float4v vpack = {v0, v1, v2, v3};
      reinterpret_cast<float4v*>(dst_vec)[0] = vpack;
    }

    __syncthreads();

    // Pre-convert four A scalars from BF16 to FP32 in registers and MFMA accumulate
    const int k0 = ty +  0;  // 0..3
    const int k1 = ty +  4;  // 4..7
    const int k2 = ty +  8;  // 8..11
    const int k3 = ty + 12;  // 12..15
    float a0 = bfloat16_to_float(sWb[tx*16 + k0]);
    float a1 = bfloat16_to_float(sWb[tx*16 + k1]);
    float a2 = bfloat16_to_float(sWb[tx*16 + k2]);
    float a3 = bfloat16_to_float(sWb[tx*16 + k3]);
    float b0 = sX[k0*16 + tx];
    float b1 = sX[k1*16 + tx];
    float b2 = sX[k2*16 + tx];
    float b3 = sX[k3*16 + tx];
    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0, b0, acc, 0, 0, 0);
    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1, b1, acc, 0, 0, 0);
    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2, b2, acc, 0, 0, 0);
    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3, b3, acc, 0, 0, 0);

    __syncthreads();
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
void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch, const PipelineStageInfo &stage);
void gpu_free_run_state(GPURunState *s);
static void gpu_clone_weights_from_peer(GPUTransformerWeights *dst,
                                        const GPUTransformerWeights *src,
                                        Config *p,
                                        const PipelineStageInfo &stage,
                                        int dst_device,
                                        int src_device,
                                        hipStream_t stream);


// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p, const PipelineStageInfo &stage);
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
  const PipelineStageInfo &stage = gpu_ctx->stage;
  int stage_layer_start = (stage.total_stages > 1) ? stage.layer_start : 0;
  int stage_layer_end = (stage.total_stages > 1) ? stage.layer_end : p->n_layers;
  if (stage_layer_end > p->n_layers) stage_layer_end = p->n_layers;
  if (stage.total_stages <= 1) stage_layer_start = 0;
  int stage_layers = stage_layer_end - stage_layer_start;
  if (stage_layers <= 0) stage_layers = p->n_layers;
  bool stage_is_first = stage.is_first() || stage.total_stages <= 1;
  bool stage_is_last = stage.is_last() || stage.total_stages <= 1;

  // Copy indices to device staging
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_slot_ids,  h_slot_ids,  batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_pos_list,  h_pos_list,  batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  if (stage_is_first) {
    HIP_CHECK(hipMemcpyAsync(gpu_ctx->d_token_ids, h_tokens, batch * sizeof(int), hipMemcpyHostToDevice, gpu_ctx->batch_stream));
  }

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
  if (stage_is_first) {
    launch_embed();
  }

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
  for (int local_l = 0; local_l < stage_layers; ++local_l) {
    int global_l = stage_layer_start + local_l;
    // RMSNorm (batched across slots)
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)local_l * (size_t)hidden_dim;
    {
      dim3 block_norm(256), grid_norm(1, batch);
      size_t shared_mem_norm = block_norm.x * sizeof(float);
      rmsnorm_kernel_float_batched<<<grid_norm, block_norm, shared_mem_norm, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.x, rms_attn_w, /*residual=*/nullptr,
        hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }

    // QKV projection (batched) via packed BF16 GEMM
    int loff = local_l * p->seq_len * kv_dim;
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)local_l *
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
      const hip_bfloat16 *w_qkv_packed = gpu_ctx->weights.w_qkv_packed16 + (size_t)local_l * (size_t)Mpad * (size_t)Kpad;
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
        gpu_ctx->state.qkv, gpu_ctx->state.kv_cache_manager->get_device_data(), local_l,
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
        gpu_ctx->state.qkv, gpu_ctx->state.kv_cache_manager->get_device_data(), local_l,
        batch, qkv_out_dim, q_size, k_size, kv_dim,
        gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
    }

    // Attention computation (FlashAttention-like, always enabled) -> tb
    {
      dim3 block(64);
      dim3 grid(1, p->n_attn_heads, batch);
      // Use shared memory for attention scores: use sliding window size for paged attention
      int max_attention_length = (p->sliding_window > 0 && (global_l % 2 == 0)) ? p->sliding_window : p->seq_len;
      // Allocate +1 element to store optional attention sink (baseline behavior) even for sliding-window layers.
      size_t shared_mem_bytes = (size_t)(max_attention_length + 1) * sizeof(float);
      
      // Use the new paged attention kernel that handles different layer types
      // Sliding window is handled by cache compaction (no explicit mask)
      paged_flash_attention_kernel_batched<<<grid, block, shared_mem_bytes, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.qkv, gpu_ctx->state.kv_cache_manager->get_device_data(), gpu_ctx->weights.attn_sinks,
        gpu_ctx->state.tb, head_dim, kv_dim, kv_mul,
        p->n_attn_heads, p->seq_len, local_l, gpu_ctx->d_pos_list, batch,
        qkv_out_dim, gpu_ctx->d_slot_ids);
    }

    // Output projection W_o (batched) via packed BF16 GEMM: tb2 = tb @ W_o + b_o
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)local_l * (size_t)hidden_dim;
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
      const hip_bfloat16 *w_o_packed = gpu_ctx->weights.w_o_packed16 + (size_t)local_l * (size_t)Mpad2 * (size_t)Kpad2;
      dim3 blockG(16, 4, 1), gridG((unsigned int)((Mpad2 + 15) / 16), (unsigned int)((Bpad2 + 15) / 16));

      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
        w_o_packed, gpu_ctx->state.xpack_bf16, gpu_ctx->state.tb2, b_o,
        Mwo, Kow, batch, Bpad2, Mwo, gpu_ctx->d_slot_ids, Mpad2, Kpad2);
    }

    // Residual add and FFN RMSNorm fused: t = RMSNorm(x += tb2)
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)local_l * (size_t)hidden_dim;
    {
      dim3 block_norm(256), grid_norm(1, batch);
      size_t shared_mem_norm = block_norm.x * sizeof(float);
      rmsnorm_kernel_float_batched<<<grid_norm, block_norm, shared_mem_norm, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.x, rms_ffn_w, /*residual=*/gpu_ctx->state.tb2,
        hidden_dim, batch, hidden_dim, gpu_ctx->d_slot_ids);
    }

    // -------- MoE batched & grouped across slots and experts --------
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)local_l * (size_t)p->n_experts;
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
      const hip_bfloat16 *w_router_packed = gpu_ctx->weights.w_router_packed16 + (size_t)local_l * (size_t)Mpadr * (size_t)Kpadr;
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
            /*W_all*/   gpu_ctx->weights.w_mlp1_packed16,
            /*Xpack*/   gpu_ctx->state.xpack_bf16,
            /*dst*/     gpu_ctx->state.gate_up_all,
            /*bias_all*/gpu_ctx->weights.b_mlp1,
            /*sizes*/   Mout, p->hidden_dim, Bmlp1, Bpad_total,
            /*stride*/  I,
            /*col_ids*/ gpu_ctx->state.d_col_ids,
            /*blk→eid*/ gpu_ctx->state.block_expert_ids,
            /*BLOCK_B*/ BLOCK_B,
            /*nexp,l*/  p->n_experts, local_l,
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
        local_l, p->n_experts, hidden_dim,
        Mpad_mlp2, Kpad_mlp2,
        gpu_ctx->state.padded_offsets + p->n_experts);
    }

    // 9) Residual fused in step 8 (mlp2 kernel accumulates directly into x)
  }

  // Fully batched final RMSNorm and logits computation
  if (stage_is_last && host_next_tokens_out != nullptr) {
    // Batched RMSNorm across all slots with proper shared memory
    dim3 block_norm(128);  // Reduced block size for better occupancy
    dim3 grid_norm(1, batch); // one block per slot; threads stride over hidden_dim
    size_t shared_mem_norm = block_norm.x * sizeof(float);
    rmsnorm_final_batched_kernel<<<grid_norm, block_norm, shared_mem_norm, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.x, gpu_ctx->weights.rms_out_w,
      hidden_dim, batch, gpu_ctx->d_slot_ids);

    // Batched logits via packed BF16 GEMM: logits = W_out @ X (batched)
    if (gpu_ctx->weights.w_out_packed16 && gpu_ctx->state.logits_fp32) {
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

  const int pipeline_stages = 4;
  const int desired_dp = 2;

  int max_devices = device_count;
  if (const char* env_nd = getenv("GETP_MAX_DEVICES")) {
    int nd = atoi(env_nd);
    if (nd > 0 && nd < max_devices) max_devices = nd;
  }

  int groups = std::max(1, std::min(desired_dp, max_devices / pipeline_stages));
  int total_devices = groups * pipeline_stages;
  if (total_devices <= 0) {
    total_devices = std::min(max_devices, pipeline_stages);
    groups = std::max(1, total_devices / pipeline_stages);
  }

  gpu_contexts.clear();
  gpu_contexts.resize(total_devices);
  pipeline_groups.clear();
  pipeline_groups.resize(groups);

  Config *cfg = &transformer->config;

  struct PendingWeightClone {
    int device_id;
    hipStream_t stream;
  };
  std::vector<PendingWeightClone> pending_weight_clones;
  pending_weight_clones.reserve(std::max(0, total_devices - pipeline_stages));

  for (auto &group : pipeline_groups) {
    group.stages.clear();
  }

  for (int g = 0; g < groups; ++g) {
    PipelineGroup &group = pipeline_groups[g];
    group.group_id = g;
    group.max_batch = INT_MAX;
    for (int s = 0; s < pipeline_stages; ++s) {
      int global_idx = g * pipeline_stages + s;
      if (global_idx >= total_devices) break;
      GPUContext &ctx = gpu_contexts[global_idx];
      HIP_CHECK(hipSetDevice(global_idx));
      ctx.device_id = global_idx;
      ctx.stage.group_id = g;
      ctx.stage.stage_id = s;
      ctx.stage.total_stages = pipeline_stages;
      ctx.stage.layer_start = (cfg->n_layers * s) / pipeline_stages;
      ctx.stage.layer_end = (cfg->n_layers * (s + 1)) / pipeline_stages;
      if (s == pipeline_stages - 1) ctx.stage.layer_end = cfg->n_layers;

      const char *env_mb = getenv("GETP_MAX_BATCH");
      int mb = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 8;
      ctx.max_batch = mb;
      group.max_batch = std::min(group.max_batch, mb);

      group.stages.push_back(&ctx);
    }
    if (group.max_batch == INT_MAX) group.max_batch = 0;
    for (size_t idx = 1; idx < group.stages.size(); ++idx) {
      GPUContext *prev = group.stages[idx - 1];
      GPUContext *curr = group.stages[idx];
      enable_peer_access_if_possible(prev->device_id, curr->device_id);
      enable_peer_access_if_possible(curr->device_id, prev->device_id);
    }
    for (auto *stage_ctx : group.stages) {
      stage_ctx->stage.total_stages = (int)group.stages.size();
    }
  }

  if (!pipeline_groups.empty()) {
    PipelineGroup &base_group = pipeline_groups.front();
    for (int s = 0; s < pipeline_stages; ++s) {
      if (s >= (int)base_group.stages.size()) break;
      GPUContext *base_ctx = base_group.stages[s];
      HIP_CHECK(hipSetDevice(base_ctx->device_id));
      log_gpu_mem(base_ctx->device_id, "[GPU] Before memory allocation on device:");
      gpu_copy_weights(&base_ctx->weights, &transformer->weights, cfg, base_ctx->stage);
      log_gpu_mem(base_ctx->device_id, "[GPU] After weight allocation on device:");

      for (int g = 1; g < groups; ++g) {
        PipelineGroup &clone_group = pipeline_groups[g];
        if (s >= (int)clone_group.stages.size()) continue;
        GPUContext *clone_ctx = clone_group.stages[s];
        log_gpu_mem(clone_ctx->device_id, "[GPU] Before memory allocation on device:");
        enable_peer_access_if_possible(clone_ctx->device_id, base_ctx->device_id);
        enable_peer_access_if_possible(base_ctx->device_id, clone_ctx->device_id);
        hipStream_t clone_stream;
        HIP_CHECK(hipStreamCreateWithFlags(&clone_stream, hipStreamNonBlocking));
        gpu_clone_weights_from_peer(&clone_ctx->weights, &base_ctx->weights, cfg, clone_ctx->stage,
                                    clone_ctx->device_id, base_ctx->device_id, clone_stream);
        pending_weight_clones.push_back({clone_ctx->device_id, clone_stream});
        log_gpu_mem(clone_ctx->device_id, "[GPU] After weight allocation on device:");
      }
    }
  }

  for (auto &pending : pending_weight_clones) {
    HIP_CHECK(hipSetDevice(pending.device_id));
    HIP_CHECK(hipStreamSynchronize(pending.stream));
    HIP_CHECK(hipStreamDestroy(pending.stream));
  }

  for (auto &ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(ctx.device_id));
    gpu_malloc_run_state(&ctx.state, cfg, ctx.max_batch, ctx.stage);
    log_gpu_mem(ctx.device_id, "[GPU] After run state allocation on device:");

    HIP_CHECK(hipStreamCreateWithFlags(&ctx.batch_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&ctx.comm_stream, hipStreamNonBlocking));
    HIP_CHECK(hipMalloc(&ctx.d_slot_ids,  ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_pos_list,  ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_token_ids, ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_next_tokens, ctx.max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_batch_iota, ctx.max_batch * sizeof(int)));
    {
      dim3 threads(256);
      dim3 blocks((unsigned int)((ctx.max_batch + threads.x - 1) / threads.x));
      iota_kernel<<<blocks, threads>>>(ctx.d_batch_iota, ctx.max_batch);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
    }
  }

  printf("[GPU] Warm-up initialized %d pipeline group(s) × %d stage(s).\n", (int)pipeline_groups.size(), pipeline_stages);
  printf("[Attention] Using FlashAttention-like optimized kernel (always on)\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    if (gpu_ctx.batch_stream) HIP_CHECK(hipStreamDestroy(gpu_ctx.batch_stream));
    if (gpu_ctx.comm_stream) HIP_CHECK(hipStreamDestroy(gpu_ctx.comm_stream));
    if (gpu_ctx.d_slot_ids)    HIP_CHECK(hipFree(gpu_ctx.d_slot_ids));
    if (gpu_ctx.d_pos_list)    HIP_CHECK(hipFree(gpu_ctx.d_pos_list));
    if (gpu_ctx.d_token_ids)   HIP_CHECK(hipFree(gpu_ctx.d_token_ids));
    if (gpu_ctx.d_next_tokens) HIP_CHECK(hipFree(gpu_ctx.d_next_tokens));
    if (gpu_ctx.d_batch_iota)  HIP_CHECK(hipFree(gpu_ctx.d_batch_iota));
    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx.state);
  }
  gpu_contexts.clear();
  pipeline_groups.clear();
  destroy_host_copy_pipeline();
  printf("[GPU] Cleanup done.\n");
}

// ------------------------- Driver (generate & inference) -------------------------



long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (gpu_contexts.empty()) warm_up(transformer, tokenizer);
  const int num_groups = (int)pipeline_groups.size();
  if (requests->num_reqs <= 0 || num_groups <= 0) return 0;
  // Global request index for all workers (lock-free producer)
  std::atomic<int> next_idx{0};

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_groups);
  
  // Continuous batching per pipeline group worker
  for (int g = 0; g < num_groups; ++g) {
    workers.emplace_back([&, g]() {
      PipelineGroup &group = pipeline_groups[g];
      if (group.stages.empty()) return;
      GPUContext *first_ctx = group.stages.front();
      HIP_CHECK(hipSetDevice(first_ctx->device_id));
      t_ctx = first_ctx;

      int batch_capacity = first_ctx->max_batch;
      for (auto *stage_ctx : group.stages) {
        batch_capacity = std::min(batch_capacity, stage_ctx->max_batch);
      }
      if (batch_capacity <= 0) return;
      std::vector<Slot> slots(batch_capacity);
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
      for (int i = 0; i < batch_capacity; ++i) {
        acquire_slot(i);
      }

      int active_cnt = 0;
      for (auto &s : slots) if (s.active) active_cnt++;

      while (active_cnt > 0) {
        // Build batch of current active slots
        std::vector<int> slot_ids; slot_ids.reserve(batch_capacity);
        std::vector<int> tokens;   tokens.reserve(batch_capacity);
        std::vector<int> poses;    poses.reserve(batch_capacity);
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

        auto run_pipeline = [&]() {
          const int stage_count = (int)group.stages.size();
          if (stage_count == 0) return;

          const int hidden_dim = transformer->config.hidden_dim;
          const size_t slot_bytes = (size_t)hidden_dim * sizeof(float);

          int micro_pref = 16;
          if (const char *env_micro = getenv("GETP_PIPE_MICRO")) {
            int parsed = atoi(env_micro);
            if (parsed > 0) micro_pref = parsed;
          }
          micro_pref = std::max(1, micro_pref);
          int micro_batch = std::min(micro_pref, B);
          int micro_count = (B + micro_batch - 1) / micro_batch;
          int total_ticks = micro_count + stage_count - 1;

          std::vector<std::vector<hipEvent_t>> ev_compute(stage_count);
          std::vector<std::vector<hipEvent_t>> ev_copy(stage_count);

          for (int s = 0; s < stage_count; ++s) {
            GPUContext *ctx = group.stages[s];
            HIP_CHECK(hipSetDevice(ctx->device_id));
            ev_compute[s].resize(micro_count);
            for (int m = 0; m < micro_count; ++m) {
              HIP_CHECK(hipEventCreateWithFlags(&ev_compute[s][m], hipEventDisableTiming));
            }
            if (s + 1 < stage_count) {
              ev_copy[s].resize(micro_count);
              for (int m = 0; m < micro_count; ++m) {
                HIP_CHECK(hipEventCreateWithFlags(&ev_copy[s][m], hipEventDisableTiming));
              }
            }
          }

          for (int tick = 0; tick < total_ticks; ++tick) {
            for (int s = 0; s < stage_count; ++s) {
              int micro_idx = tick - s;
              if (micro_idx < 0 || micro_idx >= micro_count) continue;

              int start = micro_idx * micro_batch;
              int micro_B = std::min(micro_batch, B - start);
              if (micro_B <= 0) continue;

              GPUContext *ctx = group.stages[s];
              HIP_CHECK(hipSetDevice(ctx->device_id));

              if (s > 0) {
                HIP_CHECK(hipStreamWaitEvent(ctx->batch_stream, ev_copy[s - 1][micro_idx], 0));
              }

              int *next_ptr = ctx->stage.is_last() ? next_tokens_host.data() + start : nullptr;
              gpu_forward_batch_step(transformer, ctx,
                                     tokens.data() + start,
                                     poses.data() + start,
                                     slot_ids.data() + start,
                                     micro_B,
                                     next_ptr);

              HIP_CHECK(hipEventRecord(ev_compute[s][micro_idx], ctx->batch_stream));

              if (s + 1 < stage_count) {
                GPUContext *next_ctx = group.stages[s + 1];
                HIP_CHECK(hipStreamWaitEvent(ctx->comm_stream, ev_compute[s][micro_idx], 0));

                const int *micro_slots = slot_ids.data() + start;
                size_t run_start = 0;
                while (run_start < (size_t)micro_B) {
                  int base_slot = micro_slots[run_start];
                  size_t run_end = run_start + 1;
                  while (run_end < (size_t)micro_B &&
                         micro_slots[run_end] == micro_slots[run_end - 1] + 1) {
                    ++run_end;
                  }
                  size_t run_slots = run_end - run_start;
                  float *src_ptr = ctx->state.x + (size_t)base_slot * (size_t)hidden_dim;
                  float *dst_ptr = next_ctx->state.x + (size_t)base_slot * (size_t)hidden_dim;
                  size_t bytes = run_slots * slot_bytes;
                  HIP_CHECK(hipMemcpyPeerAsync(dst_ptr, next_ctx->device_id,
                                               src_ptr, ctx->device_id,
                                               bytes, ctx->comm_stream));
                  run_start = run_end;
                }

                HIP_CHECK(hipEventRecord(ev_copy[s][micro_idx], ctx->comm_stream));
              }
            }
          }

          for (int s = 0; s < stage_count; ++s) {
            GPUContext *ctx = group.stages[s];
            HIP_CHECK(hipSetDevice(ctx->device_id));
            HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
            if (ctx->comm_stream) {
              HIP_CHECK(hipStreamSynchronize(ctx->comm_stream));
            }
          }

          for (int s = 0; s < stage_count; ++s) {
            GPUContext *ctx = group.stages[s];
            HIP_CHECK(hipSetDevice(ctx->device_id));
            for (hipEvent_t &ev : ev_compute[s]) {
              if (ev) {
                HIP_CHECK(hipEventDestroy(ev));
                ev = nullptr;
              }
            }
            for (hipEvent_t &ev : ev_copy[s]) {
              if (ev) {
                HIP_CHECK(hipEventDestroy(ev));
                ev = nullptr;
              }
            }
          }
        };

        run_pipeline();

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




void gpu_malloc_run_state(GPURunState *s, Config *p, int max_batch, const PipelineStageInfo &stage) {
  int kv_dim = p->head_dim * p->n_kv_heads;
  size_t B = (size_t)max_batch;
  int stage_layers = (stage.total_stages > 1) ? stage.layer_count() : p->n_layers;
  if (stage_layers <= 0) stage_layers = p->n_layers;

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
  s->kv_cache_manager = new KVCacheManager(max_batch, stage_layers, p->seq_len, kv_dim, p->sliding_window, stage.layer_start);
  
  // Removed legacy 'logits' buffer

  // Mask not needed for paged attention path; keep null to avoid large alloc
  s->mask = NULL;

  HIP_CHECK(hipMalloc(&s->cos_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sin_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));

  // Note: acc_buffer no longer needed since we use float state directly
  s->acc_buffer = nullptr;
  if (stage.is_last() || stage.total_stages <= 1) {
    HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));
  } else {
    s->logits_fp32 = nullptr;
  }

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
  // HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)p->intermediate_dim * sizeof(float)));

  // mlp1_out_all is unused in fused path; skip allocation to save memory
  s->mlp1_out_all = nullptr;
  HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax_pad * (size_t)p->intermediate_dim * sizeof(float)));

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
  if (s->logits_fp32) HIP_CHECK(hipFree(s->logits_fp32));
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

static void gpu_clone_weights_from_peer(GPUTransformerWeights *dst,
                                        const GPUTransformerWeights *src,
                                        Config *p,
                                        const PipelineStageInfo &stage,
                                        int dst_device,
                                        int src_device,
                                        hipStream_t stream) {
  *dst = {};

  int head_dim      = p->head_dim;
  int total_layers  = p->n_layers;
  int n_experts     = p->n_experts;
  int hidden_dim    = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size    = p->vocab_size;
  int n_attn_heads  = p->n_attn_heads;
  int n_kv_heads    = p->n_kv_heads;

  int layer_start = (stage.total_stages > 1) ? stage.layer_start : 0;
  if (layer_start < 0) layer_start = 0;
  if (layer_start > total_layers) layer_start = total_layers;
  int layer_end = (stage.total_stages > 1) ? stage.layer_end : total_layers;
  if (layer_end > total_layers) layer_end = total_layers;
  int n_layers = layer_end - layer_start;
  if (stage.total_stages <= 1 || n_layers <= 0) {
    layer_start = 0;
    layer_end = total_layers;
    n_layers = total_layers;
  }

  bool stage_is_first = stage.is_first() || stage.total_stages <= 1;
  bool stage_is_last = stage.is_last() || stage.total_stages <= 1;

  int prev_device = -1;
  HIP_CHECK(hipGetDevice(&prev_device));
  HIP_CHECK(hipSetDevice(dst_device));

  if (stage_is_first) {
    size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
    clone_peer_bf16_array(&dst->token_embedding_table, src->token_embedding_table,
                          token_embed_count, dst_device, src_device, stream);
  }

  if (stage_is_last) {
    size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
    clone_peer_bf16_array(&dst->out, src->out,
                          token_embed_count, dst_device, src_device, stream);
    if (src->w_out_packed16) {
      int M_out = vocab_size;
      int K_out = hidden_dim;
      int Mpad_out = round_up16_host(M_out);
      int Kpad_out = round_up16_host(K_out);
      size_t packed = (size_t)Mpad_out * (size_t)Kpad_out;
      clone_peer_bf16_array(&dst->w_out_packed16, src->w_out_packed16,
                            packed, dst_device, src_device, stream);
    }
    clone_peer_bf16_array(&dst->rms_out_w, src->rms_out_w,
                          (size_t)hidden_dim, dst_device, src_device, stream);
  }

  if (n_layers > 0) {
    size_t rms_count = (size_t)n_layers * (size_t)hidden_dim;
    clone_peer_bf16_array(&dst->rms_attn_w, src->rms_attn_w,
                          rms_count, dst_device, src_device, stream);
    clone_peer_bf16_array(&dst->rms_ffn_w, src->rms_ffn_w,
                          rms_count, dst_device, src_device, stream);

    int M_qkv = head_dim * n_attn_heads + 2 * head_dim * n_kv_heads;
    int K_qkv = hidden_dim;
    int Mpad_qkv = round_up16_host(M_qkv);
    int Kpad_qkv = round_up16_host(K_qkv);
    size_t qkv_elems = (size_t)n_layers * (size_t)Mpad_qkv * (size_t)Kpad_qkv;
    clone_peer_bf16_array(&dst->w_qkv_packed16, src->w_qkv_packed16,
                          qkv_elems, dst_device, src_device, stream);
    size_t qkv_b_count = (size_t)n_layers *
                         ((size_t)head_dim * (size_t)n_attn_heads +
                          (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
    clone_peer_bf16_array(&dst->b_qkv, src->b_qkv,
                          qkv_b_count, dst_device, src_device, stream);

    int M_o = hidden_dim;
    int K_o = head_dim * n_attn_heads;
    int Mpad_o = round_up16_host(M_o);
    int Kpad_o = round_up16_host(K_o);
    size_t o_elems = (size_t)n_layers * (size_t)Mpad_o * (size_t)Kpad_o;
    clone_peer_bf16_array(&dst->w_o_packed16, src->w_o_packed16,
                          o_elems, dst_device, src_device, stream);
    size_t b_o_count = (size_t)n_layers * (size_t)hidden_dim;
    clone_peer_bf16_array(&dst->b_o, src->b_o,
                          b_o_count, dst_device, src_device, stream);
    size_t attn_sinks_count = (size_t)n_layers * (size_t)n_attn_heads;
    clone_peer_bf16_array(&dst->attn_sinks, src->attn_sinks,
                          attn_sinks_count, dst_device, src_device, stream);

    int M_router = n_experts;
    int K_router = hidden_dim;
    int Mpad_router = round_up16_host(M_router);
    int Kpad_router = round_up16_host(K_router);
    size_t router_elems = (size_t)n_layers * (size_t)Mpad_router * (size_t)Kpad_router;
    clone_peer_bf16_array(&dst->w_router_packed16, src->w_router_packed16,
                          router_elems, dst_device, src_device, stream);
    size_t router_b_count = (size_t)n_layers * (size_t)n_experts;
    clone_peer_bf16_array(&dst->b_router, src->b_router,
                          router_b_count, dst_device, src_device, stream);

    int layers_flat = n_layers * n_experts;
    if (layers_flat > 0) {
      int M_mlp1 = 2 * intermediate_dim;
      int K_mlp1 = hidden_dim;
      int Mpad_mlp1 = round_up16_host(M_mlp1);
      int Kpad_mlp1 = round_up16_host(K_mlp1);
      size_t mlp1_elems = (size_t)layers_flat * (size_t)Mpad_mlp1 * (size_t)Kpad_mlp1;
      clone_peer_bf16_array(&dst->w_mlp1_packed16, src->w_mlp1_packed16,
                            mlp1_elems, dst_device, src_device, stream);
      size_t mlp1_b_count = (size_t)layers_flat * (size_t)2 * (size_t)intermediate_dim;
      clone_peer_bf16_array(&dst->b_mlp1, src->b_mlp1,
                            mlp1_b_count, dst_device, src_device, stream);

      int M_mlp2 = hidden_dim;
      int K_mlp2 = intermediate_dim;
      int Mpad_mlp2 = round_up16_host(M_mlp2);
      int Kpad_mlp2 = round_up16_host(K_mlp2);
      size_t mlp2_elems = (size_t)layers_flat * (size_t)Mpad_mlp2 * (size_t)Kpad_mlp2;
      clone_peer_bf16_array(&dst->w_mlp2_packed16, src->w_mlp2_packed16,
                            mlp2_elems, dst_device, src_device, stream);
      size_t mlp2_b_count = (size_t)layers_flat * (size_t)hidden_dim;
      clone_peer_bf16_array(&dst->b_mlp2, src->b_mlp2,
                            mlp2_b_count, dst_device, src_device, stream);
    }
  }

  if (prev_device >= 0 && prev_device != dst_device) {
    HIP_CHECK(hipSetDevice(prev_device));
  }

}

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p, const PipelineStageInfo &stage) {
  int dev = 0; HIP_CHECK(hipGetDevice(&dev));
  int head_dim      = p->head_dim;
  int total_layers  = p->n_layers;
  int n_experts     = p->n_experts;
  int hidden_dim    = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size    = p->vocab_size;
  int n_attn_heads  = p->n_attn_heads;
  int n_kv_heads    = p->n_kv_heads;

  int layer_start = (stage.total_stages > 1) ? stage.layer_start : 0;
  if (layer_start < 0) layer_start = 0;
  if (layer_start > total_layers) layer_start = total_layers;
  int layer_end = (stage.total_stages > 1) ? stage.layer_end : total_layers;
  if (layer_end > total_layers) layer_end = total_layers;
  int n_layers = layer_end - layer_start;
  if (stage.total_stages <= 1 || n_layers <= 0) {
    layer_start = 0;
    layer_end = total_layers;
    n_layers = total_layers;
  }

  bool stage_is_first = stage.is_first() || stage.total_stages <= 1;
  bool stage_is_last = stage.is_last() || stage.total_stages <= 1;

  // Ensure unused un-packed pointers are null (we use packed versions for these)
  gpu_w->token_embedding_table = nullptr;
  gpu_w->out = nullptr;
  gpu_w->w_out_packed16 = nullptr;
  gpu_w->rms_out_w = nullptr;
  gpu_w->w_qkv = nullptr;
  gpu_w->w_o = nullptr;
  gpu_w->w_router = nullptr;
  gpu_w->w_mlp1 = nullptr; // we will use packed storage for mlp1

  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    allocate_and_convert_array(cpu_data, count, gpu_data);
  };

  auto pack_weights_fp32_rowmajor_to_bf16_tiles16 = [&](const float *cpu_data,
                                                        hip_bfloat16 **gpu_packed,
                                                        int layers, int M, int K,
                                                        size_t src_stride) {
    if (layers <= 0) {
      *gpu_packed = nullptr;
      return;
    }
    int Mpad = round_up16_host(M);
    int Kpad = round_up16_host(K);
    size_t per_layer_elems = (size_t)Mpad * (size_t)Kpad;
    HIP_CHECK(hipMalloc(gpu_packed, (size_t)layers * per_layer_elems * sizeof(hip_bfloat16)));
    pack_matrices_cpu_to_device(cpu_data, layers, M, K, src_stride, *gpu_packed);
  };


  // Counts as size_t to avoid overflow
  size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
  if (stage_is_first) {
    printf("[GPU#%d] Copy token embeddings...\n", dev);
    copy_and_convert(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, token_embed_count);
  }
  if (stage_is_last) {
    printf("[GPU#%d] Copy final projection...\n", dev);
    copy_and_convert(cpu_w->out, &gpu_w->out, token_embed_count);
    printf("[GPU#%d] Pack OUT (vocab x hidden) into 16x16 BF16 tiles...\n", dev);
    int M_out = vocab_size;
    int K_out = hidden_dim;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(cpu_w->out, &gpu_w->w_out_packed16, 1, M_out, K_out,
                                               (size_t)M_out * (size_t)K_out);
    size_t rms_out_count = (size_t)hidden_dim;
    copy_and_convert(cpu_w->rms_out_w, &gpu_w->rms_out_w, rms_out_count);
  }

  size_t rms_count = (size_t)n_layers * (size_t)hidden_dim;
  if (rms_count > 0) {
    printf("[GPU#%d] Copy RMS weights (layers %d-%d)...\n", dev, layer_start, layer_end);
    const float *rms_attn_src = cpu_w->rms_attn_w + (size_t)layer_start * (size_t)hidden_dim;
    const float *rms_ffn_src  = cpu_w->rms_ffn_w  + (size_t)layer_start * (size_t)hidden_dim;
    copy_and_convert(rms_attn_src, &gpu_w->rms_attn_w, rms_count);
    copy_and_convert(rms_ffn_src,  &gpu_w->rms_ffn_w,  rms_count);
  } else {
    gpu_w->rms_attn_w = nullptr;
    gpu_w->rms_ffn_w = nullptr;
  }

  // Pack selected projection weights (QKV, W_O, Router) into 16x16 BF16 tiles
  printf("[GPU#%d] Pack QKV...\n", dev);
  int M_qkv = head_dim * n_attn_heads + 2 * head_dim * n_kv_heads;
  int K_qkv = hidden_dim;
  if (n_layers > 0) {
    const float *w_qkv_src = cpu_w->w_qkv + (size_t)layer_start * (size_t)M_qkv * (size_t)K_qkv;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(w_qkv_src, &gpu_w->w_qkv_packed16, n_layers, M_qkv, K_qkv,
                                               (size_t)M_qkv * (size_t)K_qkv);
    size_t qkv_b_count = (size_t)n_layers *
                         ((size_t)head_dim * (size_t)n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
    const float *b_qkv_src = cpu_w->b_qkv + (size_t)layer_start * (((size_t)head_dim * (size_t)n_attn_heads) + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
    copy_and_convert(b_qkv_src, &gpu_w->b_qkv, qkv_b_count);
  } else {
    gpu_w->w_qkv_packed16 = nullptr;
    gpu_w->b_qkv = nullptr;
  }

  printf("[GPU#%d] Pack W_O...\n", dev);
  int M_o = hidden_dim;
  int K_o = head_dim * n_attn_heads;
  if (n_layers > 0) {
    const float *w_o_src = cpu_w->w_o + (size_t)layer_start * (size_t)M_o * (size_t)K_o;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(w_o_src, &gpu_w->w_o_packed16, n_layers, M_o, K_o,
                                               (size_t)M_o * (size_t)K_o);
    size_t o_b_count = (size_t)n_layers * (size_t)hidden_dim;
    const float *b_o_src = cpu_w->b_o + (size_t)layer_start * (size_t)hidden_dim;
    copy_and_convert(b_o_src, &gpu_w->b_o, o_b_count);
    size_t attn_sinks_count = (size_t)n_layers * (size_t)n_attn_heads;
    const float *attn_sinks_src = cpu_w->attn_sinks + (size_t)layer_start * (size_t)n_attn_heads;
    copy_and_convert(attn_sinks_src, &gpu_w->attn_sinks, attn_sinks_count);
  } else {
    gpu_w->w_o_packed16 = nullptr;
    gpu_w->b_o = nullptr;
    gpu_w->attn_sinks = nullptr;
  }

  printf("[GPU#%d] Pack Router...\n", dev);
  int M_router = n_experts;
  int K_router = hidden_dim;
  if (n_layers > 0) {
    const float *w_router_src = cpu_w->w_router + (size_t)layer_start * (size_t)M_router * (size_t)K_router;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(w_router_src, &gpu_w->w_router_packed16, n_layers, M_router, K_router,
                                               (size_t)M_router * (size_t)K_router);
    size_t router_b_count = (size_t)n_layers * (size_t)n_experts;
    const float *b_router_src = cpu_w->b_router + (size_t)layer_start * (size_t)n_experts;
    copy_and_convert(b_router_src, &gpu_w->b_router, router_b_count);
  } else {
    gpu_w->w_router_packed16 = nullptr;
    gpu_w->b_router = nullptr;
  }

  // Pack MLP1 per (layer,expert) into 16x16 BF16 tiles
  printf("[GPU#%d] Pack MLP1 (layers=%d, experts=%d)...\n", dev, n_layers, n_experts);
  int M_mlp1 = 2 * intermediate_dim;
  int K_mlp1 = hidden_dim;
  if (n_layers > 0) {
    const float *mlp1_src_base = cpu_w->w_mlp1 + (size_t)layer_start * (size_t)n_experts * (size_t)M_mlp1 * (size_t)K_mlp1;
    int layers_flat = n_layers * n_experts;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(mlp1_src_base, &gpu_w->w_mlp1_packed16,
                                               layers_flat, M_mlp1, K_mlp1,
                                               (size_t)M_mlp1 * (size_t)K_mlp1);
    size_t mlp1_b_count = (size_t)layers_flat * (size_t)2 * (size_t)intermediate_dim;
    const float *b_mlp1_src = cpu_w->b_mlp1 + (size_t)layer_start * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim;
    copy_and_convert(b_mlp1_src, &gpu_w->b_mlp1, mlp1_b_count);
  } else {
    gpu_w->w_mlp1_packed16 = nullptr;
    gpu_w->b_mlp1 = nullptr;
  }

  printf("[GPU#%d] Pack MLP2 (layers=%d, experts=%d)...\n", dev, n_layers, n_experts);
  if (n_layers > 0) {
    int M_mlp2 = hidden_dim;
    int K_mlp2 = intermediate_dim;
    const float *mlp2_src_base = cpu_w->w_mlp2 + (size_t)layer_start * (size_t)n_experts * (size_t)M_mlp2 * (size_t)K_mlp2;
    int layers_flat = n_layers * n_experts;
    pack_weights_fp32_rowmajor_to_bf16_tiles16(mlp2_src_base, &gpu_w->w_mlp2_packed16,
                                               layers_flat, M_mlp2, K_mlp2,
                                               (size_t)M_mlp2 * (size_t)K_mlp2);

    gpu_w->w_mlp2 = nullptr;

    size_t mlp2_b_count = (size_t)layers_flat * (size_t)hidden_dim;
    const float *b_mlp2_src = cpu_w->b_mlp2 + (size_t)layer_start * (size_t)n_experts * (size_t)hidden_dim;
    copy_and_convert(b_mlp2_src, &gpu_w->b_mlp2, mlp2_b_count);
  } else {
    gpu_w->w_mlp2_packed16 = nullptr;
    gpu_w->b_mlp2 = nullptr;
    gpu_w->w_mlp2 = nullptr;
  }
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
  if (gpu_w->w_mlp1)                HIP_CHECK(hipFree(gpu_w->w_mlp1));
  if (gpu_w->w_mlp1_packed16)       HIP_CHECK(hipFree(gpu_w->w_mlp1_packed16));
  if (gpu_w->b_mlp1)                HIP_CHECK(hipFree(gpu_w->b_mlp1));
  if (gpu_w->w_mlp2)                HIP_CHECK(hipFree(gpu_w->w_mlp2));
  if (gpu_w->w_mlp2_packed16)       HIP_CHECK(hipFree(gpu_w->w_mlp2_packed16));
  if (gpu_w->b_mlp2)                HIP_CHECK(hipFree(gpu_w->b_mlp2));
  if (gpu_w->w_qkv_packed16)        HIP_CHECK(hipFree(gpu_w->w_qkv_packed16));
  if (gpu_w->w_o_packed16)          HIP_CHECK(hipFree(gpu_w->w_o_packed16));
  if (gpu_w->w_router_packed16)     HIP_CHECK(hipFree(gpu_w->w_router_packed16));
}
