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
  float *q;
  float *k;
  float *v;
  float *logits;      // (unused for final FP32 logits, but kept)
  float *key_cache;
  float *value_cache;
  float *mask;
  float *cos_vals;
  float *sin_vals;
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
  float *gate_up_all;     // [max_batch * experts_per_token][intermediate_dim]
  float *tb2_peer;        // cross-GPU accumulation buffers (leader)
  float *e_agg_peer;
  float *router_score_full;
  // MFMA packed buffers
  float *xpack_f32;       // Packed input buffer for MFMA [K_max x Bpad]
} GPURunState;

struct ParallelLayout {
  int tp_size = 1;
  int ep_size = 1;
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

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __host__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
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

__host__ __device__ __forceinline__ float decode_e8m0(uint8_t code) {
  int exp_unbiased = static_cast<int>(code) - 127;
  return ldexpf(1.0f, exp_unbiased);
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

// ------------------------- Kernels -------------------------

// FP32 -> BF16 (device)
__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
}

// BF16 -> FP32 (device)
__global__ void convert_bf16_to_fp32_kernel(const hip_bfloat16 *bf16_data, float *fp32_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) fp32_data[tid] = bfloat16_to_float(bf16_data[tid]);
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
  int tile_m = m / 16;
  int tile_k = c / 16;
  int im = m % 16;
  int ik = c % 16;
  size_t tiles_per_row = (size_t)(Kpad + 15) / 16;
  size_t dst_index = ((size_t)tile_m * tiles_per_row + (size_t)tile_k) * (size_t)256 + (size_t)im * 16 + (size_t)ik;
  dst_packed[dst_index] = float_to_bfloat16(src_fp32[(size_t)r * (size_t)K + (size_t)c]);
}

// Grouped MLP1 matvec using MFMA and packed BF16 weights.
// For each selection m (sid,eid), compute y = W[layer,eid] @ x[sid] + b[layer,eid]
// W is packed as 16x16 BF16 tiles (Mpad x Kpad). Accumulate in FP32.
// Implementation uses an MFMA 16x16x4f32 micro-kernel, replicating the input vector across 16 columns.
// Only column tx==0 produces stores; other columns are computed but discarded.
__global__ __launch_bounds__(64)
void mlp1_grouped_mfma_matvec_kernel(
    float* __restrict__ mlp1_out_all,        // [M selections][out_dim]
    const float* __restrict__ t,             // [B slots][hidden_dim]
    const hip_bfloat16* __restrict__ w_packed, // [n_layers*n_experts][Mpad*Kpad]
    const hip_bfloat16* __restrict__ b_mlp1, // [n_layers*n_experts][out_dim]
    const int* __restrict__ sel_sids,        // [M]
    const int* __restrict__ sel_eids,        // [M]
    int selections_M,
    int hidden_dim, int out_dim,
    int layer, int n_experts,
    int Mpad, int Kpad)
{
  const int tx = threadIdx.x;  // 0..15 (columns)
  const int ty = threadIdx.y;  // 0..3  (row groups)
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;        // along M (rows)
  const int msel   = blockIdx.y;        // selection index
  if (msel >= selections_M) return;

  const int base_row = tile_m * 16;

  const int sid = sel_sids[msel];
  const int eid = sel_eids[msel];

  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W = w_packed + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = b_mlp1 + mat_index * (size_t)out_dim;

  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float        sWf[16 * 16];
  __shared__ __align__(16) float        sX[16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  const int K = hidden_dim;
  const int Ktiles = (Kpad + 15) / 16;

  for (int kk = 0; kk < Ktiles; ++kk) {
    if (lane < 32) {
      const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W + w_tile_base;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
    }

    if (lane < 64) {
      const int r   = lane % 16;        // K row within tile
      const int c4  = (lane / 16) * 4;  // column block index {0,4,8,12}
      const int k_row = kk * 16 + r;
      float v = 0.f;
      if (k_row < K) {
        const float* __restrict__ x_b = t + (size_t)sid * (size_t)hidden_dim;
        v = x_b[k_row];
      }
      float* dst = sX + r * 16 + c4;
      dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = v;
    }

    __syncthreads();

    int idx4 = lane * 4;
    if (idx4 < 256) {
      sWf[idx4 + 0] = bfloat16_to_float(sWb[idx4 + 0]);
      sWf[idx4 + 1] = bfloat16_to_float(sWb[idx4 + 1]);
      sWf[idx4 + 2] = bfloat16_to_float(sWb[idx4 + 2]);
      sWf[idx4 + 3] = bfloat16_to_float(sWb[idx4 + 3]);
    }

    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4; // 0..15
      float amk = sWf[tx * 16 + k4];   // A: row=tx, col=k4
      float bkn = sX[k4 * 16 + tx];    // B: row=k4, col=tx
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }

    __syncthreads();
  }

  if (tx == 0) {
    float* out_vec = mlp1_out_all + (size_t)msel * (size_t)out_dim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = base_row + (ty * 4 + i);
      if (row < out_dim) {
        const float biasv = B ? bfloat16_to_float(B[row]) : 0.f;
        out_vec[row] = dmn[i] + biasv;
      }
    }
  }
}

// MFMA MLP1 directly from MXFP4-packed weights (no BF16 storage).
// Dequantizes per element on-the-fly using e8m0 scales and 4-bit levels.
__global__ __launch_bounds__(64)
void mlp1_grouped_mfma_matvec_mxfp4_kernel(
    float* __restrict__ mlp1_out_all,        // [M selections][out_dim]
    const float* __restrict__ t,             // [B slots][hidden_dim]
    const uint8_t* __restrict__ w_data,      // MXFP4 nibbles (row-major)
    const uint8_t* __restrict__ w_scales,    // e8m0 per-32-block (row-major)
    const hip_bfloat16* __restrict__ b_mlp1, // [n_layers*n_experts][out_dim]
    const int* __restrict__ sel_sids,        // [M]
    const int* __restrict__ sel_eids,        // [M]
    int selections_M,
    int hidden_dim, int out_dim,
    int layer, int n_experts)
{
  const int tx = threadIdx.x;  // 0..15 (columns)
  const int ty = threadIdx.y;  // 0..3  (row groups)
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;        // along M (rows)
  const int msel   = blockIdx.y;        // selection index
  if (msel >= selections_M) return;

  const int base_row = tile_m * 16;

  const int sid = sel_sids[msel];
  const int eid = sel_eids[msel];

  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const size_t row_stride = (size_t)hidden_dim;
  const size_t base = mat_index * (size_t)out_dim * row_stride; // for scales (row-major)
  const hip_bfloat16* __restrict__ B = b_mlp1 + mat_index * (size_t)out_dim;

  __shared__ __align__(16) float sWf[16 * 16];
  __shared__ __align__(16) float sX[16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  const int K = hidden_dim;
  const int Ktiles = (K + 15) / 16;

  // Compute packed matrix base (tile-major nibble layout)
  const int tiles_m = (out_dim + 15) / 16;
  const int tiles_k = (hidden_dim + 15) / 16;
  const size_t packed_mat_bytes = (size_t)tiles_m * (size_t)tiles_k * (size_t)128; // 128 bytes per 16x16 tile (256 nibbles)
  const uint8_t* __restrict__ Wpack = w_data + mat_index * packed_mat_bytes;

  for (int kk = 0; kk < Ktiles; ++kk) {
    // Fill W tile [16x16] by dequantizing from MXFP4
    int idx4 = lane * 4;
    if (idx4 < 256) {
      #pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        int te = idx4 + t4;           // 0..255 within tile
        int im = te / 16;             // row in tile
        int ik = te - im * 16;        // col in tile
        int row = base_row + im;
        int col = kk * 16 + ik;
        float wv = 0.f;
        if (row < out_dim && col < K) {
          size_t elem_index = base + (size_t)row * row_stride + (size_t)col; // for scale indexing
          uint8_t scale_code = w_scales[elem_index / MXFP4_BLOCK_SIZE];
          float scale = decode_e8m0(scale_code);
          // tile-packed lookup
          size_t tiles_per_row = (size_t)tiles_k;
          size_t tile_linear = (size_t)tile_m * tiles_per_row + (size_t)kk;
          size_t tile_base_byte = tile_linear * (size_t)128;
          size_t byte_off = tile_base_byte + ((size_t)te >> 1);
          uint8_t packed = Wpack[byte_off];
          uint8_t nibble = (te & 1) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xF);
          wv = decode_mxfp4(nibble) * scale;
        }
        sWf[im * 16 + ik] = wv;
      }
    }

    // Build X tile [16x16] by replicating vector entries across 16 columns
    if (lane < 64) {
      const int r   = lane % 16;        // K row within tile
      const int c4  = (lane / 16) * 4;  // column block index {0,4,8,12}
      const int k_row = kk * 16 + r;
      float v = 0.f;
      if (k_row < K) {
        const float* __restrict__ x_b = t + (size_t)sid * (size_t)hidden_dim;
        v = x_b[k_row];
      }
      float* dst = sX + r * 16 + c4;
      dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = v;
    }

    __syncthreads();

    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4; // 0..15
      float amk = sWf[tx * 16 + k4];   // A: row=tx, col=k4
      float bkn = sX[k4 * 16 + tx];    // B: row=k4, col=tx
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }

    __syncthreads();
  }

  if (tx == 0) {
    float* out_vec = mlp1_out_all + (size_t)msel * (size_t)out_dim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = base_row + (ty * 4 + i);
      if (row < out_dim) {
        const float biasv = B ? bfloat16_to_float(B[row]) : 0.f;
        out_vec[row] = dmn[i] + biasv;
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
// Only the data (nibbles) are reordered; scales remain in original row-major block order.
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

__global__ void rmsnorm_kernel_bf16(hip_bfloat16 *o, const hip_bfloat16 *x, const hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
      float xv = bfloat16_to_float(x[j]);
      ss += xv * xv;
    }
    ss = ss / size + 1e-5f;
    shared_ss = 1.0f / sqrtf(ss);
  }
  __syncthreads();
  if (tid < size) {
    float xv = bfloat16_to_float(x[tid]);
    float wv = bfloat16_to_float(weight[tid]);
    o[tid] = float_to_bfloat16(wv * (shared_ss * xv));
  }
}

__global__ void matmul_kernel_bf16(hip_bfloat16 *xout, const hip_bfloat16 *x, const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      float wv = bfloat16_to_float(w[i * n + j]);
      float xv = bfloat16_to_float(x[j]);
      val += wv * xv;
    }
    xout[i] = float_to_bfloat16(val);
  }
}

// Grouped MLP2 matvec using MFMA and tile-packed MXFP4 weights.
// For each selection m: y = W[layer,eid] @ g[m] + b[layer,eid]; then atomicAdd into e_agg[sid] with weight sel_w[m].
__global__ __launch_bounds__(64)
void mlp2_accum_grouped_mfma_matvec_mxfp4_kernel(
    float* __restrict__ e_agg,                // [slots, hidden_dim]
    const float* __restrict__ gate_up_all,    // [M, intermediate_dim]
    const uint8_t* __restrict__ w_data,       // tile-packed MXFP4 nibbles (16x16)
    const uint8_t* __restrict__ w_scales,     // row-major e8m0 per-32 block
    const hip_bfloat16* __restrict__ b_mlp2,  // [n_layers*n_experts][hidden_dim]
    const int* __restrict__ sel_sids,         // [M]
    const int* __restrict__ sel_eids,         // [M]
    const float* __restrict__ sel_w,          // [M]
    int selections_M,
    int hidden_dim, int intermediate_dim,
    int layer, int n_experts,
    int eagg_stride) {

  const int tx = threadIdx.x;  // 0..15
  const int ty = threadIdx.y;  // 0..3
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;        // along output rows (hidden_dim)
  const int msel   = blockIdx.y;        // selection index
  if (msel >= selections_M) return;

  const int base_row = tile_m * 16;
  const int sid = sel_sids[msel];
  const int eid = sel_eids[msel];
  const float ew = sel_w[msel];

  const int Mout = hidden_dim;
  const int Kdim = intermediate_dim;
  const int tiles_k = (Kdim + 15) / 16;

  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const size_t row_stride = (size_t)intermediate_dim;
  const size_t base_scale = mat_index * (size_t)Mout * row_stride; // for scales (row-major indexing)
  const int tiles_m = (Mout + 15) / 16;
  const size_t packed_mat_bytes = (size_t)tiles_m * (size_t)tiles_k * (size_t)128;
  const uint8_t* __restrict__ Wpack = w_data + mat_index * packed_mat_bytes;
  const hip_bfloat16* __restrict__ B = b_mlp2 + mat_index * (size_t)hidden_dim;

  __shared__ __align__(16) float sWf[16 * 16];
  __shared__ __align__(16) float sX[16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f, 0.f, 0.f, 0.f};

  for (int kk = 0; kk < tiles_k; ++kk) {
    int idx4 = lane * 4;
    if (idx4 < 256) {
      #pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        int te = idx4 + t4;           // 0..255 within tile
        int im = te / 16;             // row in tile
        int ik = te - im * 16;        // col in tile
        int row = base_row + im;
        int col = kk * 16 + ik;
        float wv = 0.f;
        if (row < Mout && col < Kdim) {
          size_t elem_index = base_scale + (size_t)row * row_stride + (size_t)col; // for scale indexing
          uint8_t scale_code = w_scales[elem_index / MXFP4_BLOCK_SIZE];
          float scale = decode_e8m0(scale_code);
          size_t tiles_per_row = (size_t)tiles_k;
          size_t tile_linear = (size_t)tile_m * tiles_per_row + (size_t)kk;
          size_t tile_base_byte = tile_linear * (size_t)128;
          size_t byte_off = tile_base_byte + ((size_t)te >> 1);
          uint8_t packed = Wpack[byte_off];
          uint8_t nibble = (te & 1) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xF);
          wv = decode_mxfp4(nibble) * scale;
        }
        sWf[im * 16 + ik] = wv;
      }
    }

    // Build X tile [16x16] by replicating vector entries across 16 columns from gate_up_all[msel]
    if (lane < 64) {
      const int r   = lane % 16;        // K row within tile
      const int c4  = (lane / 16) * 4;  // column block index {0,4,8,12}
      const int k_row = kk * 16 + r;
      float v = 0.f;
      if (k_row < Kdim) {
        const float* __restrict__ g = gate_up_all + (size_t)msel * (size_t)intermediate_dim;
        v = g[k_row];
      }
      float* dst = sX + r * 16 + c4;
      dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = v;
    }

    __syncthreads();

    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4; // 0..15
      float a0 = sWf[tx * 16 + k4];
      float b0 = sX[k4 * 16 + tx];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0, b0, acc, 0, 0, 0);
    }

    __syncthreads();
  }

  if (tx == 0) {
    const int row0 = base_row + ty * 4 + 0;
    const int row1 = base_row + ty * 4 + 1;
    const int row2 = base_row + ty * 4 + 2;
    const int row3 = base_row + ty * 4 + 3;
    float* out_slot = e_agg + (size_t)sid * (size_t)eagg_stride;
    if (row0 < Mout) {
      float y = acc[0] + (B ? bfloat16_to_float(B[row0]) : 0.f);
      atomicAdd(&out_slot[row0], ew * y);
    }
    if (row1 < Mout) {
      float y = acc[1] + (B ? bfloat16_to_float(B[row1]) : 0.f);
      atomicAdd(&out_slot[row1], ew * y);
    }
    if (row2 < Mout) {
      float y = acc[2] + (B ? bfloat16_to_float(B[row2]) : 0.f);
      atomicAdd(&out_slot[row2], ew * y);
    }
    if (row3 < Mout) {
      float y = acc[3] + (B ? bfloat16_to_float(B[row3]) : 0.f);
      atomicAdd(&out_slot[row3], ew * y);
    }
  }
}


__global__ void add_bias_kernel_bf16(hip_bfloat16 *output, const hip_bfloat16 *bias, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float ov = bfloat16_to_float(output[tid]);
    float bv = bfloat16_to_float(bias[tid]);
    output[tid] = float_to_bfloat16(ov + bv);
  }
}

__global__ void copy_kernel_bf16(hip_bfloat16 *dst, const hip_bfloat16 *src, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) dst[tid] = src[tid];
}

// NOTE: serial softmax (thread 0). OK for small sizes; simple & safe.
__global__ void softmax_kernel_bf16(hip_bfloat16 *x, int size) {
  __shared__ float max_val;
  __shared__ float sum_val;
  if (threadIdx.x == 0) {
    if (size <= 0) return;
    max_val = bfloat16_to_float(x[0]);
    for (int i = 1; i < size; i++) {
      float v = bfloat16_to_float(x[i]);
      if (v > max_val) max_val = v;
    }
    sum_val = 0.0f;
    for (int i = 0; i < size; i++) {
      float v = expf(bfloat16_to_float(x[i]) - max_val);
      x[i] = float_to_bfloat16(v);
      sum_val += v;
    }
    float inv = 1.0f / sum_val;
    for (int i = 0; i < size; i++) {
      float v = bfloat16_to_float(x[i]) * inv;
      x[i] = float_to_bfloat16(v);
    }
  }
}

__global__ void compute_cos_sin_kernel_bf16(hip_bfloat16 *cos_out, hip_bfloat16 *sin_out, int pos,
                                            float base, int head_dim, float scaling_factor,
                                            float initial_context_length, float ntk_beta, float ntk_alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim / 2;
  if (tid < d_half) {
    float freq = powf(base, ((float)(2 * tid)) / (float)head_dim);
    float inv_freq;
    if (scaling_factor > 1.0f) {
      float low  = d_half * logf(initial_context_length / (ntk_beta  * 2.0f * M_PI)) / logf(base);
      float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
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
    float val = (float)pos * inv_freq;
    cos_out[tid] = float_to_bfloat16(cosf(val) * concentration);
    sin_out[tid] = float_to_bfloat16(sinf(val) * concentration);
  }
}

__global__ void apply_rotary_emb_kernel_bf16(hip_bfloat16 *x, const hip_bfloat16 *cos, const hip_bfloat16 *sin,
                                             int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (h < n_heads && i < half) {
    float x1 = bfloat16_to_float(x[h * head_dim + i]);
    float x2 = bfloat16_to_float(x[h * head_dim + half + i]);
    float c  = bfloat16_to_float(cos[i]);
    float s  = bfloat16_to_float(sin[i]);
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    x[h * head_dim + i]           = float_to_bfloat16(o1);
    x[h * head_dim + half + i]    = float_to_bfloat16(o2);
  }
}

__global__ void attention_scores_kernel_bf16(hip_bfloat16 *att, const hip_bfloat16 *q, const hip_bfloat16 *key_cache,
                                             int head_dim, int loff, int kv_dim, int kv_mul,
                                             int h, int pos, int seq_len, const hip_bfloat16 *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const hip_bfloat16 *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      float qv = bfloat16_to_float(q[i]);
      float kv = bfloat16_to_float(k[i]);
      score += qv * kv;
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      float mv = bfloat16_to_float(mask[pos * seq_len + t]);
      score += mv;
    }
    att[t] = float_to_bfloat16(score);
  }
}

__global__ void attention_weighted_sum_kernel_bf16(hip_bfloat16 *tb, const hip_bfloat16 *att, const hip_bfloat16 *value_cache,
                                                   int head_dim, int loff, int kv_dim, int kv_mul,
                                                   int h, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < head_dim) {
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const hip_bfloat16 *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      float a  = bfloat16_to_float(att[t]);
      float vv = bfloat16_to_float(v[i]);
      sum += a * vv;
    }
    tb[h * head_dim + i] = float_to_bfloat16(sum);
  }
}

__global__ void swiglu_kernel_bf16(hip_bfloat16 *gate_up, const hip_bfloat16 *gate, const hip_bfloat16 *up,
                                   int intermediate_dim, float swiglu_limit) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    float val   = bfloat16_to_float(gate[tid]);
    float upval = bfloat16_to_float(up[tid]);
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    gate_up[tid] = float_to_bfloat16(val);
  }
}

__global__ void split_gate_up_kernel_bf16(hip_bfloat16 *gate, hip_bfloat16 *up, const hip_bfloat16 *mlp1_out, int intermediate_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    gate[tid] = mlp1_out[2 * tid];
    up[tid]   = mlp1_out[2 * tid + 1];
  }
}

__global__ void weighted_sum_kernel_bf16(hip_bfloat16 *e_agg, const hip_bfloat16 *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float agg = bfloat16_to_float(e_agg[tid]);
    float add = bfloat16_to_float(tb2[tid]);
    e_agg[tid] = float_to_bfloat16(agg + add * expert_w);
  }
}

__global__ void residual_connection_kernel_bf16(hip_bfloat16 *x, const hip_bfloat16 *residual, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float xv  = bfloat16_to_float(x[tid]);
    float rv  = bfloat16_to_float(residual[tid]);
    x[tid] = float_to_bfloat16(xv + rv);
  }
}

// Final output: W (BF16) * x (FP32) -> logits (FP32); avoid big w_out_fp32 buffer
__global__ void matvec_wbf16_xfp32_outfp32(float *out, const float *x,
                                           const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
      sum += (float)w[i * n + j] * x[j];
    }
    out[i] = sum;
  }
}

// ========================= FLOAT KERNELS =========================
// New kernels for float RunState buffers

__global__ void rmsnorm_kernel_float(float *o, const float *x, const hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
      float xv = x[j];
      ss += xv * xv;
    }
    ss = ss / size + 1e-5f;
    shared_ss = 1.0f / sqrtf(ss);
  }
  __syncthreads();
  if (tid < size) {
    float xv = x[tid];
    float wv = bfloat16_to_float(weight[tid]);
    o[tid] = wv * (shared_ss * xv);
  }
}

__global__ void matmul_kernel_float(float *xout, const float *x, const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      float wv = bfloat16_to_float(w[i * n + j]);
      float xv = x[j];
      val += wv * xv;
    }
    xout[i] = val;
  }
}

// Batched variant: same weights across batch, different inputs/outputs
__global__ void matmul_kernel_float_batched(float *xout, const float *x,
                                            const hip_bfloat16 *w, int n, int d,
                                            int batch, int out_stride, int in_stride,
                                            const int *slot_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  int b = blockIdx.y; // batch index
  if (i < d && b < batch) {
    float val = 0.0f;
    int sid = slot_ids[b];
    const float *x_b = x + (size_t)sid * (size_t)in_stride;
    for (int j = 0; j < n; j++) {
      float wv = bfloat16_to_float(w[i * n + j]);
      float xv = x_b[j];
      val += wv * xv;
    }
    float *out_s = xout + (size_t)sid * (size_t)out_stride;
    out_s[i] = val;
  }
}

// Pack FP32 row-major matrix to FP32 16x16 tiles for MFMA
__global__ void pack_fp32_rowmajor_to_fp32_tiles16_kernel(
    const float *src_fp32, int row_offset, int rows,
    int M, int K,
    float *dst_packed, int Mpad, int Kpad) {
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
  dst_packed[dst_index] = src_fp32[(size_t)r * (size_t)K + (size_t)c];
}

// Pack weights từ BF16 row-major thành FP32 tiles 16x16
__global__ void pack_bf16_rowmajor_to_fp32_tiles16_kernel(
    const hip_bfloat16 *src_bf16, int row_offset, int rows,
    int M, int K,
    float *dst_packed, int Mpad, int Kpad) {
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
  dst_packed[dst_index] = bfloat16_to_float(src_bf16[(size_t)r * (size_t)K + (size_t)c]);
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

// MFMA-based GEMM: W_packed (BF16 tiles) * X_packed (FP32) -> Output (FP32) với bias
__global__ __launch_bounds__(64)
void gemm16x16_mfma_f32_packedW_batched_scatter_bias_kernel(
    const hip_bfloat16* __restrict__ W_packed,    // [Mpad x Kpad] packed thành tiles 16x16 BF16
    const float* __restrict__ Xpack,       // [K x Bpad] FP32, stride = Bpad  
    float* __restrict__ dst,               // scatter theo slot_ids
    const hip_bfloat16* __restrict__ bias, // [M] hoặc nullptr
    int M, int K, int B, int Bpad,
    int out_stride,                        // stride cho output = M
    const int* __restrict__ slot_ids,
    int Mpad, int Kpad) {
    
  const int tx = threadIdx.x;  // 0..15
  const int ty = threadIdx.y;  // 0..3
  const int lane = ty * 16 + tx; // 0..63
  const int lane4 = lane * 4;
  const int row_in_tile = lane & 15;
  const int col_group = lane >> 4; // 0..3
  
  const int tile_m = blockIdx.x;       // output tile row
  const int b      = blockIdx.y;       // batch index
  
  if (b >= B) return;
  
  const int base_row = tile_m * 16;
  const int slot_id = slot_ids[b];
  
  __shared__ __align__(16) float sW[16 * 16];
  __shared__ __align__(16) float sX[16 * 16];
  
  float4v acc = {0.f, 0.f, 0.f, 0.f};
  
  const int K_tiles = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)Kpad / 16;
  const size_t tile_elems = 256;
  const hip_bfloat16* w_tile_row_base = W_packed + ((size_t)tile_m * tiles_per_row) * tile_elems;
  const float* sW_lane = sW + tx * 16 + ty;
  const float* sX_lane = sX + ty * 16 + tx;
  const int x_stride = Bpad;
  
  for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
    // Load W tile [16x16] từ packed format
    const hip_bfloat16* w_src = w_tile_row_base + (size_t)k_tile * tile_elems;
    reinterpret_cast<float4v*>(sW)[lane] = load_bf16x4_as_float4v(w_src + lane4);
    
    // Load X tile [16x16] và replicate across columns
    const int k_row = k_tile * 16 + row_in_tile;
    float v = 0.f;
    if (k_row < K) {
      v = Xpack[(size_t)k_row * (size_t)x_stride + (size_t)b];
    }
    reinterpret_cast<float4v*>(sX + row_in_tile * 16)[col_group] = make_float4v(v);
    
    __syncthreads();
    
    // MFMA computation: 4 groups of MFMA
    #pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
      float a_val = sW_lane[k4 * 4];
      float b_val = sX_lane[k4 * 64];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_val, acc, 0, 0, 0);
    }
    
    __syncthreads();
  }
  
  // Write results với bias (chỉ tx==0 write)
  if (tx == 0) {
    float* out = dst + (size_t)slot_id * (size_t)out_stride;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = base_row + ty * 4 + i;
      if (row < M) {
        float bias_val = bias ? bfloat16_to_float(bias[row]) : 0.0f;
        out[row] = acc[i] + bias_val;
      }
    }
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

  // Chuẩn bị slot_id cho từng cột batch trong nhóm
  // (đọc trực tiếp từ global trong phase write để tiết kiệm LDS)
  auto slot_of = [&](int local_col) -> int {
    // local_col: 0..15
    int b = b_base + local_col;
    return (b < B) ? slot_ids[b] : -1;
  };

  // Shared ping-pong
  __shared__ __align__(16) float sW[2][16 * 16];
  __shared__ __align__(16) float sX[2][16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  using u32x4   = __attribute__((ext_vector_type(4))) unsigned int;

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

  float4v acc = {0.f, 0.f, 0.f, 0.f};

  const int Ktiles        = (Kpad + 15) / 16;
  const size_t tiles_per_row = (size_t)Kpad / 16;       // do Kpad bội 16
  const size_t tile_elems    = 256;                     // 16*16 BF16/FP32
  const hip_bfloat16* w_tile_row_base =
      W_packed + ((size_t)tile_m * tiles_per_row) * tile_elems;

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
      // mỗi lane ghi 4 cột (col_group xác định cụm {0,4,8,12})
      // để 64 lanes phủ 16*16 phần tử: lane i ghi 4 phần tử kế tiếp theo cột
      const int c0 = col_group * 4 + 0;
      const int c1 = col_group * 4 + 1;
      const int c2 = col_group * 4 + 2;
      const int c3 = col_group * 4 + 3;
      const int xs = Bpad;

      // đọc an toàn trong nhóm cuối
      xv[0] = (c0 < cols_this) ? Xpack[(size_t)k_row0 * xs + (size_t)(b_base + c0)] : 0.f;
      xv[1] = (c1 < cols_this) ? Xpack[(size_t)k_row0 * xs + (size_t)(b_base + c1)] : 0.f;
      xv[2] = (c2 < cols_this) ? Xpack[(size_t)k_row0 * xs + (size_t)(b_base + c2)] : 0.f;
      xv[3] = (c3 < cols_this) ? Xpack[(size_t)k_row0 * xs + (size_t)(b_base + c3)] : 0.f;
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
        const int c0 = col_group * 4 + 0;
        const int c1 = col_group * 4 + 1;
        const int c2 = col_group * 4 + 2;
        const int c3 = col_group * 4 + 3;
        const int xs = Bpad;

        xv[0] = (c0 < cols_this) ? Xpack[(size_t)k_row * xs + (size_t)(b_base + c0)] : 0.f;
        xv[1] = (c1 < cols_this) ? Xpack[(size_t)k_row * xs + (size_t)(b_base + c1)] : 0.f;
        xv[2] = (c2 < cols_this) ? Xpack[(size_t)k_row * xs + (size_t)(b_base + c2)] : 0.f;
        xv[3] = (c3 < cols_this) ? Xpack[(size_t)k_row * xs + (size_t)(b_base + c3)] : 0.f;
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
    const int sid     = slot_of(tx);
    if (sid >= 0) {
      float* __restrict__ out = dst + (size_t)sid * (size_t)out_stride;

      if (r0 < M) { float v = acc[0]; if (bias) v += bfloat16_to_float(bias[r0]); out[r0] = v; }
      if (r1 < M) { float v = acc[1]; if (bias) v += bfloat16_to_float(bias[r1]); out[r1] = v; }
      if (r2 < M) { float v = acc[2]; if (bias) v += bfloat16_to_float(bias[r2]); out[r2] = v; }
      if (r3 < M) { float v = acc[3]; if (bias) v += bfloat16_to_float(bias[r3]); out[r3] = v; }
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


__global__ void add_bias_kernel_float(float *output, const hip_bfloat16 *bias, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float ov = output[tid];
    float bv = bfloat16_to_float(bias[tid]);
    output[tid] = ov + bv;
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

__global__ void copy_kernel_float(float *dst, const float *src, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) dst[tid] = src[tid];
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

__global__ void softmax_kernel_float(float *x, int size) {
  __shared__ float max_val;
  __shared__ float sum_val;
  if (threadIdx.x == 0) {
    if (size <= 0) return;
    max_val = x[0];
    for (int i = 1; i < size; i++) {
      float v = x[i];
      if (v > max_val) max_val = v;
    }
    sum_val = 0.0f;
    for (int i = 0; i < size; i++) {
      float v = expf(x[i] - max_val);
      x[i] = v;
      sum_val += v;
    }
    float inv = 1.0f / sum_val;
    for (int i = 0; i < size; i++) {
      x[i] *= inv;
    }
  }
}

// Serial top-k selection on device for a single vector of length n.
// Keeps K best (value, index) pairs in descending order.
__global__ void topk_kernel_float(const float *scores, int n, int k, float *topv, int *topi) {
  if (threadIdx.x != 0) return;
  // Initialize outputs
  for (int i = 0; i < k; ++i) {
    topv[i] = -INFINITY;
    topi[i] = -1;
  }
  // One-pass insertion into top-k buffers
  for (int j = 0; j < n; ++j) {
    float v = scores[j];
    // find position where v should be inserted among current topk
    int pos = -1;
    for (int t = 0; t < k; ++t) {
      if (v > topv[t]) { pos = t; break; }
    }
    if (pos >= 0) {
      // shift down [pos..k-2] -> [pos+1..k-1]
      for (int t = k - 1; t > pos; --t) {
        topv[t] = topv[t - 1];
        topi[t] = topi[t - 1];
      }
      topv[pos] = v;
      topi[pos] = j;
    }
  }
}

__global__ void compute_cos_sin_kernel_float(float *cos_out, float *sin_out, int pos,
                                            float base, int head_dim, float scaling_factor,
                                            float initial_context_length, float ntk_beta, float ntk_alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim / 2;
  if (tid < d_half) {
    float freq = powf(base, ((float)(2 * tid)) / (float)head_dim);
    float inv_freq;
    if (scaling_factor > 1.0f) {
      float low  = d_half * logf(initial_context_length / (ntk_beta  * 2.0f * M_PI)) / logf(base);
      float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
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
    float val = (float)pos * inv_freq;
    cos_out[tid] = cosf(val) * concentration;
    sin_out[tid] = sinf(val) * concentration;
  }
}

__global__ void apply_rotary_emb_kernel_float(float *x, const float *cos, const float *sin,
                                             int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (h < n_heads && i < half) {
    float x1 = x[h * head_dim + i];
    float x2 = x[h * head_dim + half + i];
    float c  = cos[i];
    float s  = sin[i];
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    x[h * head_dim + i]           = o1;
    x[h * head_dim + half + i]    = o2;
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

__global__ void attention_scores_kernel_float(float *att, const float *q, const float *key_cache,
                                             int head_dim, int loff, int kv_dim, int kv_mul,
                                             int h, int pos, int seq_len, const float *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      float qv = q[i];
      float kv = k[i];
      score += qv * kv;
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      float mv = mask[pos * seq_len + t];
      score += mv;
    }
    att[t] = score;
  }
}

__global__ void attention_weighted_sum_kernel_float(float *tb, const float *att, const float *value_cache,
                                                   int head_dim, int loff, int kv_dim, int kv_mul,
                                                   int h, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < head_dim) {
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      float a  = att[t];
      float vv = v[i];
      sum += a * vv;
    }
    tb[h * head_dim + i] = sum;
  }
}

// Write attention sink value directly on device: reads BF16 sink and writes float
__global__ void add_attn_sink_from_bf16_kernel_float(float *att, int write_index,
                                                     const hip_bfloat16 *attn_sinks,
                                                     int layer, int n_attn_heads, int head) {
  if (threadIdx.x == 0) {
    size_t sink_idx = (size_t)layer * (size_t)n_attn_heads + (size_t)head;
    att[write_index] = bfloat16_to_float(attn_sinks[sink_idx]);
  }
}

__global__ void swiglu_kernel_float(float *gate_up, const float *gate, const float *up,
                                   int intermediate_dim, float swiglu_limit) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    float val   = gate[tid];
    float upval = up[tid];
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    gate_up[tid] = val;
  }
}

__global__ void split_gate_up_kernel_float(float *gate, float *up, const float *mlp1_out, int intermediate_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < intermediate_dim) {
    gate[tid] = mlp1_out[2 * tid];
    up[tid]   = mlp1_out[2 * tid + 1];
  }
}

__global__ void weighted_sum_kernel_float(float *e_agg, const float *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float agg = e_agg[tid];
    float add = tb2[tid];
    e_agg[tid] = agg + add * expert_w;
  }
}

// Atomic variant for parallel accumulation across experts/streams
__global__ void weighted_sum_atomic_kernel_float(float *e_agg, const float *tb2, float expert_w, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float add = tb2[tid] * expert_w;
    atomicAdd(&e_agg[tid], add);
  }
}

__global__ void residual_connection_kernel_float(float *x, const float *residual, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float xv  = x[tid];
    float rv  = residual[tid];
    x[tid] = xv + rv;
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

// Grouped MLP1: for each (sid,e) pair compute y = W_mlp1[e] * t[sid] + b_mlp1[e]
__global__ void mlp1_grouped_kernel_float(float *mlp1_out_all,
                                          const float *t,
                                          const uint8_t *w_mlp1_data,
                                          const uint8_t *w_mlp1_scales,
                                          const hip_bfloat16 *b_mlp1,
                                          const int *sel_sids, const int *sel_eids,
                                          int M, int hidden_dim, int intermediate_dim,
                                          int layer, int n_experts) {
  int m = blockIdx.y; // selection idx
  if (m >= M) return;

  // Cache input vector x for this selection into shared memory once per block
  extern __shared__ float x_shared[];

  int sid = sel_sids[m];
  int eid = sel_eids[m];
  const float *x = t + (size_t)sid * (size_t)hidden_dim;

  // Cooperative load of x into shared memory
  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    x_shared[j] = x[j];
  }
  __syncthreads();

  int o = blockIdx.x * blockDim.x + threadIdx.x; // output idx in [0, 2*intermediate_dim)
  int out_dim = 2 * intermediate_dim;
  if (o < out_dim) {
    size_t base = ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)out_dim * (size_t)hidden_dim;
    size_t row_start = base + (size_t)o * (size_t)hidden_dim;
    float sum = 0.0f;
    int j = 0;
    while (j < hidden_dim) {
      size_t elem_index = row_start + (size_t)j;
      uint8_t scale_byte = w_mlp1_scales[elem_index / MXFP4_BLOCK_SIZE];
      float scale = decode_e8m0(scale_byte);
      int block_offset = static_cast<int>(elem_index % MXFP4_BLOCK_SIZE);
      int span = hidden_dim - j;
      int remaining = MXFP4_BLOCK_SIZE - block_offset;
      if (span > remaining) span = remaining;
      for (int t_idx = 0; t_idx < span; ++t_idx) {
        size_t idx = elem_index + (size_t)t_idx;
        uint8_t packed = w_mlp1_data[idx >> 1];
        uint8_t nibble = (idx & 1ULL) ? (packed >> 4) : (packed & 0xF);
        float wv = decode_mxfp4(nibble) * scale;
        sum += wv * x_shared[j + t_idx];
      }
      j += span;
    }
    const hip_bfloat16 *b = b_mlp1 + ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)out_dim;
    mlp1_out_all[(size_t)m * (size_t)out_dim + (size_t)o] = sum + bfloat16_to_float(b[o]);
  }
}

// Split + SwiGLU batched for all selections
__global__ void swiglu_grouped_kernel_float(float *gate_up_all, const float *mlp1_out_all,
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
    gate_up_all[(size_t)m * (size_t)intermediate_dim + (size_t)i] = v;
  }
}

// Grouped MLP2 + accumulate: for each (sid,e) pair compute y = W_mlp2[e] * gate_up[m] + b_mlp2[e]
// and atomically accumulate into e_agg[sid]
__global__ void mlp2_accum_grouped_kernel_float(float *e_agg,
                                                const float *gate_up_all,
                                                const uint8_t *w_mlp2_data,
                                                const uint8_t *w_mlp2_scales,
                                                const hip_bfloat16 *b_mlp2,
                                                const int *sel_sids, const int *sel_eids, const float *sel_w,
                                                int M, int hidden_dim, int intermediate_dim,
                                                int layer, int n_experts, int eagg_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // output idx in [0, hidden_dim)
  int m = blockIdx.y;
  if (m < M && i < hidden_dim) {
    int sid = sel_sids[m];
    int eid = sel_eids[m];
    float ew = sel_w[m];
    const float *g = gate_up_all + (size_t)m * (size_t)intermediate_dim;
    size_t base = ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)hidden_dim * (size_t)intermediate_dim;
    size_t row_start = base + (size_t)i * (size_t)intermediate_dim;
    float sum = 0.0f;
    int j = 0;
    while (j < intermediate_dim) {
      size_t elem_index = row_start + (size_t)j;
      uint8_t scale_byte = w_mlp2_scales[elem_index / MXFP4_BLOCK_SIZE];
      float scale = decode_e8m0(scale_byte);
      int block_offset = static_cast<int>(elem_index % MXFP4_BLOCK_SIZE);
      int span = intermediate_dim - j;
      int remaining = MXFP4_BLOCK_SIZE - block_offset;
      if (span > remaining) span = remaining;
      for (int t_idx = 0; t_idx < span; ++t_idx) {
        size_t idx = elem_index + (size_t)t_idx;
        uint8_t packed = w_mlp2_data[idx >> 1];
        uint8_t nibble = (idx & 1ULL) ? (packed >> 4) : (packed & 0xF);
        float wv = decode_mxfp4(nibble) * scale;
        sum += wv * g[j + t_idx];
      }
      j += span;
    }
    const hip_bfloat16 *b = b_mlp2 + ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)hidden_dim;
    float y = sum + bfloat16_to_float(b[i]);
    atomicAdd(&e_agg[(size_t)sid * (size_t)eagg_stride + (size_t)i], ew * y);
  }
}

__global__ void bf16_to_float_kernel(float *out, const hip_bfloat16 *in, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    out[tid] = bfloat16_to_float(in[tid]);
  }
}

// FlashAttention-like single-sample kernel (used for testing/reference)
__global__ void flash_attention_kernel_optimized(
    const float *all_q,           // Q cho tất cả heads [n_heads * head_dim]
    const float *key_cache,       // K cache [layers * seq_len * kv_dim]
    const float *value_cache,     // V cache [layers * seq_len * kv_dim]
    const hip_bfloat16 *attn_sinks, // Attention sinks [layers * n_heads]
    float *all_output,            // Output [n_heads * head_dim]
    const float *mask,            // Sliding window mask (optional)
    int head_dim, int loff, int kv_dim, int kv_mul,
    int n_heads, int pos, int seq_len, int layer,
    bool can_add_sink,
    int tile_size) {              // Unused for now

  int h = blockIdx.y;  // Head index
  if (h >= n_heads) return;

  const float *q_head = all_q + (size_t)h * (size_t)head_dim;
  float *output = all_output + (size_t)h * (size_t)head_dim;

  // Shared memory buffer for attention scores
  extern __shared__ float shared_att[];
  
  // Phase 1: Compute attention scores
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float *k = key_cache + (size_t)loff + (size_t)t * (size_t)kv_dim + (size_t)(h / kv_mul) * (size_t)head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += q_head[i] * k[i];
    }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) {
      score += mask[(size_t)pos * (size_t)seq_len + (size_t)t];
    }
    shared_att[t] = score;
  }
  __syncthreads();

  // Phase 2: Add attention sink
  if (threadIdx.x == 0 && can_add_sink && (pos + 1) < seq_len) {
    size_t sink_idx = (size_t)layer * (size_t)n_heads + (size_t)h;
    shared_att[pos + 1] = bfloat16_to_float(attn_sinks[sink_idx]);
  }
  __syncthreads();

  // Phase 3: Softmax
  if (threadIdx.x == 0) {
    int len = can_add_sink ? (pos + 2) : (pos + 1);
    if (len > 0) {
      float max_val = shared_att[0];
      for (int i = 1; i < len; i++) max_val = fmaxf(max_val, shared_att[i]);
      float sum_val = 0.0f;
      for (int i = 0; i < len; i++) { 
        float v = expf(shared_att[i] - max_val); 
        shared_att[i] = v; 
        sum_val += v; 
      }
      float inv = 1.0f / (sum_val + 1e-8f); // Add epsilon for stability
      for (int i = 0; i < len; i++) shared_att[i] *= inv;
    }
  }
  __syncthreads();

  // Phase 4: Weighted sum with values
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float *v = value_cache + (size_t)loff + (size_t)t * (size_t)kv_dim + (size_t)(h / kv_mul) * (size_t)head_dim;
      sum += shared_att[t] * v[i];
    }
    output[i] = sum;
  }


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
// (Fallback multi-head attention and env-based FlashAttention config removed to simplify code.)


__global__ void sample_argmax_kernel(const float *logits, int *next_tokens, int batch, int vocab_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float *logit = logits + b * vocab_size;
    float max_val = logit[0];
    int max_idx = 0;
    for (int i = 1; i < vocab_size; ++i) {
        if (logit[i] > max_val) {
            max_val = logit[i];
            max_idx = i;
        }
    }
    next_tokens[b] = max_idx;
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

__global__ void logits_final_batched_kernel(const float *x,
                                          const hip_bfloat16 *weights_out,
                                          float *logits,
                                          int hidden_dim,
                                          int vocab_size,
                                          int batch,
                                          const int *slot_ids) {
    int v = blockIdx.x;  // vocab index
    int b = blockIdx.y;  // batch index
    
    if (v >= vocab_size || b >= batch) return;
    
    int sid = slot_ids[b];
    const float *x_slot = x + (size_t)sid * (size_t)hidden_dim;
    float *logits_slot = logits + (size_t)b * (size_t)vocab_size;
    
    // Use shared memory efficiently
    extern __shared__ float s_partial[];
    
    // Initialize partial sum
    float sum = 0.0f;
    
    // Each thread processes strided elements
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x_val = x_slot[i];
        float w_val = bfloat16_to_float(weights_out[v * hidden_dim + i]);
        sum += x_val * w_val;
    }
    
    // Store in shared memory
    s_partial[threadIdx.x] = sum;
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // First thread writes result for this vocab entry
    if (threadIdx.x == 0) {
        logits_slot[v] = s_partial[0];
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
  size_t packed_tiles_bytes_per_mat = ((Mpad * Kpad) + 1) / 2; // nibble-packed
  size_t total_mats = (size_t)n_layers * (size_t)ctx->local_experts;
  size_t total_packed_bytes = total_mats * packed_tiles_bytes_per_mat;
  uint8_t *w_mlp1_data_packed = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp1_data_packed), total_packed_bytes));
  HIP_CHECK(hipMemsetAsync(w_mlp1_data_packed, 0, total_packed_bytes, ws.quant_stream));

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
        dim3 grid((bytes + block.x - 1) / block.x);
        pack_mxfp4_row_to_tile16_kernel<<<grid, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp1_data + src_offset_bytes,
          w_mlp1_data_packed + dst_offset_bytes,
          rows, cols);
        HIP_CHECK(hipGetLastError());
      }
    }
    HIP_CHECK(hipStreamSynchronize(ws.quant_stream));
  }
  // Reuse w_mlp1_data pointer to refer to tile-packed layout to keep memory flat.
  HIP_CHECK(hipFree(gpu_w->w_mlp1_data));
  gpu_w->w_mlp1_data = w_mlp1_data_packed;
  // Note: scales remain in row-major block order; we index with original linear element indices.

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
    size_t packed_tiles_bytes_per_mat = ((Mpad2 * Kpad2) + 1) / 2; // nibble-packed
    size_t total_mats = (size_t)n_layers * (size_t)ctx->local_experts;
    size_t total_packed_bytes = total_mats * packed_tiles_bytes_per_mat;
    uint8_t *w_mlp2_data_packed = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&w_mlp2_data_packed), total_packed_bytes));
    HIP_CHECK(hipMemsetAsync(w_mlp2_data_packed, 0, total_packed_bytes, ws.quant_stream));

    dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < ctx->local_experts; ++e) {
        size_t mat_idx = (size_t)l * (size_t)ctx->local_experts + (size_t)e;
        size_t elems = rows * cols;
        size_t bytes = (elems + 1) / 2;
        size_t src_offset_bytes = ((size_t)l * (size_t)ctx->local_experts + (size_t)e) * bytes;
        size_t dst_offset_bytes = mat_idx * packed_tiles_bytes_per_mat;
        dim3 grid((unsigned int)((bytes + block.x - 1) / block.x));
        pack_mxfp4_row_to_tile16_kernel<<<grid, block, 0, ws.quant_stream>>>(
          gpu_w->w_mlp2_data + src_offset_bytes,
          w_mlp2_data_packed + dst_offset_bytes,
          rows, cols);
        HIP_CHECK(hipGetLastError());
      }
    }
    HIP_CHECK(hipStreamSynchronize(ws.quant_stream));
    HIP_CHECK(hipFree(gpu_w->w_mlp2_data));
    gpu_w->w_mlp2_data = w_mlp2_data_packed;
    // Note: scales stay row-major; kernel uses logical indices for scale lookup.
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

  HIP_CHECK(hipStreamSynchronize(ws.copy_stream));
  HIP_CHECK(hipStreamSynchronize(ws.convert_stream));
  HIP_CHECK(hipStreamSynchronize(ws.quant_stream));

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
    HIP_CHECK(hipStreamSynchronize(ws.convert_stream));
    add_log(name + "_packed16", total_elems * sizeof(hip_bfloat16));
  };

  // Pack QKV weights
  pack_weights_for_mfma(gpu_w->w_qkv, &gpu_w->w_qkv_packed16, n_layers, ctx->local_qkv_dim, hidden_dim, "w_qkv");
  
  // Pack W_o weights  
  pack_weights_for_mfma(gpu_w->w_o, &gpu_w->w_o_packed16, n_layers, hidden_dim, ctx->local_q_dim, "w_o");
  
  // Pack router weights (only on leader)
  if (ctx->is_leader) {
    pack_weights_for_mfma(gpu_w->w_router, &gpu_w->w_router_packed16, n_layers, p->n_experts, hidden_dim, "w_router");
  } else {
    gpu_w->w_router_packed16 = nullptr;
  }

  if (!ws.logs.empty()) {
    printf("[Weights][GPU%d] copy summary:\\n", ctx->device_id);
    for (const auto &entry : ws.logs) {
      printf("  %-22s %.3f GB\\n", entry.first.c_str(), to_gb(entry.second));
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

  std::vector<GPUContext*> ctxs;
  ctxs.reserve(parallel_layout.tp_size);
  for (int r = 0; r < parallel_layout.tp_size; ++r) {
    ctxs.push_back(&gpu_contexts[r]);
  }
  GPUContext *leader = ctxs[0];
  (void)gpu_ctx; // leader orchestrates shards; incoming pointer kept for compatibility

  // Stage indices to every GPU
  for (GPUContext *ctx : ctxs) {
    HIP_CHECK(hipSetDevice(ctx->device_id));
    HIP_CHECK(hipMemcpyAsync(ctx->d_slot_ids,  h_slot_ids,  batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
    HIP_CHECK(hipMemcpyAsync(ctx->d_pos_list,  h_pos_list,  batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
    HIP_CHECK(hipMemcpyAsync(ctx->d_token_ids, h_tokens,    batch * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
    HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
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
    HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
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
    HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
  }

  std::vector<size_t> kv_block_sizes(ctxs.size());
  for (size_t idx = 0; idx < ctxs.size(); ++idx) {
    GPUContext *ctx = ctxs[idx];
    kv_block_sizes[idx] = (size_t)p->n_layers * (size_t)p->seq_len * (size_t)ctx->local_kv_dim;
  }

  struct KVWriterV {
    static __global__ void run(const float *qkv, float *value_cache, int batch, int q_stride,
                               int v_offset, int v_size, int kv_dim, int loff,
                               size_t kv_block_size, const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < v_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        const float *src = qkv + (size_t)sid * (size_t)q_stride + (size_t)v_offset;
        float *dst = value_cache + (size_t)sid * kv_block_size + (size_t)loff + (size_t)pos * (size_t)kv_dim;
        dst[i] = src[i];
      }
    }
  };
  struct KVWriterK {
    static __global__ void run(const float *qkv, float *key_cache, int batch, int q_stride,
                               int k_offset, int k_size, int kv_dim, int loff,
                               size_t kv_block_size, const int *pos_list, const int *slot_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int b = blockIdx.y;
      if (i < k_size && b < batch) {
        int sid = slot_ids[b];
        int pos = pos_list[b];
        const float *src = qkv + (size_t)sid * (size_t)q_stride + (size_t)k_offset;
        float *dst = key_cache + (size_t)sid * kv_block_size + (size_t)loff + (size_t)pos * (size_t)kv_dim;
        dst[i] = src[i];
      }
    }
  };

  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (attention)
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      const hip_bfloat16 *rms_attn_w = ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
      for (int bi = 0; bi < batch; ++bi) {
        int sid = h_slot_ids[bi];
        dim3 block(256);
        dim3 grid((hidden_dim + block.x - 1) / block.x);
        rmsnorm_kernel_float<<<grid, block, 0, ctx->batch_stream>>>(
          ctx->state.t + (size_t)sid * (size_t)hidden_dim,
          ctx->state.x + (size_t)sid * (size_t)hidden_dim,
          rms_attn_w, hidden_dim);
      }
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
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
      
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // Cache write + RoPE per shard
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      int loff = l * p->seq_len * ctx->local_kv_dim;
      int q_stride = ctx->local_qkv_dim;
      int q_size = ctx->local_q_dim;
      int k_size = ctx->local_kv_dim;
      int v_size = ctx->local_kv_dim;
      size_t kv_block = kv_block_sizes[idx];

      dim3 block_v(256), grid_v((v_size + block_v.x - 1) / block_v.x, batch);
      KVWriterV::run<<<grid_v, block_v, 0, ctx->batch_stream>>>(ctx->state.qkv, ctx->state.value_cache,
        batch, q_stride, q_size + k_size, v_size, ctx->local_kv_dim, loff, kv_block, ctx->d_pos_list, ctx->d_slot_ids);

      dim3 grid_heads(ctx->local_q_heads, batch);
      apply_rotary_emb_kernel_float_batched<<<grid_heads, head_dim / 2, 0, ctx->batch_stream>>>(
        ctx->state.qkv, ctx->state.cos_vals, ctx->state.sin_vals,
        ctx->local_q_heads, head_dim, batch, q_stride, head_dim / 2, ctx->d_slot_ids);

      dim3 grid_heads_k(ctx->local_kv_heads, batch);
      apply_rotary_emb_kernel_float_batched<<<grid_heads_k, head_dim / 2, 0, ctx->batch_stream>>>(
        ctx->state.qkv + q_size, ctx->state.cos_vals, ctx->state.sin_vals,
        ctx->local_kv_heads, head_dim, batch, q_stride, head_dim / 2, ctx->d_slot_ids);

      dim3 block_k(256), grid_k((k_size + block_k.x - 1) / block_k.x, batch);
      KVWriterK::run<<<grid_k, block_k, 0, ctx->batch_stream>>>(ctx->state.qkv, ctx->state.key_cache,
        batch, q_stride, q_size, k_size, ctx->local_kv_dim, loff, kv_block, ctx->d_pos_list, ctx->d_slot_ids);

      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // Flash attention per shard
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      HIP_CHECK(hipSetDevice(ctx->device_id));
      bool can_add_sink = true;
      const float *layer_mask = (p->sliding_window > 0 && (l % 2 == 0)) ? ctx->state.mask : NULL;
      size_t shared_mem_bytes = (size_t)p->seq_len * sizeof(float);
      dim3 block(256);
      dim3 grid(1, ctx->local_q_heads, batch);
      flash_attention_kernel_optimized_batched<<<grid, block, shared_mem_bytes, ctx->batch_stream>>>(
        ctx->state.qkv, ctx->state.key_cache, ctx->state.value_cache, ctx->weights.attn_sinks,
        ctx->state.tb, layer_mask, head_dim, l * p->seq_len * ctx->local_kv_dim, ctx->local_kv_dim, kv_mul,
        ctx->local_q_heads, p->seq_len, l, ctx->d_pos_list, can_add_sink, batch,
        ctx->local_qkv_dim, (int)kv_block_sizes[idx], ctx->d_slot_ids, 0);
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // Output projection without bias using MFMA
    for (GPUContext *ctx : ctxs) {
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
      
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // Sum partial tb2 on leader and broadcast
    if (ctxs.size() > 1) {
      HIP_CHECK(hipSetDevice(leader->device_id));
      HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
      size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
      for (size_t idx = 1; idx < ctxs.size(); ++idx) {
        GPUContext *peer = ctxs[idx];
        HIP_CHECK(hipSetDevice(peer->device_id));
        HIP_CHECK(hipStreamSynchronize(peer->batch_stream));
        HIP_CHECK(hipMemcpyPeer(leader->state.tb2_peer, leader->device_id,
                                peer->state.tb2, peer->device_id, bytes));
        HIP_CHECK(hipSetDevice(leader->device_id));
        dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
        add_selected_kernel_float<<<grid, block, 0, leader->batch_stream>>>(
          leader->state.tb2, leader->state.tb2_peer, hidden_dim, batch, leader->d_slot_ids);
        HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
      }
    }

    // Add bias once on leader and broadcast
    HIP_CHECK(hipSetDevice(leader->device_id));
    {
      const hip_bfloat16 *b_o = leader->weights.b_o + (size_t)l * (size_t)hidden_dim;
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      add_bias_kernel_float_batched<<<grid, block, 0, leader->batch_stream>>>(
        leader->state.tb2, b_o, hidden_dim, batch, hidden_dim, leader->d_slot_ids);
      HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
    }

    if (ctxs.size() > 1) {
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
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // MoE RMSNorm per slot
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      const hip_bfloat16 *rms_ffn_w = ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
      for (int bi = 0; bi < batch; ++bi) {
        int sid = h_slot_ids[bi];
        dim3 block(256), grid((hidden_dim + block.x - 1) / block.x);
        rmsnorm_kernel_float<<<grid, block, 0, ctx->batch_stream>>>(
          ctx->state.t + (size_t)sid * (size_t)hidden_dim,
          ctx->state.x + (size_t)sid * (size_t)hidden_dim,
          rms_ffn_w, hidden_dim);
      }
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
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
      HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
    }

    // Zero MoE accumulation buffers on every shard
    for (GPUContext *ctx : ctxs) {
      HIP_CHECK(hipSetDevice(ctx->device_id));
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      zero_selected_slots_kernel_float<<<grid, block, 0, ctx->batch_stream>>>(
        ctx->state.e_agg, hidden_dim, ctx->d_slot_ids, batch);
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
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
      HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
    }

    std::vector<int> host_sel_sids(M);
    std::vector<int> host_sel_eids(M);
    std::vector<float> host_sel_w(M);
    HIP_CHECK(hipMemcpy(host_sel_sids.data(), leader->state.sel_sids, M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_sel_eids.data(), leader->state.sel_eids, M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_sel_w.data(), leader->state.sel_w, M * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<int> local_M(ctxs.size(), 0);
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
      local_M[idx] = (int)sel_sids_local.size();
      HIP_CHECK(hipSetDevice(ctx->device_id));
      if (!sel_sids_local.empty()) {
        HIP_CHECK(hipMemcpyAsync(ctx->state.sel_sids, sel_sids_local.data(), sel_sids_local.size() * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
        HIP_CHECK(hipMemcpyAsync(ctx->state.sel_eids, sel_eids_local.data(), sel_eids_local.size() * sizeof(int), hipMemcpyHostToDevice, ctx->batch_stream));
        HIP_CHECK(hipMemcpyAsync(ctx->state.sel_w, sel_w_local.data(), sel_w_local.size() * sizeof(float), hipMemcpyHostToDevice, ctx->batch_stream));
      }
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // Grouped MoE compute per shard
    for (size_t idx = 0; idx < ctxs.size(); ++idx) {
      GPUContext *ctx = ctxs[idx];
      int M_local = local_M[idx];
      if (M_local == 0) continue;
      HIP_CHECK(hipSetDevice(ctx->device_id));
      // Launch MFMA matvec over 16-row tiles in output dimension
      const int out_dim = 2 * p->intermediate_dim;
      dim3 block1(16,4,1); // 64 threads per block for 16x16x4 MFMA
      dim3 grid1((out_dim + 15) / 16, M_local, 1);
      mlp1_grouped_mfma_matvec_mxfp4_kernel<<<grid1, block1, 0, ctx->batch_stream>>>(
        ctx->state.mlp1_out_all, ctx->state.t,
        ctx->weights.w_mlp1_data, ctx->weights.w_mlp1_scales,
        ctx->weights.b_mlp1,
        ctx->state.sel_sids, ctx->state.sel_eids,
        M_local, hidden_dim, out_dim,
        l, ctx->local_experts);

      dim3 block2(256), grid2((p->intermediate_dim + block2.x - 1) / block2.x, M_local);
      swiglu_grouped_kernel_float<<<grid2, block2, 0, ctx->batch_stream>>>(
        ctx->state.gate_up_all, ctx->state.mlp1_out_all,
        M_local, p->intermediate_dim, p->swiglu_limit);

      // Launch MFMA MLP2 accumulation over 16-row tiles in hidden_dim
      dim3 block3(16,4,1);
      dim3 grid3((hidden_dim + 15) / 16, M_local, 1);
      mlp2_accum_grouped_mfma_matvec_mxfp4_kernel<<<grid3, block3, 0, ctx->batch_stream>>>(
        ctx->state.e_agg, ctx->state.gate_up_all,
        ctx->weights.w_mlp2_data, ctx->weights.w_mlp2_scales,
        ctx->weights.b_mlp2,
        ctx->state.sel_sids, ctx->state.sel_eids, ctx->state.sel_w,
        M_local, hidden_dim, p->intermediate_dim,
        l, ctx->local_experts, hidden_dim);
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
    }

    // Sum e_agg on leader and broadcast
    if (ctxs.size() > 1) {
      HIP_CHECK(hipSetDevice(leader->device_id));
      HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
      size_t bytes = (size_t)leader->max_batch * (size_t)hidden_dim * sizeof(float);
      for (size_t idx = 1; idx < ctxs.size(); ++idx) {
        GPUContext *peer = ctxs[idx];
        HIP_CHECK(hipSetDevice(peer->device_id));
        HIP_CHECK(hipStreamSynchronize(peer->batch_stream));
        HIP_CHECK(hipMemcpyPeer(leader->state.e_agg_peer, leader->device_id,
                                peer->state.e_agg, peer->device_id, bytes));
        HIP_CHECK(hipSetDevice(leader->device_id));
        dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
        add_selected_kernel_float<<<grid, block, 0, leader->batch_stream>>>(
          leader->state.e_agg, leader->state.e_agg_peer, hidden_dim, batch, leader->d_slot_ids);
        HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
      }
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
      HIP_CHECK(hipStreamSynchronize(ctx->batch_stream));
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

    dim3 block_logits(128);
    dim3 grid_logits(p->vocab_size, batch);
    size_t shared_logits = block_logits.x * sizeof(float);
    logits_final_batched_kernel<<<grid_logits, block_logits, shared_logits, leader->batch_stream>>>(
      leader->state.x, leader->weights.out,
      leader->state.logits_fp32, hidden_dim,
      p->vocab_size, batch, leader->d_slot_ids);

    dim3 block_argmax(64);
    dim3 grid_argmax((batch + block_argmax.x - 1) / block_argmax.x);
    sample_argmax_kernel<<<grid_argmax, block_argmax, 0, leader->batch_stream>>>(
      leader->state.logits_fp32, leader->d_next_tokens, batch, p->vocab_size);

    HIP_CHECK(hipMemcpyAsync(host_next_tokens_out, leader->d_next_tokens,
                             batch * sizeof(int), hipMemcpyDeviceToHost, leader->batch_stream));
    HIP_CHECK(hipStreamSynchronize(leader->batch_stream));
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
}

// ------------------------- Warm-up & Finish -------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "No HIP devices found.\n");
    exit(EXIT_FAILURE);
  }

  Config *cfg = &transformer->config;

  int desired_tp = device_count;
  const char *env_tp = getenv("GETP_TP_SIZE");
  if (env_tp && atoi(env_tp) > 0) desired_tp = atoi(env_tp);
  if (desired_tp < 1) desired_tp = 1;
  if (desired_tp > device_count) desired_tp = device_count;

  int desired_ep = desired_tp;
  const char *env_ep = getenv("GETP_EP_SIZE");
  if (env_ep && atoi(env_ep) > 0) desired_ep = atoi(env_ep);
  if (desired_ep < 1) desired_ep = 1;
  if (desired_ep > device_count) desired_ep = device_count;

  if (desired_tp != desired_ep) {
    fprintf(stderr, "[GPU] Only matching TP/EP sizes are supported (got TP=%d, EP=%d).\n",
            desired_tp, desired_ep);
    exit(EXIT_FAILURE);
  }

  parallel_layout.tp_size = desired_tp;
  parallel_layout.ep_size = desired_ep;
  int used_devices = parallel_layout.tp_size;

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

  auto load_mask_if_needed = [&](GPUContext &ctx) {
    if (cfg->sliding_window <= 0 || ctx.state.mask == NULL) return;
    int S = cfg->seq_len;
    std::vector<float> cpu_mask((size_t)S * (size_t)S);
    for (int i = 0; i < S; i++) {
      for (int j = 0; j < S; j++) {
        cpu_mask[(size_t)i * (size_t)S + (size_t)j] = (i - j >= cfg->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(ctx.state.mask, cpu_mask.data(), (size_t)S * (size_t)S * sizeof(float), hipMemcpyHostToDevice));
  };

  for (int d = 0; d < used_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    GPUContext &ctx = gpu_contexts[d];
    ctx.device_id = d;
    ctx.tp_rank = d;
    ctx.tp_size = parallel_layout.tp_size;
    ctx.ep_rank = d;
    ctx.ep_size = parallel_layout.ep_size;
    ctx.head_offset_q = d * local_q_heads;
    ctx.head_offset_kv = d * local_kv_heads;
    ctx.expert_offset = d * local_experts;
    ctx.local_q_heads = local_q_heads;
    ctx.local_kv_heads = local_kv_heads;
    ctx.local_experts = local_experts;
    ctx.local_q_dim = ctx.local_q_heads * head_dim;
    ctx.local_kv_dim = ctx.local_kv_heads * head_dim;
    ctx.local_qkv_dim = ctx.local_q_dim + 2 * ctx.local_kv_dim;
    ctx.is_leader = (d == 0);

    const char *env_mb = getenv("GETP_MAX_BATCH");
    int mb = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 8;
    ctx.max_batch = mb;

    WeightCopyWorkspace &ws = ctx.copy_ws;
    size_t stage_mb = 256;
    const char *env_stage = getenv("GETP_STAGE_MB");
    if (env_stage && atoi(env_stage) > 0) stage_mb = (size_t)atoi(env_stage);
    ws.capacity_elems = (stage_mb * (size_t)1024 * (size_t)1024) / sizeof(float);
    if (ws.capacity_elems == 0) ws.capacity_elems = (size_t)65536;
    size_t stage_bytes = ws.capacity_elems * sizeof(float);
    HIP_CHECK(hipStreamCreateWithFlags(&ws.copy_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&ws.convert_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&ws.quant_stream, hipStreamNonBlocking));
    ws.logs.clear();
    ws.logs.reserve(32);
    for (int i = 0; i < 2; ++i) {
      HIP_CHECK(hipHostMalloc((void**)&ws.host_stage[i], stage_bytes));
      HIP_CHECK(hipMalloc((void**)&ws.device_stage[i], stage_bytes));
      HIP_CHECK(hipEventCreateWithFlags(&ws.copy_ready[i], hipEventDisableTiming));
      HIP_CHECK(hipEventCreateWithFlags(&ws.stage_free[i], hipEventDisableTiming));
    }
    for (int i = 0; i < 2; ++i) {
      HIP_CHECK(hipEventRecord(ws.stage_free[i], ws.quant_stream));
    }

    gpu_copy_weights(&ctx, &transformer->weights, cfg);
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

    ctx.num_expert_streams = MAX_EXPERT_STREAMS;
    for (int s = 0; s < ctx.num_expert_streams; ++s) {
      HIP_CHECK(hipStreamCreate(&ctx.expert_streams[s]));
    }

    load_mask_if_needed(ctx);
  }

  if (parallel_layout.tp_size > 1) {
    for (int src = 0; src < parallel_layout.tp_size; ++src) {
      HIP_CHECK(hipSetDevice(src));
      for (int dst = 0; dst < parallel_layout.tp_size; ++dst) {
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

  printf("[GPU] Warm-up done on %d device(s). BF16 weights loaded to VRAM with TP=%d, EP=%d.\n",
         used_devices, parallel_layout.tp_size, parallel_layout.ep_size);
  printf("[Attention] Using FlashAttention-like optimized kernel (always on)\n");
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
  const int num_devices = 1; // use leader context to orchestrate tensor/expert parallel shard set
  // Global request index for all workers (lock-free producer)
  std::atomic<int> next_idx{0};

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  // Continuous batching per GPU worker
  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[0].device_id));
      GPUContext *ctx = &gpu_contexts[0];
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




void gpu_malloc_run_state(GPUContext *ctx, Config *p, int max_batch) {
  GPURunState *s = &ctx->state;
  size_t B = (size_t)max_batch;
  int hidden_dim = p->hidden_dim;
  int local_q_dim = ctx->local_q_dim;
  int local_kv_dim = ctx->local_kv_dim;
  int local_qkv_dim = ctx->local_qkv_dim;

  HIP_CHECK(hipMalloc(&s->x,   B * (size_t)hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->t,   B * (size_t)hidden_dim * sizeof(float)));
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
  HIP_CHECK(hipMalloc(&s->q,   B * (size_t)local_q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->k,   B * (size_t)local_kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->v,   B * (size_t)local_kv_dim * sizeof(float)));

  size_t kv_cache_elems = (size_t)max_batch * (size_t)p->n_layers * (size_t)p->seq_len * (size_t)local_kv_dim;
  HIP_CHECK(hipMalloc(&s->key_cache,   kv_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->value_cache, kv_cache_elems * sizeof(float)));

  if (ctx->is_leader) {
    HIP_CHECK(hipMalloc(&s->logits,      B * (size_t)p->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));
  } else {
    s->logits = nullptr;
    s->logits_fp32 = nullptr;
  }

  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(float)));
  } else {
    s->mask = NULL;
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
  HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)p->intermediate_dim * sizeof(float)));

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
  GPURunState *s = &ctx->state;
  HIP_CHECK(hipFree(s->x));
  HIP_CHECK(hipFree(s->t));
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
  HIP_CHECK(hipFree(s->q));
  HIP_CHECK(hipFree(s->k));
  HIP_CHECK(hipFree(s->v));
  HIP_CHECK(hipFree(s->key_cache));
  HIP_CHECK(hipFree(s->value_cache));
  if (s->logits) HIP_CHECK(hipFree(s->logits));
  if (s->mask) HIP_CHECK(hipFree(s->mask));
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
  HIP_CHECK(hipFree(s->mlp1_out_all));
  HIP_CHECK(hipFree(s->gate_up_all));
  if (s->tb2_peer) HIP_CHECK(hipFree(s->tb2_peer));
  if (s->e_agg_peer) HIP_CHECK(hipFree(s->e_agg_peer));
  HIP_CHECK(hipFree(s->xpack_f32));
}

void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  HIP_CHECK(hipFree(gpu_w->token_embedding_table));
  HIP_CHECK(hipFree(gpu_w->out));
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
