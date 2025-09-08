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


// 128-bit alias cho load/stor 16B
using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;



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
  hip_bfloat16 *w_mlp1;
  hip_bfloat16 *w_mlp2;
  hip_bfloat16 *b_mlp1;
  hip_bfloat16 *b_mlp2;
  // Packed (16x16 tiling) BF16 weights for GEMM core (convert to FP32 per tile in shared memory)
  hip_bfloat16 *w_qkv_packed16;   // [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_o_packed16;     // [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_router_packed16;// [n_layers][Mpad * Kpad]
  hip_bfloat16 *w_mlp1_packed16;  // [n_layers*n_experts][Mpad * Kpad]
  hip_bfloat16 *w_mlp2_packed16;  // [n_layers*n_experts][Mpad * Kpad] for GEMM-based MLP2
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
  // Packed input buffer for GEMM (FP32), size: K_max x Bpad
  float *xpack_f32;
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
  // Expert-grouped MoE staging
  int   *sel_sids_grouped;   // grouped by expert
  int   *sel_eids_grouped;   // grouped by expert
  float *sel_w_grouped;      // grouped by expert
  int   *eid_counts;         // [n_experts]
  int   *eid_offsets;        // [n_experts + 1]
  int   *eid_cursors;        // [n_experts] for scatter
  // --- New: Block-aligned MoE staging for MLP1 ---
  int   *sorted_indices;     // [M_pad]  index m gốc (hoặc -1 cho pad)
  int   *sorted_sids;        // [M_pad]  sid theo thứ tự đã sắp xếp + đệm
  int   *sorted_eids;        // [M_pad]  eid theo thứ tự đã sắp xếp + đệm
  float *sorted_w;           // [M_pad]  trọng số top-k; 0.0f cho pad
  int   *padded_offsets;     // [n_experts + 1] prefix-sum sau khi align block
  int   *block_expert_ids;   // [M_pad / BLOCK_B] expert id cho mỗi block  (BLOCK_B=16)
  int   *d_col_ids;          // [M_pad]  iota: 0..M_pad-1 dùng để scatter output theo cột
  float *mlp1_Ypack;         // [M_out x M_pad] với M_out = 2*intermediate_dim (layout: col-major theo "slot_id")

} GPURunState;

// Per-GPU device context
typedef struct {
  int device_id;
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

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __host__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
}
__host__ __device__ __forceinline__ int round_up16(int x) { return ((x + 15) / 16) * 16; }

// ------------------------- Kernels -------------------------

// FP32 -> BF16 (device)
__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
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

// Fused MLP1 (matvec) + SwiGLU for a single expert group.
// For each selection m (sid), compute y = W[layer,eid] @ x[sid] + b[layer,eid]
// then apply SwiGLU to produce gate_up (size = intermediate_dim).
// W is packed as 16x16 BF16 tiles (Mpad x Kpad). Accumulate in FP32.
__global__ __launch_bounds__(64)
void mlp1_swiglu_grouped_mfma_matvec_kernel(
    float* __restrict__ gate_up_out,       // [M selections][intermediate_dim]
    const float* __restrict__ t,           // [B slots][hidden_dim]
    const hip_bfloat16* __restrict__ w_packed, // [n_layers*n_experts][Mpad*Kpad]
    const hip_bfloat16* __restrict__ b_mlp1,   // [n_layers*n_experts][2*intermediate_dim]
    const int* __restrict__ sel_sids,      // [M]
    int selections_M,
    int hidden_dim, int intermediate_dim,
    int layer, int n_experts, int eid,
    int Mpad, int Kpad,
    float swiglu_limit)
{
  const int tx = threadIdx.x;  // 0..15
  const int ty = threadIdx.y;  // 0..3
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;  // along output rows
  const int msel   = blockIdx.y;  // selection index within this expert group
  if (msel >= selections_M) return;

  const int base_row = tile_m * 16;
  const int out_dim = 2 * intermediate_dim;

  // Selection sid (slot id)
  const int sid = sel_sids[msel];

  // Per-(layer,eid) base pointers
  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W = w_packed + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = b_mlp1 + mat_index * (size_t)out_dim;

  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float        sWf[16 * 16];
  __shared__ __align__(16) float        sX[16 * 16];
  __shared__ __align__(16) float        sY[16];       // store accumulated rows for this 16x1 tile

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  const int K = hidden_dim;
  const int Ktiles = (Kpad + 15) / 16;

  for (int kk = 0; kk < Ktiles; ++kk) {
    // Load W tile [16x16] packed BF16
    if (lane < 32) {
      const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W + w_tile_base;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
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

    // Convert W BF16 -> FP32 in shared
    {
      int idx4 = lane * 4;
      if (idx4 < 256) {
        sWf[idx4 + 0] = bfloat16_to_float(sWb[idx4 + 0]);
        sWf[idx4 + 1] = bfloat16_to_float(sWb[idx4 + 1]);
        sWf[idx4 + 2] = bfloat16_to_float(sWb[idx4 + 2]);
        sWf[idx4 + 3] = bfloat16_to_float(sWb[idx4 + 3]);
      }
    }

    // 4 micro-steps across the 16-wide K tile
    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4; // 0..15
      float amk = sWf[tx * 16 + k4];   // A: row=tx, col=k4
      float bkn = sX[k4 * 16 + tx];    // B: row=k4, col=tx
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }

    __syncthreads();
  }

  // Store accumulators into shared rows and then fuse SwiGLU on even rows
  if (tx == 0) {
    // Write accum rows for this tile
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int rloc = ty * 4 + i;     // 0..15
      const int row  = base_row + rloc;
      if (row < out_dim) {
        sY[rloc] = dmn[i];
      }
    }
  }

  __syncthreads();

  if (tx == 0) {
    const float alpha = 1.702f;
    float* out_vec = gate_up_out + (size_t)msel * (size_t)intermediate_dim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int rloc = ty * 4 + i;     // 0..15
      const int row  = base_row + rloc; // 0..out_dim-1
      if ((row & 1) == 0) { // even -> gate
        int gate_idx = row >> 1;
        if (gate_idx < intermediate_dim && (row + 1) < out_dim) {
          float gate = sY[rloc];
          float up   = sY[rloc + 1];
          // Add bias
          if (B) {
            gate += bfloat16_to_float(B[row]);
            up   += bfloat16_to_float(B[row + 1]);
          }
          // Clamp
          if (gate >  swiglu_limit) gate =  swiglu_limit;
          if (up   >  swiglu_limit) up   =  swiglu_limit;
          if (up   < -swiglu_limit) up   = -swiglu_limit;
          // SwiGLU
          gate *= (1.0f / (1.0f + expf(-alpha * gate)));
          float v = gate * (up + 1.0f);
          out_vec[gate_idx] = v;
        }
      }
    }
  }
}
// Grouped MLP2 matvec using MFMA and packed BF16 weights with accumulation into e_agg.
// For each selection m (sid,eid), compute y = W_mlp2[layer,eid] @ gate_up_all[m] + b_mlp2[layer,eid]
// then atomically accumulate e_agg[sid] += sel_w[m] * y.
// W is packed as 16x16 BF16 tiles (Mpad x Kpad). Accumulate in FP32.
// Implementation replicates the input vector across 16 columns; only tx==0 stores results.
// Same as above, but atomically accumulates directly into x (residual add fused).
__global__ __launch_bounds__(64)
void mlp2_grouped_mfma_matvec_residual_kernel(
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
    int Mpad, int Kpad)
{
  const int tx = threadIdx.x;  // 0..15 (columns)
  const int ty = threadIdx.y;  // 0..3  (row groups)
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;  // along hidden_dim
  const int msel   = blockIdx.y;  // column id in sorted/padded arrays
  if (msel >= Btotal) return;

  const int base_row = tile_m * 16;

  // Resolve selection sid/eid and expert weight; skip padding
  const int valid_idx = sorted_indices[msel];
  if (valid_idx < 0) return;
  const int sid = sorted_sids[msel];
  const int eid = sorted_eids[msel];
  const float ew = sorted_w[msel];

  // Pointers per (layer,eid)
  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W = w_packed_mlp2 + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = b_mlp2 + mat_index * (size_t)hidden_dim;

  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float        sWf[16 * 16];
  __shared__ __align__(16) float        sX[16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  const int K = intermediate_dim;
  const int Ktiles = (Kpad + 15) / 16;

  for (int kk = 0; kk < Ktiles; ++kk) {
    if (lane < 32) {
      const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W + w_tile_base;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
    }

    if (lane < 64) {
      const int r   = lane % 16;
      const int c4  = (lane / 16) * 4;    // 0,4,8,12
      const int k_row = kk * 16 + r;
      float* dst_vec = sX + r * 16 + c4;
      float val = 0.0f;
      if (k_row < K) {
        const float* g = gate_up_all + (size_t)msel * (size_t)intermediate_dim;
        val = g[k_row];
      }
      float4v vrep = {val, val, val, val};
      reinterpret_cast<float4v*>(dst_vec)[0] = vrep;
    }

    __syncthreads();

    int idx4 = lane * 4;
    if (idx4 < 256) {
      sWf[idx4 + 0] = bfloat16_to_float(sWb[idx4 + 0]);
      sWf[idx4 + 1] = bfloat16_to_float(sWb[idx4 + 1]);
      sWf[idx4 + 2] = bfloat16_to_float(sWb[idx4 + 2]);
      sWf[idx4 + 3] = bfloat16_to_float(sWb[idx4 + 3]);
    }

    __syncthreads();

    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4;
      float amk = sWf[tx * 16 + k4];
      float bkn = sX[k4 * 16 + tx];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }

    __syncthreads();
  }

  if (tx == 0) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = base_row + (ty * 4 + i);
      if (row < hidden_dim) {
        const float biasv = B ? bfloat16_to_float(B[row]) : 0.f;
        float y = dmn[i] + biasv;
        atomicAdd(&x[(size_t)sid * (size_t)x_stride + (size_t)row], ew * y);
      }
    }
  }
}

// ========================= FLOAT KERNELS =========================



// ------------------------- New BF16 GEMM path (packed 16x16) -------------------------



// GEMM MFMA FP32: W_packed [Mpad x Kpad] (tile 16x16 contiguous), Xpack [K x Bpad] (row-major K, stride=Bpad)
// Accumulate FP32, add BF16 bias, and scatter to per-slot outputs via slot_ids.
// Each block computes one 16x16 tile with 64 threads (dim3 block(16,4,1)).
__global__ __launch_bounds__(64)
void gemm16x16_mfma_f32_packedW_batched_scatter_bias(
    const hip_bfloat16* __restrict__ W_packed,
    const float* __restrict__ Xpack,
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

  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float sWf[16 * 16];
  __shared__ __align__(16) float sX[16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  for (int kk = 0; kk < Ktiles; ++kk) {
    // Load W tile BF16: 16x16 contiguous (256 bf16 -> 512B). 32 lanes copy 16 bytes each.
    if (lane < 32) {
      const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256; // 256 elements (bf16)
      const hip_bfloat16* w_src = W_packed + w_tile_base;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
    }

    // Load X tile FP32: 16x16 floats (64 lanes copy 16B each = 4 floats)
    if (lane < 64) {
      const int r   = lane % 16;          // row within tile
      const int c4  = (lane / 16) * 4;    // column block of 4 floats: 0,4,8,12
      const int k_row = kk * 16 + r;      // actual K row

      float* dst_vec = sX + r * 16 + c4;

      const bool in_k = (k_row < K);
      const bool in_b = (base_col + c4) < Bpad;

      if (in_k && in_b) {
        const float* src_vec = Xpack + (size_t)k_row * (size_t)Bpad + (size_t)base_col + (size_t)c4;
        reinterpret_cast<u32x4*>(dst_vec)[0] = reinterpret_cast<const u32x4*>(src_vec)[0];
      } else {
        reinterpret_cast<u32x4*>(dst_vec)[0] = u32x4{0u,0u,0u,0u};
      }
    }

    __syncthreads();

    // Convert W tile BF16 -> FP32 in shared memory (64 lanes x 4 elems)
    {
      int idx4 = lane * 4;
      if (idx4 < 256) {
        sWf[idx4 + 0] = bfloat16_to_float(sWb[idx4 + 0]);
        sWf[idx4 + 1] = bfloat16_to_float(sWb[idx4 + 1]);
        sWf[idx4 + 2] = bfloat16_to_float(sWb[idx4 + 2]);
        sWf[idx4 + 3] = bfloat16_to_float(sWb[idx4 + 3]);
      }
    }

    // Iterate 4 micro-steps over the 16-wide K tile using MFMA 16x16x4 f32
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k4 = ty + 4 * t; // 0..15
      // A: row = tx (M), col = k4
      float amk = sWf[tx * 16 + k4];
      // B: row = k4, col = tx (N)
      float bkn = sX[k4 * 16 + tx];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }

    __syncthreads();
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
  // Thread mapping same as MFMA 16x16x4f32 kernel: blockDim(16,4,1)
  const int tx = threadIdx.x;  // 0..15 (columns)
  const int ty = threadIdx.y;  // 0..3  (row groups)
  const int lane = ty * 16 + tx; // 0..63

  const int tile_m = blockIdx.x;        // along M (rows)
  const int msel   = blockIdx.y;        // selection index
  if (msel >= selections_M) return;

  const int base_row = tile_m * 16;

  // Resolve selection sid/eid
  const int sid = sel_sids[msel];
  const int eid = sel_eids[msel];

  // Per-(layer,eid) base pointers
  const size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
  const size_t mat_index = (size_t)layer * (size_t)n_experts + (size_t)eid;
  const hip_bfloat16* __restrict__ W = w_packed + mat_index * per_mat_elems;
  const hip_bfloat16* __restrict__ B = b_mlp1 + mat_index * (size_t)out_dim;

  // Shared tiles
  __shared__ __align__(16) hip_bfloat16 sWb[16 * 16];
  __shared__ __align__(16) float        sWf[16 * 16];
  __shared__ __align__(16) float        sX[16 * 16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v dmn = {0.f, 0.f, 0.f, 0.f};

  const int K = hidden_dim;
  const int Ktiles = (Kpad + 15) / 16;

  for (int kk = 0; kk < Ktiles; ++kk) {
    // Load W tile [16x16] from packed BF16 storage
    if (lane < 32) {
      const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
      const size_t w_tile_base = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* w_src = W + w_tile_base;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(w_src)[lane];
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
      // replicate v into 4 consecutive columns
      float* dst = sX + r * 16 + c4;
      dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = v;
    }

    __syncthreads();

    // Convert W BF16 -> FP32 in shared
    {
      int idx4 = lane * 4;
      if (idx4 < 256) {
        sWf[idx4 + 0] = bfloat16_to_float(sWb[idx4 + 0]);
        sWf[idx4 + 1] = bfloat16_to_float(sWb[idx4 + 1]);
        sWf[idx4 + 2] = bfloat16_to_float(sWb[idx4 + 2]);
        sWf[idx4 + 3] = bfloat16_to_float(sWb[idx4 + 3]);
      }
    }

    // 4 micro-steps across the 16-wide K tile
    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const int k4 = ty + 4 * t4; // 0..15
      float amk = sWf[tx * 16 + k4];   // A: row=tx, col=k4
      float bkn = sX[k4 * 16 + tx];    // B: row=k4, col=tx
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }

    __syncthreads();
  }

  // Store only for tx==0 (column 0) to produce a matvec result
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

// Scatter selections into grouped arrays by expert id using atomic cursors
__global__ void scatter_group_by_eid_kernel(const int *sel_sids_in, const int *sel_eids_in, const float *sel_w_in, int M,
                                            const int *eid_offsets, int *eid_cursors,
                                            int *sel_sids_out, int *sel_eids_out, float *sel_w_out, int n_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M) {
    int e = sel_eids_in[idx];
    if (e >= 0 && e < n_experts) {
      int pos = atomicAdd(&eid_cursors[e], 1);
      int out_idx = eid_offsets[e] + pos;
      sel_sids_out[out_idx] = sel_sids_in[idx];
      sel_eids_out[out_idx] = e;
      sel_w_out[out_idx]    = sel_w_in[idx];
    }
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

// Optimized vectorized bfloat16 to float conversion
__device__ __forceinline__ float4 bfloat16_to_float4(const hip_bfloat16 *bf16_ptr) {
  float4 result;
  result.x = bfloat16_to_float(bf16_ptr[0]);
  result.y = bfloat16_to_float(bf16_ptr[1]);
  result.z = bfloat16_to_float(bf16_ptr[2]);
  result.w = bfloat16_to_float(bf16_ptr[3]);
  return result;
}

// Grouped MLP2 + accumulate: for each (sid,e) pair compute y = W_mlp2[e] * gate_up[m] + b_mlp2[e]
// and atomically accumulate into e_agg[sid] - OPTIMIZED VERSION
__global__ void mlp2_accum_grouped_kernel_float(float *e_agg,
                                                const float *gate_up_all,
                                                const hip_bfloat16 *w_mlp2, const hip_bfloat16 *b_mlp2,
                                                const int *sel_sids, const int *sel_eids, const float *sel_w,
                                                int M, int hidden_dim, int intermediate_dim,
                                                int layer, int n_experts, int eagg_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // output idx in [0, hidden_dim)
  int m = blockIdx.y;
  if (m < M && i < hidden_dim) {
    int sid = sel_sids[m];
    int eid = sel_eids[m];
    float ew = sel_w[m];
    if (ew == 0.0f) return; // PAD: bỏ qua hoàn toàn
    const float *g = gate_up_all + (size_t)m * (size_t)intermediate_dim;
    size_t base = ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)hidden_dim * (size_t)intermediate_dim;
    const hip_bfloat16 *w = w_mlp2 + base + (size_t)i * (size_t)intermediate_dim;
    
    float sum = 0.0f;
    
    // Vectorized computation with 4-element unrolling
    int j = 0;
    for (; j + 3 < intermediate_dim; j += 4) {
      // Load 4 bfloat16 weights and convert to float4
      float4 w_vec = bfloat16_to_float4(&w[j]);
      // Load 4 gate values as float4
      float4 g_vec = make_float4(g[j], g[j+1], g[j+2], g[j+3]);
      
      // Compute dot product
      sum += w_vec.x * g_vec.x + w_vec.y * g_vec.y + 
             w_vec.z * g_vec.z + w_vec.w * g_vec.w;
    }
    
    // Handle remaining elements
    for (; j < intermediate_dim; ++j) {
      sum += bfloat16_to_float(w[j]) * g[j];
    }
    
    const hip_bfloat16 *b = b_mlp2 + ((size_t)layer * (size_t)n_experts + (size_t)eid) * (size_t)hidden_dim;
    float y = sum + bfloat16_to_float(b[i]);
    atomicAdd(&e_agg[(size_t)sid * (size_t)eagg_stride + (size_t)i], ew * y);
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
__global__ __launch_bounds__(64)
void gemm16x16_mlp1_fused_swiglu_scatter(
    const hip_bfloat16* __restrict__ W_packed,
    const float*        __restrict__ Xpack,          // base cho đoạn cột [c0, c1)
    float*              __restrict__ gate_up_out,    // scatter theo column id
    const hip_bfloat16* __restrict__ bias,           // [Mout], nullable
    // Kích thước logic
    int Mout, int K, int Be, int Bpad,
    // Ghi ra I = Mout/2 phần tử theo "row_half"
    int out_stride,                                   // = intermediate_dim
    const int* __restrict__ slot_ids,                // ánh xạ cột local -> col_id toàn cục
    // padding
    int Mpad, int Kpad,
    float swiglu_limit)
{
  const int tx = threadIdx.x;   // 0..15 (N trong tile)
  const int ty = threadIdx.y;   // 0..3
  const int lane = ty*16 + tx;  // 0..63

  const int tile_m  = blockIdx.x;     // theo M (rows)
  const int tile_n  = blockIdx.y;     // theo N (cols)
  const int row0    = tile_m * 16;
  const int col0    = tile_n * 16;

  if (col0 >= Be) return;

  // LDS tiles
  __shared__ __align__(16) hip_bfloat16 sWb[16*16];
  __shared__ __align__(16) float        sWf[16*16];
  __shared__ __align__(16) float        sX [16*16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f,0.f,0.f,0.f};

  // W base (đã chọn đúng (layer,e) ở caller)
  const int Ktiles = (Kpad + 15)/16;
  const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);
  // X base (đã offset c0 ở caller): stride theo Bpad
  for (int kk = 0; kk < Ktiles; ++kk) {
    // 512B W tile → 32 lanes * 16B
    if (lane < 32) {
      const size_t w_tile = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      const hip_bfloat16* src = W_packed + w_tile;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(src)[lane];
    }
    // X: 16x16 floats
    if (lane < 64) {
      const int r  = lane % 16;
      const int c4 = (lane / 16) * 4;          // 0,4,8,12
      const int k_row = kk*16 + r;
      float* dst = sX + r*16 + c4;
      const bool in_k = (k_row < K);
      const bool in_b = (col0 + c4) < Be;
      if (in_k && in_b) {
        const float* src = Xpack + (size_t)k_row * (size_t)Bpad + (size_t)col0 + (size_t)c4;
        reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
      } else {
        reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
      }
    }
    __syncthreads();

    // BF16->FP32 cho W trong LDS
    {
      int i4 = lane*4;
      if (i4 < 256) {
        sWf[i4+0] = bfloat16_to_float(sWb[i4+0]);
        sWf[i4+1] = bfloat16_to_float(sWb[i4+1]);
        sWf[i4+2] = bfloat16_to_float(sWb[i4+2]);
        sWf[i4+3] = bfloat16_to_float(sWb[i4+3]);
      }
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k4 = ty + 4*t;           // 0..15
      float a = sWf[tx*16 + k4];         // A(M,K)
      float b = sX [k4*16 + tx];         // B(K,N)
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0,0,0);
    }
    __syncthreads();
  }

  // Epilogue fused: (2I) → I (SwiGLU) và scatter theo column id
const int col = col0 + tx;
if (col >= Be) return;

const float alpha = 1.702f;
const int   I     = Mout >> 1;

// Lấy column-id toàn cục để scatter
const int col_id = slot_ids[col];
float* out_base  = gate_up_out + (size_t)col_id * (size_t)out_stride;

// Helper đọc phần tử acc theo chỉ số (tránh lấy địa chỉ phần tử vector)
auto comp4 = [&](int idx) __device__ -> float {
  // acc là float4v: {x,y,z,w}
  return (idx == 0) ? acc.x :
         (idx == 1) ? acc.y :
         (idx == 2) ? acc.z : acc.w;
};

// Mỗi thread có 4 hàng liên tiếp: (ty*4 + i), i=0..3
// Cặp even/odd luôn nằm trong cùng thread: (0,1) và (2,3).
#pragma unroll
for (int i = 0; i < 4; i += 2) {                     // i = 0, 2
  const int r  = row0 + (ty*4 + i);                  // even row
  const int rp = r + 1;                              // odd row (cặp với r)
  if (r < Mout && rp < Mout) {
    float gate = comp4(i);
    float up   = comp4(i + 1);
    if (bias) {
      gate += bfloat16_to_float(bias[r]);
      up   += bfloat16_to_float(bias[rp]);
    }
    // clamp nhẹ để ổn định số học
    gate = fminf( swiglu_limit, gate);
    up   = fminf( swiglu_limit, fmaxf(-swiglu_limit, up));
    // SwiGLU: σ(α·gate) * gate * (up + 1)
    gate *= (1.f / (1.f + expf(-alpha * gate)));
    const int row_half = r >> 1;                     // 0..I-1
    if (row_half < I) {
      out_base[row_half] = gate * (up + 1.f);
    }
  }
}
}

__global__ void pack_gateup_to_xpack_kernel(const float* __restrict__ gate_up_all,
                                            float* __restrict__ Xpack,
                                            int I, int Be, int Bpad,
                                            int col0) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)I * (size_t)Bpad;
  if (idx >= tot) return;
  int i = (int)(idx / Bpad);
  int b = (int)(idx % Bpad);
  if (b < Be) {
    int col_id = col0 + b; // cột toàn cục trong gate_up_all
    Xpack[(size_t)i * (size_t)Bpad + (size_t)b] =
        gate_up_all[(size_t)col_id * (size_t)I + (size_t)i];
  } else {
    Xpack[(size_t)i * (size_t)Bpad + (size_t)b] = 0.f;
  }
}

__global__ __launch_bounds__(64)
void gemm16x16_mlp2_f32_scatter_accum(
    const hip_bfloat16* __restrict__ W_packed,  // (H x I) packed 16x16
    const float*        __restrict__ Xpack,     // (I x Bpad) FP32
    float*              __restrict__ e_agg,     // scatter-accumulate
    const hip_bfloat16* __restrict__ bias,      // [H], nullable
    const float*        __restrict__ ew_cols,   // [Be] expert weights per column
    const int*          __restrict__ slot_ids,  // [Be] col→sid
    // sizes
    int H, int I, int Be, int Bpad,
    int out_stride,                             // hidden_dim
    int Mpad, int Kpad)
{
  const int tx = threadIdx.x;      // 0..15 (N)
  const int ty = threadIdx.y;      // 0..3
  const int lane = ty*16 + tx;

  const int tile_m = blockIdx.x;   // rows
  const int tile_n = blockIdx.y;   // cols
  const int row0   = tile_m * 16;
  const int col0   = tile_n * 16;

  if (col0 >= Be) return;

  __shared__ __align__(16) hip_bfloat16 sWb[16*16];
  __shared__ __align__(16) float        sWf[16*16];
  __shared__ __align__(16) float        sX [16*16];

  using float4v = __attribute__((ext_vector_type(4))) float;
  float4v acc = {0.f,0.f,0.f,0.f};

  const int Ktiles = (Kpad + 15)/16;
  const size_t tilesPerRow = (size_t)((Kpad + 15) / 16);

  for (int kk = 0; kk < Ktiles; ++kk) {
    if (lane < 32) {
      const size_t w_tile = ((size_t)tile_m * tilesPerRow + (size_t)kk) * (size_t)256;
      reinterpret_cast<u32x4*>(sWb)[lane] = reinterpret_cast<const u32x4*>(W_packed + w_tile)[lane];
    }
    if (lane < 64) {
      const int r  = lane % 16;          // K row
      const int c4 = (lane / 16) * 4;    // 0,4,8,12
      const int k_row = kk*16 + r;
      float* dst = sX + r*16 + c4;

      const bool in_k = (k_row < I);
      const bool in_b = (col0 + c4) < Bpad;
      if (in_k && in_b) {
        const float* src = Xpack + (size_t)k_row * (size_t)Bpad + (size_t)col0 + (size_t)c4;
        reinterpret_cast<u32x4*>(dst)[0] = reinterpret_cast<const u32x4*>(src)[0];
      } else {
        reinterpret_cast<u32x4*>(dst)[0] = u32x4{0u,0u,0u,0u};
      }
    }
    __syncthreads();

    // BF16 -> FP32 (W)
    {
      int i4 = lane*4;
      if (i4 < 256) {
        sWf[i4+0] = bfloat16_to_float(sWb[i4+0]);
        sWf[i4+1] = bfloat16_to_float(sWb[i4+1]);
        sWf[i4+2] = bfloat16_to_float(sWb[i4+2]);
        sWf[i4+3] = bfloat16_to_float(sWb[i4+3]);
      }
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k4 = ty + 4*t;
      float a = sWf[tx*16 + k4];        // A(H,I)
      float b = sX [k4*16 + tx];        // B(I,Be)
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0,0,0);
    }
    __syncthreads();
  }

  const int col = col0 + tx;
  if (col >= Be) return;

  const float ew = ew_cols[col];
  if (ew == 0.f) return; // pad

  const int sid = slot_ids[col];
  float* out_s  = e_agg + (size_t)sid * (size_t)out_stride;

  // helper để truy cập phần tử acc
  auto comp4 = [&](int idx) __device__ -> float {
    return (idx==0)?acc.x : (idx==1)?acc.y : (idx==2)?acc.z : acc.w;
  };

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = row0 + (ty*4 + i);
    if (row < H) {
      float v = comp4(i);
      if (bias) v += bfloat16_to_float(bias[row]);
      atomicAdd(&out_s[row], ew * v);
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
    // Gather/pack inputs t -> xpack_f32
    {
      size_t total = (size_t)Kqkv * (size_t)Bpad;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_fp32_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.xpack_f32, Kqkv, batch, Bpad, hidden_dim, gpu_ctx->d_slot_ids);
    }
    // GEMM W_qkv_packed x Xpack -> qkv (scattered per slot) + bias
    {
      const hip_bfloat16 *w_qkv_packed = gpu_ctx->weights.w_qkv_packed16 + (size_t)l * (size_t)Mpad * (size_t)Kpad;
      dim3 grid((unsigned)(Mpad + 15) / 16, (unsigned)(Bpad + 15) / 16);
      dim3 block(16, 4, 1);
      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        w_qkv_packed, gpu_ctx->state.xpack_f32, gpu_ctx->state.qkv, b_qkv,
        qkv_out_dim, Kqkv, batch, Bpad, qkv_out_dim, gpu_ctx->d_slot_ids, Mpad, Kpad);
    }

    // Write V to cache (before RoPE)
    {
      dim3 block(256), grid((v_size + block.x - 1) / block.x, batch);
      KVWriterV::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.qkv, gpu_ctx->state.value_cache,
        batch, qkv_out_dim, q_size + k_size, v_size, kv_dim, loff, kv_block_size, gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
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

    // Write K (after RoPE)
    {
      dim3 block(256), grid((k_size + block.x - 1) / block.x, batch);
      KVWriterK::run<<<grid, block, 0, gpu_ctx->batch_stream>>>(gpu_ctx->state.qkv, gpu_ctx->state.key_cache,
        batch, qkv_out_dim, q_size, k_size, kv_dim, loff, kv_block_size, gpu_ctx->d_pos_list, gpu_ctx->d_slot_ids);
    }

    // Attention computation (FlashAttention-like, always enabled) -> tb
    {
      dim3 block(256);
      dim3 grid(1, p->n_attn_heads, batch);
      bool can_add_sink = true;
      const float *layer_mask = (p->sliding_window > 0 && (l % 2 == 0)) ? gpu_ctx->state.mask : NULL;

      // Use shared memory only for attention scores: seq_len floats
      size_t shared_mem_bytes = (size_t)p->seq_len * sizeof(float);
      flash_attention_kernel_optimized_batched<<<grid, block, shared_mem_bytes, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.qkv, gpu_ctx->state.key_cache, gpu_ctx->state.value_cache, gpu_ctx->weights.attn_sinks,
        gpu_ctx->state.tb, layer_mask, head_dim, loff, kv_dim, kv_mul,
        p->n_attn_heads, p->seq_len, l, gpu_ctx->d_pos_list, can_add_sink, batch,
        qkv_out_dim, kv_block_size, gpu_ctx->d_slot_ids, 0);
    }

    // Output projection W_o (batched) via packed BF16 GEMM: tb2 = tb @ W_o + b_o
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    {
      int Mwo = hidden_dim;
      int Kow = head_dim * p->n_attn_heads;
      int Bpad2 = ((batch + 15) / 16) * 16;
      int Mpad2 = ((Mwo + 15) / 16) * 16;
      int Kpad2 = ((Kow + 15) / 16) * 16;
      // Gather/pack tb -> xpack_f32
      size_t total = (size_t)Kow * (size_t)Bpad2;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_fp32_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.tb, gpu_ctx->state.xpack_f32, Kow, batch, Bpad2, Kow, gpu_ctx->d_slot_ids);
      const hip_bfloat16 *w_o_packed = gpu_ctx->weights.w_o_packed16 + (size_t)l * (size_t)Mpad2 * (size_t)Kpad2;
      dim3 blockG(16, 4, 1), gridG((unsigned int)((Mpad2 + 15) / 16), (unsigned int)((Bpad2 + 15) / 16));

      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
        w_o_packed, gpu_ctx->state.xpack_f32, gpu_ctx->state.tb2, b_o,
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
      // Gather/pack t -> xpack_f32
      size_t total = (size_t)Kr * (size_t)Bpadr;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_fp32_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.t, gpu_ctx->state.xpack_f32, Kr, batch, Bpadr, hidden_dim, gpu_ctx->d_slot_ids);
      const hip_bfloat16 *w_router_packed = gpu_ctx->weights.w_router_packed16 + (size_t)l * (size_t)Mpadr * (size_t)Kpadr;
      dim3 blockG(16, 4, 1), gridG((unsigned)(Mpadr + 15) / 16, (unsigned)(Bpadr + 15) / 16);
      gemm16x16_mfma_f32_packedW_batched_scatter_bias<<<gridG, blockG, 0, gpu_ctx->batch_stream>>>(
        w_router_packed, gpu_ctx->state.xpack_f32, gpu_ctx->state.router_score, b_router,
        Mr, Kr, batch, Bpadr, Mr, gpu_ctx->d_slot_ids, Mpadr, Kpadr);
    }
    // 3) Top-k + softmax on device for all slots (fused per-slot kernel)
    {
      dim3 grid(batch), block(1);
      topk_softmax_batched_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.router_score, p->n_experts, p->experts_per_token,
        gpu_ctx->state.topk_v, gpu_ctx->state.topk_i,
        p->n_experts, p->experts_per_token,
        gpu_ctx->d_slot_ids, batch);
    }
    // 4) Zero e_agg for selected slots
    {
      dim3 block(256), grid((hidden_dim + block.x - 1) / block.x, batch);
      zero_selected_slots_kernel_float<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.e_agg, hidden_dim, gpu_ctx->d_slot_ids, batch);
    }
    // 5) Gather top-k selections into flattened arrays [batch * K]
    int M = batch * p->experts_per_token;
    {
      dim3 block(256), grid((M + block.x - 1) / block.x);
      gather_topk_pairs_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.topk_v, gpu_ctx->state.topk_i,
        gpu_ctx->d_slot_ids, batch, p->experts_per_token,
        gpu_ctx->state.sel_w, gpu_ctx->state.sel_eids, gpu_ctx->state.sel_sids);
    }
    // 6) Group selections by expert (eid) into grouped arrays
    {
      // Zero counts and cursors
      HIP_CHECK(hipMemsetAsync(gpu_ctx->state.eid_counts,  0, (size_t)p->n_experts * sizeof(int), gpu_ctx->batch_stream));
      HIP_CHECK(hipMemsetAsync(gpu_ctx->state.eid_cursors, 0, (size_t)p->n_experts * sizeof(int), gpu_ctx->batch_stream));
      // Histogram counts
      dim3 blockH(256), gridH((M + blockH.x - 1) / blockH.x);
      histogram_eid_counts_kernel<<<gridH, blockH, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.sel_eids, M, gpu_ctx->state.eid_counts, p->n_experts);
      // 6') Xây prefix-sum đã đệm theo BLOCK_B
      const int BLOCK_B = (getenv("GETP_MOE_BLOCK") ? std::max(1, atoi(getenv("GETP_MOE_BLOCK"))) : 16);
      build_padded_offsets_kernel<<<1,1,0,gpu_ctx->batch_stream>>>(
          gpu_ctx->state.eid_counts,
          gpu_ctx->state.padded_offsets,
          p->n_experts, BLOCK_B);

      // Lấy total_post_pad từ padded_offsets[n_experts]
      // int total_post_pad = 0;
      HIP_CHECK(hipMemcpyAsync(&total_post_pad,
                              gpu_ctx->state.padded_offsets + p->n_experts,
                              sizeof(int),
                              hipMemcpyDeviceToHost,
                              gpu_ctx->batch_stream));
      HIP_CHECK(hipStreamSynchronize(gpu_ctx->batch_stream));

      // Reset cursors để scatter
      HIP_CHECK(hipMemsetAsync(gpu_ctx->state.eid_cursors, 0,
                              (size_t)p->n_experts * sizeof(int),
                              gpu_ctx->batch_stream));

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

      // 6'''') Chuẩn bị iota cột toàn cục 0..total_post_pad-1
      {
        int threads = 256;
        int blocks  = (total_post_pad + threads - 1) / threads;
        iota_kernel<<<blocks, threads, 0, gpu_ctx->batch_stream>>>(
            gpu_ctx->state.d_col_ids, total_post_pad);
      }

    }
    {
        // 7') Pack X cho MLP1 theo thứ tự đã sort+pad
    int Kmlp1 = p->hidden_dim;
    int Bmlp1 = total_post_pad;
    int Bpad_total = round_up16(Bmlp1); // Bpad dùng khi pack (toàn cục)
    {
      size_t total = (size_t)Kmlp1 * (size_t)Bpad_total;
      dim3 block(256), grid((unsigned int)((total + block.x - 1) / block.x));
      gather_pack_x_fp32_to_fp32_kernel<<<grid, block, 0, gpu_ctx->batch_stream>>>(
          gpu_ctx->state.t,            // nguồn: x đã RMSNorm-FFN per-slot
          gpu_ctx->state.xpack_f32,    // đích:  [K x Bpad_total]
          Kmlp1,
          Bmlp1,
          Bpad_total,
          p->hidden_dim,
          gpu_ctx->state.sorted_sids   // cột b → sid = sorted_sids[b]
      );
    }
        // 7'') GEMM MLP1 per-expert (fused epilogue SwiGLU → ghi thẳng gate_up_all)
    {
      const int Mout       = 2 * p->intermediate_dim;
      const int I         = p->intermediate_dim;
      const int Mpad_mlp1 = round_up16(Mout);
      const int Kpad_mlp1 = round_up16(p->hidden_dim);

      std::vector<int> h_padded(p->n_experts + 1);
      HIP_CHECK(hipMemcpyAsync(h_padded.data(),
                              gpu_ctx->state.padded_offsets,
                              (size_t)(p->n_experts + 1) * sizeof(int),
                              hipMemcpyDeviceToHost, gpu_ctx->batch_stream));
      HIP_CHECK(hipStreamSynchronize(gpu_ctx->batch_stream));

      for (int e = 0; e < p->n_experts; ++e) {
        int c0 = h_padded[e];
        int c1 = h_padded[e+1];
        int Be = c1 - c0;
        if (Be <= 0) continue;

        dim3 blockG(16, 4, 1);
        dim3 gridG((unsigned)((Mpad_mlp1 + 15) / 16),
                  (unsigned)((Be + 15) / 16));

        hipStream_t es = gpu_ctx->expert_streams[
            (gpu_ctx->num_expert_streams > 0) ? (e % gpu_ctx->num_expert_streams) : 0];

        const size_t per_mat_elems = (size_t)Mpad_mlp1 * (size_t)Kpad_mlp1;
        const size_t mat_index = (size_t)l * (size_t)p->n_experts + (size_t)e;
        const hip_bfloat16* W = gpu_ctx->weights.w_mlp1_packed16 + mat_index * per_mat_elems;
        const hip_bfloat16* B = gpu_ctx->weights.b_mlp1          + mat_index * (size_t)Mout;

        gemm16x16_mlp1_fused_swiglu_scatter<<<gridG, blockG, 0, es>>>(
            /*W_packed*/ W,
            /*Xpack   */ gpu_ctx->state.xpack_f32 + (size_t)c0,
            /*dst     */ gpu_ctx->state.gate_up_all,
            /*bias    */ B,
            /*Mout,K,Be,Bpad*/ Mout, p->hidden_dim, Be, Bpad_total,
            /*out_stride(I)  */ I,
            /*slot_ids*/ gpu_ctx->state.d_col_ids + c0,
            /*pads*/     Mpad_mlp1, Kpad_mlp1,
            /*limit*/    p->swiglu_limit);
      }

      // Đồng bộ các expert-streams về batch_stream
      std::vector<hipEvent_t> evts(gpu_ctx->num_expert_streams);
      for (int s = 0; s < gpu_ctx->num_expert_streams; ++s) {
        evts[s] = nullptr;
        if (gpu_ctx->expert_streams[s]) {
          HIP_CHECK(hipEventCreateWithFlags(&evts[s], hipEventDisableTiming));
          HIP_CHECK(hipEventRecord(evts[s], gpu_ctx->expert_streams[s]));
        }
      }
      for (int s = 0; s < gpu_ctx->num_expert_streams; ++s) {
        if (evts[s]) {
          HIP_CHECK(hipStreamWaitEvent(gpu_ctx->batch_stream, evts[s], 0));
          HIP_CHECK(hipEventDestroy(evts[s]));
        }
      }
    }

    // [BỎ HẲN] 7''') swiglu_grouped_kernel_float(...)
        
    }

    // 8) Grouped MLP2 via MFMA matvec (packed) with in-kernel accumulation
    {
      int Mpad_mlp2 = ((hidden_dim + 15) / 16) * 16;
      int Kpad_mlp2 = ((p->intermediate_dim + 15) / 16) * 16;
      dim3 blockG2(16, 4, 1);
      // Use total_post_pad (Btotal) columns produced by fused MLP1 scatter
      int Btotal = total_post_pad;
      dim3 gridG2((unsigned int)((Mpad_mlp2 + 15) / 16), (unsigned int)Btotal);
      mlp2_grouped_mfma_matvec_residual_kernel<<<gridG2, blockG2, 0, gpu_ctx->batch_stream>>>(
        gpu_ctx->state.x,
        gpu_ctx->state.gate_up_all,
        gpu_ctx->weights.w_mlp2_packed16,
        gpu_ctx->weights.b_mlp2,
        gpu_ctx->state.sorted_sids,
        gpu_ctx->state.sorted_eids,
        gpu_ctx->state.sorted_w,
        gpu_ctx->state.sorted_indices,
        Btotal, hidden_dim, p->intermediate_dim,
        l, p->n_experts, hidden_dim,
        Mpad_mlp2, Kpad_mlp2);
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

    // Batched logits computation across all slots with optimized block size
    dim3 block_logits(128);  // Reduced block size for better shared memory usage
    dim3 grid_logits(p->vocab_size, batch);
    size_t shared_mem_logits = block_logits.x * sizeof(float);
    logits_final_batched_kernel<<<grid_logits, block_logits, shared_mem_logits, gpu_ctx->batch_stream>>>(
      gpu_ctx->state.x, gpu_ctx->weights.out,
      gpu_ctx->state.logits_fp32, hidden_dim,
      p->vocab_size, batch, gpu_ctx->d_slot_ids);

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
    // Configure per-GPU max batch from env, default 8
    const char *env_mb = getenv("GETP_MAX_BATCH");
    int mb = (env_mb && atoi(env_mb) > 0) ? atoi(env_mb) : 8;
    gpu_contexts[d].max_batch = mb;
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config, gpu_contexts[d].max_batch);
    // Create per-slot persistent streams
    gpu_contexts[d].slot_streams.resize(gpu_contexts[d].max_batch);
    for (int s = 0; s < gpu_contexts[d].max_batch; ++s) {
      HIP_CHECK(hipStreamCreate(&gpu_contexts[d].slot_streams[s]));
    }
    // Create batch stream and staging device buffers
    HIP_CHECK(hipStreamCreate(&gpu_contexts[d].batch_stream));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_slot_ids,  gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_pos_list,  gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_token_ids, gpu_contexts[d].max_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu_contexts[d].d_next_tokens, gpu_contexts[d].max_batch * sizeof(int)));
    // Create expert streams (up to MAX_EXPERT_STREAMS). Allow override via GETP_NUM_EXPERT_STREAMS
    int nes = MAX_EXPERT_STREAMS;
    if (const char* env_es = getenv("GETP_NUM_EXPERT_STREAMS")) {
      int v = atoi(env_es);
      if (v > 0) nes = std::min(v, MAX_EXPERT_STREAMS);
    }
    gpu_contexts[d].num_expert_streams = nes;
    for (int s = 0; s < gpu_contexts[d].num_expert_streams; ++s) {
      HIP_CHECK(hipStreamCreate(&gpu_contexts[d].expert_streams[s]));
    }

    if (transformer->config.sliding_window > 0 && gpu_contexts[d].state.mask != NULL) {
      int S = transformer->config.seq_len;
      std::vector<float> cpu_mask((size_t)S * (size_t)S);
      for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
          cpu_mask[(size_t)i * (size_t)S + (size_t)j] = (i - j >= transformer->config.sliding_window) ? -INFINITY : 0.0f;
        }
      }
      // Copy mask directly as float (no conversion needed)
      HIP_CHECK(hipMemcpy(gpu_contexts[d].state.mask, cpu_mask.data(), (size_t)S * (size_t)S * sizeof(float), hipMemcpyHostToDevice));
    }
  }
  
  printf("[GPU] Warm-up done on %d device(s). Packed BF16 tiles for QKV/W_O/Router/MLP1 loaded.\n", max_devices);
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
    for (int s = 0; s < gpu_ctx.num_expert_streams; ++s) {
      if (gpu_ctx.expert_streams[s] != nullptr) HIP_CHECK(hipStreamDestroy(gpu_ctx.expert_streams[s]));
    }
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

  HIP_CHECK(hipMalloc(&s->mlp1_out, B * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate,     B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up,       B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up,  B * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->e_agg,    B * (size_t)p->hidden_dim * sizeof(float)));

  HIP_CHECK(hipMalloc(&s->qkv, B * (size_t)p->head_dim * (size_t)(p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->q,   B * (size_t)p->n_attn_heads * (size_t)p->head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->k,   B * (size_t)p->n_kv_heads   * (size_t)p->head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->v,   B * (size_t)p->n_kv_heads   * (size_t)p->head_dim * sizeof(float)));

  // Allocate KV cache per-slot for continuous batching
  size_t kv_cache_elems = (size_t)max_batch * (size_t)p->n_layers * (size_t)p->seq_len * (size_t)kv_dim;
  HIP_CHECK(hipMalloc(&s->key_cache,   kv_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->value_cache, kv_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits,      B * (size_t)p->vocab_size * sizeof(float)));

  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(float)));
  } else {
    s->mask = NULL;
  }

  HIP_CHECK(hipMalloc(&s->cos_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->sin_vals, B * (size_t)(p->head_dim / 2) * sizeof(float)));

  // Note: acc_buffer no longer needed since we use float state directly
  s->acc_buffer = nullptr;
  HIP_CHECK(hipMalloc(&s->logits_fp32, B * (size_t)p->vocab_size * sizeof(float)));

  // Per-stream expert scratch buffers
  HIP_CHECK(hipMalloc(&s->mlp1_out_s, MAX_EXPERT_STREAMS * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_s,     MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up_s,       MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up_s,  MAX_EXPERT_STREAMS * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2_s,      MAX_EXPERT_STREAMS * (size_t)p->hidden_dim * sizeof(float)));

  // Grouped MoE staging buffers
  size_t Mmax = (size_t)max_batch * (size_t)p->experts_per_token;
  HIP_CHECK(hipMalloc(&s->sel_sids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_eids,     Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_w,        Mmax * sizeof(float)));

  // Expert-grouped arrays and counters
  HIP_CHECK(hipMalloc(&s->sel_sids_grouped, Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_eids_grouped, Mmax * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->sel_w_grouped,    Mmax * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->eid_counts,   (size_t)p->n_experts * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->eid_offsets, ((size_t)p->n_experts + 1) * sizeof(int)));
  HIP_CHECK(hipMalloc(&s->eid_cursors,  (size_t)p->n_experts * sizeof(int)));


  const int BLOCK_B = (getenv("GETP_MOE_BLOCK") ? std::max(1, atoi(getenv("GETP_MOE_BLOCK"))) : 16);

  int Bpad_batch = ((max_batch + 15) / 16) * 16;  // cho QKV/W_O/Router
  // size_t Mmax      = (size_t)max_batch * (size_t)p->experts_per_token;               // B tối đa của selections
  size_t Mmax_pad  = Mmax + (size_t)p->n_experts * (size_t)(BLOCK_B - 1);            // trừ hao pad tối đa
  int Bpad_mlp1    = (int)round_up16((int)Mmax_pad);                                  // cho pack MLP1
  int Bpad_pack    = std::max(Bpad_batch, Bpad_mlp1);

  // size_t Kmax_xpack = std::max(
  //     (size_t)p->hidden_dim,
  //     (size_t)p->head_dim * (size_t)p->n_attn_heads
  // );
    size_t Kmax_xpack = std::max({
      (size_t)p->hidden_dim,
      (size_t)p->head_dim * (size_t)p->n_attn_heads,
      (size_t)p->intermediate_dim    // NEW
  });

  HIP_CHECK(hipMalloc(&s->xpack_f32, Kmax_xpack * (size_t)Bpad_pack * sizeof(float)));
  // HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax * 2 * (size_t)p->intermediate_dim * sizeof(float)));
  // HIP_CHECK(hipMalloc(&s->gate_up_all,  Mmax * (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->mlp1_out_all, Mmax_pad * 2 * (size_t)p->intermediate_dim * sizeof(float)));
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
  HIP_CHECK(hipMalloc(&s->mlp1_Ypack, Mout * Mmax_pad * sizeof(float)));


}


void gpu_free_run_state(GPURunState *s) {
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
  HIP_CHECK(hipFree(s->logits));
  HIP_CHECK(hipFree(s->mask));
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  if (s->acc_buffer) HIP_CHECK(hipFree(s->acc_buffer));  // acc_buffer may be nullptr
  HIP_CHECK(hipFree(s->logits_fp32));
  HIP_CHECK(hipFree(s->mlp1_out_s));
  HIP_CHECK(hipFree(s->gate_s));
  HIP_CHECK(hipFree(s->up_s));
  HIP_CHECK(hipFree(s->gate_up_s));
  HIP_CHECK(hipFree(s->tb2_s));
  // Grouped MoE staging buffers
  HIP_CHECK(hipFree(s->sel_sids));
  HIP_CHECK(hipFree(s->sel_eids));
  HIP_CHECK(hipFree(s->sel_w));
  HIP_CHECK(hipFree(s->mlp1_out_all));
  HIP_CHECK(hipFree(s->gate_up_all));
  HIP_CHECK(hipFree(s->sel_sids_grouped));
  HIP_CHECK(hipFree(s->sel_eids_grouped));
  HIP_CHECK(hipFree(s->sel_w_grouped));
  HIP_CHECK(hipFree(s->eid_counts));
  HIP_CHECK(hipFree(s->eid_offsets));
  HIP_CHECK(hipFree(s->eid_cursors));
  HIP_CHECK(hipFree(s->xpack_f32));
  HIP_CHECK(hipFree(s->sorted_indices));
  HIP_CHECK(hipFree(s->sorted_sids));
  HIP_CHECK(hipFree(s->sorted_eids));
  HIP_CHECK(hipFree(s->sorted_w));
  HIP_CHECK(hipFree(s->padded_offsets));
  HIP_CHECK(hipFree(s->block_expert_ids));
  HIP_CHECK(hipFree(s->d_col_ids));
  HIP_CHECK(hipFree(s->mlp1_Ypack));

}

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int dev = 0; hipGetDevice(&dev);
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
  gpu_w->w_mlp1 = nullptr; // we will use packed storage for mlp1

  // Helper: host FP32 -> device staging FP32 (tile) -> device BF16 dest
  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    const size_t bf16_size = count * sizeof(hip_bfloat16);
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Try to pin host memory region for faster HtoD copies
    hipError_t pin_ret = hipHostRegister((void*)cpu_data, count * sizeof(float), 0);

    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024; // 512 MB
    size_t staging_elems = std::max((size_t)1, STAGING_BYTES / sizeof(float));

    const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
    if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;

    float *d_stage_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&d_stage_fp32, staging_elems * sizeof(float)));

    size_t done = 0;
    const dim3 block(256);
    while (done < count) {
      size_t tile = std::min(staging_elems, count - done);
      size_t tile_bytes_fp32 = tile * sizeof(float);

      HIP_CHECK(hipMemcpyAsync(d_stage_fp32, cpu_data + done, tile_bytes_fp32,
                               hipMemcpyHostToDevice, stream));

      int n = static_cast<int>(tile);
      dim3 grid((n + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(d_stage_fp32, *gpu_data + done, n);
      HIP_CHECK(hipGetLastError());

      done += tile;
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_stage_fp32));
    HIP_CHECK(hipStreamDestroy(stream));

    if (pin_ret == hipSuccess) {
      hipHostUnregister((void*)cpu_data);
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

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Pin entire host buffer for faster streaming
    size_t total_f32 = (size_t)layers * (size_t)M * (size_t)K;
    hipError_t pin_ret = hipHostRegister((void*)cpu_data, total_f32 * sizeof(float), 0);

    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024; // 512 MB
    size_t staging_elems = std::max((size_t)1, STAGING_BYTES / sizeof(float));
    const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
    if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;

    size_t max_rows = staging_elems / (size_t)K;
    if (max_rows < 1) max_rows = 1;

    float *d_stage_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&d_stage_fp32, (size_t)max_rows * (size_t)K * sizeof(float)));

    const dim3 block(256);
    for (int l = 0; l < layers; ++l) {
      const float *layer_src = cpu_data + (size_t)l * (size_t)M * (size_t)K;
      hip_bfloat16 *layer_dst = *gpu_packed + (size_t)l * (size_t)per_layer_elems;
      for (int row_off = 0; row_off < M; row_off += (int)max_rows) {
        int rows = std::min((int)max_rows, M - row_off);
        size_t tile_bytes = (size_t)rows * (size_t)K * sizeof(float);
        HIP_CHECK(hipMemcpyAsync(d_stage_fp32, layer_src + (size_t)row_off * (size_t)K,
                                 tile_bytes, hipMemcpyHostToDevice, stream));
        size_t total = (size_t)rows * (size_t)K;
        dim3 grid((unsigned int)((total + block.x - 1) / block.x));
        pack_convert_fp32_to_bf16_tiles16_kernel<<<grid, block, 0, stream>>>(
            d_stage_fp32, row_off, rows, M, K, layer_dst, Mpad, Kpad);
        HIP_CHECK(hipGetLastError());
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_stage_fp32));
    HIP_CHECK(hipStreamDestroy(stream));

    if (pin_ret == hipSuccess) {
      hipHostUnregister((void*)cpu_data);
    }
  };


  // Counts as size_t to avoid overflow
  size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
  printf("[GPU#%d] Copy embed/out...\n", dev);
  copy_and_convert(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, token_embed_count);
  copy_and_convert(cpu_w->out,                    &gpu_w->out,                    token_embed_count);

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

  // Pack MLP1 per (layer,expert) into 16x16 BF16 tiles
  printf("[GPU#%d] Pack MLP1 (layers=%d, experts=%d)...\n", dev, n_layers, n_experts);
  int M_mlp1 = 2 * intermediate_dim;
  int K_mlp1 = hidden_dim;
  // Treat each (layer,expert) as a separate matrix in a flattened array of length n_layers*n_experts
  {
    // Pin host W_mlp1 for faster streaming
    size_t total_f32 = (size_t)n_layers * (size_t)n_experts * (size_t)M_mlp1 * (size_t)K_mlp1;
    hipError_t pin_ret_mlp1 = hipHostRegister((void*)cpu_w->w_mlp1, total_f32 * sizeof(float), 0);
    int layers_flat = n_layers * n_experts;
    int Mpad = round_up16(M_mlp1);
    int Kpad = round_up16(K_mlp1);
    size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
    HIP_CHECK(hipMalloc(&gpu_w->w_mlp1_packed16, (size_t)layers_flat * per_mat_elems * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemset(gpu_w->w_mlp1_packed16, 0, (size_t)layers_flat * per_mat_elems * sizeof(hip_bfloat16)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024;
    size_t staging_elems = std::max((size_t)1, STAGING_BYTES / sizeof(float));
    const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
    if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;
    size_t max_rows = staging_elems / (size_t)K_mlp1;
    if (max_rows < 1) max_rows = 1;
    float *d_stage_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&d_stage_fp32, (size_t)max_rows * (size_t)K_mlp1 * sizeof(float)));
    const dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      // Simple progress indicator per layer
      // fprintf(stdout, "[GPU#%d]  MLP1 layer %d/%d\n", dev, l+1, n_layers);
      for (int e = 0; e < n_experts; ++e) {
        int idx = l * n_experts + e;
        const float *src = cpu_w->w_mlp1 + (size_t)l * (size_t)n_experts * (size_t)M_mlp1 * (size_t)K_mlp1
                                         + (size_t)e * (size_t)M_mlp1 * (size_t)K_mlp1;
        hip_bfloat16 *dst = gpu_w->w_mlp1_packed16 + (size_t)idx * per_mat_elems;
        for (int row_off = 0; row_off < M_mlp1; row_off += (int)max_rows) {
          int rows = std::min((int)max_rows, M_mlp1 - row_off);
          size_t tile_bytes = (size_t)rows * (size_t)K_mlp1 * sizeof(float);
          HIP_CHECK(hipMemcpyAsync(d_stage_fp32, src + (size_t)row_off * (size_t)K_mlp1,
                                   tile_bytes, hipMemcpyHostToDevice, stream));
          size_t total = (size_t)rows * (size_t)K_mlp1;
          dim3 grid((unsigned int)((total + block.x - 1) / block.x));
          pack_convert_fp32_to_bf16_tiles16_kernel<<<grid, block, 0, stream>>>(
              d_stage_fp32, row_off, rows, M_mlp1, K_mlp1, dst, Mpad, Kpad);
          HIP_CHECK(hipGetLastError());
        }
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_stage_fp32));
    HIP_CHECK(hipStreamDestroy(stream));
    if (pin_ret_mlp1 == hipSuccess) {
      hipHostUnregister((void*)cpu_w->w_mlp1);
    }
  }

  size_t mlp1_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->b_mlp1, &gpu_w->b_mlp1, mlp1_b_count);

  // Pack MLP2 per (layer,expert) into 16x16 BF16 tiles: [hidden_dim x intermediate_dim]
  int M_mlp2 = hidden_dim;
  int K_mlp2 = intermediate_dim;
  {
    int layers_flat = n_layers * n_experts;
    int Mpad = round_up16(M_mlp2);
    int Kpad = round_up16(K_mlp2);
    size_t per_mat_elems = (size_t)Mpad * (size_t)Kpad;
    HIP_CHECK(hipMalloc(&gpu_w->w_mlp2_packed16, (size_t)layers_flat * per_mat_elems * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemset(gpu_w->w_mlp2_packed16, 0, (size_t)layers_flat * per_mat_elems * sizeof(hip_bfloat16)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    const size_t STAGING_BYTES = (size_t)512 * 1024 * 1024;
    size_t staging_elems = std::max((size_t)1, STAGING_BYTES / sizeof(float));
    const size_t INT_MAX_ELEMS = (size_t)std::numeric_limits<int>::max() - 1024;
    if (staging_elems > INT_MAX_ELEMS) staging_elems = INT_MAX_ELEMS;
    size_t max_rows = staging_elems / (size_t)K_mlp2;
    if (max_rows < 1) max_rows = 1;
    float *d_stage_fp32 = nullptr;
    HIP_CHECK(hipMalloc(&d_stage_fp32, (size_t)max_rows * (size_t)K_mlp2 * sizeof(float)));
    const dim3 block(256);
    for (int l = 0; l < n_layers; ++l) {
      for (int e = 0; e < n_experts; ++e) {
        int idx = l * n_experts + e;
        const float *src = cpu_w->w_mlp2 + (size_t)l * (size_t)n_experts * (size_t)M_mlp2 * (size_t)K_mlp2
                                         + (size_t)e * (size_t)M_mlp2 * (size_t)K_mlp2;
        hip_bfloat16 *dst = gpu_w->w_mlp2_packed16 + (size_t)idx * per_mat_elems;
        for (int row_off = 0; row_off < M_mlp2; row_off += (int)max_rows) {
          int rows = std::min((int)max_rows, M_mlp2 - row_off);
          size_t tile_bytes = (size_t)rows * (size_t)K_mlp2 * sizeof(float);
          HIP_CHECK(hipMemcpyAsync(d_stage_fp32, src + (size_t)row_off * (size_t)K_mlp2,
                                   tile_bytes, hipMemcpyHostToDevice, stream));
          size_t total = (size_t)rows * (size_t)K_mlp2;
          dim3 grid((unsigned int)((total + block.x - 1) / block.x));
          pack_convert_fp32_to_bf16_tiles16_kernel<<<grid, block, 0, stream>>>(
              d_stage_fp32, row_off, rows, M_mlp2, K_mlp2, dst, Mpad, Kpad);
          HIP_CHECK(hipGetLastError());
        }
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_stage_fp32));
    HIP_CHECK(hipStreamDestroy(stream));
  }
  size_t mlp2_w_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->w_mlp2, &gpu_w->w_mlp2, mlp2_w_count);

  size_t mlp2_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_mlp2, &gpu_w->b_mlp2, mlp2_b_count);
}

void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  if (gpu_w->token_embedding_table) HIP_CHECK(hipFree(gpu_w->token_embedding_table));
  if (gpu_w->out)                   HIP_CHECK(hipFree(gpu_w->out));
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
