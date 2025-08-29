// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode, with dynamic batching (grid.y = batch index).
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (e.g., MI250), BF16 weights/KV; FP32 accumulate.
// - Warm-up loads weights with chunked copy+convert to avoid OOM.
// - Runtime fixes: bound pos to seq_len; guard attention sink write.
// - Dynamic batching: per-GPU slots, 2 streams (decode high prio, prefill low prio),
//   batched kernels for RMSNorm/MatVec/AddBias/Final matvec; attention kept per-slot (safer).
// - Keeps function signatures: warm_up, inference, finish.


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

struct GPUTransformerWeights {
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
};

struct GPURunState {
  hip_bfloat16 *x;
  hip_bfloat16 *t;
  hip_bfloat16 *tb;
  hip_bfloat16 *tb2;
  hip_bfloat16 *router_score;
  hip_bfloat16 *topk_v;
  int *topk_i;
  hip_bfloat16 *mlp1_out;
  hip_bfloat16 *gate;
  hip_bfloat16 *up;
  hip_bfloat16 *gate_up;
  hip_bfloat16 *e_agg;
  hip_bfloat16 *qkv;
  hip_bfloat16 *q;
  hip_bfloat16 *k;
  hip_bfloat16 *v;
  hip_bfloat16 *att;         // size: n_attn_heads * seq_len
  hip_bfloat16 *logits;      // (unused for final FP32 logits, but kept)
  hip_bfloat16 *key_cache;
  hip_bfloat16 *value_cache;
  hip_bfloat16 *mask;
  hip_bfloat16 *cos_vals;
  hip_bfloat16 *sin_vals;
  // FP32 buffers for accumulation and precision-sensitive ops
  float *acc_buffer;         // x in FP32 for output matvec
  float *logits_fp32;        // final logits
};

#ifndef MAX_SLOTS
#define MAX_SLOTS 8
#endif

// Trạng thái 1 slot trên 1 GPU
enum SlotStage { SLOT_EMPTY=0, SLOT_PREFILL, SLOT_PREFILL_WAIT, SLOT_DECODE, SLOT_DONE };

struct Slot {
  bool occupied = false;
  int  req_idx  = -1;
  std::vector<int> prompt;   // prompt tokens (host)
  int pos = 0;               // vị trí toàn trình (0..seq_len-1)
  int token = 0;             // token hiện tại ở vị trí pos
  int num_prompt_tokens = 0; // số token của prompt
  int gen_written = 0;       // đã ghi ra bao nhiêu token sinh
  SlotStage stage = SLOT_EMPTY;
};

struct GPUContext {
  int device_id;
  GPUTransformerWeights weights;
  GPURunState state; // (legacy) vẫn giữ để tương thích
  // ---- dynamic batching ----
  int num_slots = 0;
  hipStream_t stream_decode = nullptr; // ưu tiên cao
  hipStream_t stream_prefill = nullptr;// ưu tiên thấp
  std::vector<GPURunState> slot_states;    // mỗi slot 1 runstate riêng
  std::vector<hipEvent_t>  prefill_done;   // event báo prefill xong cho mỗi slot

  // device arrays of pointers (B <= num_slots)
  hip_bfloat16 **d_x_list=nullptr, **d_t_list=nullptr, **d_tb_list=nullptr, **d_tb2_list=nullptr;
  hip_bfloat16 **d_qkv_list=nullptr, **d_q_list=nullptr, **d_k_list=nullptr, **d_v_list=nullptr;
  hip_bfloat16 **d_router_list=nullptr, **d_mlp1_out_list=nullptr, **d_gate_list=nullptr, **d_up_list=nullptr, **d_gate_up_list=nullptr, **d_e_agg_list=nullptr;
  hip_bfloat16 **d_att_list=nullptr, **d_cos_list=nullptr, **d_sin_list=nullptr;
  float        **d_acc_list=nullptr, **d_logitsfp32_list=nullptr;

  // top-k per-slot pointer lists (for batched MOE)
  int           **d_topk_i_list=nullptr;       // [B] -> int*
  hip_bfloat16  **d_topk_v_list=nullptr;       // [B] -> bf16*

  // MOE weight/bias pointer lists per-slot (built per K)
  hip_bfloat16 **d_w1_list=nullptr, **d_w2_list=nullptr;
  hip_bfloat16 **d_b1_list=nullptr, **d_b2_list=nullptr;

  // host staging (reused per step)
  std::vector<hip_bfloat16*> h_x_list, h_t_list, h_tb_list, h_tb2_list;
  std::vector<hip_bfloat16*> h_qkv_list, h_q_list, h_k_list, h_v_list;
  std::vector<hip_bfloat16*> h_router_list, h_mlp1_out_list, h_gate_list, h_up_list, h_gate_up_list, h_e_agg_list;
  std::vector<hip_bfloat16*> h_att_list, h_cos_list, h_sin_list;
  std::vector<float*>         h_acc_list, h_logitsfp32_list;
  // host staging for top-k per-slot
  std::vector<int*>           h_topk_i_list;
  std::vector<hip_bfloat16*>  h_topk_v_list;
  // host staging for MOE weight/bias pointers (optional, we build on device but keep vectors for size)
  std::vector<hip_bfloat16*>  h_w1_list, h_w2_list;
  std::vector<hip_bfloat16*>  h_b1_list, h_b2_list;
};

void prefill(Transformer *transformer, int token, int pos,
             GPUContext *gpu_ctx, GPURunState *s, hipStream_t stream);

static std::vector<GPUContext> gpu_contexts;
static thread_local GPUContext* t_ctx = nullptr;

// ------------------------- Simple blocking MPMC queue -------------------------

template <typename T>
class MPMCQueue {
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_ = false;
public:
  void push(const T& v) {
    std::unique_lock<std::mutex> lk(m_);
    if (closed_) return;
    q_.push_back(v);
    cv_.notify_one();
  }
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&]{ return closed_ || !q_.empty(); });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }
  void close() {
    std::unique_lock<std::mutex> lk(m_);
    closed_ = true;
    cv_.notify_all();
  }
};

// ------------------------- BF16 utils -------------------------

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) { return hip_bfloat16(f); }
__device__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) { return float(bf); }

// ------------------------- Router TopK Softmax Kernel -------------------------

__global__ void router_topk_softmax_kernel_bf16(const hip_bfloat16 *router_scores, 
                                                hip_bfloat16 *topk_values,
                                                int *topk_indices, 
                                                int num_experts, 
                                                int k) {
    extern __shared__ float shared_mem[];
    float *values = shared_mem;
    int *indices = (int*)(values + blockDim.x);
    
    // Convert all scores to float and initialize local indices
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        values[i] = bfloat16_to_float(router_scores[i]);
        indices[i] = i;
    }
    __syncthreads();
    
    // Simple parallel selection sort for top-k
    for (int i = 0; i < k; i++) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int j = threadIdx.x + i; j < num_experts; j += blockDim.x) {
            if (values[j] > max_val) {
                max_val = values[j];
                max_idx = j;
            }
        }
        
        // Reduce to find max across threads
        __shared__ float thread_max_vals[256];
        __shared__ int thread_max_idxs[256];
        
        thread_max_vals[threadIdx.x] = max_val;
        thread_max_idxs[threadIdx.x] = max_idx;
        __syncthreads();
        
        // Reduction to find max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                if (thread_max_vals[threadIdx.x] < thread_max_vals[threadIdx.x + stride]) {
                    thread_max_vals[threadIdx.x] = thread_max_vals[threadIdx.x + stride];
                    thread_max_idxs[threadIdx.x] = thread_max_idxs[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        
        // Thread 0 has the max now
        if (threadIdx.x == 0) {
            // Swap the max element with the i-th element
            float temp_val = values[i];
            int temp_idx = indices[i];
            
            values[i] = values[thread_max_idxs[0]];
            indices[i] = indices[thread_max_idxs[0]];
            
            values[thread_max_idxs[0]] = temp_val;
            indices[thread_max_idxs[0]] = temp_idx;
        }
        __syncthreads();
    }
    
    // Apply softmax to top-k values
    if (threadIdx.x == 0) {
        // Find max for numerical stability
        float max_val = values[0];
        for (int i = 1; i < k; i++) {
            if (values[i] > max_val) max_val = values[i];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            values[i] = expf(values[i] - max_val);
            sum += values[i];
        }
        
        // Normalize and convert back to bf16
        float scale = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < k; i++) {
            topk_values[i] = float_to_bfloat16(values[i] * scale);
            topk_indices[i] = indices[i];
        }
    }
}

// ------------------------- Kernels (single-slot) -------------------------

__global__ void convert_fp32_to_bf16_kernel(const float *fp32_data, hip_bfloat16 *bf16_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) bf16_data[tid] = float_to_bfloat16(fp32_data[tid]);
}

__global__ void convert_bf16_to_fp32_kernel(const hip_bfloat16 *bf16_data, float *fp32_data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) fp32_data[tid] = bfloat16_to_float(bf16_data[tid]);
}

__global__ void rmsnorm_kernel_bf16(hip_bfloat16 *o, const hip_bfloat16 *x, const hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { float xv = bfloat16_to_float(x[j]); ss += xv * xv; }
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
    for (int j = 0; j < n; j++) { val += bfloat16_to_float(w[i * n + j]) * bfloat16_to_float(x[j]); }
    xout[i] = float_to_bfloat16(val);
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

__global__ void softmax_kernel_bf16(hip_bfloat16 *x, int size) {
  __shared__ float max_val;
  __shared__ float sum_val;
  if (threadIdx.x == 0) {
    if (size <= 0) return;
    max_val = bfloat16_to_float(x[0]);
    for (int i = 1; i < size; i++) { float v = bfloat16_to_float(x[i]); if (v > max_val) max_val = v; }
    sum_val = 0.0f;
    for (int i = 0; i < size; i++) { float v = expf(bfloat16_to_float(x[i]) - max_val); x[i] = float_to_bfloat16(v); sum_val += v; }
    float inv = 1.0f / (sum_val + 1e-12f);
    for (int i = 0; i < size; i++) { float v = bfloat16_to_float(x[i]) * inv; x[i] = float_to_bfloat16(v); }
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
      if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
      float mask = 1.0f - ramp;
      inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
    } else { inv_freq = 1.0f / freq; }
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
    x[h * head_dim + i]        = float_to_bfloat16(o1);
    x[h * head_dim + half + i] = float_to_bfloat16(o2);
  }
}

__global__ void attention_scores_kernel_bf16(hip_bfloat16 *att, const hip_bfloat16 *q, const hip_bfloat16 *key_cache,
                                             int head_dim, int loff, int kv_dim, int kv_mul,
                                             int h, int pos, int seq_len, const hip_bfloat16 *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t <= pos) {
    const hip_bfloat16 *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) { score += bfloat16_to_float(q[i]) * bfloat16_to_float(k[i]); }
    score /= sqrtf((float)head_dim);
    if (mask != NULL) { float mv = bfloat16_to_float(mask[pos * seq_len + t]); score += mv; }
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
      sum += bfloat16_to_float(att[t]) * bfloat16_to_float(v[i]);
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
  if (tid < intermediate_dim) { gate[tid] = mlp1_out[2 * tid]; up[tid] = mlp1_out[2 * tid + 1]; }
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
  if (tid < size) { x[tid] = float_to_bfloat16(bfloat16_to_float(x[tid]) + bfloat16_to_float(residual[tid])); }
}

__global__ void matvec_wbf16_xfp32_outfp32(float *out, const float *x,
                                           const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  if (i < d) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) { sum += (float)w[i * n + j] * x[j]; }
    out[i] = sum;
  }
}

// ------------------------- Kernels (batched via pointer lists) -------------------------

// ------------------------- MOE (device-only) -------------------------
// Single-slot MOE forward: reads topk indices/weights on device and computes e_agg in-place (BF16)
__global__ void moe_forward_single_kernel_bf16(
    hip_bfloat16 *e_agg,
    const hip_bfloat16 *t,
    const hip_bfloat16 *w_mlp1, const hip_bfloat16 *b_mlp1,
    const hip_bfloat16 *w_mlp2, const hip_bfloat16 *b_mlp2,
    const int *topk_i, const hip_bfloat16 *topk_v, int K,
    int hidden_dim, int intermediate_dim, int n_experts, int layer_idx,
    hip_bfloat16 *gate_up, hip_bfloat16 *tb2,
    float swiglu_limit) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  // Loop over top-K experts
  for (int idx = 0; idx < K; ++idx) {
    int e = topk_i[idx];
    float ew = bfloat16_to_float(topk_v[idx]);
    // Offsets
    size_t base_mlp1 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
    size_t base_b1   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);
    size_t base_mlp2 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
    size_t base_b2   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;

    // Compute gate_up[j] for j in [0, intermediate_dim)
    for (int j = tid; j < intermediate_dim; j += nthreads) {
      // rows 2*j and 2*j+1
      const hip_bfloat16 *row_gate = w_mlp1 + base_mlp1 + (size_t)(2 * j) * (size_t)hidden_dim;
      const hip_bfloat16 *row_up   = w_mlp1 + base_mlp1 + (size_t)(2 * j + 1) * (size_t)hidden_dim;
      float sg = 0.0f, su = 0.0f;
      for (int k = 0; k < hidden_dim; ++k) {
        sg += bfloat16_to_float(row_gate[k]) * bfloat16_to_float(t[k]);
        su += bfloat16_to_float(row_up[k])   * bfloat16_to_float(t[k]);
      }
      sg += bfloat16_to_float(b_mlp1[base_b1 + 2 * (size_t)j]);
      su += bfloat16_to_float(b_mlp1[base_b1 + 2 * (size_t)j + 1]);
      if (sg   >  swiglu_limit) sg   =  swiglu_limit; // match CPU logic
      if (su   >  swiglu_limit) su   =  swiglu_limit;
      if (su   < -swiglu_limit) su   = -swiglu_limit;
      const float alpha = 1.702f;
      sg *= (1.0f / (1.0f + expf(-alpha * sg)));
      float gu = sg * (su + 1.0f);
      gate_up[j] = float_to_bfloat16(gu);
    }
    __syncthreads();

    // Compute tb2[i] = W2 * gate_up + b2, then accumulate into e_agg with weight ew
    for (int i = tid; i < hidden_dim; i += nthreads) {
      const hip_bfloat16 *row2 = w_mlp2 + base_mlp2 + (size_t)i * (size_t)intermediate_dim;
      float sum = 0.0f;
      for (int j = 0; j < intermediate_dim; ++j) {
        sum += bfloat16_to_float(row2[j]) * bfloat16_to_float(gate_up[j]);
      }
      sum += bfloat16_to_float(b_mlp2[base_b2 + (size_t)i]);
      tb2[i] = float_to_bfloat16(sum);
      float cur = bfloat16_to_float(e_agg[i]);
      e_agg[i] = float_to_bfloat16(cur + sum * ew);
    }
    __syncthreads();
  }
}

// Batched MOE: one block per batch item (grid.y = B), no host-device copies of top-k
extern "C" __global__
void moe_forward_bf16_ptrs(
    hip_bfloat16 **e_agg_list, hip_bfloat16 **t_list,
    const hip_bfloat16 *w_mlp1, const hip_bfloat16 *b_mlp1,
    const hip_bfloat16 *w_mlp2, const hip_bfloat16 *b_mlp2,
    int **topk_i_list, hip_bfloat16 **topk_v_list, int K,
    int hidden_dim, int intermediate_dim, int n_experts, int layer_idx,
    hip_bfloat16 **gate_up_list, hip_bfloat16 **tb2_list,
    float swiglu_limit) {
  int b = blockIdx.y; int tid = threadIdx.x; int nthreads = blockDim.x;
  hip_bfloat16 *e_agg = e_agg_list[b];
  const hip_bfloat16 *t = t_list[b];
  int *topk_i = topk_i_list[b];
  hip_bfloat16 *topk_v = topk_v_list[b];
  hip_bfloat16 *gate_up = gate_up_list[b];
  hip_bfloat16 *tb2 = tb2_list[b];

  // Loop over top-K experts
  for (int idx = 0; idx < K; ++idx) {
    int e = topk_i[idx];
    float ew = bfloat16_to_float(topk_v[idx]);
    size_t base_mlp1 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
    size_t base_b1   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);
    size_t base_mlp2 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
    size_t base_b2   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;

    // gate_up
    for (int j = tid; j < intermediate_dim; j += nthreads) {
      const hip_bfloat16 *row_gate = w_mlp1 + base_mlp1 + (size_t)(2 * j) * (size_t)hidden_dim;
      const hip_bfloat16 *row_up   = w_mlp1 + base_mlp1 + (size_t)(2 * j + 1) * (size_t)hidden_dim;
      float sg = 0.0f, su = 0.0f;
      for (int k = 0; k < hidden_dim; ++k) {
        sg += bfloat16_to_float(row_gate[k]) * bfloat16_to_float(t[k]);
        su += bfloat16_to_float(row_up[k])   * bfloat16_to_float(t[k]);
      }
      sg += bfloat16_to_float(b_mlp1[base_b1 + 2 * (size_t)j]);
      su += bfloat16_to_float(b_mlp1[base_b1 + 2 * (size_t)j + 1]);
      if (sg   >  swiglu_limit) sg   =  swiglu_limit;
      if (su   >  swiglu_limit) su   =  swiglu_limit;
      if (su   < -swiglu_limit) su   = -swiglu_limit;
      const float alpha = 1.702f;
      sg *= (1.0f / (1.0f + expf(-alpha * sg)));
      float gu = sg * (su + 1.0f);
      gate_up[j] = float_to_bfloat16(gu);
    }
    __syncthreads();

    // tb2 and accumulate
    for (int i = tid; i < hidden_dim; i += nthreads) {
      const hip_bfloat16 *row2 = w_mlp2 + base_mlp2 + (size_t)i * (size_t)intermediate_dim;
      float sum = 0.0f;
      for (int j = 0; j < intermediate_dim; ++j) {
        sum += bfloat16_to_float(row2[j]) * bfloat16_to_float(gate_up[j]);
      }
      sum += bfloat16_to_float(b_mlp2[base_b2 + (size_t)i]);
      tb2[i] = float_to_bfloat16(sum);
      float cur = bfloat16_to_float(e_agg[i]);
      e_agg[i] = float_to_bfloat16(cur + sum * ew);
    }
    __syncthreads();
  }
}

extern "C" __global__
void rmsnorm_kernel_bf16_ptrs(hip_bfloat16 **o_list, hip_bfloat16 **x_list,
                              const hip_bfloat16 *weight, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  hip_bfloat16 *o = o_list[b];
  hip_bfloat16 *x = x_list[b];
  __shared__ float shared_ss[1];
  if (threadIdx.x == 0) {
    float ss = 0.f;
    for (int j=0;j<size;j++){ float xv=float(x[j]); ss += xv*xv; }
    ss = ss/size + 1e-5f; shared_ss[0] = rsqrtf(ss);
  }
  __syncthreads();
  if (tid < size) { o[tid] = hip_bfloat16(float(weight[tid]) * (shared_ss[0] * float(x[tid]))); }
}

extern "C" __global__
void add_bias_kernel_bf16_ptrs(hip_bfloat16 **out_list, const hip_bfloat16 *bias, int size){
  int tid = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  hip_bfloat16 *out = out_list[b];
  if (tid < size) out[tid] = hip_bfloat16(float(out[tid]) + float(bias[tid]));
}

// Variant: bias per-slot (pointer list)
extern "C" __global__
void add_bias_kernel_bf16_ptrs_varbias(hip_bfloat16 **out_list, hip_bfloat16 **bias_list, int size){
  int tid = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  hip_bfloat16 *out = out_list[b]; const hip_bfloat16 *bias = bias_list[b];
  if (tid < size) out[tid] = hip_bfloat16(float(out[tid]) + float(bias[tid]));
}

extern "C" __global__
void matmul_kernel_bf16_ptrs(hip_bfloat16 **xout_list, hip_bfloat16 **x_list,
                             const hip_bfloat16 *w, int n, int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  hip_bfloat16 *xout = xout_list[b]; hip_bfloat16 *x = x_list[b];
  if (row < d) { float val = 0.f; for (int j=0;j<n;j++) val += float(w[row*n + j]) * float(x[j]); xout[row] = hip_bfloat16(val); }
}

// Variant: per-slot weight pointer list
extern "C" __global__
void matmul_kernel_bf16_ptrs_w(hip_bfloat16 **xout_list, hip_bfloat16 **x_list,
                               const hip_bfloat16 **w_list, int n, int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  hip_bfloat16 *xout = xout_list[b]; hip_bfloat16 *x = x_list[b]; const hip_bfloat16 *w = w_list[b];
  if (row < d) { float val = 0.f; for (int j=0;j<n;j++) val += float(w[row*n + j]) * float(x[j]); xout[row] = hip_bfloat16(val); }
}

extern "C" __global__
void matvec_wbf16_xfp32_outfp32_ptrs(float **out_list, float **x_list,
                                     const hip_bfloat16 *w, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  if (i < d) { float sum=0.f; float *x = x_list[b]; for (int j=0;j<n;++j) sum += (float)w[i*n+j]*x[j]; out_list[b][i]=sum; }
}

// Batched split gate/up
extern "C" __global__
void split_gate_up_kernel_bf16_ptrs(hip_bfloat16 **gate_list, hip_bfloat16 **up_list, hip_bfloat16 **mlp1_out_list, int intermediate_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  if (tid < intermediate_dim) { hip_bfloat16 *mlp1 = mlp1_out_list[b]; hip_bfloat16 *gate = gate_list[b]; hip_bfloat16 *up = up_list[b]; gate[tid] = mlp1[2*tid]; up[tid] = mlp1[2*tid + 1]; }
}

// Batched swiglu
extern "C" __global__
void swiglu_kernel_bf16_ptrs(hip_bfloat16 **gate_up_list, hip_bfloat16 **gate_list, hip_bfloat16 **up_list,
                             int intermediate_dim, float swiglu_limit) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  hip_bfloat16 *gate_up = gate_up_list[b]; const hip_bfloat16 *gate = gate_list[b]; const hip_bfloat16 *up = up_list[b];
  if (tid < intermediate_dim) {
    float val = float(gate[tid]); float upval = float(up[tid]);
    if (val   >  swiglu_limit) val   =  swiglu_limit;
    if (upval >  swiglu_limit) upval =  swiglu_limit;
    if (upval < -swiglu_limit) upval = -swiglu_limit;
    const float alpha = 1.702f;
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    val *= (upval + 1.0f);
    gate_up[tid] = hip_bfloat16(val);
  }
}

// Build per-slot pointers to selected MOE expert weights/biases for k_idx
extern "C" __global__
void build_moe_ptrs_kernel(
    const hip_bfloat16 *w_mlp1, const hip_bfloat16 *b_mlp1,
    const hip_bfloat16 *w_mlp2, const hip_bfloat16 *b_mlp2,
    int **topk_i_list, int k_idx, int layer_idx,
    int hidden_dim, int intermediate_dim, int n_experts,
    hip_bfloat16 **w1_out, hip_bfloat16 **b1_out,
    hip_bfloat16 **w2_out, hip_bfloat16 **b2_out) {
  int b = blockIdx.x;
  int e = topk_i_list[b][k_idx];
  size_t base_mlp1 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
  size_t base_b1   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)(2 * intermediate_dim);
  size_t base_mlp2 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
  size_t base_b2   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;
  w1_out[b] = (hip_bfloat16*)(w_mlp1 + base_mlp1);
  b1_out[b] = (hip_bfloat16*)(b_mlp1 + base_b1);
  w2_out[b] = (hip_bfloat16*)(w_mlp2 + base_mlp2);
  b2_out[b] = (hip_bfloat16*)(b_mlp2 + base_b2);
}

// Weighted sum per-slot with expert weight read from topk_v_list[b][k_idx]
extern "C" __global__
void weighted_sum_kernel_bf16_ptrs(hip_bfloat16 **e_agg_list, hip_bfloat16 **tb2_list,
                                   hip_bfloat16 **topk_v_list, int k_idx, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
  if (tid < hidden_dim) {
    hip_bfloat16 *e_agg = e_agg_list[b]; const hip_bfloat16 *tb2 = tb2_list[b];
    float ew = float(topk_v_list[b][k_idx]);
    float agg = float(e_agg[tid]); float add = float(tb2[tid]);
    e_agg[tid] = hip_bfloat16(agg + add * ew);
  }
}

// ------------------------- MOE (single-slot, row-parallel) -------------------------
__global__ void moe_mlp1_single_kernel(hip_bfloat16 *mlp1_out, const hip_bfloat16 *t,
                                       const hip_bfloat16 *w_mlp1, const hip_bfloat16 *b_mlp1,
                                       const int *topk_i, int k_idx,
                                       int hidden_dim, int intermediate_dim, int n_experts, int layer_idx) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int out_dim = 2 * intermediate_dim;
  if (row < out_dim) {
    int e = topk_i[k_idx];
    size_t base_mlp1 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)out_dim * (size_t)hidden_dim;
    size_t base_b1   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)out_dim;
    const hip_bfloat16 *row_w = w_mlp1 + base_mlp1 + (size_t)row * (size_t)hidden_dim;
    float sum = 0.f; for (int j=0;j<hidden_dim;++j) sum += float(row_w[j]) * float(t[j]);
    sum += float(b_mlp1[base_b1 + (size_t)row]);
    mlp1_out[row] = hip_bfloat16(sum);
  }
}

__global__ void moe_mlp2_single_kernel(hip_bfloat16 *tb2, const hip_bfloat16 *gate_up,
                                       const hip_bfloat16 *w_mlp2, const hip_bfloat16 *b_mlp2,
                                       const int *topk_i, int k_idx,
                                       int hidden_dim, int intermediate_dim, int n_experts, int layer_idx) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < hidden_dim) {
    int e = topk_i[k_idx];
    size_t base_mlp2 = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim * (size_t)intermediate_dim;
    size_t base_b2   = ((size_t)layer_idx * (size_t)n_experts + (size_t)e) * (size_t)hidden_dim;
    const hip_bfloat16 *row_w = w_mlp2 + base_mlp2 + (size_t)row * (size_t)intermediate_dim;
    float sum = 0.f; for (int j=0;j<intermediate_dim;++j) sum += float(row_w[j]) * float(gate_up[j]);
    sum += float(b_mlp2[base_b2 + (size_t)row]);
    tb2[row] = hip_bfloat16(sum);
  }
}

__global__ void weighted_sum_kernel_bf16_single_ptrw(hip_bfloat16 *e_agg, const hip_bfloat16 *tb2,
                                                     const hip_bfloat16 *topk_v, int k_idx, int hidden_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < hidden_dim) {
    float ew = float(topk_v[k_idx]); float agg = float(e_agg[tid]); float add = float(tb2[tid]);
    e_agg[tid] = hip_bfloat16(agg + add * ew);
  }
}

// ------------------------- CPU top-k -------------------------

void topk_cpu(float *topk_values, int *topk_indices, const float *router_score, int num_experts, int experts_per_token) {
  struct Pair { float value; int index; };
  Pair *pairs = (Pair*)malloc(num_experts * sizeof(Pair));
  for (int i = 0; i < num_experts; ++i) { pairs[i].value = router_score[i]; pairs[i].index = i; }
  for (int i = 0; i < experts_per_token; i++) {
    int max_idx = i;
    for (int j = i + 1; j < num_experts; j++) if (pairs[j].value > pairs[max_idx].value) max_idx = j;
    if (max_idx != i) { Pair t = pairs[i]; pairs[i] = pairs[max_idx]; pairs[max_idx] = t; }
    topk_values[i]  = pairs[i].value; topk_indices[i] = pairs[i].index;
  }
  free(pairs);
}

// ------------------------- Alloc/Free run state -------------------------
void gpu_malloc_run_state(GPURunState *s, Config *p);
void gpu_free_run_state(GPURunState *s);

// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
void gpu_free_weights(GPUTransformerWeights *gpu_w);

// ------------------------- Prefill (single-slot) -------------------------

void prefill(Transformer *transformer, int token, int pos,
             GPUContext *gpu_ctx, GPURunState *s, hipStream_t stream) {
  Config *p = &transformer->config;
  int head_dim         = p->head_dim;
  int hidden_dim       = p->hidden_dim;
  int kv_dim           = p->head_dim * p->n_kv_heads;
  int kv_mul           = p->n_attn_heads / p->n_kv_heads;
  int intermediate_dim = p->intermediate_dim;
  int n_experts        = p->n_experts;

  dim3 block(256);
  dim3 grid_hidden       = dim3((hidden_dim + block.x - 1) / block.x);
  dim3 grid_vocab        = dim3((p->vocab_size + block.x - 1) / block.x);
  dim3 grid_intermediate = dim3((intermediate_dim + block.x - 1) / block.x);
  dim3 grid_experts      = dim3((n_experts + block.x - 1) / block.x);

  // Token embedding
  hip_bfloat16 *token_row = gpu_ctx->weights.token_embedding_table + (size_t)token * (size_t)hidden_dim;
  HIP_CHECK(hipMemcpyAsync(s->x, token_row, hidden_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));

  // Layers
  for (int l = 0; l < p->n_layers; l++) {
    // Attention RMSNorm
    const hip_bfloat16 *rms_attn_w = gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->t, s->x, rms_attn_w, hidden_dim);

    // QKV projection
    int loff = l * p->seq_len * kv_dim;
    const hip_bfloat16 *w_qkv = gpu_ctx->weights.w_qkv + (size_t)l * (size_t)hidden_dim * ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv = gpu_ctx->weights.b_qkv + (size_t)l * ((size_t)head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);

    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 grid_qkv = dim3((qkv_out_dim + block.x - 1) / block.x);
    matmul_kernel_bf16<<<grid_qkv, block, 0, stream>>>(s->qkv, s->t, w_qkv, hidden_dim, qkv_out_dim);
    add_bias_kernel_bf16<<<grid_qkv, block, 0, stream>>>(s->qkv, b_qkv, qkv_out_dim);

    // Split Q, K, V
    int q_size = head_dim * p->n_attn_heads;
    int k_size = head_dim * p->n_kv_heads;
    int v_size = head_dim * p->n_kv_heads;

    HIP_CHECK(hipMemcpyAsync(s->q, s->qkv, q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(s->k, s->qkv + q_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(s->v, s->qkv + q_size + k_size, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));

    // Store K, V in cache
    hip_bfloat16 *k_cache_pos = s->key_cache + loff + pos * kv_dim;
    hip_bfloat16 *v_cache_pos = s->value_cache + loff + pos * kv_dim;
    HIP_CHECK(hipMemcpyAsync(k_cache_pos, s->k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(v_cache_pos, s->v, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));

    // RoPE
    float ntk_beta  = 32.0f; float ntk_alpha = 1.0f;
    dim3 grid_rope = dim3(((head_dim / 2) + block.x - 1) / block.x);
    compute_cos_sin_kernel_bf16<<<grid_rope, block, 0, stream>>>(s->cos_vals, s->sin_vals, pos, p->rope_theta, head_dim,
        p->rope_scaling_factor, p->initial_context_length, ntk_beta, ntk_alpha);

    apply_rotary_emb_kernel_bf16<<<p->n_attn_heads, head_dim/2, 0, stream>>>(s->q, s->cos_vals, s->sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb_kernel_bf16<<<p->n_kv_heads, head_dim/2, 0, stream>>>(s->k, s->cos_vals, s->sin_vals, p->n_kv_heads, head_dim);

    // Update K cache sau RoPE
    HIP_CHECK(hipMemcpyAsync(k_cache_pos, s->k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));

    // Multi-head attention
    for (int h = 0; h < p->n_attn_heads; h++) {
      hip_bfloat16 *q   = s->q + (size_t)h * (size_t)head_dim;
      hip_bfloat16 *att = s->att + (size_t)h * (size_t)p->seq_len;
      dim3 grid_att = dim3(((pos + 1) + block.x - 1) / block.x);
      attention_scores_kernel_bf16<<<grid_att, block, 0, stream>>>(
          att, q, s->key_cache, head_dim, loff, kv_dim, kv_mul, h, pos,
          p->seq_len, (p->sliding_window > 0 && l % 2 == 0) ? s->mask : NULL);

      int len_no_sink = pos + 1;
      if ((pos + 1) < p->seq_len) {
        HIP_CHECK(hipMemcpyAsync(att + pos + 1,
          gpu_ctx->weights.attn_sinks + (size_t)l * (size_t)p->n_attn_heads + (size_t)h,
          sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, stream));
        softmax_kernel_bf16<<<1, 1, 0, stream>>>(att, len_no_sink + 1);
      } else {
        softmax_kernel_bf16<<<1, 1, 0, stream>>>(att, len_no_sink);
      }

      dim3 grid_head_dim = dim3((head_dim + block.x - 1) / block.x);
      attention_weighted_sum_kernel_bf16<<<grid_head_dim, block, 0, stream>>>(
          s->tb, att, s->value_cache, head_dim, loff, kv_dim, kv_mul, h, pos);
    }

    // Output projection of attention
    const hip_bfloat16 *w_o = gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
    const hip_bfloat16 *b_o = gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    matmul_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    add_bias_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->tb2, b_o, hidden_dim);

    // Residual
    residual_connection_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->x, s->tb2, hidden_dim);

    // -------- MoE --------
    const hip_bfloat16 *rms_ffn_w = gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->t, s->x, rms_ffn_w, hidden_dim);

    const hip_bfloat16 *w_router = gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)n_experts;
    const hip_bfloat16 *b_router = gpu_ctx->weights.b_router + (size_t)l * (size_t)n_experts;

    matmul_kernel_bf16<<<grid_experts, block, 0, stream>>>(s->router_score, s->t, w_router, hidden_dim, n_experts);
    add_bias_kernel_bf16<<<grid_experts, block, 0, stream>>>(s->router_score, b_router, n_experts);

    // GPU top-k + softmax (per-slot)
    {
      const int K = ((Config*)(&transformer->config))->experts_per_token;
      const int threads = 256; size_t shmem = (size_t)threads * (sizeof(float) + sizeof(int));
      router_topk_softmax_kernel_bf16<<<1, threads, shmem, stream>>>(s->router_score, s->topk_v, s->topk_i, n_experts, K);
    }

    // Init aggregation and run MOE fully on device (no D2H), row-parallel
    HIP_CHECK(hipMemsetAsync(s->e_agg, 0, hidden_dim * sizeof(hip_bfloat16), stream));
    int K = p->experts_per_token;
    for (int ki = 0; ki < K; ++ki) {
      dim3 grid_m1((2*intermediate_dim + block.x - 1)/block.x);
      moe_mlp1_single_kernel<<<grid_m1, block, 0, stream>>>(s->mlp1_out, s->t,
          gpu_ctx->weights.w_mlp1, gpu_ctx->weights.b_mlp1,
          s->topk_i, ki,
          hidden_dim, intermediate_dim, n_experts, l);
      // split + swiglu
      split_gate_up_kernel_bf16<<<grid_intermediate, block, 0, stream>>>(s->gate, s->up, s->mlp1_out, intermediate_dim);
      swiglu_kernel_bf16<<<grid_intermediate, block, 0, stream>>>(s->gate_up, s->gate, s->up, intermediate_dim, p->swiglu_limit);
      // mlp2
      dim3 grid_hid((hidden_dim + block.x - 1)/block.x);
      moe_mlp2_single_kernel<<<grid_hid, block, 0, stream>>>(s->tb2, s->gate_up,
          gpu_ctx->weights.w_mlp2, gpu_ctx->weights.b_mlp2,
          s->topk_i, ki,
          hidden_dim, intermediate_dim, n_experts, l);
      // aggregate with per-slot weight stored on device
      weighted_sum_kernel_bf16_single_ptrw<<<grid_hid, block, 0, stream>>>(s->e_agg, s->tb2, s->topk_v, ki, hidden_dim);
    }

    residual_connection_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->x, s->e_agg, hidden_dim);
  }

  // Final RMSNorm
  dim3 blockH(256), gridH((p->hidden_dim + 255)/256);
  rmsnorm_kernel_bf16<<<gridH, blockH, 0, stream>>>(s->x, s->x, gpu_ctx->weights.rms_out_w, p->hidden_dim);
}


// ------------------------- Decode (batched) -------------------------

static int build_ptr_lists_decode(GPUContext *ctx, const std::vector<int> &active) {
  int B = (int)active.size();
  for (int bi=0; bi<B; ++bi) {
    auto &st = ctx->slot_states[ active[bi] ];
    ctx->h_x_list[bi]       = st.x;       ctx->h_t_list[bi]       = st.t;
    ctx->h_tb_list[bi]      = st.tb;      ctx->h_tb2_list[bi]     = st.tb2;
    ctx->h_qkv_list[bi]     = st.qkv;     ctx->h_q_list[bi]       = st.q;
    ctx->h_k_list[bi]       = st.k;       ctx->h_v_list[bi]       = st.v;
    ctx->h_router_list[bi]  = st.router_score;
    ctx->h_mlp1_out_list[bi]= st.mlp1_out;
    ctx->h_gate_list[bi]    = st.gate;    ctx->h_up_list[bi]      = st.up;
    ctx->h_gate_up_list[bi] = st.gate_up; ctx->h_e_agg_list[bi]   = st.e_agg;
    ctx->h_att_list[bi]     = st.att;     ctx->h_cos_list[bi]     = st.cos_vals;
    ctx->h_sin_list[bi]     = st.sin_vals;
    ctx->h_acc_list[bi]     = st.acc_buffer;
    ctx->h_logitsfp32_list[bi] = st.logits_fp32;
    // top-k pointers
    ctx->h_topk_i_list[bi]  = st.topk_i;
    ctx->h_topk_v_list[bi]  = st.topk_v;
  }
  size_t Bp   = sizeof(hip_bfloat16*) * B;
  size_t Bpf  = sizeof(float*) * B;
  // copy host->device 1 lần
  HIP_CHECK(hipMemcpyAsync(ctx->d_x_list,   ctx->h_x_list.data(),   Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_t_list,   ctx->h_t_list.data(),   Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_tb_list,  ctx->h_tb_list.data(),  Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_tb2_list, ctx->h_tb2_list.data(), Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_qkv_list, ctx->h_qkv_list.data(), Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_q_list,   ctx->h_q_list.data(),   Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_k_list,   ctx->h_k_list.data(),   Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_v_list,   ctx->h_v_list.data(),   Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_router_list,  ctx->h_router_list.data(),  Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_mlp1_out_list,ctx->h_mlp1_out_list.data(),Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_gate_list,    ctx->h_gate_list.data(),    Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_up_list,      ctx->h_up_list.data(),      Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_gate_up_list, ctx->h_gate_up_list.data(), Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_e_agg_list,   ctx->h_e_agg_list.data(),   Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_att_list,     ctx->h_att_list.data(),     Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_cos_list,     ctx->h_cos_list.data(),     Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_sin_list,     ctx->h_sin_list.data(),     Bp, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_acc_list,        ctx->h_acc_list.data(),        Bpf, hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_logitsfp32_list, ctx->h_logitsfp32_list.data(), Bpf, hipMemcpyHostToDevice, ctx->stream_decode));
  // top-k pointer lists
  HIP_CHECK(hipMemcpyAsync(ctx->d_topk_i_list, ctx->h_topk_i_list.data(), B * sizeof(int*), hipMemcpyHostToDevice, ctx->stream_decode));
  HIP_CHECK(hipMemcpyAsync(ctx->d_topk_v_list, ctx->h_topk_v_list.data(), B * sizeof(hip_bfloat16*), hipMemcpyHostToDevice, ctx->stream_decode));
  return B;
}

static void decode(Transformer *tr, GPUContext *ctx,
                   const std::vector<int> &active_slots,
                   const std::vector<int> &tokens,
                   const std::vector<int> &poses) {
  Config *p = &tr->config;
  const int B = build_ptr_lists_decode(ctx, active_slots);
  if (B == 0) return;

  dim3 block(256);
  dim3 grid_hiddenB((p->hidden_dim + block.x - 1)/block.x, B);
  dim3 grid_vocabB ((p->vocab_size + block.x - 1)/block.x, B);
  dim3 grid_interB ((p->intermediate_dim + block.x - 1)/block.x, B);

  // Token embedding copy per-slot (D2D), B nhỏ nên ổn
  for (int bi=0; bi<B; ++bi) {
    int tok  = tokens[bi];
    hip_bfloat16 *token_row = ctx->weights.token_embedding_table + (size_t)tok * (size_t)p->hidden_dim;
    HIP_CHECK(hipMemcpyAsync(ctx->h_x_list[bi], token_row, p->hidden_dim*sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
  }

  // Layers
  for (int l = 0; l < p->n_layers; l++) {
    const hip_bfloat16 *rms_attn_w = ctx->weights.rms_attn_w + (size_t)l * (size_t)p->hidden_dim;
    rmsnorm_kernel_bf16_ptrs<<<grid_hiddenB, block, 0, ctx->stream_decode>>>(ctx->d_t_list, ctx->d_x_list, rms_attn_w, p->hidden_dim);

    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    const hip_bfloat16 *w_qkv = ctx->weights.w_qkv + (size_t)l * (size_t)p->hidden_dim * ((size_t)p->head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)p->head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv = ctx->weights.b_qkv + (size_t)l * ((size_t)p->head_dim * (size_t)p->n_attn_heads + (size_t)2 * (size_t)p->head_dim * (size_t)p->n_kv_heads);
    dim3 grid_qkvB((qkv_out_dim + block.x - 1)/block.x, B);
    matmul_kernel_bf16_ptrs<<<grid_qkvB, block, 0, ctx->stream_decode>>>(ctx->d_qkv_list, ctx->d_t_list, w_qkv, p->hidden_dim, qkv_out_dim);
    add_bias_kernel_bf16_ptrs<<<grid_qkvB, block, 0, ctx->stream_decode>>>(ctx->d_qkv_list, b_qkv, qkv_out_dim);

    // Split Q,K,V per-slot
    int q_size = p->head_dim * p->n_attn_heads;
    int k_size = p->head_dim * p->n_kv_heads;
    int v_size = k_size;
    for (int bi=0; bi<B; ++bi) {
      auto &st = ctx->slot_states[ active_slots[bi] ];
      HIP_CHECK(hipMemcpyAsync(st.q, st.qkv, q_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
      HIP_CHECK(hipMemcpyAsync(st.k, st.qkv + q_size, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
      HIP_CHECK(hipMemcpyAsync(st.v, st.qkv + q_size + k_size, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
    }

    // K/V cache + RoPE + Attention per-slot (đảm bảo đúng)
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    for (int bi=0; bi<B; ++bi) {
      int slot = active_slots[bi]; auto &st = ctx->slot_states[slot]; int pos = poses[bi]; int loff = l * p->seq_len * kv_dim;
      // cache
      HIP_CHECK(hipMemcpyAsync(st.key_cache + loff + pos * kv_dim, st.k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
      HIP_CHECK(hipMemcpyAsync(st.value_cache + loff + pos * kv_dim, st.v, v_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
      // RoPE
      dim3 grid_rope((p->head_dim/2 + block.x - 1)/block.x);
      compute_cos_sin_kernel_bf16<<<grid_rope, block, 0, ctx->stream_decode>>>(st.cos_vals, st.sin_vals, pos, p->rope_theta, p->head_dim, p->rope_scaling_factor, p->initial_context_length, 32.0f, 1.0f);
      apply_rotary_emb_kernel_bf16<<<p->n_attn_heads, p->head_dim/2, 0, ctx->stream_decode>>>(st.q, st.cos_vals, st.sin_vals, p->n_attn_heads, p->head_dim);
      apply_rotary_emb_kernel_bf16<<<p->n_kv_heads, p->head_dim/2, 0, ctx->stream_decode>>>(st.k, st.cos_vals, st.sin_vals, p->n_kv_heads, p->head_dim);
      HIP_CHECK(hipMemcpyAsync(st.key_cache + loff + pos * kv_dim, st.k, k_size * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
      // Attention
      for (int h=0; h<p->n_attn_heads; ++h) {
        hip_bfloat16 *q = st.q + (size_t)h * (size_t)p->head_dim;
        hip_bfloat16 *att = st.att + (size_t)h * (size_t)p->seq_len;
        dim3 grid_att(((pos+1)+block.x-1)/block.x);
        attention_scores_kernel_bf16<<<grid_att, block, 0, ctx->stream_decode>>>(att, q, st.key_cache, p->head_dim, loff, kv_dim, kv_mul, h, pos, p->seq_len, (p->sliding_window>0 && l%2==0)? st.mask : NULL);
        int len_no_sink = pos + 1;
        if (pos + 1 < p->seq_len) {
          HIP_CHECK(hipMemcpyAsync(att + pos + 1, ctx->weights.attn_sinks + (size_t)l*(size_t)p->n_attn_heads + (size_t)h, sizeof(hip_bfloat16), hipMemcpyDeviceToDevice, ctx->stream_decode));
          softmax_kernel_bf16<<<1,1,0,ctx->stream_decode>>>(att, len_no_sink + 1);
        } else { softmax_kernel_bf16<<<1,1,0,ctx->stream_decode>>>(att, len_no_sink); }
        dim3 grid_hd((p->head_dim + block.x - 1)/block.x);
        attention_weighted_sum_kernel_bf16<<<grid_hd, block, 0, ctx->stream_decode>>>(st.tb, att, st.value_cache, p->head_dim, loff, kv_dim, kv_mul, h, pos);
      }
    }

    // Output proj + bias (batched)
    const hip_bfloat16 *w_o = ctx->weights.w_o + (size_t)l * (size_t)(p->head_dim * p->n_attn_heads) * (size_t)p->hidden_dim;
    const hip_bfloat16 *b_o = ctx->weights.b_o + (size_t)l * (size_t)p->hidden_dim;
    matmul_kernel_bf16_ptrs<<<grid_hiddenB, block, 0, ctx->stream_decode>>>(ctx->d_tb2_list, ctx->d_tb_list, w_o, p->head_dim * p->n_attn_heads, p->hidden_dim);
    add_bias_kernel_bf16_ptrs<<<grid_hiddenB, block, 0, ctx->stream_decode>>>(ctx->d_tb2_list, b_o, p->hidden_dim);

    // residual per-slot
    for (int bi=0; bi<B; ++bi) {
      auto &st = ctx->slot_states[ active_slots[bi] ]; dim3 grid_h((p->hidden_dim + block.x - 1)/block.x);
      residual_connection_kernel_bf16<<<grid_h, block, 0, ctx->stream_decode>>>(st.x, st.tb2, p->hidden_dim);
    }

    // --- MoE ---
    const hip_bfloat16 *rms_ffn_w = ctx->weights.rms_ffn_w + (size_t)l * (size_t)p->hidden_dim;
    rmsnorm_kernel_bf16_ptrs<<<grid_hiddenB, block, 0, ctx->stream_decode>>>(ctx->d_t_list, ctx->d_x_list, rms_ffn_w, p->hidden_dim);

    const hip_bfloat16 *w_router = ctx->weights.w_router + (size_t)l * (size_t)p->hidden_dim * (size_t)p->n_experts;
    const hip_bfloat16 *b_router = ctx->weights.b_router + (size_t)l * (size_t)p->n_experts;
    dim3 grid_exB((p->n_experts + block.x - 1)/block.x, B);
    matmul_kernel_bf16_ptrs<<<grid_exB, block, 0, ctx->stream_decode>>>(ctx->d_router_list, ctx->d_t_list, w_router, p->hidden_dim, p->n_experts);
    add_bias_kernel_bf16_ptrs<<<grid_exB, block, 0, ctx->stream_decode>>>(ctx->d_router_list, b_router, p->n_experts);

    // top-k per-slot (device-only) and zero aggregations
    for (int bi=0; bi<B; ++bi) {
      auto &st = ctx->slot_states[ active_slots[bi] ]; const int threads=256; size_t shmem = (size_t)threads*(sizeof(float)+sizeof(int));
      router_topk_softmax_kernel_bf16<<<1, threads, shmem, ctx->stream_decode>>>(st.router_score, st.topk_v, st.topk_i, p->n_experts, p->experts_per_token);
      HIP_CHECK(hipMemsetAsync(st.e_agg, 0, p->hidden_dim * sizeof(hip_bfloat16), ctx->stream_decode));
    }

    // MOE forward (batched, device-only) using per-slot weight/bias pointers and big grids
    for (int k = 0; k < p->experts_per_token; ++k) {
      // build expert pointers per-slot on device
      build_moe_ptrs_kernel<<<B, 1, 0, ctx->stream_decode>>>(
        ctx->weights.w_mlp1, ctx->weights.b_mlp1,
        ctx->weights.w_mlp2, ctx->weights.b_mlp2,
        ctx->d_topk_i_list, k, l,
        p->hidden_dim, p->intermediate_dim, p->n_experts,
        ctx->d_w1_list, ctx->d_b1_list,
        ctx->d_w2_list, ctx->d_b2_list);

      // mlp1: (2*intermediate_dim, hidden_dim) @ t -> mlp1_out
      dim3 grid_m1((2*p->intermediate_dim + block.x - 1)/block.x, B);
      matmul_kernel_bf16_ptrs_w<<<grid_m1, block, 0, ctx->stream_decode>>>(ctx->d_mlp1_out_list, ctx->d_t_list, (const hip_bfloat16**)ctx->d_w1_list, p->hidden_dim, 2*p->intermediate_dim);
      add_bias_kernel_bf16_ptrs_varbias<<<grid_m1, block, 0, ctx->stream_decode>>>(ctx->d_mlp1_out_list, ctx->d_b1_list, 2*p->intermediate_dim);

      // split + swiglu
      dim3 grid_int((p->intermediate_dim + block.x - 1)/block.x, B);
      split_gate_up_kernel_bf16_ptrs<<<grid_int, block, 0, ctx->stream_decode>>>(ctx->d_gate_list, ctx->d_up_list, ctx->d_mlp1_out_list, p->intermediate_dim);
      swiglu_kernel_bf16_ptrs<<<grid_int, block, 0, ctx->stream_decode>>>(ctx->d_gate_up_list, ctx->d_gate_list, ctx->d_up_list, p->intermediate_dim, p->swiglu_limit);

      // mlp2: (hidden_dim, intermediate_dim) @ gate_up -> tb2
      dim3 grid_h((p->hidden_dim + block.x - 1)/block.x, B);
      matmul_kernel_bf16_ptrs_w<<<grid_h, block, 0, ctx->stream_decode>>>(ctx->d_tb2_list, ctx->d_gate_up_list, (const hip_bfloat16**)ctx->d_w2_list, p->intermediate_dim, p->hidden_dim);
      add_bias_kernel_bf16_ptrs_varbias<<<grid_h, block, 0, ctx->stream_decode>>>(ctx->d_tb2_list, ctx->d_b2_list, p->hidden_dim);

      // aggregate with per-slot expert weight
      weighted_sum_kernel_bf16_ptrs<<<grid_h, block, 0, ctx->stream_decode>>>(ctx->d_e_agg_list, ctx->d_tb2_list, ctx->d_topk_v_list, k, p->hidden_dim);
    }

    // Residual add per-slot
    for (int bi=0; bi<B; ++bi) {
      auto &st = ctx->slot_states[ active_slots[bi] ]; dim3 grid_h2((p->hidden_dim + block.x - 1)/block.x);
      residual_connection_kernel_bf16<<<grid_h2, block, 0, ctx->stream_decode>>>(st.x, st.e_agg, p->hidden_dim);
    }
  }

  // Final rmsnorm (batched)
  rmsnorm_kernel_bf16_ptrs<<<grid_hiddenB, block, 0, ctx->stream_decode>>>(ctx->d_x_list, ctx->d_x_list, ctx->weights.rms_out_w, p->hidden_dim);

  // x(BF16)->acc(FP32) per-slot
  for (int bi=0; bi<B; ++bi) {
    auto &st = ctx->slot_states[ active_slots[bi] ]; dim3 grid_h((p->hidden_dim + block.x - 1)/block.x);
    convert_bf16_to_fp32_kernel<<<grid_h, block, 0, ctx->stream_decode>>>(st.x, st.acc_buffer, p->hidden_dim);
  }

  // logits (batched FP32)
  matvec_wbf16_xfp32_outfp32_ptrs<<<grid_vocabB, block, 0, ctx->stream_decode>>>(ctx->d_logitsfp32_list, ctx->d_acc_list, ctx->weights.out, p->hidden_dim, p->vocab_size);
}

// ------------------------- Warm-up & Finish -------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  int device_count = 0; HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) { fprintf(stderr, "No HIP devices found.\n"); exit(EXIT_FAILURE); }
  int max_devices = device_count;
  gpu_contexts.clear(); gpu_contexts.resize(max_devices);

  for (int d = 0; d < max_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    gpu_contexts[d].device_id = d;
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config);

    if (transformer->config.sliding_window > 0 && gpu_contexts[d].state.mask != NULL) {
      int S = transformer->config.seq_len; std::vector<float> cpu_mask((size_t)S*(size_t)S);
      for (int i=0;i<S;i++) for (int j=0;j<S;j++) cpu_mask[(size_t)i*(size_t)S + (size_t)j] = (i - j >= transformer->config.sliding_window) ? -INFINITY : 0.0f;
      float *gpu_mask_fp32 = nullptr; HIP_CHECK(hipMalloc(&gpu_mask_fp32, (size_t)S*(size_t)S*sizeof(float)));
      HIP_CHECK(hipMemcpy(gpu_mask_fp32, cpu_mask.data(), (size_t)S*(size_t)S*sizeof(float), hipMemcpyHostToDevice));
      dim3 block(256); dim3 grid = dim3((((size_t)S*(size_t)S) + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block>>>(gpu_mask_fp32, gpu_contexts[d].state.mask, (int)((size_t)S * (size_t)S));
      HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipFree(gpu_mask_fp32));
    }
  }
  printf("[GPU] Warm-up done on %d device(s). BF16 weights loaded to VRAM.\n", max_devices);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    
    // Cleanup streams
    if (gpu_ctx.stream_decode) HIP_CHECK(hipStreamDestroy(gpu_ctx.stream_decode));
    if (gpu_ctx.stream_prefill) HIP_CHECK(hipStreamDestroy(gpu_ctx.stream_prefill));
    
    // Cleanup slot states and events
    for (auto &s : gpu_ctx.slot_states) gpu_free_run_state(&s);
    for (auto &ev : gpu_ctx.prefill_done) if (ev) HIP_CHECK(hipEventDestroy(ev));
    gpu_ctx.slot_states.clear(); 
    gpu_ctx.prefill_done.clear();

    // Cleanup device pointer lists
    if (gpu_ctx.d_x_list)        HIP_CHECK(hipFree(gpu_ctx.d_x_list));
    if (gpu_ctx.d_t_list)        HIP_CHECK(hipFree(gpu_ctx.d_t_list));
    if (gpu_ctx.d_tb_list)       HIP_CHECK(hipFree(gpu_ctx.d_tb_list));
    if (gpu_ctx.d_tb2_list)      HIP_CHECK(hipFree(gpu_ctx.d_tb2_list));
    if (gpu_ctx.d_qkv_list)      HIP_CHECK(hipFree(gpu_ctx.d_qkv_list));
    if (gpu_ctx.d_q_list)        HIP_CHECK(hipFree(gpu_ctx.d_q_list));
    if (gpu_ctx.d_k_list)        HIP_CHECK(hipFree(gpu_ctx.d_k_list));
    if (gpu_ctx.d_v_list)        HIP_CHECK(hipFree(gpu_ctx.d_v_list));
    if (gpu_ctx.d_router_list)   HIP_CHECK(hipFree(gpu_ctx.d_router_list));
    if (gpu_ctx.d_mlp1_out_list) HIP_CHECK(hipFree(gpu_ctx.d_mlp1_out_list));
    if (gpu_ctx.d_gate_list)     HIP_CHECK(hipFree(gpu_ctx.d_gate_list));
    if (gpu_ctx.d_up_list)       HIP_CHECK(hipFree(gpu_ctx.d_up_list));
    if (gpu_ctx.d_gate_up_list)  HIP_CHECK(hipFree(gpu_ctx.d_gate_up_list));
    if (gpu_ctx.d_e_agg_list)    HIP_CHECK(hipFree(gpu_ctx.d_e_agg_list));
    if (gpu_ctx.d_att_list)      HIP_CHECK(hipFree(gpu_ctx.d_att_list));
    if (gpu_ctx.d_cos_list)      HIP_CHECK(hipFree(gpu_ctx.d_cos_list));
    if (gpu_ctx.d_sin_list)      HIP_CHECK(hipFree(gpu_ctx.d_sin_list));
    if (gpu_ctx.d_topk_i_list)   HIP_CHECK(hipFree(gpu_ctx.d_topk_i_list));
    if (gpu_ctx.d_topk_v_list)   HIP_CHECK(hipFree(gpu_ctx.d_topk_v_list));
    if (gpu_ctx.d_w1_list)       HIP_CHECK(hipFree(gpu_ctx.d_w1_list));
    if (gpu_ctx.d_w2_list)       HIP_CHECK(hipFree(gpu_ctx.d_w2_list));
    if (gpu_ctx.d_b1_list)       HIP_CHECK(hipFree(gpu_ctx.d_b1_list));
    if (gpu_ctx.d_b2_list)       HIP_CHECK(hipFree(gpu_ctx.d_b2_list));
    if (gpu_ctx.d_acc_list)      HIP_CHECK(hipFree(gpu_ctx.d_acc_list));
    if (gpu_ctx.d_logitsfp32_list) HIP_CHECK(hipFree(gpu_ctx.d_logitsfp32_list));

    // Cleanup weights and legacy state
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

  // Global job queue
  MPMCQueue<int> jobq; 
  #pragma omp parallel for
  for (int i = 0; i < requests->num_reqs; ++i) jobq.push(i);
  jobq.close();

  std::atomic<long long> total_out{0};
  std::vector<std::thread> workers; workers.reserve(num_devices);

  const int SLOTS = std::min((int)MAX_SLOTS, 4);

  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[t].device_id));
      GPUContext *ctx = &gpu_contexts[t]; t_ctx = ctx;

      // Streams with priorities
      int leastPri = 0, greatestPri = 0; 
      HIP_CHECK(hipDeviceGetStreamPriorityRange(&leastPri, &greatestPri));
      HIP_CHECK(hipStreamCreateWithPriority(&ctx->stream_decode, hipStreamDefault, greatestPri));
      HIP_CHECK(hipStreamCreateWithPriority(&ctx->stream_prefill, hipStreamDefault, leastPri));

      ctx->num_slots = SLOTS; ctx->slot_states.resize(SLOTS); ctx->prefill_done.resize(SLOTS);
      for (int s = 0; s < SLOTS; ++s) { gpu_malloc_run_state(&ctx->slot_states[s], &transformer->config); HIP_CHECK(hipEventCreateWithFlags(&ctx->prefill_done[s], hipEventDisableTiming)); }

      // alloc device pointer lists
      auto alloc_ptr_list = [&](auto ***dptr){ HIP_CHECK(hipMalloc((void**)dptr, sizeof(**dptr) * SLOTS)); };
      alloc_ptr_list(&ctx->d_x_list);      alloc_ptr_list(&ctx->d_t_list);
      alloc_ptr_list(&ctx->d_tb_list);     alloc_ptr_list(&ctx->d_tb2_list);
      alloc_ptr_list(&ctx->d_qkv_list);    alloc_ptr_list(&ctx->d_q_list);
      alloc_ptr_list(&ctx->d_k_list);      alloc_ptr_list(&ctx->d_v_list);
      alloc_ptr_list(&ctx->d_router_list); alloc_ptr_list(&ctx->d_mlp1_out_list);
      alloc_ptr_list(&ctx->d_gate_list);   alloc_ptr_list(&ctx->d_up_list);
      alloc_ptr_list(&ctx->d_gate_up_list);alloc_ptr_list(&ctx->d_e_agg_list);
      alloc_ptr_list(&ctx->d_att_list);    alloc_ptr_list(&ctx->d_cos_list);
      alloc_ptr_list(&ctx->d_sin_list);
      // top-k pointer lists
      HIP_CHECK(hipMalloc((void**)&ctx->d_topk_i_list, sizeof(int*) * SLOTS));
      HIP_CHECK(hipMalloc((void**)&ctx->d_topk_v_list, sizeof(hip_bfloat16*) * SLOTS));
      // MOE per-slot weight/bias pointer lists
      HIP_CHECK(hipMalloc((void**)&ctx->d_w1_list, sizeof(hip_bfloat16*) * SLOTS));
      HIP_CHECK(hipMalloc((void**)&ctx->d_w2_list, sizeof(hip_bfloat16*) * SLOTS));
      HIP_CHECK(hipMalloc((void**)&ctx->d_b1_list, sizeof(hip_bfloat16*) * SLOTS));
      HIP_CHECK(hipMalloc((void**)&ctx->d_b2_list, sizeof(hip_bfloat16*) * SLOTS));
      HIP_CHECK(hipMalloc((void**)&ctx->d_acc_list,        sizeof(float*) * SLOTS));
      HIP_CHECK(hipMalloc((void**)&ctx->d_logitsfp32_list, sizeof(float*) * SLOTS));

      // host staging vectors
      ctx->h_x_list.resize(SLOTS);   ctx->h_t_list.resize(SLOTS);
      ctx->h_tb_list.resize(SLOTS);  ctx->h_tb2_list.resize(SLOTS);
      ctx->h_qkv_list.resize(SLOTS); ctx->h_q_list.resize(SLOTS);
      ctx->h_k_list.resize(SLOTS);   ctx->h_v_list.resize(SLOTS);
      ctx->h_router_list.resize(SLOTS); ctx->h_mlp1_out_list.resize(SLOTS);
      ctx->h_gate_list.resize(SLOTS);   ctx->h_up_list.resize(SLOTS);
      ctx->h_gate_up_list.resize(SLOTS);ctx->h_e_agg_list.resize(SLOTS);
      ctx->h_att_list.resize(SLOTS);    ctx->h_cos_list.resize(SLOTS);
      ctx->h_sin_list.resize(SLOTS);
      ctx->h_topk_i_list.resize(SLOTS);
      ctx->h_topk_v_list.resize(SLOTS);
      ctx->h_w1_list.resize(SLOTS); ctx->h_w2_list.resize(SLOTS);
      ctx->h_b1_list.resize(SLOTS); ctx->h_b2_list.resize(SLOTS);
      ctx->h_acc_list.resize(SLOTS);    ctx->h_logitsfp32_list.resize(SLOTS);

      Sampler local_sampler; build_sampler(&local_sampler, sampler->vocab_size, sampler->temperature, sampler->topp, sampler->rng_state + 0x9E3779B97f4A7C15ULL * (t + 1));

      std::vector<Slot> slots(SLOTS);
      auto has_active = [&](){ for (auto &s: slots) if (s.occupied && s.stage != SLOT_DONE) return true; return false; };

      long long local_out = 0; bool jobq_exhausted = false;

      while (true) {
        // Fill empty slots
        for (int s = 0; s < SLOTS; ++s) {
          if (!slots[s].occupied) {
            int idx; if (jobq.pop(idx)) {
              Slot &sl = slots[s]; sl = Slot{}; sl.occupied = true; sl.req_idx = idx;
              const char *input_seq = get_str_req_ptr(requests, idx);
              int num_prompt_tokens = 0; std::vector<int> prompt_buf(strlen(input_seq) + 3);
              encode(tokenizer, input_seq, -1, -1, prompt_buf.data(), &num_prompt_tokens, transformer->config.initial_context_length);
              prompt_buf.resize(num_prompt_tokens); sl.prompt = std::move(prompt_buf);
              sl.num_prompt_tokens = num_prompt_tokens; sl.pos = 0; sl.token = sl.prompt[0]; sl.stage = SLOT_PREFILL;
            } else { jobq_exhausted = true; }
          }
        }
        if (jobq_exhausted && !has_active()) break;

        // 1) PREFILL: mỗi slot đang prefill push 1 token
        for (int s = 0; s < SLOTS; ++s) {
          Slot &sl = slots[s]; if (!sl.occupied || sl.stage != SLOT_PREFILL) continue;
          if (sl.pos + 1 >= sl.num_prompt_tokens) { HIP_CHECK(hipEventRecord(ctx->prefill_done[s], ctx->stream_prefill)); sl.stage = SLOT_PREFILL_WAIT; continue; }
          prefill(transformer, sl.token, sl.pos, ctx, &ctx->slot_states[s], ctx->stream_prefill);
          sl.pos += 1; sl.token = sl.prompt[sl.pos];
        }

        // 2) chuyển sang DECODE khi prefill xong
        for (int s = 0; s < SLOTS; ++s) {
          Slot &sl = slots[s]; if (!sl.occupied || sl.stage != SLOT_PREFILL_WAIT) continue;
          HIP_CHECK(hipStreamWaitEvent(ctx->stream_decode, ctx->prefill_done[s], 0)); sl.stage = SLOT_DECODE;
        }

        // 3) DECODE (batched decode)
        std::vector<int> active, tokens, poses; active.reserve(SLOTS); tokens.reserve(SLOTS); poses.reserve(SLOTS);
        for (int s = 0; s < SLOTS; ++s) {
          Slot &sl = slots[s]; if (!sl.occupied || sl.stage != SLOT_DECODE) continue; active.push_back(s); tokens.push_back(sl.token); poses.push_back(sl.pos);
        }
        if (!active.empty()) {
          decode(transformer, ctx, active, tokens, poses);
          HIP_CHECK(hipStreamSynchronize(ctx->stream_decode));

          for (int i=0;i<(int)active.size();++i) {
            int s = active[i]; Slot &sl = slots[s]; auto &st = ctx->slot_states[s];
            std::vector<float> h_logits(transformer->config.vocab_size);
            HIP_CHECK(hipMemcpy(h_logits.data(), st.logits_fp32, h_logits.size()*sizeof(float), hipMemcpyDeviceToHost));

            int next;
            if (sl.pos + 1 < sl.num_prompt_tokens) next = sl.prompt[sl.pos + 1];
            else { next = sample(&local_sampler, h_logits.data()); int *out_tokens = get_tok_gen_ptr(requests, sl.req_idx); out_tokens[sl.gen_written++] = next; }

            sl.pos += 1; sl.token = next;
            if (next == 199999 || next == 200002 || sl.pos >= transformer->config.seq_len || sl.gen_written >= requests->max_seq_len) {
              int *out_tokens = get_tok_gen_ptr(requests, sl.req_idx); out_tokens[sl.gen_written] = -1; local_out += (long long)sl.gen_written; sl.stage = SLOT_DONE; sl.occupied = false;
            }
          }
        }
      }

      total_out.fetch_add(local_out, std::memory_order_relaxed);

      free_sampler(&local_sampler);
    });
  }

  for (auto &th : workers) th.join();
  return total_out.load(std::memory_order_relaxed);
}

#endif // GETP_RUN

// ------------------------- Run-state alloc/free -------------------------

void gpu_malloc_run_state(GPURunState *s, Config *p) {
  int kv_dim = p->head_dim * p->n_kv_heads;

  HIP_CHECK(hipMalloc(&s->x,   p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->t,   p->hidden_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb,  p->head_dim * p->n_attn_heads * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->tb2, p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->router_score,      p->n_experts * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_v,            p->experts_per_token * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->topk_i,            p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&s->mlp1_out, 2 * p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate,     p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->up,       p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->gate_up,  p->intermediate_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->e_agg,    p->hidden_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->qkv, p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->q,   p->n_attn_heads * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->k,   p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->v,   p->n_kv_heads   * p->head_dim * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->key_cache,   p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->att,         p->n_attn_heads * p->seq_len * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->logits,      p->vocab_size * sizeof(hip_bfloat16)));

  if (p->sliding_window > 0) { HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(hip_bfloat16))); }
  else { s->mask = NULL; }

  HIP_CHECK(hipMalloc(&s->cos_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->sin_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->acc_buffer,  p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits_fp32, p->vocab_size * sizeof(float)));
}

void gpu_free_run_state(GPURunState *s) {
  if (!s) return;
  HIP_CHECK(hipFree(s->x));           HIP_CHECK(hipFree(s->t));
  HIP_CHECK(hipFree(s->tb));          HIP_CHECK(hipFree(s->tb2));
  HIP_CHECK(hipFree(s->router_score));HIP_CHECK(hipFree(s->topk_v));
  HIP_CHECK(hipFree(s->topk_i));      HIP_CHECK(hipFree(s->mlp1_out));
  HIP_CHECK(hipFree(s->gate));        HIP_CHECK(hipFree(s->up));
  HIP_CHECK(hipFree(s->gate_up));     HIP_CHECK(hipFree(s->e_agg));
  HIP_CHECK(hipFree(s->qkv));         HIP_CHECK(hipFree(s->q));
  HIP_CHECK(hipFree(s->k));           HIP_CHECK(hipFree(s->v));
  HIP_CHECK(hipFree(s->key_cache));   HIP_CHECK(hipFree(s->value_cache));
  HIP_CHECK(hipFree(s->att));         HIP_CHECK(hipFree(s->logits));
  if (s->mask) HIP_CHECK(hipFree(s->mask));
  HIP_CHECK(hipFree(s->cos_vals));    HIP_CHECK(hipFree(s->sin_vals));
  HIP_CHECK(hipFree(s->acc_buffer));  HIP_CHECK(hipFree(s->logits_fp32));
}

// ------------------------- Weights copy/free (chunked FP32->BF16) -------------------------

void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p) {
  int head_dim      = p->head_dim;
  int n_layers      = p->n_layers;
  int n_experts     = p->n_experts;
  int hidden_dim    = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int vocab_size    = p->vocab_size;
  int n_attn_heads  = p->n_attn_heads;
  int n_kv_heads    = p->n_kv_heads;

  // Helper: host FP32 -> device staging FP32 (tile) -> device BF16 dest
  auto copy_and_convert = [&](const float *cpu_data, hip_bfloat16 **gpu_data, size_t count) {
    const size_t bf16_size = count * sizeof(hip_bfloat16);
    HIP_CHECK(hipMalloc(gpu_data, bf16_size));

    auto toGB = [](size_t bytes){ return (double)bytes / GB; };
    // printf("BF16 SIZE: %.2f GB\n", toGB(bf16_size));
    // printf("FP32 SIZE: %.2f GB\n", toGB(count * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

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
  };

  // Counts as size_t to avoid overflow
  size_t token_embed_count = (size_t)vocab_size * (size_t)hidden_dim;
  copy_and_convert(cpu_w->token_embedding_table, &gpu_w->token_embedding_table, token_embed_count);
  copy_and_convert(cpu_w->out,                    &gpu_w->out,                    token_embed_count);

  size_t rms_count = (size_t)n_layers * (size_t)hidden_dim;
  copy_and_convert(cpu_w->rms_attn_w, &gpu_w->rms_attn_w, rms_count);
  copy_and_convert(cpu_w->rms_ffn_w,  &gpu_w->rms_ffn_w,  rms_count);

  size_t rms_out_count = (size_t)hidden_dim;
  copy_and_convert(cpu_w->rms_out_w,  &gpu_w->rms_out_w,  rms_out_count);

  size_t qkv_w_count = (size_t)n_layers * (size_t)hidden_dim *
                       ((size_t)head_dim * (size_t)n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
  copy_and_convert(cpu_w->w_qkv, &gpu_w->w_qkv, qkv_w_count);

  size_t qkv_b_count = (size_t)n_layers *
                       ((size_t)head_dim * (size_t)n_attn_heads + (size_t)2 * (size_t)head_dim * (size_t)n_kv_heads);
  copy_and_convert(cpu_w->b_qkv, &gpu_w->b_qkv, qkv_b_count);

  size_t o_w_count = (size_t)n_layers * (size_t)head_dim * (size_t)n_attn_heads * (size_t)hidden_dim;
  copy_and_convert(cpu_w->w_o, &gpu_w->w_o, o_w_count);

  size_t o_b_count = (size_t)n_layers * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_o, &gpu_w->b_o, o_b_count);

  size_t attn_sinks_count = (size_t)n_layers * (size_t)n_attn_heads;
  copy_and_convert(cpu_w->attn_sinks, &gpu_w->attn_sinks, attn_sinks_count);

  size_t router_w_count = (size_t)n_layers * (size_t)hidden_dim * (size_t)n_experts;
  copy_and_convert(cpu_w->w_router, &gpu_w->w_router, router_w_count);

  size_t router_b_count = (size_t)n_layers * (size_t)n_experts;
  copy_and_convert(cpu_w->b_router, &gpu_w->b_router, router_b_count);

  size_t mlp1_w_count = (size_t)n_layers * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim * (size_t)hidden_dim;
  copy_and_convert(cpu_w->w_mlp1, &gpu_w->w_mlp1, mlp1_w_count);

  size_t mlp1_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)2 * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->b_mlp1, &gpu_w->b_mlp1, mlp1_b_count);

  size_t mlp2_w_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim * (size_t)intermediate_dim;
  copy_and_convert(cpu_w->w_mlp2, &gpu_w->w_mlp2, mlp2_w_count);

  size_t mlp2_b_count = (size_t)n_layers * (size_t)n_experts * (size_t)hidden_dim;
  copy_and_convert(cpu_w->b_mlp2, &gpu_w->b_mlp2, mlp2_b_count);
}
void gpu_free_weights(GPUTransformerWeights *gpu_w) {
  HIP_CHECK(hipFree(gpu_w->token_embedding_table)); HIP_CHECK(hipFree(gpu_w->out));
  HIP_CHECK(hipFree(gpu_w->rms_attn_w)); HIP_CHECK(hipFree(gpu_w->rms_ffn_w)); HIP_CHECK(hipFree(gpu_w->rms_out_w));
  HIP_CHECK(hipFree(gpu_w->w_qkv)); HIP_CHECK(hipFree(gpu_w->b_qkv));
  HIP_CHECK(hipFree(gpu_w->w_o)); HIP_CHECK(hipFree(gpu_w->b_o));
  HIP_CHECK(hipFree(gpu_w->attn_sinks));
  HIP_CHECK(hipFree(gpu_w->w_router)); HIP_CHECK(hipFree(gpu_w->b_router));
  HIP_CHECK(hipFree(gpu_w->w_mlp1));   HIP_CHECK(hipFree(gpu_w->b_mlp1));
  HIP_CHECK(hipFree(gpu_w->w_mlp2));   HIP_CHECK(hipFree(gpu_w->b_mlp2));
}
