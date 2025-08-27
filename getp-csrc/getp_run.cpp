// getp-csrc/getp_run.cpp
// GPU (HIP) BF16 end-to-end for getp mode.
// - No external GPU libs. All kernels written here.
// - ROCm/HIP (e.g., MI250), BF16 weights/KV; FP32 accumulate.
// - Warm-up loads weights with chunked copy+convert to avoid OOM.
// - Runtime fixes: bound pos to seq_len; guard attention sink write.

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
} GPUTransformerWeights;

typedef struct {
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
} GPURunState;

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

typedef struct {
  int device_id;
  GPUTransformerWeights weights;
  GPURunState state; // (legacy) vẫn giữ để tương thích, nhưng scheduler mới không dùng
  // ---- mới ----
  int num_slots = 0;
  hipStream_t stream_decode = nullptr; // ưu tiên cao
  hipStream_t stream_prefill = nullptr;// ưu tiên thấp
  std::vector<GPURunState> slot_states;    // mỗi slot 1 runstate riêng
  std::vector<hipEvent_t>  prefill_done;   // event báo prefill xong cho mỗi slot
} GPUContext;
float *gpu_forward_streamed(Transformer *transformer, int token, int pos,
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
    if (closed_) return;           // không nhận nữa nếu đã đóng
    q_.push_back(v);
    cv_.notify_one();
  }
  // pop() chờ đến khi có phần tử hoặc queue đóng & rỗng
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&]{ return closed_ || !q_.empty(); });
    if (q_.empty()) return false;  // closed & empty
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

__device__ __forceinline__ hip_bfloat16 float_to_bfloat16(float f) {
  return hip_bfloat16(f);
}
__device__ __forceinline__ float bfloat16_to_float(hip_bfloat16 bf) {
  return float(bf);
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

// Find top-K (small K, e.g., 1..8) over BF16 scores on device and softmax them.
// One block processes ONE vector of length n_experts.
// Complexity: O(K * n_experts), OK for typical MoE sizes (<= 64~256).
__global__ void router_topk_softmax_kernel_bf16(const hip_bfloat16 *router_bf16,
                                                hip_bfloat16 *topk_v_bf16,
                                                int *topk_i,
                                                int n_experts, int K) {
  extern __shared__ float shared[]; // shared buffer for reductions
  float *scratch_val = shared;      // size blockDim.x
  int   *scratch_idx = (int*)(shared + blockDim.x);

  // Keep chosen indices in shared (K is small)
  __shared__ int chosen_idx[16]; // assume K <= 16
  __shared__ float chosen_val[16];

  int tid = threadIdx.x;

  // Helper: read value (float) at j, ignoring already-chosen indices
  auto read_score = [&](int j)->float {
    // reject if j already chosen
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (t >= K) break;
      if (j == chosen_idx[t]) return -INFINITY;
    }
    return float(router_bf16[j]);
  };

  for (int k = 0; k < K; ++k) {
    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int j = tid; j < n_experts; j += blockDim.x) {
      float v = read_score(j);
      if (v > local_max) { local_max = v; local_idx = j; }
    }
    scratch_val[tid] = local_max;
    scratch_idx[tid] = local_idx;
    __syncthreads();

    // Reduce to global max
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
      if (tid < offset) {
        if (scratch_val[tid + offset] > scratch_val[tid]) {
          scratch_val[tid] = scratch_val[tid + offset];
          scratch_idx[tid] = scratch_idx[tid + offset];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      chosen_val[k] = scratch_val[0];
      chosen_idx[k] = scratch_idx[0] < 0 ? 0 : scratch_idx[0];
    }
    __syncthreads();
  }

  // Softmax on top-K (thread 0 is fine; K small)
  if (tid == 0) {
    float maxv = chosen_val[0];
    for (int i = 1; i < K; ++i) if (chosen_val[i] > maxv) maxv = chosen_val[i];
    float sum = 0.f;
    for (int i = 0; i < K; ++i) { chosen_val[i] = expf(chosen_val[i] - maxv); sum += chosen_val[i]; }
    float inv = 1.f / (sum + 1e-12f);
    for (int i = 0; i < K; ++i) {
      topk_i[i]       = chosen_idx[i];
      topk_v_bf16[i]  = hip_bfloat16(chosen_val[i] * inv);
    }
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
void gpu_malloc_run_state(GPURunState *s, Config *p);
void gpu_free_run_state(GPURunState *s);


// ------------------------- Chunked copy+convert of weights -------------------------
void gpu_copy_weights(GPUTransformerWeights *gpu_w, TransformerWeights *cpu_w, Config *p);
void gpu_free_weights(GPUTransformerWeights *gpu_w);



// ------------------------- Forward -------------------------




float *gpu_forward_streamed(Transformer *transformer, int token, int pos,
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
  hip_bfloat16 *token_row =
      gpu_ctx->weights.token_embedding_table + (size_t)token * (size_t)hidden_dim;
  HIP_CHECK(hipMemcpyAsync(s->x, token_row, hidden_dim * sizeof(hip_bfloat16),
                           hipMemcpyDeviceToDevice, stream));

  // Layers
  for (int l = 0; l < p->n_layers; l++) {
    // Attention RMSNorm
    const hip_bfloat16 *rms_attn_w =
        gpu_ctx->weights.rms_attn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->t, s->x, rms_attn_w, hidden_dim);

    // QKV projection
    int loff = l * p->seq_len * kv_dim;
    const hip_bfloat16 *w_qkv =
        gpu_ctx->weights.w_qkv +
        (size_t)l * (size_t)hidden_dim *
            ((size_t)head_dim * (size_t)p->n_attn_heads +
             (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);
    const hip_bfloat16 *b_qkv =
        gpu_ctx->weights.b_qkv +
        (size_t)l * ((size_t)head_dim * (size_t)p->n_attn_heads +
                     (size_t)2 * (size_t)head_dim * (size_t)p->n_kv_heads);

    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 grid_qkv = dim3((qkv_out_dim + block.x - 1) / block.x);
    matmul_kernel_bf16<<<grid_qkv, block, 0, stream>>>(s->qkv, s->t, w_qkv, hidden_dim, qkv_out_dim);
    add_bias_kernel_bf16<<<grid_qkv, block, 0, stream>>>(s->qkv, b_qkv, qkv_out_dim);

    // Split Q, K, V
    int q_size = head_dim * p->n_attn_heads;
    int k_size = head_dim * p->n_kv_heads;
    int v_size = head_dim * p->n_kv_heads;

    HIP_CHECK(hipMemcpyAsync(s->q, s->qkv, q_size * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(s->k, s->qkv + q_size, k_size * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(s->v, s->qkv + q_size + k_size, v_size * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));

    // Store K, V in cache
    hip_bfloat16 *k_cache_pos = s->key_cache + loff + pos * kv_dim;
    hip_bfloat16 *v_cache_pos = s->value_cache + loff + pos * kv_dim;
    HIP_CHECK(hipMemcpyAsync(k_cache_pos, s->k, k_size * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(v_cache_pos, s->v, v_size * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));

    // RoPE
    float ntk_beta  = 32.0f;
    float ntk_alpha = 1.0f;
    dim3 grid_rope = dim3(((head_dim / 2) + block.x - 1) / block.x);
    compute_cos_sin_kernel_bf16<<<grid_rope, block, 0, stream>>>(
        s->cos_vals, s->sin_vals, pos, p->rope_theta, head_dim,
        p->rope_scaling_factor, p->initial_context_length, ntk_beta, ntk_alpha);

    apply_rotary_emb_kernel_bf16<<<p->n_attn_heads, head_dim/2, 0, stream>>>(
        s->q, s->cos_vals, s->sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb_kernel_bf16<<<p->n_kv_heads, head_dim/2, 0, stream>>>(
        s->k, s->cos_vals, s->sin_vals, p->n_kv_heads, head_dim);

    // Update K cache sau RoPE
    HIP_CHECK(hipMemcpyAsync(k_cache_pos, s->k, k_size * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));

    // Multi-head attention
    for (int h = 0; h < p->n_attn_heads; h++) {
      hip_bfloat16 *q   = s->q + (size_t)h * (size_t)head_dim;
      hip_bfloat16 *att = s->att + (size_t)h * (size_t)p->seq_len;

      // Scores
      dim3 grid_att = dim3(((pos + 1) + block.x - 1) / block.x);
      attention_scores_kernel_bf16<<<grid_att, block, 0, stream>>>(
          att, q, s->key_cache, head_dim, loff, kv_dim, kv_mul, h, pos,
          p->seq_len,
          (p->sliding_window > 0 && l % 2 == 0) ? s->mask : NULL);

      // Attention sink an toàn (giới hạn)
      int len_no_sink = pos + 1;    // t = 0..pos
      bool can_add_sink = (pos + 1) < p->seq_len;
      if (can_add_sink) {
        HIP_CHECK(hipMemcpyAsync(
          att + pos + 1,
          gpu_ctx->weights.attn_sinks + (size_t)l * (size_t)p->n_attn_heads + (size_t)h,
          sizeof(hip_bfloat16),
          hipMemcpyDeviceToDevice, stream /* hoặc bỏ 'stream' ở bản sync */));
        softmax_kernel_bf16<<<1, 1, 0, stream>>>(att, len_no_sink + 1);

      } else {
        softmax_kernel_bf16<<<1, 1, 0, stream>>>(att, len_no_sink);
      }

      // Weighted sum (0..pos)
      dim3 grid_head_dim = dim3((head_dim + block.x - 1) / block.x);
      attention_weighted_sum_kernel_bf16<<<grid_head_dim, block, 0, stream>>>(
          s->tb, att, s->value_cache, head_dim, loff, kv_dim, kv_mul, h, pos);
    }

    // Output projection of attention
    const hip_bfloat16 *w_o =
        gpu_ctx->weights.w_o + (size_t)l * (size_t)(head_dim * p->n_attn_heads) * (size_t)hidden_dim;
    const hip_bfloat16 *b_o =
        gpu_ctx->weights.b_o + (size_t)l * (size_t)hidden_dim;
    matmul_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->tb2, s->tb, w_o,
                                                          head_dim * p->n_attn_heads, hidden_dim);
    add_bias_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->tb2, b_o, hidden_dim);

    // Residual
    residual_connection_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->x, s->tb2, hidden_dim);

    // -------- MoE --------
    // FFN RMSNorm
    const hip_bfloat16 *rms_ffn_w =
        gpu_ctx->weights.rms_ffn_w + (size_t)l * (size_t)hidden_dim;
    rmsnorm_kernel_bf16<<<grid_hidden, block, 0, stream>>>(s->t, s->x, rms_ffn_w, hidden_dim);

    // Router
    const hip_bfloat16 *w_router =
        gpu_ctx->weights.w_router + (size_t)l * (size_t)hidden_dim * (size_t)n_experts;
    const hip_bfloat16 *b_router =
        gpu_ctx->weights.b_router + (size_t)l * (size_t)n_experts;

    matmul_kernel_bf16<<<grid_experts, block, 0, stream>>>(s->router_score, s->t, w_router, hidden_dim, n_experts);
    add_bias_kernel_bf16<<<grid_experts, block, 0, stream>>>(s->router_score, b_router, n_experts);

    // GPU top-k + softmax (no host sync)
    {
      const int K = ((Config*)(&transformer->config))->experts_per_token;
      const int threads = 256;
      size_t shmem = (size_t)threads * (sizeof(float) + sizeof(int));
      router_topk_softmax_kernel_bf16<<<1, threads, shmem, stream>>>(
          s->router_score, s->topk_v, s->topk_i, n_experts, K);
    }

    // --- NEW: copy topk indices & weights về host để chọn expert và truyền expert_w ---
    const int K = p->experts_per_token;
    // chuyển BF16 -> FP32 trên device rồi mới copy host (an toàn, K nhỏ)
    float *topk_v_fp32_dev = nullptr;
    HIP_CHECK(hipMalloc(&topk_v_fp32_dev, K * sizeof(float)));
    dim3 grid_topk = dim3((K + block.x - 1) / block.x);
    convert_bf16_to_fp32_kernel<<<grid_topk, block, 0, stream>>>(s->topk_v, topk_v_fp32_dev, K);

    std::vector<int>   h_topk_i(K);
    std::vector<float> h_topk_v(K);
    HIP_CHECK(hipMemcpyAsync(h_topk_i.data(), s->topk_i, K * sizeof(int),
                            hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_topk_v.data(), topk_v_fp32_dev, K * sizeof(float),
                            hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(topk_v_fp32_dev));


    // Init aggregation
    HIP_CHECK(hipMemsetAsync(s->e_agg, 0, hidden_dim * sizeof(hip_bfloat16), stream));
    
    // Experts
    for (int idx = 0; idx < p->experts_per_token; idx++) {
      const int   e         = h_topk_i[idx];
      const float expert_w  = h_topk_v[idx];

      const hip_bfloat16 *w_mlp1 =
          gpu_ctx->weights.w_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) *
                                    (size_t)(2 * intermediate_dim) * (size_t)hidden_dim;
      const hip_bfloat16 *b_mlp1 =
          gpu_ctx->weights.b_mlp1 + ((size_t)l * (size_t)n_experts + (size_t)e) *
                                    (size_t)(2 * intermediate_dim);

      dim3 grid_mlp1 = dim3(((2 * intermediate_dim) + block.x - 1) / block.x);
      matmul_kernel_bf16<<<grid_mlp1, block, 0, stream>>>(
          s->mlp1_out, s->t, w_mlp1, hidden_dim, 2 * intermediate_dim);
      add_bias_kernel_bf16<<<grid_mlp1, block, 0, stream>>>(
          s->mlp1_out, b_mlp1, 2 * intermediate_dim);

      split_gate_up_kernel_bf16<<<grid_intermediate, block, 0, stream>>>(
          s->gate, s->up, s->mlp1_out, intermediate_dim);

      swiglu_kernel_bf16<<<grid_intermediate, block, 0, stream>>>(
          s->gate_up, s->gate, s->up, intermediate_dim, p->swiglu_limit);

      const hip_bfloat16 *w_mlp2 =
          gpu_ctx->weights.w_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) *
                                    (size_t)hidden_dim * (size_t)intermediate_dim;
      const hip_bfloat16 *b_mlp2 =
          gpu_ctx->weights.b_mlp2 + ((size_t)l * (size_t)n_experts + (size_t)e) *
                                    (size_t)hidden_dim;

      matmul_kernel_bf16<<<grid_hidden, block, 0, stream>>>(
          s->tb2, s->gate_up, w_mlp2, intermediate_dim, hidden_dim);
      add_bias_kernel_bf16<<<grid_hidden, block, 0, stream>>>(
          s->tb2, b_mlp2, hidden_dim);

      weighted_sum_kernel_bf16<<<grid_hidden, block, 0, stream>>>(
          s->e_agg, s->tb2, expert_w, hidden_dim);
    }


    residual_connection_kernel_bf16<<<grid_hidden, block, 0, stream>>>(
        s->x, s->e_agg, hidden_dim);

    // Cleanup temporaries (GPU)
    // HIP_CHECK(hipFree(gpu_router_score_fp32));
    // HIP_CHECK(hipFree(gpu_topk_v_bf16));
    // HIP_CHECK(hipFree(gpu_topk_v_fp32));
  }

  // Final RMSNorm
  rmsnorm_kernel_bf16<<<grid_hidden, block, 0, stream>>>(
      s->x, s->x, gpu_ctx->weights.rms_out_w, hidden_dim);

  // Convert x to FP32
  convert_bf16_to_fp32_kernel<<<grid_hidden, block, 0, stream>>>(
      s->x, s->acc_buffer, hidden_dim);

  // Output logits: W(BF16) * x(FP32) -> logits (FP32)
  dim3 grid_vocab2 = dim3((p->vocab_size + block.x - 1) / block.x);
  matvec_wbf16_xfp32_outfp32<<<grid_vocab2, block, 0, stream>>>(
      s->logits_fp32, s->acc_buffer, gpu_ctx->weights.out, /*n=*/hidden_dim, /*d=*/p->vocab_size);

  // Copy logits về host (đồng bộ 1 lần trước khi return)
  thread_local float *cpu_logits = NULL;
  if (cpu_logits == NULL) cpu_logits = (float*)malloc(p->vocab_size * sizeof(float));
  HIP_CHECK(hipMemcpyAsync(cpu_logits, s->logits_fp32,
                           p->vocab_size * sizeof(float),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  return cpu_logits;
}

// ------------------------- Warm-up & Finish -------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "No HIP devices found.\n");
    exit(EXIT_FAILURE);
  }
  int max_devices = device_count;
  gpu_contexts.clear();
  gpu_contexts.resize(max_devices);

  for (int d = 0; d < max_devices; ++d) {
    HIP_CHECK(hipSetDevice(d));
    gpu_contexts[d].device_id = d;
    gpu_copy_weights(&gpu_contexts[d].weights, &transformer->weights, &transformer->config);
    gpu_malloc_run_state(&gpu_contexts[d].state, &transformer->config);

    if (transformer->config.sliding_window > 0 && gpu_contexts[d].state.mask != NULL) {
      int S = transformer->config.seq_len;
      std::vector<float> cpu_mask((size_t)S * (size_t)S);
      for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
          cpu_mask[(size_t)i * (size_t)S + (size_t)j] = (i - j >= transformer->config.sliding_window) ? -INFINITY : 0.0f;
        }
      }
      float *gpu_mask_fp32 = nullptr;
      HIP_CHECK(hipMalloc(&gpu_mask_fp32, (size_t)S * (size_t)S * sizeof(float)));
      HIP_CHECK(hipMemcpy(gpu_mask_fp32, cpu_mask.data(), (size_t)S * (size_t)S * sizeof(float), hipMemcpyHostToDevice));

      dim3 block(256);
      dim3 grid = dim3((((size_t)S * (size_t)S) + block.x - 1) / block.x);
      convert_fp32_to_bf16_kernel<<<grid, block>>>(gpu_mask_fp32, gpu_contexts[d].state.mask, (int)((size_t)S * (size_t)S));
      HIP_CHECK(hipDeviceSynchronize());

      HIP_CHECK(hipFree(gpu_mask_fp32));
    }
  }
  printf("[GPU] Warm-up done on %d device(s). BF16 weights loaded to VRAM.\n", max_devices);
}



void finish(Transformer *transformer, Tokenizer *tokenizer) {
  for (auto &gpu_ctx : gpu_contexts) {
    HIP_CHECK(hipSetDevice(gpu_ctx.device_id));
    // dọn streams/events/slot_states nếu còn
    if (gpu_ctx.stream_decode) hipStreamDestroy(gpu_ctx.stream_decode);
    if (gpu_ctx.stream_prefill) hipStreamDestroy(gpu_ctx.stream_prefill);
    for (auto &s : gpu_ctx.slot_states) gpu_free_run_state(&s);
    for (auto &ev : gpu_ctx.prefill_done) if (ev) hipEventDestroy(ev);
    gpu_ctx.slot_states.clear();
    gpu_ctx.prefill_done.clear();

    gpu_free_weights(&gpu_ctx.weights);
    gpu_free_run_state(&gpu_ctx.state); // legacy
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
  std::vector<std::thread> workers;
  workers.reserve(num_devices);

  // Tham số slot/gpu (có thể cho vào config hay flag)
  const int SLOTS = std::min((int)MAX_SLOTS,  4);

  for (int t = 0; t < num_devices; ++t) {
    workers.emplace_back([&, t]() {
      HIP_CHECK(hipSetDevice(gpu_contexts[t].device_id));
      GPUContext *ctx = &gpu_contexts[t];
      t_ctx = ctx; // thread-local

      // Tạo 2 stream với độ ưu tiên
      int leastPri = 0, greatestPri = 0;
      hipDeviceGetStreamPriorityRange(&leastPri, &greatestPri);
      // HIP: số nhỏ hơn thường là ưu tiên cao hơn; ở đây mình dùng greatestPri cho decode:
      HIP_CHECK(hipStreamCreateWithPriority(&ctx->stream_decode, hipStreamDefault, greatestPri));
      HIP_CHECK(hipStreamCreateWithPriority(&ctx->stream_prefill, hipStreamDefault, leastPri));

      ctx->num_slots = SLOTS;
      ctx->slot_states.resize(SLOTS);
      ctx->prefill_done.resize(SLOTS);

      for (int s = 0; s < SLOTS; ++s) {
        gpu_malloc_run_state(&ctx->slot_states[s], &transformer->config);
        HIP_CHECK(hipEventCreateWithFlags(&ctx->prefill_done[s], hipEventDisableTiming));
      }

      Sampler local_sampler;
      build_sampler(&local_sampler, sampler->vocab_size, sampler->temperature,
                    sampler->topp, sampler->rng_state + 0x9E3779B97f4A7C15ULL * (t + 1));

      std::vector<Slot> slots(SLOTS);
      auto has_active = [&](){
        for (auto &s: slots) if (s.occupied && s.stage != SLOT_DONE) return true;
        return false;
      };

      long long local_out = 0;
      bool jobq_exhausted = false;

      while (true) {
        // Nạp job mới vào các slot trống
        for (int s = 0; s < SLOTS; ++s) {
          if (!slots[s].occupied) {
            int idx;
            if (jobq.pop(idx)) {
              Slot &sl = slots[s];
              sl = Slot{}; // reset
              sl.occupied = true;
              sl.req_idx  = idx;

              const char *input_seq = get_str_req_ptr(requests, idx);
              // Encode prompt
              int num_prompt_tokens = 0;
              std::vector<int> prompt_buf(strlen(input_seq) + 3);
              encode(tokenizer, input_seq, -1, -1, prompt_buf.data(), &num_prompt_tokens,
                     transformer->config.initial_context_length);
              prompt_buf.resize(num_prompt_tokens);
              sl.prompt = std::move(prompt_buf);
              sl.num_prompt_tokens = num_prompt_tokens;
              sl.pos   = 0;
              sl.token = sl.prompt[0];
              sl.stage = SLOT_PREFILL;
            } else {
              jobq_exhausted = true;
            }
          }
        }

        // Điều kiện thoát: hết job và không còn slot active
        if (jobq_exhausted && !has_active()) break;

        // 1) PREFILL: mỗi slot đang PREFILL đẩy đúng 1 token/lần vào stream_prefill
        for (int s = 0; s < SLOTS; ++s) {
          Slot &sl = slots[s];
          if (!sl.occupied || sl.stage != SLOT_PREFILL) continue;

          // Nếu đã nạp hết prompt -> ghi event và chuyển sang PREFILL_WAIT
          if (sl.pos + 1 >= sl.num_prompt_tokens) {
            HIP_CHECK(hipEventRecord(ctx->prefill_done[s], ctx->stream_prefill));
            sl.stage = SLOT_PREFILL_WAIT;
            continue;
          }

          // Chạy 1 bước forward trên stream_prefill (không cần đọc logits, không sample)
          (void)gpu_forward_streamed(transformer, sl.token, sl.pos, ctx,
                                     &ctx->slot_states[s], ctx->stream_prefill);

          sl.pos  += 1;
          sl.token = sl.prompt[sl.pos]; // dùng token kế tiếp trong prompt
        }

        // 2) Chuyển slot đã prefill xong sang decode (đảm bảo chờ event ở stream_decode)
        for (int s = 0; s < SLOTS; ++s) {
          Slot &sl = slots[s];
          if (!sl.occupied || sl.stage != SLOT_PREFILL_WAIT) continue;
          HIP_CHECK(hipStreamWaitEvent(ctx->stream_decode, ctx->prefill_done[s], 0));
          sl.stage = SLOT_DECODE;
        }

        // 3) DECODE: mỗi slot active decode 1 token/lần (round-robin đơn giản)
        for (int s = 0; s < SLOTS; ++s) {
          Slot &sl = slots[s];
          if (!sl.occupied || sl.stage != SLOT_DECODE) continue;

          float *logits = gpu_forward_streamed(transformer, sl.token, sl.pos, ctx,
                                               &ctx->slot_states[s], ctx->stream_decode);
          int next;
          if (sl.pos + 1 < sl.num_prompt_tokens) {
            // (Trường hợp hiếm: prompt rất dài vẫn còn—nhưng slot đã qua PREFILL_WAIT, nên
            // về logic bình thường sẽ không vào nhánh này)
            next = sl.prompt[sl.pos + 1];
          } else {
            next = sample(&local_sampler, logits);
            int *out_tokens = get_tok_gen_ptr(requests, sl.req_idx);
            out_tokens[sl.gen_written++] = next;
          }

          sl.pos  += 1;
          sl.token = next;

          // EOS hoặc đạt giới hạn
          if (next == 199999 || next == 200002 ||
              sl.pos >= transformer->config.seq_len ||
              sl.gen_written >= requests->max_seq_len) {
            int *out_tokens = get_tok_gen_ptr(requests, sl.req_idx);
            out_tokens[sl.gen_written] = -1;
            local_out += (long long)sl.gen_written;
            sl.stage = SLOT_DONE;
            sl.occupied = false; // giải phóng slot ngay để nạp job mới ở vòng sau
          }
        }
      }

      total_out.fetch_add(local_out, std::memory_order_relaxed);

      // Cleanup per-GPU
      for (int s = 0; s < SLOTS; ++s) {
        gpu_free_run_state(&ctx->slot_states[s]);
        HIP_CHECK(hipEventDestroy(ctx->prefill_done[s]));
      }
      ctx->slot_states.clear();
      ctx->prefill_done.clear();
      HIP_CHECK(hipStreamDestroy(ctx->stream_decode));
      HIP_CHECK(hipStreamDestroy(ctx->stream_prefill));

      free_sampler(&local_sampler);
    });
  }

  for (auto &th : workers) th.join();
  return total_out.load(std::memory_order_relaxed);
}


#endif // GETP_RUN




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

  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(hip_bfloat16)));
  } else {
    s->mask = NULL;
  }

  HIP_CHECK(hipMalloc(&s->cos_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&s->sin_vals, (p->head_dim / 2) * sizeof(hip_bfloat16)));

  HIP_CHECK(hipMalloc(&s->acc_buffer,  p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits_fp32, p->vocab_size * sizeof(float)));
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
  HIP_CHECK(hipFree(s->att));
  HIP_CHECK(hipFree(s->logits));
  HIP_CHECK(hipFree(s->mask));
  HIP_CHECK(hipFree(s->cos_vals));
  HIP_CHECK(hipFree(s->sin_vals));
  HIP_CHECK(hipFree(s->acc_buffer));
  HIP_CHECK(hipFree(s->logits_fp32));
}

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
  HIP_CHECK(hipFree(gpu_w->w_mlp1));
  HIP_CHECK(hipFree(gpu_w->b_mlp1));
  HIP_CHECK(hipFree(gpu_w->w_mlp2));
  HIP_CHECK(hipFree(gpu_w->b_mlp2));
}
