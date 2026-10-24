// Feeds each *_exact GPU kernel the same input as the matching CPU code in
// run.cpp / engine/cpu_forward.hpp and counts outputs that differ in any bit.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
//     getp-csrc/tests/stage_check.cpp tokenizer.cpp -o stage_check
// Run: tools/gpulock.sh <hip id> -- ./stage_check <model folder>
#define TESTING
#include "../../run.cpp"

template <class T>
static T *to_gpu(const T *h, size_t n) {
  T *d;
  HIP_CHECK(hipMalloc(&d, n * sizeof(T)));
  HIP_CHECK(hipMemcpy(d, h, n * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

static std::vector<float> from_gpu(const float *d, size_t n) {
  std::vector<float> h(n);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost));
  return h;
}

static int report(const char *name, const float *cpu, const float *gpu, size_t n) {
  size_t diff = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; i++)
    if (memcmp(&cpu[i], &gpu[i], 4) != 0) {
      diff++;
      max_abs = fmaxf(max_abs, fabsf(cpu[i] - gpu[i]));
    }
  printf("%-28s %8zu / %8zu differ, max |diff| %g\n", name, diff, n, max_abs);
  return diff != 0;
}

int main(int argc, char **argv) {
  Transformer t;
  hf_build_transformer(&t, argv[1]);
  const Config &c = t.config;
  const TransformerWeights &w = t.weights;
  const int H = c.hidden_dim, I = c.intermediate_dim, E = c.n_experts;
  const int D = c.head_dim, ATT = D * c.n_attn_heads, KVD = D * c.n_kv_heads;
  const int QKV = ATT + 2 * KVD, V = c.vocab_size;
  HIP_CHECK(hipSetDevice(0));
  GpuWorker *gw = gpu_create_worker(0, &t, true, 1, 64);
  const GpuLayer &g = gw->layers[0];
  int bad = 0;

  // embedding + rmsnorm
  const int token = 12194;
  std::vector<float> x(w.token_embedding_table + (size_t)token * H,
                       w.token_embedding_table + (size_t)token * H + H);
  int *d_tok = to_gpu(&token, 1);
  float *d_x;
  HIP_CHECK(hipMalloc(&d_x, H * 4));
  embed_kernel<<<1, 256>>>(gw->emb, d_tok, d_x, H);
  bad |= report("embedding", x.data(), from_gpu(d_x, H).data(), H);

  std::vector<float> tn(H);
  rmsnorm(tn.data(), x.data(), w.rms_attn_w, H);
  float *d_t;
  HIP_CHECK(hipMalloc(&d_t, H * 4));
  rmsnorm_exact<<<1, 256, H * 4>>>(d_x, g.rms_attn, d_t, H);
  bad |= report("rmsnorm", tn.data(), from_gpu(d_t, H).data(), H);

  // qkv matmul + bias, on the CPU's normed input
  float *d_tc = to_gpu(tn.data(), H);
  std::vector<float> qkv(QKV);
  matmul(qkv.data(), tn.data(), w.w_qkv, H, QKV);
  for (int i = 0; i < QKV; i++)
    qkv[i] += w.b_qkv[i];
  float *d_qkv;
  HIP_CHECK(hipMalloc(&d_qkv, QKV * 4));
  gemm_bf16_exact(0, d_tc, 1, g.w_qkv, g.b_qkv, d_qkv, QKV, H);
  bad |= report("qkv matmul + bias", qkv.data(), from_gpu(d_qkv, QKV).data(), QKV);

  // RoPE at position 37 on the CPU's qkv
  const int pos = 37, half = D / 2;
  float *d_qkv2 = to_gpu(qkv.data(), QKV);
  std::vector<float> cosv(half), sinv(half);
  compute_cos_sin(pos, c.rope_theta, D, c.rope_scaling_factor,
                  c.initial_context_length, 32.0f, 1.0f, cosv.data(), sinv.data());
  std::vector<float> q(qkv.begin(), qkv.begin() + ATT), k(qkv.begin() + ATT, qkv.begin() + ATT + KVD);
  apply_rotary_emb(q.data(), cosv.data(), sinv.data(), c.n_attn_heads, D);
  apply_rotary_emb(k.data(), cosv.data(), sinv.data(), c.n_kv_heads, D);
  rope_exact<<<1, 256>>>(d_qkv2, QKV, gw->cos_tab + pos * half, gw->sin_tab + pos * half,
                         c.n_attn_heads + c.n_kv_heads, D);
  std::vector<float> gq = from_gpu(d_qkv2, QKV);
  bad |= report("cos table", cosv.data(), from_gpu(gw->cos_tab + pos * half, half).data(), half);
  bad |= report("rope q", q.data(), gq.data(), ATT);
  bad |= report("rope k", k.data(), gq.data() + ATT, KVD);

  // router matmul + bias + topk + softmax
  std::vector<float> rs(E);
  matmul(rs.data(), tn.data(), w.w_router, H, E);
  for (int i = 0; i < E; i++)
    rs[i] += w.b_router[i];
  float *d_rs;
  HIP_CHECK(hipMalloc(&d_rs, E * 4));
  gemm_bf16_exact(0, d_tc, 1, g.w_router, g.b_router, d_rs, E, H);
  bad |= report("router matmul + bias", rs.data(), from_gpu(d_rs, E).data(), E);
  std::vector<float> tv(c.experts_per_token);
  std::vector<int> ti(c.experts_per_token);
  topk(tv.data(), ti.data(), rs.data(), E, c.experts_per_token);
  softmax(tv.data(), c.experts_per_token);
  float *d_tw;
  int *d_te;
  HIP_CHECK(hipMalloc(&d_tw, 16 * 4));
  HIP_CHECK(hipMalloc(&d_te, 16 * 4));
  float *d_rs_cpu = to_gpu(rs.data(), E);
  router_topk_exact<<<1, 64>>>(d_rs_cpu, E, c.experts_per_token, d_tw, d_te, 1);
  bad |= report("topk weights", tv.data(), from_gpu(d_tw, c.experts_per_token).data(), c.experts_per_token);

  // expert 3 of layer 0: mlp1 + bias, swiglu, mlp2 + bias
  const int e = 3;
  std::vector<float> m1(2 * I);
  const HfLayerExperts &ex = g_hf.experts[0];
  matmul_mxfp4(m1.data(), tn.data(), ex.mlp1_blocks + (size_t)e * 2 * I * (H / 32) * 16,
               ex.mlp1_scales + (size_t)e * 2 * I * (H / 32), H, 2 * I);
  for (int i = 0; i < 2 * I; i++)
    m1[i] += w.b_mlp1[(size_t)e * 2 * I + i];
  std::vector<int> offs(E + 1, 0);
  for (int i = e + 1; i <= E; i++)
    offs[i] = 1;
  int *d_offs = to_gpu(offs.data(), E + 1);
  float *d_m1;
  HIP_CHECK(hipMalloc(&d_m1, 2 * I * 4));
  gemm_grouped_mxfp4_exact(0, d_tc, d_offs, 1, E, g.mlp1_blocks, g.mlp1_scales, g.b_mlp1, d_m1, 2 * I, H);
  bad |= report("expert mlp1 + bias", m1.data(), from_gpu(d_m1, 2 * I).data(), 2 * I);

  std::vector<float> gu(I);
  for (int j = 0; j < I; j++) {
    float val = m1[2 * j], up = m1[2 * j + 1];
    if (val > c.swiglu_limit) val = c.swiglu_limit;
    if (up > c.swiglu_limit) up = c.swiglu_limit;
    if (up < -c.swiglu_limit) up = -c.swiglu_limit;
    val *= (1.0f / (1.0f + expf(-1.702f * val)));
    val *= (up + 1.0f);
    gu[j] = val;
  }
  float *d_m1c = to_gpu(m1.data(), 2 * I), *d_gu;
  HIP_CHECK(hipMalloc(&d_gu, I * 4));
  swiglu_exact<<<(I + 255) / 256, 256>>>(d_m1c, d_gu, 1, I, c.swiglu_limit);
  bad |= report("swiglu", gu.data(), from_gpu(d_gu, I).data(), I);

  // final rmsnorm + LM head on the CPU's normed input
  std::vector<float> logits(V);
  matmul(logits.data(), tn.data(), w.out, H, V);
  float *d_logits;
  HIP_CHECK(hipMalloc(&d_logits, (size_t)V * 4));
  gemm_bf16_exact(0, d_tc, 1, gw->unemb, nullptr, d_logits, V, H);
  bad |= report("lm head", logits.data(), from_gpu(d_logits, V).data(), V);

  printf(bad ? "some stages differ\n" : "all stages bit-identical\n");
  return bad;
}
