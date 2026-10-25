// Statistics of the value vectors the fast path stores in its int8 V cache, on
// real prompts: how peaky they are (largest |v| over the root mean square),
// the int8 rounding noise that gives, and what two exact rewrites would give:
//   bias: quantize v - b (b = the value part of the QKV bias) and add b back
//         after attention (out += b * (sum of the position weights));
//   smooth: divide each dim by a fixed per-(layer, head, dim) size before
//         quantizing and multiply the output dim back.
// Values are read back as scale * code (within half a step of the real ones).
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
//     getp-csrc/tests/attn_vstats.cpp tokenizer.cpp -o attn_vstats
// Run (needs tokenizer.bin in the current folder):
//   tools/gpulock.sh <hip id> -- ./attn_vstats <model folder> [n=64] [steps=400]
#define TESTING
#include "../../run.cpp"

#include <algorithm>
#include <fstream>

int main(int argc, char **argv) {
  const int n = argc > 2 ? atoi(argv[2]) : 64;
  const int steps = argc > 3 ? atoi(argv[3]) : 400;
  Transformer t;
  hf_build_transformer(&t, argv[1]);
  const Config &c = t.config;
  Tokenizer tok;
  read_tokenizer(&tok, "tokenizer.bin", c.vocab_size);
  std::ifstream in("evaluation/input.txt");
  std::string line;
  std::getline(in, line);
  std::vector<std::vector<int>> prompts;
  while ((int)prompts.size() < n && std::getline(in, line)) {
    std::vector<int> buf(line.size() + 3);
    int np = 0;
    encode(&tok, line.c_str(), -1, -1, buf.data(), &np, c.initial_context_length);
    buf.resize(np);
    prompts.push_back(buf);
  }
  HIP_CHECK(hipSetDevice(0));
  GpuWorker *w = gpu_new_worker(0, &t, false, 0, c.n_experts);
  gpu_alloc_state(w, n, 1024);
  std::vector<int> cur(n), next(n);
  for (int b = 0; b < n; b++)
    cur[b] = prompts[b][0];
  for (int pos = 0; pos < steps; pos++) {
    HIP_CHECK(hipMemcpy(w->tokens, cur.data(), n * sizeof(int), hipMemcpyHostToDevice));
    gpu_step(w, n, pos);
    HIP_CHECK(hipMemcpy(next.data(), w->next, n * sizeof(int), hipMemcpyDeviceToHost));
    for (int b = 0; b < n; b++)
      cur[b] = pos + 1 < (int)prompts[b].size() ? prompts[b][pos + 1] : next[b];
  }

  const int D = c.head_dim, NKV = c.n_kv_heads, KVD = D * NKV;
  const int QKV = D * (c.n_attn_heads + 2 * NKV);
  printf("%d prompts, %d steps; per layer, mean over all stored value vectors:\n", n, steps);
  printf("%-5s %-6s %9s %9s | %-26s | %9s %9s\n", "layer", "kind", "max/rms", "noise", "noise: bias / smooth / both",
         "top dim", "its share");
  double tot[4] = {0, 0, 0, 0};
  long long tot_n = 0;
  for (int l = 0; l < c.n_layers; l++) {
    const GpuLayer &g = w->layers[l];
    const size_t cache = (size_t)n * g.slots * KVD;
    std::vector<int8_t> vq(cache);
    std::vector<float> vs((size_t)n * g.slots * NKV), bias(QKV);
    HIP_CHECK(hipMemcpy(vq.data(), g.vq8, cache, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(vs.data(), g.vs, vs.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(bias.data(), g.b_qkv, QKV * 4, hipMemcpyDeviceToHost));
    const float *bv = bias.data() + D * c.n_attn_heads + KVD;
    const int used = g.window > 0 ? std::min(steps, g.window) : steps;
    // per (head, dim) root mean square of v and of v - b, for "smooth"
    std::vector<double> ss((size_t)KVD, 0.0), ssb((size_t)KVD, 0.0);
    std::vector<long long> top((size_t)KVD, 0);
    auto value = [&](int b, int s, int h, int i) {
      return vs[vs_index_chunk64(b, s, h, g.slots, KVD)] *
             (float)vq[vq_index_chunk64(b, s, h, i, g.slots, KVD)];
    };
    for (int b = 0; b < n; b++)
      for (int s = 0; s < used; s++)
        for (int h = 0; h < NKV; h++) {
          int am = 0;
          float best = -1;
          for (int i = 0; i < D; i++) {
            const float v = value(b, s, h, i);
            ss[h * D + i] += (double)v * v;
            ssb[h * D + i] += (double)(v - bv[h * D + i]) * (v - bv[h * D + i]);
            if (fabsf(v) > best) {
              best = fabsf(v);
              am = i;
            }
          }
          top[h * D + am]++;
        }
    const double cnt = (double)n * used;
    for (int i = 0; i < KVD; i++) {
      ss[i] = sqrt(ss[i] / cnt) + 1e-12;
      ssb[i] = sqrt(ssb[i] / cnt) + 1e-12;
    }
    // noise = rms(int8 rounding error) / rms(v), rounding error ~ step / sqrt(12)
    double peak = 0, nz[4] = {0, 0, 0, 0};
    for (int b = 0; b < n; b++)
      for (int s = 0; s < used; s++)
        for (int h = 0; h < NKV; h++) {
          double r2 = 0, amax = 0, amax_b = 0, amax_s = 0, amax_sb = 0, f2 = 0, f2b = 0;
          for (int i = 0; i < D; i++) {
            const double v = value(b, s, h, i), vb = v - bv[h * D + i];
            r2 += v * v;
            amax = std::max(amax, fabs(v));
            amax_b = std::max(amax_b, fabs(vb));
            amax_s = std::max(amax_s, fabs(v) / ss[h * D + i]);
            amax_sb = std::max(amax_sb, fabs(vb) / ssb[h * D + i]);
            f2 += ss[h * D + i] * ss[h * D + i];
            f2b += ssb[h * D + i] * ssb[h * D + i];
          }
          const double rms = sqrt(r2 / D) + 1e-30;
          peak += amax / rms;
          const double k = 1.0 / (127.0 * sqrt(12.0));
          nz[0] += amax * k / rms;
          nz[1] += amax_b * k / rms;
          nz[2] += amax_s * k * sqrt(f2 / D) / rms;
          nz[3] += amax_sb * k * sqrt(f2b / D) / rms;
        }
    long long best = 0;
    int bi = 0;
    for (int i = 0; i < KVD; i++)
      if (top[i] > best) {
        best = top[i];
        bi = i;
      }
    const double nv = cnt * NKV;
    printf("%-5d %-6s %9.2f %8.2f%% | %7.2f%% %7.2f%% %7.2f%% | %4d/%-4d %8.1f%%\n", l,
           g.window ? "window" : "full", peak / nv, 100 * nz[0] / nv, 100 * nz[1] / nv,
           100 * nz[2] / nv, 100 * nz[3] / nv, bi / D, bi % D, 100.0 * best / (cnt));
    for (int j = 0; j < 4; j++)
      tot[j] += nz[j];
    tot_n += (long long)nv;
  }
  printf("all layers: noise %.2f%%, bias %.2f%%, smooth %.2f%%, both %.2f%%\n", 100 * tot[0] / tot_n,
         100 * tot[1] / tot_n, 100 * tot[2] / tot_n, 100 * tot[3] / tot_n);
  return 0;
}
