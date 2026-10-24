// Runs the first n prompts of a getp input file on one GPU (gpu_step) and on
// the CPU (hf_cpu_forward), feeding both the same tokens, and compares the
// logits of every sequence at every step bit for bit.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
//     getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
// Run (needs tokenizer.bin in the current folder):
//   tools/gpulock.sh <hip id> -- ./gpu_vs_cpu <model folder> [n=4] [steps=32] [input=evaluation/input.txt]
#define TESTING
#include "../../run.cpp"

#include <fstream>

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model folder> [n] [steps] [input file]\n", argv[0]);
    return 1;
  }
  const int n = argc > 2 ? atoi(argv[2]) : 4;
  const int steps = argc > 3 ? atoi(argv[3]) : 32;
  const char *input = argc > 4 ? argv[4] : "evaluation/input.txt";

  Transformer base;
  hf_build_transformer(&base, argv[1]);
  const Config &c = base.config;
  Tokenizer tok;
  read_tokenizer(&tok, "tokenizer.bin", c.vocab_size);

  std::ifstream in(input);
  std::string line;
  std::getline(in, line); // request count
  std::vector<std::vector<int>> prompts;
  while ((int)prompts.size() < n && std::getline(in, line)) {
    std::vector<int> buf(line.size() + 3);
    int np = 0;
    encode(&tok, line.c_str(), -1, -1, buf.data(), &np, c.initial_context_length);
    buf.resize(np);
    prompts.push_back(buf);
  }

  GpuWorker *w = gpu_create_worker(0, &base, true, n, steps);
  std::vector<Transformer> cpu(n, base);
  for (auto &t : cpu)
    malloc_run_state(&t.state, &t.config);
  // LAYERS=L runs only the first L layers on both sides, to find where they part.
  if (getenv("LAYERS")) {
    w->cfg.n_layers = atoi(getenv("LAYERS"));
    for (auto &t : cpu)
      t.config.n_layers = w->cfg.n_layers;
  }

  std::vector<int> cur(n), next(n);
  for (int b = 0; b < n; b++)
    cur[b] = prompts[b][0];
  std::vector<float> gpu_logits((size_t)n * c.vocab_size);
  long long total_diff = 0;
  for (int pos = 0; pos < steps; pos++) {
    HIP_CHECK(hipMemcpy(w->tokens, cur.data(), n * sizeof(int), hipMemcpyHostToDevice));
    gpu_step(w, n, pos);
    HIP_CHECK(hipMemcpy(gpu_logits.data(), w->logits, gpu_logits.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(next.data(), w->next, n * sizeof(int), hipMemcpyDeviceToHost));
    int step_diff = 0, argmax_diff = 0;
    float max_abs = 0.0f;
    for (int b = 0; b < n; b++) {
      const float *cl = hf_cpu_forward(&cpu[b], cur[b], pos);
      const float *gl = &gpu_logits[(size_t)b * c.vocab_size];
      for (int i = 0; i < c.vocab_size; i++)
        if (memcmp(&cl[i], &gl[i], sizeof(float)) != 0) {
          step_diff++;
          max_abs = fmaxf(max_abs, fabsf(cl[i] - gl[i]));
        }
      argmax_diff += sample_argmax((float *)cl, c.vocab_size) != next[b];
      if (getenv("LAYERS")) { // attention output of the last layer run
        const int ATT = c.head_dim * c.n_attn_heads;
        std::vector<float> ga(ATT);
        HIP_CHECK(hipMemcpy(ga.data(), w->att + (size_t)b * ATT, ATT * 4, hipMemcpyDeviceToHost));
        int ad = 0;
        for (int i = 0; i < ATT; i++)
          ad += memcmp(&ga[i], &cpu[b].state.tb[i], 4) != 0;
        printf("  seq %d: attention output differing %d / %d\n", b, ad, ATT);
        { // o-proj (CPU recomputed from its own attention output) and ffn norm
          const int H = c.hidden_dim, Kk = c.experts_per_token;
          const int L = w->cfg.n_layers - 1;
          std::vector<float> proj(H), gp(H), gt(H);
          matmul(proj.data(), cpu[b].state.tb, cpu[b].weights.w_o + (size_t)L * H * ATT, ATT, H);
          for (int i = 0; i < H; i++)
            proj[i] += cpu[b].weights.b_o[(size_t)L * H + i];
          HIP_CHECK(hipMemcpy(gp.data(), w->proj + (size_t)b * H, H * 4, hipMemcpyDeviceToHost));
          int pair_row0;
          HIP_CHECK(hipMemcpy(&pair_row0, w->pair_row + (size_t)b * Kk, 4, hipMemcpyDeviceToHost));
          HIP_CHECK(hipMemcpy(gt.data(), w->Xg + (size_t)pair_row0 * H, H * 4, hipMemcpyDeviceToHost));
          int pd = 0, td = 0;
          for (int i = 0; i < H; i++) {
            pd += memcmp(&proj[i], &gp[i], 4) != 0;
            td += memcmp(&cpu[b].state.t[i], &gt[i], 4) != 0;
          }
          printf("  o-proj differing %d / %d, ffn-norm output differing %d / %d\n", pd, H, td, H);
        }
        const int E = c.n_experts, K = c.experts_per_token;
        std::vector<float> gr(E), gw(K);
        std::vector<int> ge(K);
        HIP_CHECK(hipMemcpy(gr.data(), w->router + (size_t)b * E, E * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(gw.data(), w->topk_w + (size_t)b * K, K * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ge.data(), w->topk_e + (size_t)b * K, K * 4, hipMemcpyDeviceToHost));
        int rd = 0;
        for (int i = 0; i < E; i++)
          rd += memcmp(&gr[i], &cpu[b].state.router_score[i], 4) != 0;
        printf("  router scores differing %d / %d; experts cpu/gpu:", rd, E);
        for (int j = 0; j < K; j++)
          printf(" %d:%.9g/%d:%.9g", cpu[b].state.topk_i[j], cpu[b].state.topk_v[j], ge[j], gw[j]);
        printf("\n");
      }
    }
    printf("pos %3d: logits differing %d / %d, max |diff| %g, argmax differing %d\n",
           pos, step_diff, n * c.vocab_size, max_abs, argmax_diff);
    fflush(stdout);
    total_diff += step_diff;
    for (int b = 0; b < n; b++)
      cur[b] = pos + 1 < (int)prompts[b].size() ? prompts[b][pos + 1] : next[b];
  }
  printf(total_diff == 0 ? "PASS: GPU logits bit-identical to the CPU\n"
                         : "FAIL: logits differ\n");
  return total_diff == 0 ? 0 : 1;
}
