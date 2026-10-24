// Runs the first n prompts of a getp input file on the GPU exact path and on
// the CPU (hf_cpu_forward), feeding both the same tokens, and compares the
// logits of every sequence at every step bit for bit.
//
// EP=2 splits the experts between GPUs 0 and 1 (as getp does for 120b) and the
// prompts between them; otherwise GPU 0 holds the whole model.
// LAYERS=L runs only the first L layers on both sides, to find where they part.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
//     getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
// Run (needs tokenizer.bin in the current folder):
//   tools/gpulock.sh <hip id(s)> -- ./gpu_vs_cpu <model folder> [n=4] [steps=32] [input=evaluation/input.txt]
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
  const int ep = env_int("EP", 1);

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

  // member i serves sequences first[i] .. first[i] + count[i] - 1
  GpuGroup g;
  std::vector<int> first(ep), count(ep);
  for (int i = 0; i < ep; i++) {
    count[i] = n / ep + (i < n % ep ? 1 : 0);
    first[i] = i == 0 ? 0 : first[i - 1] + count[i - 1];
    GpuWorker *w = gpu_new_worker(i, &base, true, i * (c.n_experts / ep), c.n_experts / ep);
    gpu_alloc_state(w, std::max(count[i], 1), steps);
    g.m.push_back(w);
  }
  if (ep == 2)
    gpu_pair(g.m[0], g.m[1]);
  std::vector<Transformer> cpu(n, base);
  for (auto &t : cpu)
    malloc_run_state(&t.state, &t.config);
  if (getenv("LAYERS")) {
    for (GpuWorker *w : g.m)
      w->cfg.n_layers = atoi(getenv("LAYERS"));
    for (auto &t : cpu)
      t.config.n_layers = atoi(getenv("LAYERS"));
  }

  std::vector<int> cur(n), next(n);
  for (int b = 0; b < n; b++)
    cur[b] = prompts[b][0];
  std::vector<float> gpu_logits((size_t)n * c.vocab_size);
  long long total_diff = 0;
  for (int pos = 0; pos < steps; pos++) {
    for (int i = 0; i < ep; i++) {
      HIP_CHECK(hipSetDevice(g.m[i]->device));
      HIP_CHECK(hipMemcpy(g.m[i]->tokens, &cur[first[i]], count[i] * sizeof(int),
                          hipMemcpyHostToDevice));
    }
    gpu_step_group(g, count.data(), pos);
    for (int i = 0; i < ep; i++) {
      HIP_CHECK(hipSetDevice(g.m[i]->device));
      HIP_CHECK(hipMemcpy(&gpu_logits[(size_t)first[i] * c.vocab_size], g.m[i]->logits,
                          (size_t)count[i] * c.vocab_size * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&next[first[i]], g.m[i]->next, count[i] * sizeof(int),
                          hipMemcpyDeviceToHost));
    }
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
