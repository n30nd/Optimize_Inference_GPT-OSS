// Checks that hf_cpu_forward() (expert weights decoded from MXFP4 on the fly)
// gives bit-identical logits to run.cpp's forward() on fully expanded fp32
// weights, which is what an exported .bin file holds.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
//     getp-csrc/tests/cpu_forward_equiv.cpp tokenizer.cpp -o cpu_forward_equiv
// Run:
//   ./cpu_forward_equiv <model folder> [steps]
//   ./cpu_forward_equiv <model folder> --dump-expert <layer> <expert> <out.f32>
//     writes that expert's decoded mlp1 weight (fp32, row-major) for
//     getp-csrc/tests/check_mxfp4_dequant.py
#define TESTING
#include "../../run.cpp"

static int dump_expert(Transformer *t, int layer, int expert, const char *out) {
  Config *c = &t->config;
  const size_t rows = 2ull * c->intermediate_dim;
  const int groups = c->hidden_dim / 32;
  const HfLayerExperts &x = g_hf.experts[layer];
  std::vector<float> buf(rows * c->hidden_dim);
  mxfp4_to_f32(buf.data(), x.mlp1_blocks + expert * rows * groups * 16,
               x.mlp1_scales + expert * rows * groups, rows, c->hidden_dim);
  FILE *f = fopen(out, "wb");
  if (!f || fwrite(buf.data(), sizeof(float), buf.size(), f) != buf.size()) {
    fprintf(stderr, "cannot write %s\n", out);
    return 1;
  }
  fclose(f);
  printf("wrote layer %d expert %d mlp1 (%zu x %d fp32) to %s\n", layer, expert,
         rows, c->hidden_dim, out);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model folder> [steps] | --dump-expert L E out\n",
            argv[0]);
    return 1;
  }
  Transformer a;
  hf_build_transformer(&a, argv[1]);
  if (argc >= 6 && strcmp(argv[2], "--dump-expert") == 0)
    return dump_expert(&a, atoi(argv[3]), atoi(argv[4]), argv[5]);
  int steps = argc >= 3 ? atoi(argv[2]) : 8;

  // b shares a's config and fp32 dense weights, gets its own run state and
  // fully expanded fp32 expert weights.
  Transformer b = a;
  malloc_run_state(&b.state, &b.config);
  Config *c = &a.config;
  const size_t E = c->n_experts, H = c->hidden_dim, I = c->intermediate_dim;
  b.weights.w_mlp1 = (float *)malloc(c->n_layers * E * 2 * I * H * sizeof(float));
  b.weights.w_mlp2 = (float *)malloc(c->n_layers * E * H * I * sizeof(float));
  if (!b.weights.w_mlp1 || !b.weights.w_mlp2) {
    fprintf(stderr, "out of memory for fp32 experts\n");
    return 1;
  }
  long t0 = time_in_ms();
  for (int l = 0; l < c->n_layers; l++) {
    const HfLayerExperts &x = g_hf.experts[l];
    mxfp4_to_f32(b.weights.w_mlp1 + l * E * 2 * I * H, x.mlp1_blocks,
                 x.mlp1_scales, E * 2 * I, H);
    mxfp4_to_f32(b.weights.w_mlp2 + l * E * H * I, x.mlp2_blocks,
                 x.mlp2_scales, E * H, I);
  }
  printf("expanded experts to fp32 (%.1f GB) in %.1f s\n",
         (c->n_layers * E * 3 * I * H * 4) / 1e9, (time_in_ms() - t0) / 1000.0);

  // "Hello world" = 13225 2375, then greedy continuation.
  int prompt[2] = {13225, 2375};
  int token = prompt[0];
  int total_diff = 0;
  for (int pos = 0; pos < steps; pos++) {
    float *la = hf_cpu_forward(&a, token, pos);
    float *lb = forward(&b, token, pos);
    int diff = 0;
    float max_abs = 0.0f;
    for (int i = 0; i < c->vocab_size; i++) {
      if (memcmp(&la[i], &lb[i], sizeof(float)) != 0) {
        diff++;
        max_abs = fmaxf(max_abs, fabsf(la[i] - lb[i]));
      }
    }
    int next = sample_argmax(la, c->vocab_size);
    printf("pos %2d  token %6d  next %6d  logits differing: %d/%d (max |diff| %g)\n",
           pos, token, next, diff, c->vocab_size, max_abs);
    fflush(stdout);
    total_diff += diff;
    token = pos + 1 < 2 ? prompt[pos + 1] : next;
  }
  printf(total_diff == 0 ? "PASS: bit-identical logits\n"
                         : "FAIL: logits differ\n");
  return total_diff == 0 ? 0 : 1;
}
