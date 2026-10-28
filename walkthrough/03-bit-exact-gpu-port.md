# Lesson 03: The first GPU version, giving exactly the CPU's numbers

| | |
|---|---|
| Reference commit | `e824840` "Run getp on the GPUs, reproducing the CPU arithmetic bit for bit" |
| Starts from | `e8c283e` (lesson 02) |
| To type | 978 lines of program code in 7 files; 434 lines of tests (optional); 491 lines of docs (optional) |
| Shortcut | `git cherry-pick e824840` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 3,348 tok/s (was 3.3 on the CPU: ×1,000) |

**In plain words:** we move the work to the 8 GPU dies. Each die gets a full copy of the model
and one eighth of the prompts, and works on 512 prompts at the same time instead of one. Every
calculation is copied so carefully that the GPU gives exactly the same numbers as the CPU, down
to the last bit. That makes any later mistake easy to see.

## 1. Goal

`getp` runs on all GPUs with the same answers as the CPU program, and the program is organised
so that later lessons can replace one kernel at a time.

## 2. Why this, why now

- The CPU does 3.3 tok/s: the official input would take about two weeks.
- Correctness first: with a GPU version that equals the CPU bit for bit, any difference in a
  later lesson is a bug, not "rounding noise". This paid off at once (section 6: a wrong square
  root in HIP itself).

## 3. The technique

New words (HIP basics first):
- **HIP** — AMD's C++ language for GPU code, almost the same as NVIDIA's CUDA.
- **thread / block / grid** — a kernel runs as many threads; threads are grouped in blocks
  (here 64 or 256 threads) that can share fast memory and wait for each other; all blocks of
  one launch form the grid. `kernel<<<grid, block, shared bytes, stream>>>(…)` starts it.
- **wavefront** — 64 threads that the hardware runs in lockstep, one instruction for all.
- **LDS** — a small fast memory (64 KB) inside each compute unit, shared by one block.
- **stream** — a queue of GPU work; kernels in one stream run one after another.
- **batch** — the prompts handled together. **Static batching**: 512 prompts start together
  and all advance one position per step until the slowest is done.
- **data parallel** — every GCD has the whole model and its own share of the prompts; the
  GCDs never talk to each other.
- **KV cache** — for every prompt and layer, the keys and values of all earlier positions, kept
  so attention does not recompute them. **Attention** — each new token looks back at all earlier
  tokens of its prompt and takes a weighted mix of them.
- **experts / router / top-4** — in each layer a small multiply (the router) scores the 32
  experts for each token; the 4 best are used, weighted by a softmax of their scores.
- **grouping by expert** — sorting the (token, expert) pairs so that each expert multiplies all
  its tokens at once and its weights are read once per step, not once per token.
- **bit-identical** — every bit of every number is the same. **FMA** — multiply and add with
  one rounding instead of two. **fp64** — 64-bit numbers; the CPU code adds its products in fp64.

The plan:

```
4,096 prompts ─► GCD 0: prompts 0–511 · GCD 1: 512–1023 · … · GCD 7: 3,584–4,095
                 one CPU thread per GCD, each GCD holding the whole model (data parallel)

one step on one GCD (all 512 rows advance one position):
 tokens[512] → embed → 24 × layer → final norm → LM head (201,088 scores) → argmax → next[512]

 layer: rmsnorm → qkv multiply → RoPE → write KV cache → attention → out multiply → add
        → rmsnorm → router multiply → top-4 → group by expert → expert multiply 1
        → SwiGLU → expert multiply 2 → weighted add back into the row
```

How the first matrix multiply splits the work (`gemm_tile_exact`):

```
one block (256 threads) = one 64 × 64 tile of the output
thread t: tx = t % 16, ty = t / 16  → owns the 4 × 4 outputs at rows ty·4.., columns tx·4..
loop over k in slices of 32:
   all 256 threads copy a 64×32 slice of A and of W into LDS ; __syncthreads()
   each thread: 32 × 16 products, each fp32 product added into an fp64 sum, in k order
   __syncthreads()
round each fp64 sum to fp32 once, then add the bias  ← exactly what run.cpp's matmul() does
```

Copying the CPU's rounding: we read the CPU machine code (`objdump -d run`) to see where the
compiler used FMA and where it rounded, and copied each step (`docs/04` section "How it works").
Only additions need a fixed order; products, maxima and per-element work still run in parallel.

## 4. The change, file by file (typing order)

Type in parts; after each part, read it again against `git show e824840 -- <file>`.

| Part | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| A | `getp-csrc/kernels/exact_math.hpp` | 61 | `expf_cpu` (glibc's exp algorithm), `sqrt_cpu` and `div_cpu` (computed in fp64, rounded once) | yes |
| A | `getp-csrc/kernels/misc.hpp` | 129 | `embed_kernel`, `rmsnorm_exact`, `rope_exact`, `kv_write_kernel`, `add_inplace_kernel`, `argmax_kernel` | yes |
| B | `getp-csrc/kernels/gemm.hpp` | 141 | `gemm_tile_exact` (above), the bf16 and MXFP4 weight readers, `gemm_bf16_exact`, `gemm_grouped_mxfp4_exact` (one grid layer per expert) | yes |
| C | `getp-csrc/kernels/attention.hpp` | 83 | `attention_exact`: one 64-thread block per (head, prompt), scores in LDS, softmax with the head's sink | yes |
| D | `getp-csrc/kernels/moe.hpp` | 152 | `router_topk_exact`, `moe_group_kernel`, `moe_gather_kernel`, `swiglu_exact`, `moe_combine_exact` | yes |
| E | `getp-csrc/engine/gpu_engine.hpp` | 398 | weights to the GPU, memory for 512 prompts, `gpu_step` (the layer loop), `gpu_run_wave` (the token loop), one thread per GCD | yes |
| E | `getp-csrc/getp_run.cpp` | +14 −12 | `warm_up` / `inference` / `finish` call the GPU engine; `GPTOSS_CPU=1` keeps the CPU path | yes |
| F | `getp-csrc/tests/stage_check.cpp` | 158 | each `*_exact` kernel against its CPU code | optional |
| F | `getp-csrc/tests/gpu_vs_cpu.cpp` | 126 | the whole model, GPU vs CPU, every logit | optional |
| F | `getp-csrc/tests/expf_exhaustive.cpp`, `fp32_ops_exhaustive.cpp` | 66 + 84 | `exp` and square root on all 2^32 inputs | optional |
| — | `docs/01`–`docs/06` | 491 | the model, the MI250, batching, this port, data parallelism, MoE grouping, for newcomers | optional (read them!) |

Spots to read while typing (line numbers at `e824840`):
- `getp-csrc/kernels/gemm.hpp:48` `gemm_tile_exact` — the `__dadd_rn(acc, (double)__fmul_rn(a, b))`
  line is the whole "copy the CPU" idea in one line.
- `getp-csrc/kernels/exact_math.hpp:58` `sqrt_cpu` — why not just `__fsqrt_rn` (section 6).
- `getp-csrc/kernels/moe.hpp:58` `moe_group_kernel` — counting, then placing rows per expert.
- `getp-csrc/engine/gpu_engine.hpp:228` `gpu_step` — the whole model as ~20 kernel launches per
  layer; `:278` `gpu_run_wave` — the host loop: upload tokens, step, download next tokens.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick e824840
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout e824840 -- $(git show --name-only --format= e824840)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat e824840
git show e824840 -- getp-csrc/kernels/gemm.hpp
git checkout e824840 -- docs/          # docs are optional to type

git diff e824840 -- $(git show --name-only --format= e824840)   # must print nothing
make runomp
```

Quick check (one GCD, about 2 minutes; this exact build is slow):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12            # expected: ad7fdd092a55
python3 tools/ref_match.py $W/out64.txt evaluation/references/output_20b_token_ids.txt
```

Tests (build commands are in each file's first lines; run from the repo root):

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/stage_check.cpp tokenizer.cpp -o stage_check
tools/gpulock.sh 2 -- ./stage_check $M20
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
tools/gpulock.sh 2 -- ./gpu_vs_cpu $M20            # 4 prompts × 32 steps, every logit compared
```

Expected last lines (measured on the reference build):

```
stage_check:          all stages bit-identical                                   (~8 s)
gpu_vs_cpu:           PASS: GPU logits bit-identical to the CPU                  (~55 s)
expf_exhaustive:      0 mismatches over 2^32 inputs (16777214 NaN inputs agree as NaN)   (~4 s)
fp32_ops_exhaustive:  sqrtf / __fdiv_rn(1,x) / 1.0f/x / sqrt_cpu / div_cpu(1,x): 0 mismatches each   (~12 s)
```

The two exhaustive tests are built the same way without `tokenizer.cpp` and `-march=native`
(see their first lines). They check the functions we use; the 7.5 % count for `__fsqrt_rn` in
section 6 comes from `docs/04`.

Full run (8 GCDs, about 20 minutes at this speed):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-03 --score
```

Expected: about 3,348 tok/s, METEOR 0.9967, BERTScore 0.99998, fingerprint `5017002309c8`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Run getp on the GPUs, reproducing the CPU arithmetic bit for bit"`

## 6. Result

| | before | after |
|---|---:|---:|
| 20b tok/s, 8 GCDs, 4,096 prompts | 3.3 (CPU) | **3,348** |
| One step at 512 prompts, one GCD | — | 1.08 s |
| METEOR / BERTScore | — | 0.9967 / 0.99998 |
| Answers equal to the reference, line for line | — | 4,062 of 4,096 |
| Logits vs the CPU, 4 prompts × 32 steps, all 24 layers | — | bit-identical |

What the bit-for-bit rule found:

| Check, all 2^32 fp32 inputs | Differences from the CPU |
|---|---:|
| our `expf_cpu` | 0 |
| HIP `sqrtf` | 0 |
| **HIP `__fsqrt_rn`** ("correctly rounded" square root) | **323,287,622 (7.5 %)** |
| our `sqrt_cpu` (fp64 square root, rounded to fp32 once) | 0 |

`__fsqrt_rn` made the RMSNorm scale differ now and then, which moved every logit a little. It
was found by cutting the model to 1 layer, comparing each stage of the first wrong step, then
testing the square root on every possible input. In a "close enough" port it would have looked
like rounding noise.

Why 34 answers still differ from the reference: this version equals the CPU program bit for
bit, so the CPU program would differ on the same 34. The reference answers were made with
slightly different arithmetic, and one flipped token changes the rest of an answer.

## 7. Decisions and reasons

- **Decided:** the first GPU version copies the CPU's arithmetic exactly.
  **Why:** any later difference is then a bug, not rounding; it found the square-root fault.
  **Not chosen:** a fast bf16 version first, checked only by METEOR — a fault like the square
  root would have hidden inside "slightly different but fine" scores.
- **Decided:** static batching, 512 prompts per GCD, all moving one position per step.
  **Why:**
  - one prompt alone uses each weight it reads only once, about 200 times less work per byte
    read than the GPU needs to keep its calculators busy (`docs/02`);
  - AMD's library multiply takes only 1.2× longer for 64 times more rows;
  - 4,096 prompts ÷ 8 GCDs = 512;
  - 95 % of the reference answers run to the full length, so waiting for the slowest wastes
    little.
  **Not chosen:** one prompt at a time; *continuous batching* (a finished prompt's seat is given
  to a new prompt at once) — here every prompt is already in the first batch, so there is no
  queue to fill from.
- **Decided:** 20b runs data parallel: a full copy on each of the 8 GCDs.
  **Why:** the model (13.7 GB) fits next to a 29 GB fp32 KV cache for 512 prompts, and the GCDs
  need no traffic between them. **Not chosen:** splitting the model over GCDs (needed later for
  120b, lesson 05).
- **Decided:** group tokens by expert from the very first version.
  **Why:** without grouping, 512 tokens × 4 experts would read 27 GB of expert weights per layer;
  grouped, 0.42 GB — 64 times less.

## 8. Questions to think about

1. `gemm_tile_exact` keeps 16 fp64 sums per thread and adds products in k order. Which part of
   that makes it slow, and which part is needed only for bit-exactness?
2. Why may products, maxima and per-element work run in any order, while additions may not?
3. `gpu_run_wave` copies 512 tokens to the GPU and 512 back every step. How much data is that,
   and why is it not the problem here?

## 9. Discussion notes

(empty: we write here during the walk-through)
