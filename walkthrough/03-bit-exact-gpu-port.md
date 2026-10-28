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

### 9.1 `exact_math.hpp`: why our own `exp` and square root, not HIP's (2026-10-28)

**In plain words:** HIP has its own `expf` and square root, and they are good: at most one last
bit off. But this lesson's rule is "the GPU gives the same bits as the CPU", and HIP's `expf`
breaks it on almost 1 input in 100. So for `exp` we copied the CPU library's own code (glibc) and
only translated it to HIP. For square root we use a short trick of our own that is right on every
input whatever the compiler settings are.

New words:
- **glibc** — the standard C library on Linux. The CPU program's `expf` comes from it (version 2.39
  here).
- **bit pattern** — the 32 (or 64) zeros and ones that store a number. "Same bits" = same number,
  exactly.

**Measured: HIP's own functions vs the CPU, all 2^32 fp32 inputs (GPU 0)**

| GPU function | differs from the CPU (normal build) | with the two "faster fp32" flags below |
|---|---:|---:|
| HIP `expf` | 33,606,109 (0.78 %), each by 1 last bit | — |
| HIP `__expf` (the fast one) | 117,997,392 (2.75 %), some about 2× off for tiny results | — |
| our `expf_cpu` | 0 (`tests/expf_exhaustive.cpp`) | — |
| HIP `sqrtf` | 0 | 338,779,618 (7.89 %) |
| HIP `__fsqrt_rn` | 323,287,622 (7.53 %) | — |
| our `sqrt_cpu` | 0 | not run; it works in fp64, which these fp32 flags do not touch |
| `1.0f / x` | 0 | 498,565,540 (11.61 %) |

The last column is a build with `-fno-hip-fp32-correctly-rounded-divide-sqrt` (fp32 square root
and division take a quicker, less exact path) and `-fgpu-flush-denormals-to-zero` (the tiniest
fp32 numbers count as 0). Fast-math builds turn on this kind of flag. So HIP's `sqrtf` is exact
only as long as nobody changes the build flags.

**Why one last bit matters here.** `expf` is used in the attention softmax, the router softmax and
SwiGLU. A one-bit difference in one weight moves the logits a little; when two tokens are nearly
tied, the other one wins, and the rest of the answer changes. Worse, `gpu_vs_cpu` compares every
logit with `==`; with HIP's `expf` it would always fail, and we could no longer tell a real bug
from "just rounding". That is the whole point of this lesson (section 2).

**Which part is copied and which is ours**

| Lines | What | Where it comes from | Need to understand the inside? |
|---|---|---|---|
| 12-53 table + `expf_cpu` | e^x | **Copied** from glibc 2.39, `sysdeps/ieee754/flt-32/e_expf.c` (itself from ARM optimized-routines `math/expf.c`, MIT licence): same table, same 5 constants, same steps in the same order | **No** |
| 58 `sqrt_cpu`, 59-61 `div_cpu` | square root, divide | **Ours** | **Yes**, line by line below |

**`expf_cpu`: copied, treat it as a closed box.**
- What we changed: only the language. Each `a * b`, `a + b`, `a - b` became `__dmul_rn`,
  `__dadd_rn`, `__dsub_rn`, and each `a * b + c` that the CPU does as one multiply-add became
  `__fma_rn`. Why: with plain `*` and `+` the GPU compiler may merge a multiply and an add into one
  step (one rounding instead of two), which changes the last bit. Each `_rn` function is exactly
  one operation, rounded once to the nearest number, and the compiler may not merge them.
- Proof it was copied: the table and all 5 constants are found byte for byte inside this machine's
  CPU math library, `/lib/x86_64-linux-gnu/libm.so.6` (checked 2026-10-28).
- Proof the copy is right: `tests/expf_exhaustive.cpp`, 0 differences over all 2^32 inputs.
- What to tell others: "it is glibc's `expf`, ported to HIP, tested equal on every input".

For someone who wants to check the copy against the original:

| glibc `e_expf.c` | `exact_math.hpp` lines |
|---|---|
| `if (abstop >= top12 (88.0f)) { … }` (too big, too small, NaN) | 30-41 |
| `z = InvLn2N * xd;` | 42 |
| `kd = z + SHIFT; ki = asuint64 (kd); kd -= SHIFT;` | 43-45 |
| `r = z - kd;` | 46 |
| `t = T[ki % N]; t += ki << (52 - EXP2F_TABLE_BITS); s = asdouble (t);` | 47-48 |
| `z = C[0] * r + C[1]; r2 = r * r; y = C[2] * r + 1; y = z * r2 + y;` | 49-51 |
| `y = y * s; return (float) y;` | 52 |

**`sqrt_cpu` and `div_cpu`: ours, line by line.**

```cpp
__device__ inline float sqrt_cpu(float x) { return (float)__dsqrt_rn((double)x); }
```

| Piece | Meaning |
|---|---|
| `__device__` | this function runs on the GPU; kernels call it |
| `inline` | the compiler pastes the body where it is called, so calling it costs nothing |
| `(double)x` | turn the 32-bit number into a 64-bit one. Nothing is lost: every fp32 number fits in fp64 exactly |
| `__dsqrt_rn(…)` | 64-bit square root, rounded to the nearest 64-bit number |
| `(float)…` | round that 64-bit answer to the nearest 32-bit number |

Why this equals the CPU: the CPU's `sqrtf` takes the exact square root and rounds it **once** to
fp32. We round **twice** (to fp64, then to fp32). Two roundings can go wrong in general. Everyday
example: 1.49 rounded to one decimal is 1.5, then to a whole number is 2; but 1.49 rounded straight
to a whole number is 1. That cannot happen here: fp64 keeps 53 binary digits, more than 2 × 24 + 2
= 50 (fp32 has 24), and for square root that is proven to be enough (Figueroa 1995, in `docs/04`).
The test agrees: `fp32_ops_exhaustive.cpp`, 0 differences over all 2^32 inputs.

Where it is used: the RMSNorm scale `1 / sqrt(mean of squares + eps)` (`misc.hpp:44`) and the
attention scale `sqrt(64)` (`attention.hpp:34`).

```cpp
__device__ inline float div_cpu(float a, float b) {
  return (float)__ddiv_rn((double)a, (double)b);
}
```

The same idea for `a / b`: both to fp64 (exact), divide in fp64 rounding once, round to fp32 once.
Tested equal to the CPU's `1.0f / x` on all 2^32 inputs. But no kernel calls it: they use HIP's
`__fdiv_rn(1.0f, x)`, which also tests equal. It could be deleted.

**Honest notes:**
- With today's build flags HIP's `sqrtf` would also have worked (0 differences). `sqrt_cpu` is
  insurance: it stays exact under any fp32 flag, and it was written right after `__fsqrt_rn` (whose
  name promises "round to nearest") turned out wrong on 7.5 % of inputs.
- The cost is small: on the MI250, fp64 adds and multiplies run as fast as fp32 ones (not on most
  gaming GPUs). The slow part of this lesson is the matrix multiply (section 8, question 1).
- This file is only for the exact reference version. The fast kernels of later lessons use HIP's
  fast `__expf` (`getp-gpu`: `fast_attention.hpp:504`, `fast_misc.hpp:139`) and are judged by
  METEOR/BERTScore and by how many tokens still match this exact version.

### 9.2 Running `tests/expf_exhaustive.cpp` (2026-10-28)

**In plain words:** the test makes the GPU compute `expf_cpu` for every possible fp32 number (all
4,294,967,296 of them), lets the CPU compute its own `expf` for the same numbers, and counts how
many answers differ in any bit. It takes about 3 seconds.

Run from the repo root (here `/home/namvu/gpt-oss`, branch `main`; the test and `exact_math.hpp`
are unchanged there since `e824840`):

```bash
cd /home/namvu/gpt-oss
python3 tools/gpus.py --free 1          # prints a free GPU number, e.g. 0
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a \
  getp-csrc/tests/expf_exhaustive.cpp -o expf_exhaustive
tools/gpulock.sh 0 -- ./expf_exhaustive
```

| Piece | Meaning |
|---|---|
| `hipcc` | AMD's compiler for HIP code; builds the CPU part and the GPU part into one program |
| `-O3` | optimise fully |
| `-fopenmp` | lets the CPU's checking loop use all CPU cores at once |
| `--offload-arch=gfx90a` | build the GPU part for the MI250's chip (gfx90a) |
| `tools/gpulock.sh 0 --` | waits until none of our other jobs uses GPU 0, then runs the program on GPU 0 only |

`expf_exhaustive` is listed in `.gitignore`, so the built program does not show up in `git status`.

What happens inside (16 rounds of 2^28 = 268 million inputs each):
1. GPU: thread `i` turns the bit pattern `base + i` into a float `x` and stores `expf_cpu(x)`.
2. The results are copied back to the CPU.
3. CPU, all cores: the CPU's `expf(x)` for the same `x`; compare the two answers bit by bit.
   If both answers are NaN ("not a number"), it counts as agreeing.
4. It prints the first differing input of a round, if there is one.

Result (GPU 0, 3.1 s):

```
0 mismatches over 2^32 inputs (16777214 NaN inputs agree as NaN)
```

16,777,214 = 2^24 − 2: the number of fp32 bit patterns that mean NaN. NaN in, NaN out on both
sides; their bit patterns are allowed to differ.

> Line numbers in 9.3–9.7: `kernels/misc.hpp`, `kernels/gemm.hpp`, `run.cpp` and
> `tests/stage_check.cpp` are the same at `e824840` and at `main` (`4687903`). `gpu_engine.hpp`
> line numbers are from `main` (`4687903`). All paths are under `getp-csrc/` unless they start
> with `run.cpp`.

### 9.3 `tests/stage_check.cpp`: the embedding check, line by line (2026-10-28)

**In plain words:** pick one word (token 12194), look up its row of 2880 numbers in the model's
table on the CPU, ask the GPU to do the same lookup, bring the GPU's row back, and compare the two
rows number by number. Any difference in any bit counts as a failure.

New words:
- **token** — a word or piece of a word, written as a number.
- **embedding table** — one row of numbers for every possible token: `[vocab, H]`, H = 2880.
- **host / device** — the CPU with its memory / the GPU with its own separate memory. Names that
  start with `d_` hold GPU addresses.
- **pointer** — a number that says *where* something is in memory.

The table in memory is one long line, row after row. Row 12194 starts at `12194·H` and ends just
before `12194·H + H`.

| Line (52-59) | What happens | After this line |
|---|---|---|
| `const int token = 12194;` | a fixed choice, so every run tests the same row | one int on the CPU |
| `std::vector<float> x(start, start + H);` | the two addresses mean "copy from here up to (not including) here"; the vector copies the 2880 numbers between them and the addresses are forgotten. `(size_t)` keeps the multiply in 64 bits | `x` = a CPU copy of row 12194: the **expected answer**. The CPU table was widened from bf16 to fp32 at load time (`engine/hf_model.hpp:212`) |
| `int *d_tok = to_gpu(&token, 1);` | `to_gpu` (lines 11-17): `hipMalloc` reserves room on the GPU, `hipMemcpy` copies 4 bytes there | the GPU holds 12194; `d_tok` says where |
| `float *d_x;` | an empty address variable | nothing in memory changes |
| `HIP_CHECK(hipMalloc(&d_x, H * 4));` | reserve 2880 × 4 = 11,520 bytes on the GPU; `HIP_CHECK` stops with a message if the call fails | an empty space for one row |
| `embed_kernel<<<1, 256>>>(…)` | 1 block of 256 threads; the kernel (`misc.hpp:16-23`) reads row 12194 of the GPU's bf16 table and widens each value to fp32 by putting its 16 bits on top of 16 zeros, the same rule the CPU used. The CPU does **not** wait here | (once the GPU is done) `d_x` = the GPU's row |
| `bad \|= report(…, x.data(), from_gpu(d_x, H).data(), H);` | `from_gpu` (19-24) first **waits for the GPU**, then copies the row back. `report` (26-36) compares the 4 raw bytes of each number (`memcmp`, so "almost equal" counts as different) and prints `embedding 0 / 2880 differ`. `bad \|=` keeps `bad` at 1 once any check failed | one line printed |

`x.data()` = the address of `x`'s first number. `report` takes addresses (`const float *cpu`), not
vectors, so both lists are handed over as "where your numbers start" plus the count `H`.

The expected result is 0 differences: both sides start from the same 16-bit values and widen them
the same way. A difference would mean a wrong row, missed positions, or a different conversion.

### 9.4 `rmsnorm_exact` (`misc.hpp:28-50`) (2026-10-28)

**In plain words:** make a row of 2880 numbers "the right size": find how big they are on average,
divide every number by that, multiply each by its own weight. 256 threads share the squaring and
the scaling; one thread does the adding, in the CPU's order.

**1. Operation**

```
Input   X [B, 2880] fp32  (x),  w [2880] fp32  (weight)
Output  Y [B, 2880] fp32  (out, new buffer)
Math    Y[b, j] = w[j] · X[b, j] / sqrt( (Σ_{k=0..2879} X[b, k]²) / 2880 + ε )     ε = 1e-5f
```

Copies `rmsnorm()` in `run.cpp:320-333`. Launch: `<<<B, 256, H·4>>>` (one block per row,
11,520 bytes of shared memory for the squares).

**2. Threads**

| level | count | owns |
|---|---|---|
| grid | B blocks | block b = row b |
| block | 256 threads | the 2880 numbers of the row |
| thread | 2880 / 256 = 11.25 → threads 0-63 do 12, threads 64-255 do 11 | `j = t, t+256, …` |

Phases: (1) all threads square their numbers into shared memory `sq`; barrier; (2) **thread 0
alone** adds all 2880 squares in order, divides, adds ε, takes 1/√ and stores `inv` in shared
memory; barrier; (3) all threads write `w[j]·(inv·X[b,j])`.

**3. Details (block 2, thread 70)**
- Row 2 starts at `2·2880 = 5760`. Thread 70 owns `j = 70, 326, 582, …, 2630` (11 numbers), i.e.
  `x[5830], x[6086], …, x[8390]`, in phase 1 and phase 3 (same split).
- Thread 0 needs all 2880 squares, made 12 itself, so **2868 came from the other 255 threads**:
  that is why `sq` is shared memory and why the barrier on line 37 exists. `inv` is made by one
  thread and read by 256: shared memory plus the barrier on line 46.

| GPU (line) | CPU `run.cpp` | why |
|---|---|---|
| `__fmul_rn(x, x)` (36) | `x[j] * x[j]` (324), float | product rounded once to fp32 |
| `__dadd_rn(ss, sq[j])` in order (40-41) | `ss += …`, `ss` is double (323-324) | **same order** |
| `__ddiv_rn(ss, H)` (42) | `ss /= size` (326) | |
| `__dadd_rn(ss, (double)1e-5f)` (43) | `ss += 1e-5f` (327) | `1e-5f` is really 0.0000099999997475…, and the GPU must add that exact value |
| `__fdiv_rn(1, sqrt_cpu((float)ss))` (44) | `1.0f / sqrtf(ss)` (328) | `sqrtf` takes a float, so `ss` is cut to float first; see 9.1 for `sqrt_cpu` |
| `(float)(w·(inv·x))` in double (49) | `weight[j] * (ss * x[j])` (331) | double multiplies, one rounding to float at the end |

**Why only one thread adds:** `(a+b)+c` and `a+(b+c)` can differ in the last bit. A tree sum (the
usual fast GPU way) gave a different double result than the in-order sum in **30 of 100** random
rows of 2880 (checked in Python). Slow (2880 adds in a row) but exact.

Tiny example (H = 4): `x = [1, 2, 3, 4]`, `w = [1, 1, 0.5, 2]` → squares sum 30, ÷4 = 7.5, +ε,
√ = 2.7386146, inv = 0.36514813 → `Y = [0.36514813, 0.73029625, 0.5477222, 2.921185]`, the same as
the direct formula in double to float accuracy (checked in Python).

### 9.5 `rope_exact` (`misc.hpp:52-70`): RoPE, math first (2026-10-28)

**In plain words:** mark each word with *where* it is in the text. In every group of 64 numbers,
pair number `i` with number `i+32` and turn the pair like a clock hand by an angle that depends on
the word's position. Later, when two words are compared, the result depends only on how far apart
they are.

New words:
- **RoPE (rotary position embedding)** — "turn pairs of numbers by a position-dependent angle".
- **head** — one group of 64 numbers; the model looks at a word in 64 (query) separate ways.
- **query (q) / key (k) / value (v)** — three lists made from each word by the qkv multiply. A query
  asks "what am I looking for?", a key says "what do I contain?", a value is "what I hand over".
  Only q and k are turned.
- **step** — one pass through the whole model for one more word. **position p** — that word's
  place in the text, starting at 0. All B texts in a batch move one word per step together
  (static batching), so they share one p.

**Math for one pair.** Turning the point (x, y) by angle θ:
`x' = x·cos θ − y·sin θ`, `y' = y·cos θ + x·sin θ`. Example: (1, 0) turned by 90° is (0, 1).

**1. Operation**

```
Input   X [B, 5120] fp32 (qkv)          one row per word: q = 64 heads × 64 (0..4095),
                                         k = 8 heads × 64 (4096..4607), v = 512 (not touched)
        c [32], s [32] fp32 (cosv, sinv) row p of cos_tab [P, 32] and sin_tab [P, 32]
Output  Y [B, 5120]                      written over X (in place)

X[b, h, j] = qkv[b·5120 + h·64 + j],  h = 0..71 (64 q heads + 8 k heads),  j = 0..63

Math    Y[b, h, i]    = X[b, h, i]·c[i]    − X[b, h, i+32]·s[i]       i = 0..31
        Y[b, h, i+32] = X[b, h, i+32]·c[i] + X[b, h, i]·s[i]
        Y[b, 4608:5120] = X[b, 4608:5120]

        c[i] = 1.3466·cos(p·ω_i),  s[i] = 1.3466·sin(p·ω_i)
```

- **ω_i** = the fixed "speed" of pair i: `ω_i = 150000^(−i/32)` (ω_0 = 1, ω_1 ≈ 0.69,
  ω_6 ≈ 0.107, ω_31 ≈ 0.0000003). gpt-oss slows pairs 18-31 by 32× and blends pairs 9-17 (YaRN,
  `run.cpp:438-465`) so it works on longer texts.
- **1.3466** = 0.1·ln 32 + 1, a fixed stretch (`run.cpp:440`).
- **cos_tab / sin_tab [P, 32]** — P = the most positions this run handles (`max_pos`; 64 in the
  test). Row p holds the 32 c (or s) values for position p. Made **once, on the CPU**, with
  `run.cpp`'s own `compute_cos_sin()` (`gpu_engine.hpp:388-396`), then copied to the GPU. At step
  p the kernel gets row p (`gpu_engine.hpp:476`).

```
cos_tab           i = 0     i = 1     i = 6     i = 31
row p = 0         1.3466    1.3466    1.3466    1.3466
row p = 1         0.7276    1.0394    1.3389    1.3466
row p = 37        1.0307    1.2594   -0.9203    1.3466
sin_tab
row p = 0         0         0         0         0
row p = 1         1.1331    0.8562    0.1438    ≈0
row p = 37       -0.8666    0.4768   -0.9830    ≈0
```

**Why: only the distance matters.** The model later scores q against k by multiplying pairs and
adding ("dot product"). If q is turned by p_q·ω and k by p_k·ω, the score depends only on
(p_q − p_k)·ω. One pair, ω = 0.5, q = (1, 2), k = (3, 1) (checked in Python):

| q at | k at | distance | score |
|---|---|---|---|
| 5 | 3 | 2 | −1.506 |
| 12 | 10 | 2 | −1.506 |
| 3 | 5 | −2 | 6.909 |

Fast pairs tell apart small distances, slow pairs large ones, like the hands of a clock.

**2. Threads**

| level | count | owns |
|---|---|---|
| grid | B blocks | block b = row b |
| block | 256 threads | 72 heads × 32 pairs = 2304 pairs |
| thread | 2304 / 256 = **9 pairs** | `idx = t, t+256, …, t+2048`; `h = idx / 32`, `i = idx % 32` |

One lap of 256 covers 8 heads. No shared memory, no barrier: each pair is read and written by one
thread only.

**3. Details (block 2, thread 70)**
- Row 2 starts at `2·5120 = 10240`. Each lap adds 256 = 8·32, so **i is always 6** for thread 70
  and only the head moves: h = 2, 10, 18, …, 58 (q), then 66 (k head 2). Lap 1: `h = 10`, `x1 =
  qkv[10886]`, `x2 = qkv[10918]`, both read before either is written (so in place is safe).
- Rounding: the CPU build (`-march=native`) turns `x1*c - x2*s` into one FMA (fused multiply-add:
  `a·b + c` with one rounding): `x2*s` is rounded, then `x1*c − that` in one step. The GPU spells
  out the same steps with `__fmul_rn` and `__fmaf_rn`. On 20,000 random pairs (exact rounding in
  Python), "fuse the first product" differs from "no fusing" in **4,888** and from "fuse the
  second product" in **6,626**.
- The angles come from the CPU's table, not from `cosf`/`sinf` on the GPU, so they are exact too.
  The test checks the table (`stage_check.cpp:91`).

Tiny example (one head of 4, pairs (0,2) and (1,3), plain cos/sin): `X = [1, 2, 3, 4]`,
`c = [0.6, 1]`, `s = [0.8, 0]` → `Y = [−1.8, 2, 2.6, 4]`. The length is kept (√10 before and
after) and pair 1 (angle 0) does not move (checked in Python).

### 9.6 `gemm_bf16_exact` (`gemm.hpp:47-129`): the exact matrix multiply (2026-10-28)

**In plain words:** each output number is one input row times one weight row, added up, plus a
bias. The output is cut into 64 × 64 squares; one block of 256 threads fills each square, 16
numbers per thread. The adding is done in the CPU's order and precision.

**1. Operation** (the qkv use)

```
Input   A    [M, K] = [B, 2880]     fp32   (w->t, after RMSNorm)
        W    [N, K] = [5120, 2880]  bf16   (g.w_qkv)
        bias [N]    = [5120]        fp32   (g.b_qkv, may be null)
Output  C    [M, N] = [B, 5120]     fp32   (w->qkv, new buffer)
Math    C[m, n] = ( Σ_{k=0..K-1} A[m, k] · W[n, k] ) + bias[n]
```

W is stored `[N, K]`: one row of weights per output, so it is **row of A · row of W**.
`Bf16Rows` (`gemm.hpp:23-29`) reads `W[n·K + k]` and widens it to fp32. Copies `matmul()`
(`run.cpp:355-368`) plus the bias loop (`run.cpp:568-570`).

| use | M | N | K |
|---|---|---|---|
| qkv | B | 5120 | 2880 |
| attention output | B | 2880 | 4096 |
| router | B | n_experts | 2880 |
| last layer (word scores) | B | vocab | 2880 (no bias) |

**2. Threads** (example M = 512)

| level | count | owns |
|---|---|---|
| grid | (5120/64) × (512/64) = **80 × 8 = 640 blocks** | block (bx, by) = rows `by·64…+63`, columns `bx·64…+63` |
| block | 256 threads | 64 × 64 = 4096 outputs |
| thread | **16 outputs**, a 4 × 4 square | `tx = tid % 16`, `ty = tid / 16` |
| k loop | 2880 / 32 = **90 rounds** | each round: copy a 64 × 32 slice of A and of W to shared memory, barrier, use it, barrier |

Rows ≥ M and columns ≥ N are loaded as 0 and never written; with M = 1 (the test) 63 of every 64
rows are padding.

**Why `ty * 4 + i` and `tx * 4 + i` (lines 72-73).** The 64 × 64 tile is a 16 × 16 grid of 4 × 4
squares. `ty` (0-15) picks the row group, so its first row is `ty·4`; `+ i` (i = 0-3) walks its 4
rows. Same for `tx` and the columns. Thread 18: `ty = 1`, `tx = 2` → rows 4-7, columns 8-11 of
the tile. At each `kk` it reads 4 A values (its rows) and 4 W values (its columns) and makes all
4 × 4 = 16 products (`acc[i][j] += a[i]·b[j]`, line 77): 8 reads, 16 products.

```
             tx=0   tx=1   tx=2  ...  tx=15     (4 columns each)
ty=0  [ t0 ] [ t1 ] [ t2 ] ... [t15 ]
ty=1  [ t16] [ t17] [★t18] ... [t31 ]           (4 rows each)
 ...
ty=15 [t240] ...               [t255]
```

**3. Details (block bx = 3, by = 2, thread 18, round 2)**
- Tile starts at `m0 = 128`, `n0 = 192`; round 2 covers k = 64-95.
- **Load** (62-67), a different split: `idx = 18, 274, …, 1810` → `r = 0, 8, …, 56`, `kk = 18`.
  Thread 18 copies `A[128+r, 82]` and `W[192+r, 82]`: 8 + 8 values, mostly *not* the ones it uses.
- **Compute/write**: `C[132…135, 200…203]`.
- Per round it reads 128 A + 128 W values from shared memory and copied 16 itself, so **240 came
  from other threads**: hence `As`, `Ws` and the barrier on line 68. The barrier on line 79 stops
  a fast thread from loading the next round over data others are still reading. Each A value is
  used by the 16 threads with the same `ty`, each W value by the 16 with the same `tx`.

| GPU (line) | CPU | why |
|---|---|---|
| `__fmul_rn(a, b)` (77) | `w[…] * x[j]` (`run.cpp:364`) | product rounded to fp32 |
| `__dadd_rn(acc, …)` (77) | `double val` (`run.cpp:362`) | sum in fp64 |
| k0 rounds, then kk = 0…31 | `for j` | increasing k, the CPU's order |
| `(float)acc` (90) | `xout[i] = val` (366) | one rounding to fp32 |
| `__fadd_rn(c, bias)` (92) | `qkv[i] += b[i]` (569) | bias added after, in fp32 |

Measured on 1000 random outputs with K = 2880 (Python): an fp32 sum differs from the CPU in
**967**; an unrounded product (e.g. an fp64 FMA) differs in **439**. Both details are needed.

Tiny example (M = N = 2, K = 3): `A = [[1,2,3],[4,5,6]]`, `W = [[1,0,2],[0.5,1,−1]]`,
`bias = [0.25, −1]` → `C = [[7.25, −1.5], [16.25, 0]]`, the same as `A @ Wᵀ + bias` (checked in
Python).

### 9.7 `As[GEMM_BK][GEMM_BM + 1]`: why the +1, and why this order (2026-10-28)

**In plain words:** shared memory is split into 32 lanes. Threads that use the same lane at the
same time wait in line. Without the +1, the 32 threads of one load all land in the same lane. The
+1 shifts each row by one lane. The `[BK][BM]` order itself makes no difference here (measured).

New words:
- **LDS (local data share)** — the block's shared memory (`__shared__`).
- **bank** — LDS has 32 banks, 4 bytes wide; a float at position `a` (in floats) is in bank
  `a mod 32`.
- **bank conflict** — several of the 32 threads hit the same bank at *different* positions and
  take turns ("32-way" = 32 turns). Many threads reading the *same* position is a broadcast, one turn.
- **coalesced read** — neighbouring threads read neighbouring addresses in global memory, so it is
  fetched as one piece.

**Why +1.** The load gives neighbouring threads neighbouring k (`kk = t % 32`), so threads 0-31
read `A[m, k0…k0+31]`, 32 floats side by side: coalesced. They write `As[kk][r]` at position
`kk·rowlength + r`:
- rowlength 64: `64 mod 32 = 0`, so all 32 threads are in bank r: **32-way conflict**.
- rowlength 65: `65 mod 32 = 1`, so thread kk is in bank `r + kk`: 32 different banks, no conflict.

Cost: `32 × 65 × 4 = 8,320` bytes instead of 8,192 (the compiled code puts `Ws` at offset 8320).
Letting neighbouring threads walk `r` instead would avoid the conflict without padding, but then
they would read global memory 2880 floats apart, which is much slower.

**Why `[BK][BM]`: style, not speed.** Worst conflict per 32 threads (Python count):

| layout | load write | read `a[i]` (72) | read `b[j]` (73) |
|---|---|---|---|
| `[32][64]` no pad | **32-way** | 1 | 2-way |
| `[32][65]` (**the code**) | 1 | 1 | 2-way |
| `[64][32]` no pad | 1 | 2-way | **16-way** |
| `[64][33]` | 1 | 1 | 2-way |

Without padding both orders are bad, in different places. With padding they are equal. Compiled
both for gfx90a (`hipcc -O3 --cuda-device-only -S`): the same LDS instructions, the same number of
them, only the offsets differ. `[BK][BM]` just reads like the math: for each `kk`, add (column k
of the A tile) × (row k of the W tile) to the 64 × 64 result. It is the common layout in GEMM
tutorials.
