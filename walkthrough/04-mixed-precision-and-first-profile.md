# Lesson 04: Smaller numbers for the big multiplications, and the first time profile

| | |
|---|---|
| Reference commit | `3163e03` "Add a mixed-precision fast path for getp" |
| Starts from | `e824840` (lesson 03) |
| To type | 467 lines of program code in 6 files; 2 lines of tests; 81 lines of docs (optional) |
| Shortcut | `git cherry-pick 3163e03` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 5,217 tok/s (was 3,348: +56 %) |

**In plain words:** we store the numbers that go into the big multiplications at half size, but
still add up the results at full size. That alone makes the program 56 % faster, and the answers
stay good. Then, for the first time, we measure how long each part of one step takes. The
measurement decides what to fix next: the expert multiplications take two thirds of the time.

## 1. Goal

1. A second, faster path (the default from now on), with the exact path of lesson 03 still
   available through `GPTOSS_EXACT=1`.
2. Every matrix multiply and the attention behind a fixed function, so a later lesson can
   replace the inside without touching the rest: `gemm_bf16`, `gemm_grouped_mxfp4`,
   `attention_bf16`.
3. A time profile that says which of them to rewrite first.

## 2. Why this, why now

- The GPU's **matrix cores** (calculators built for small matrix multiplies) only take 16-bit
  inputs such as bf16, and are 8 times faster than the ordinary fp32 units: 181 vs 22.6 TFLOP/s
  on one GCD. The exact path can never use them.
- Half the bytes per number means half the bytes to read. A bf16 KV cache needs 14.5 GB instead
  of 29 GB at 512 prompts per GCD.
- Other teams on the same course used bf16 and passed the quality gates (METEOR 0.30–0.55).
- Before rewriting any kernel we must know where the time goes. Guessing wastes days.

## 3. The technique

New words:
- **mixed precision** — keeping some numbers in a small format (bf16) and others in a large one
  (fp32), choosing per value.
- **residual stream** — the 2,880 numbers per token that every layer adds its result into; it
  carries the model's state through all 24 layers, so it stays fp32.
- **accumulate** — add up many products. Always in fp32 here: a 2,880-term sum in bf16 would
  pile up rounding errors.
- **round to nearest even** — how fp32 becomes bf16 (`f32_to_bf16` in `fast_common.hpp`).
  Just cutting off the low 16 bits would push every value towards zero.
- **profile** — a record of how long each kernel ran. **rocprofv3** — AMD's tool that records
  every kernel launch with its start and end time (`--kernel-trace`).

```
                 fp32 x  (residual stream, never rounded)
                   │
     RMSNorm ──► bf16 t ──► multiply (bf16 inputs, fp32 sums) ──► fp32 result
                                              │
     attention reads the bf16 KV cache, computes in fp32 ─► bf16 output ─► next multiply
                                              │
                 fp32 x += fp32 result
```

The kernels in this commit are still simple (plain fp32 multiply-adds on LDS tiles, like lesson
03 but without fp64). The speed-up comes only from moving half the bytes.

Reading a profile: add up each kernel's time over a run, divide by the number of steps, and sort.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_common.hpp` | 20 | `bf16_t`, `bf16_to_f32`, `f32_to_bf16` (round to nearest even, NaN stays NaN) | yes |
| 2 | `getp-csrc/kernels/fast_misc.hpp` | 103 | `rmsnorm_bf16_kernel`, `rope_kv_bf16_kernel` (RoPE, then K and V into the bf16 cache), `swiglu_bf16_kernel`, `gather_bf16_kernel`, `moe_combine_kernel` | yes |
| 3 | `getp-csrc/kernels/fast_gemm.hpp` | 67 | `gemm_bf16`: bf16 × bf16, fp32 sums, 64 × 64 LDS tiles | yes |
| 4 | `getp-csrc/kernels/fast_moe_gemm.hpp` | 81 | `gemm_grouped_mxfp4`: MXFP4 decoded to fp32 in LDS, fp32 multiply-adds | yes |
| 5 | `getp-csrc/kernels/fast_attention.hpp` | 87 | `attention_bf16`: lesson 03's attention shape with fp32 math and a bf16 cache | yes |
| 6 | `getp-csrc/engine/gpu_engine.hpp` | +109 −27 | `exact` flag; bf16 buffers and cache; `gpu_step` split into `gpu_step_exact` and `gpu_step_fast` | yes |
| 7 | `getp-csrc/tests/gpu_vs_cpu.cpp`, `stage_check.cpp` | 1 + 1 | ask for the exact path | optional |
| 8 | `docs/10-bf16-mixed-precision.md` | 81 | this idea, for newcomers | optional (read it) |

Spots to read while typing (line numbers at `3163e03`):
- `getp-csrc/kernels/fast_common.hpp:14` `f32_to_bf16` — the rounding in 4 lines.
- `getp-csrc/engine/gpu_engine.hpp`: put `gpu_step_fast` next to `gpu_step_exact` — the same 20
  launches per layer; only the buffers (`t16`, `att16`, `Xg16`, `Gg16`, `kc16`, `vc16`) and the
  kernel names differ.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick 3163e03
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout 3163e03 -- $(git show --name-only --format= 3163e03)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 3163e03
git diff 3163e03 -- $(git show --name-only --format= 3163e03)   # must print nothing
make runomp
```

Quick check, both paths (one GCD, about a minute each):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12            # expected: b4c73f55a422 (fast path, new)
GPTOSS_EXACT=1 HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64x.txt -n 128
sha256sum $W/out64x.txt | cut -c1-12           # expected: ad7fdd092a55 (exact path, as in lesson 03)
```

The profile (one GCD, 512 prompts, 32 positions, about 30 s):

```bash
(echo 512; sed -n 2,513p evaluation/input.txt) > $W/in512.txt
tools/gpulock.sh 2 -- rocprofv3 --kernel-trace --output-format csv -d $W/prof04 -o p -- \
  ./run $M20 -m getp -i $W/in512.txt -o $W/out512.txt -n 32
python3 walkthrough/kernel_shares.py $W/prof04/p_kernel_trace.csv
```

What it printed on the reference build:

```
busy 21.56 s, 32 steps, 673.9 ms per step
   share  total s   calls   us/call  kernel
   68.2%   14.707    1536    9574.9  gemm_grouped_mxfp4_f32acc_kernel     ← the expert multiplies
   28.7%    6.179    2336    2645.1  gemm_f32acc_kernel<Bf16Rows>         ← qkv, out, router, LM head
    1.1%    0.245     768     319.4  moe_group_kernel
    1.0%    0.218     768     284.3  attention_bf16_kernel
    0.2%    0.047     768      60.8  swiglu_bf16_kernel
    …       (9 more kernels, each 0.2 % or less)
```

**The same kind of run as a picture (Perfetto).** rocprofv3 can also write a `.pftrace` file that
the web page https://ui.perfetto.dev draws as a timeline: one row per CPU thread and per GPU
stream, one box per call. Keep the run short; 64 prompts × 16 steps gives a 3 MB file in ~15 s:

```bash
tools/gpulock.sh 2 -- rocprofv3 --runtime-trace --output-format pftrace -d $W/prof04pf -o p -- \
  ./run $M20 -m getp -i $W/in64.txt -o $W/out64pf.txt -n 16
ls -lh $W/prof04pf/p_results.pftrace
```

- `--runtime-trace` records the kernels **and** the HIP calls on the CPU side (`hipLaunchKernel`,
  `hipMemcpy`, …) and the memory copies. `--kernel-trace` alone gives only the kernels.
- `--output-format csv pftrace` writes both kinds of file in one run, if you also want
  `kernel_shares.py`.

Bring the file to your own computer. From a terminal **on your computer** (not the server):
`scp mi250-02:/home/namvu/walk-runs/prof04pf/p_results.pftrace .`
— or in VS Code: File → Add Folder to Workspace → `/home/namvu/walk-runs`, then right-click the
file → Download.

Open https://ui.perfetto.dev → "Open trace file" (left menu) → pick the file. The page reads the
file inside your browser; it is not uploaded. Keys: `W`/`S` zoom in/out, `A`/`D` move left/right.
Click a box for its details; drag across a row to get, at the bottom, a table of every name in that
range with its count and total time.

The rows you will see (from a test run of lesson 03's program, 16 steps):

| Row | What is in it |
|---|---|
| `THREAD 1` | the main thread: 365 × `hipMemcpy` (uploading the weights, ~4.7 s), `hipMalloc` |
| `STREAM 0` | the GPU side of those uploads (`MEMORY_COPY_HOST_TO_DEVICE`) |
| `THREAD 2` | the GPU's worker thread: ~390 `hipLaunchKernel` per step (about 60 µs each in this recorded run), then a long `hipMemcpyAsync` per step (`gpu_engine.hpp:304`): the CPU waiting until the step's kernels are done |
| `STREAM 1` | the kernels, back to back: zoom into one step to see the 24 layers, each a long expert multiply with small kernels between |

Full run (8 GCDs, about 13 minutes):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-04 --score
```

Expected: about 5,217 tok/s, METEOR 0.552, BERTScore 0.979, fingerprint `40dd9b429d66`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Add a mixed-precision fast path for getp"`

## 6. Result

| 20b, 8 GCDs, 4,096 prompts | exact (lesson 03) | mixed precision |
|---|---:|---:|
| tok/s | 3,348 | **5,217** (+56 %) |
| one step at 512 prompts, one GCD | 1.08 s | 0.67 s |
| METEOR (gate ≥ 0.30) | 0.9967 | 0.552 |
| BERTScore (gate ≥ 0.90) | 0.99998 | 0.979 |
| answers identical to the reference | 4,062 (99.2 %) | 344 (8.4 %); median first difference at token 138 |

Only 8 % of answers stay word for word the same, yet the scores stay high: when two next tokens
are nearly tied, rounding picks the other one, and the answer goes on differently but just as
well. BERTScore 0.979 says the meaning stays nearly the same.

What the profile says, in numbers:
- The expert multiplies do about 2.45 trillion operations per step (mlp1 2,048 rows × 5,760 ×
  2,880 × 2, plus mlp2 half that, × 24 layers) at about 5 TFLOP/s: 9.6 ms per call. AMD's
  library needs about 1.3 ms for the same work. The matrix cores could do 181 TFLOP/s.
- The dense multiplies (29 %) have the same problem, smaller.
- Attention is only 1 % here because these 32 positions are short histories. Its cost grows
  with the position: near position 1,023 this simple kernel reads every key and value 8 times
  (≈ 8.6 GB per full layer, ~100 ms per step near the end of the answers).
- Everything else together: under 3 %.

## 7. Decisions and reasons

- **Decided:** bf16 inputs to every multiply and a bf16 KV cache; fp32 sums and residual stream.
  **Why:** matrix cores need bf16 (8× the fp32 rate); half the bytes; other teams passed with
  bf16; checked here by the scores (0.552 / 0.979, far above the gates).
  **Not chosen:** fp32 everywhere (no matrix cores); int8 or fp8 already now (bigger quality
  risk before the basics are fast; lesson 18 comes back to it).
- **Decided — the order of work after this profile:**
  1. the expert multiply (68 %) — the biggest share and the furthest from the library (9.6 vs
     ~1.3 ms per call);
  2. the dense multiply (29 %);
  3. attention, although only 1 % here, because its cost grows with the answer's length and it
     reads every key and value 8 times;
  4. the small kernels last (< 3 % together).
  **Why:** fix what takes the time, largest first. The three big kernels sit behind fixed
  functions, so they were rewritten side by side and landed in the order they were ready:
  attention (lesson 06), dense multiply (08), expert multiply (10).
  **Not chosen:** tuning the small kernels first — at most 3 % to win.
- **Decided:** leave continuous batching and PagedAttention for later (both explained in
  `docs/03` and lesson 16).
  **Why:** 95 % of the reference answers run to 1,024 tokens, and all 4,096 prompts fit in
  memory at once (bf16 KV cache 14.5 GB per GCD), so the possible gain is a few percent.

## 8. Questions to think about

1. Why is it safe to round the input of a multiply to bf16 but not the residual stream?
2. The expert multiply does ~2.45 trillion operations per step. At 181 TFLOP/s, what is the
   least time it could take? What else could stop a kernel from reaching that?
3. Attention is 1 % of this profile. Why would it be wrong to conclude "attention does not
   matter", and how would you measure it properly?

## 9. Discussion notes

### 9.1 What "behind a fixed function" means (section 1, goal 2)

"Fixed function" means a function whose **name and inputs never change again**, while what is
inside it may be thrown away and rewritten. The rest of the program only ever calls that name.

`gemm_bf16` is not a kernel. It is an ordinary function that runs on the CPU (a *host function*)
and whose only job is to launch the kernel. At `3163e03`
(`getp-csrc/kernels/fast_gemm.hpp:63-67`):

```cpp
static void gemm_bf16(hipStream_t s, const bf16_t *A, int M, const bf16_t *W,
                      const float *bias, float *C, int N, int K) {
  dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  gemm_f32acc_kernel<<<grid, 256, 0, s>>>(A, M, Bf16Rows{W, K}, bias, C, N, K);
}
```

- Line 1–2, the doorway: what goes in (`A` M×K bf16, `W` N×K bf16, `bias`), what comes out
  (`C` M×N fp32), and on which queue of GPU work (`s`, the stream).
- Line 3–4, the room behind it: pick the grid size and launch one simple kernel.

`gpu_engine.hpp` (the code that runs the 24 layers) calls only the doorway, never the kernel:

```
gpu_engine.hpp (3163e03)                       fast_gemm.hpp
  line 318  gemm_bf16(... w_qkv ...)    ─┐
  line 325  gemm_bf16(... w_o ...)      ─┤      gemm_bf16(...)        ← stays the same
  line 329  gemm_bf16(... w_router ...) ─┼───►  ─────────────────
  line 343  gemm_bf16(... unemb ...)    ─┘      how it does it        ← lessons rewrite this
```

The payoff, in the later commits:

| Lesson | Commit | Rewrote | `fast_*.hpp` grew by | Touched `gpu_engine.hpp`? |
|---|---|---|---:|---|
| 06 | `03e8e9d` | `attention_bf16` | +381 | no |
| 08 | `30dd744` | `gemm_bf16` | +558 | no |
| 10 | `9813b7a` | `gemm_grouped_mxfp4` | +672 | no |

At `30dd744` the inside of `gemm_bf16` picks one of several matrix-core kernels by the shape
`M, N, K`, yet its first two lines are the same as at `3163e03`, character for character. So:
- the three rewrites could be worked on side by side, each in its own file, and land in any order
  (section 7);
- each rewrite is tested on its own (`gemm_test.cpp`, `attn_test.cpp`, `moe_gemm_test.cpp`) against
  the same inputs and outputs the simple version had;
- if a rewrite is wrong, only that file is suspect.

The simple kernels of this lesson are therefore placeholders: correct and slow, with the doorway
already in its final shape.

### 9.2 The trace file and `kernel_shares.py`

`rocprofv3 --kernel-trace` writes one row per kernel launch. Our run (lesson 03 build, HIP GPU 2,
512 prompts, 32 steps) has 12,601 rows. The columns, with values from that run:

| Column | What it is | In our run |
|---|---|---|
| `Kind` | type of record | always `KERNEL_DISPATCH` (one kernel launch) |
| `Agent_Id` | which processor ran it; rocprofv3 numbers every one, see `p_agent_info.csv` | `Agent 4`: agents 0–1 are the two CPUs, 2–9 the 8 GCDs in HIP order, so 4 = HIP GPU 2 |
| `Queue_Id` | the hardware queue (list in memory the GPU takes launch orders from) | 2 for the 121 copies at load time, 1 for the model's kernels |
| `Stream_Id` | the HIP stream | 0 = default stream (`hipMemcpy` at load), 1 = `w->stream` (`gpu_engine.hpp:215`) |
| `Thread_Id` | Linux number of the CPU thread that launched it | 1395177 = main thread (loading), 1395509 = this GPU's worker thread (`gpu_engine.hpp:363`) |
| `Dispatch_Id` | launch counter | 1 … 12,601 |
| `Kernel_Id` | one number per different kernel | 15 different kernels |
| `Kernel_Name` | full C++ name with argument types | the script keeps only the part before `(` |
| `Correlation_Id` | links this row to the CPU-side call, if those are traced too (`--hip-trace`) | equal to `Dispatch_Id` here |
| `Start_Timestamp`, `End_Timestamp` | nanoseconds on the GPU's clock; only End − Start means something | first copy: 6,560 ns |
| `LDS_Block_Size` | bytes of `__shared__` memory per block, rounded up | gemm 16,896 (two 32 × 65 float tiles = 16,640) |
| `Scratch_Size` | bytes per thread pushed out to slow memory because registers ran out | 0 everywhere (good) |
| `VGPR_Count` | vector registers per thread (each thread has its own) | gemm 108, attention 120 |
| `Accum_VGPR_Count` | AGPRs: a second register set on MI250, made for matrix-core results | used here as spare storage (attention 128) |
| `SGPR_Count` | scalar registers: one value shared by the whole wave of 64 threads | 16–112 |
| `Workgroup_Size_X/Y/Z` | threads per block (`blockDim`) | 256 or 64 |
| `Grid_Size_X/Y/Z` | **total threads**, not blocks | embed `<<<512, 256>>>` → 131,072 = 512 × 256 |

The last 7 columns are fixed when the kernel is compiled or launched; the same kernel shows the
same values on every row. Registers matter later: the more a thread uses, the fewer threads fit on
a compute unit at once.

What the script does: per agent and per kernel name, add up `End − Start` and count the rows;
`busy` = sum over all kernels; one step = one call of `embed_kernel`; sort by time, print share,
seconds, calls and µs per call. `busy` is a sum of kernel times, not clock time — with one stream
that is the same (here 33.97 s busy vs 33.98 s from the first to the last model kernel: the GPU
was idle for about 15 ms in all).

Our lesson 03 run printed 1,061.7 ms per step, with the same order as the reference profile in
section 5: expert multiply 69.9 %, dense multiply 27.8 %, attention 0.7 %. The lesson page's exact
step time is 1.08 s, so this is the "before" picture for this lesson.

### 9.3 Full-length profile of lesson 03's program (before `3163e03`)

The same profile, but without `-n 32`, so all 1,024 steps run (lesson 03 build, still at
`35f01e8`, HIP GPU 2, 512 prompts). It took 20 minutes:

```
$ tools/gpulock.sh 2 -- rocprofv3 --kernel-trace --output-format csv -d $W/prof04 -o p -- \
    ./run $M20 -m getp -i $W/in512.txt -o $W/out512.txt
...
elapsed time(s): 1199.450000, achieved throughput TPS (tok/s): 423.077244
...
$ python3 walkthrough/kernel_shares.py $W/prof04/p_kernel_trace.csv
Agent 4: busy 1198.80 s, 1024 steps, 1170.7 ms per step
   share  total s   calls   us/call  kernel
   64.7%  776.128   49152   15790.4  gemm_grouped_mxfp4_exact_kernel
   25.2%  302.511   74752    4046.9  gemm_exact_kernel<Bf16Rows>
    8.5%  102.491   24576    4170.3  attention_exact
    0.7%    7.830   24576     318.6  moe_group_kernel
    0.3%    3.137   50176      62.5  rmsnorm_exact
    0.2%    1.871   24576      76.1  swiglu_exact
    0.1%    1.247   24576      50.7  router_topk_exact
    0.1%    1.236   24576      50.3  moe_combine_exact
    0.1%    0.853   24576      34.7  moe_gather_kernel
    0.0%    0.448    1024     437.5  argmax_kernel
    0.0%    0.423   24576      17.2  rope_exact
    0.0%    0.408   24576      16.6  add_inplace_kernel
    0.0%    0.196   24576       8.0  kv_write_kernel
    0.0%    0.014    1024      13.5  embed_kernel
    0.0%    0.011    2169       5.0  __amd_rocclr_copyBuffer
```

(The model settings and rocprofv3's own log lines are left out. This run wrote over the 32-step
`p_kernel_trace.csv` of 9.2.)

Compared with the 32-step run of 9.2:

| | 32 steps (9.2) | 1,024 steps |
|---|---:|---:|
| ms per step | 1,061.7 | 1,170.7 |
| expert multiply, µs per call | 15,452 | 15,790 |
| dense multiply, µs per call | 4,039 | 4,047 |
| attention, µs per call | 328 | **4,170** (×12.7) |
| attention share | 0.7 % | **8.5 %** |

The multiplies cost the same per call at every step; attention does not. Splitting the trace by
step (each step starts at its `embed_kernel`) and by layer:

| step | whole step, ms | attention, ms | in the 12 sliding-window layers | in the 12 full layers |
|---:|---:|---:|---:|---:|
| 1 | 1,035 | 5.0 | 2.5 | 2.5 |
| 127 | 1,135 | 43.5 | 22.8 | 20.7 |
| 255 | 1,152 | 63.0 | 22.9 | 40.1 |
| 511 | 1,169 | 102.3 | 22.9 | 79.4 |
| 767 | 1,201 | 142.0 | 22.9 | 119.1 |
| 1,023 | 1,255 | **185.1** | 22.9 | **162.2** |

- gpt-oss alternates two kinds of layer. The even layers (0, 2, …) only look back 128 positions
  (`sliding window: 128`, `gpu_engine.hpp:168`), so their attention stops growing at step 128:
  22.9 ms from then on.
- The odd layers look back at every earlier position, so their time grows in a straight line with
  the step: about 0.16 ms more per step, 162 ms at the last step.
- At the last step attention is 185 of 1,255 ms = **15 %** of the step, not 0.7 %.

This answers question 3 of section 8: a profile of 32 short steps hides attention, because it
measures only the start of the answers, where the histories are short. To see it properly, profile
the full answer length (or profile single steps late in the answer), and split the time by step.
Once the multiplies get 8–10× faster (lessons 08 and 10), the same 185 ms would be a much larger
share, which is why attention is third on the list of section 7 although the short profile gave
it 1 %.

### 9.4 Full-length profile after the cherry-pick (fast path)

Same command as 9.3, on HIP GPU 1. The `./run` was built from `f48c918`, which already holds
lesson 05 on top of lesson 04 (`8a5abb0`). For 20b on one GCD that changes only the three small
MoE kernels (`moe_group_ep_kernel`, `moe_gather_ep_kernel`, `moe_combine_ep_kernel` instead of
`moe_group_kernel`, `gather_bf16_kernel`, `moe_combine_kernel`); lesson 05 does not touch the
`fast_*.hpp` kernels, so the three big kernels are this lesson's.

```
$ tools/gpulock.sh 1 -- rocprofv3 --kernel-trace --output-format csv -d $W/prof04 -o p -- \
    ./run $M20 -m getp -i $W/in512.txt -o $W/out512.txt
...
getp: fast path, 1 GPU(s) in 1 group(s) of 1, 512 requests per GPU per wave, KV cache for 1024 positions
...
elapsed time(s): 772.801000, achieved throughput TPS (tok/s): 659.004064
...
$ python3 walkthrough/kernel_shares.py $W/prof04/p_kernel_trace.csv
Agent 3: busy 772.28 s, 1024 steps, 754.2 ms per step
   share  total s   calls   us/call  kernel
   62.6%  483.564   49152    9838.1  gemm_grouped_mxfp4_f32acc_kernel
   25.6%  197.846   74752    2646.7  gemm_f32acc_kernel<Bf16Rows>
    9.9%   76.543   24576    3114.5  attention_bf16_kernel
    0.9%    7.333   24576     298.4  moe_group_ep_kernel
    0.2%    1.531   24576      62.3  swiglu_bf16_kernel
    0.2%    1.276   24576      51.9  router_topk_exact
    0.1%    1.121   24576      45.6  moe_combine_ep_kernel<false>
    0.1%    0.923   50176      18.4  rmsnorm_bf16_kernel
    0.1%    0.753   24576      30.7  moe_gather_ep_kernel<unsigned short>
    0.1%    0.504   24576      20.5  rope_kv_bf16_kernel
    0.1%    0.441    1024     430.8  argmax_kernel
    0.1%    0.414   24576      16.8  add_inplace_kernel
    0.0%    0.015    1024      14.2  embed_kernel
    0.0%    0.012    2169       5.6  __amd_rocclr_copyBuffer
```

(Model settings and rocprofv3's log lines left out; this run wrote over the trace of 9.3.)

The first 32 steps of this trace average **674.3 ms**, the reference's 673.9 ms of section 5.

Before (9.3, exact path) → after (fast path), 512 prompts, 1,024 steps, one GCD:

| | exact (9.3) | fast (this run) | change |
|---|---:|---:|---:|
| tok/s (one GCD, under rocprofv3) | 423 | 659 | **+56 %** |
| ms per step | 1,170.7 | 754.2 | −36 % |
| expert multiply, µs per call | 15,790 | 9,838 | −38 % |
| dense multiply, µs per call | 4,047 | 2,647 | −35 % |
| attention, µs per call | 4,170 | 3,115 | −25 % |
| attention share | 8.5 % | 9.9 % | |

+56 % on one GCD is the same gain as the 8-GCD run of section 6 (3,348 → 5,217 tok/s).

Attention by step and layer, as in 9.3:

| step | whole step, ms | attention, ms | 12 sliding-window layers | 12 full layers |
|---:|---:|---:|---:|---:|
| 1 | 660 | 4.4 | 2.2 | 2.2 |
| 127 | 715 | 32.7 | 17.3 | 15.5 |
| 255 | 733 | 46.7 | 17.3 | 29.4 |
| 511 | 754 | 75.8 | 17.3 | 58.5 |
| 767 | 781 | 104.6 | 17.3 | 87.3 |
| 1,023 | 815 | **132.8** | 17.3 | **115.6** |

- The multiplies got faster by more than attention did, so attention's share went **up**: 16 % of
  the last step (133 of 815 ms), against 15 % before.
- The order of section 7 still holds: expert multiply (63 %), dense multiply (26 %), attention
  (10 %, growing with the step), the rest under 2 %.
