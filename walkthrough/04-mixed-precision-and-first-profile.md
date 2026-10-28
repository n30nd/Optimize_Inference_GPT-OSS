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

(empty: we write here during the walk-through)
