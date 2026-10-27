# Lesson 11: Do the step after the first expert multiply inside it

| | |
|---|---|
| Reference commit | `5d9e8e7` "Fuse SwiGLU (and, without a partner GCD, the row gather) into mlp1" |
| Starts from | `9813b7a` (lesson 10) |
| To type | 24 lines of program code in 2 files (22 in the engine, 2 comment lines in the kernel); no tests or docs |
| Speed after | 20b 64,099 → **66,710** tok/s (+4.1 %); 120b measured together with lesson 13 (35,815) |

**In plain words:** the first expert multiply wrote a big table of results to memory, and the
next small step read it straight back, changed it a little and wrote a smaller table. Lesson 10
already built a version of the multiply that does that small step itself, before anything leaves
the chip. This lesson switches it on. Nothing about the answers changes; the big table is simply
never written.

## 1. Goal

For each layer, replace three kernels (copy each token's row into expert order, mlp1, SwiGLU)
by one, where the shapes allow it, with exactly the same bits.

## 2. Why this, why now

- After lesson 10, the expert multiplies were still the biggest cost of a step.
- mlp1 writes its fp32 output, 5,760 numbers per row: at 512 requests that is 2,048 rows ×
  5,760 × 4 bytes = **47 MB per layer**, written once and read back once by SwiGLU, which then
  writes half as many bf16 numbers.
- The gather (copying each token's row into the expert order before mlp1) is another full pass
  over the inputs.
- The fused entry points were already written and tested in lesson 10 (`moe_gemm_test`: "mlp1
  fused vs separate kernels … 0 differ"), so switching them on was cheap and safe.

## 3. The technique

New words:
- **fusion (kernel fusion)** — doing two steps in one kernel, so the result of the first stays in
  registers and never makes the round trip through memory.
- **epilogue** — the end of a multiply kernel, after the sums are done: add the bias, then store.
  Extra work placed there is "free" in memory traffic.
- **SwiGLU** — gpt-oss's activation between mlp1 and mlp2. mlp1's 5,760 outputs per row are 2,880
  pairs (gate, up); each pair becomes one number: with `g = min(gate, 7)` and
  `u = up` limited to −7…7, the result is `g · sigmoid(1.702 · g) · (u + 1)`.
- **gather** — copying rows from one table into a new order. Here: token rows into expert order
  (`row_pair` says which token each expert row belongs to).
- **partner GCD** — for 120b, the other GCD of the same card, which holds the other half of the
  experts (lesson 05).

```
 before, per layer (20b):                          after:
   gather   t16 ──► Xg16 (expert order, bf16)        gemm_grouped_mxfp4_gather_swiglu:
   mlp1     Xg16 ──► Yg   (fp32, 47 MB)  write         A rows read straight from t16
   SwiGLU   Yg   ──► Gg16 (bf16)         read+write      (row r = token row_pair[r] / 4)
   mlp2     Gg16 ──► Zg                                 MFMAs as in lesson 10
                                                        epilogue: + bias, SwiGLU, store bf16 Gg16
                                                      mlp2 unchanged

 in the epilogue, gate and up sit in neighbouring lanes:
   lane j (even) holds column 2q (gate), lane j+1 holds column 2q+1 (up)
   up = __shfl_xor(value, 1)   → the even lane computes SwiGLU and stores G[row, q]
```

Same bits: the epilogue uses exactly the arithmetic of `swiglu_bf16_kernel`, on exactly the same
fp32 sums, so every stored bf16 is identical.

Why the gather is fused only without a partner: with a partner (120b), an expert's rows come from
two tables — this GCD's own tokens and the partner's tokens, read from the partner's memory. The
fused kernel reads its rows from one table. 120b keeps the separate gather and uses the fused
mlp1 + SwiGLU without it. (Lesson 26 puts both GCDs' rows into one table, and then 120b fuses the
gather too.)

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/engine/gpu_engine.hpp` | +22 −6 | `layer_experts()`: pick the fused path when `moe_mfma_ok` holds; without a partner call `gemm_grouped_mxfp4_gather_swiglu`, with one call the gather then `gemm_grouped_mxfp4_swiglu`; otherwise the old three kernels | yes |
| 2 | `getp-csrc/kernels/fast_moe_gemm.hpp` | +2 −2 | two comments: the fused entry points are now used | yes |

Read while typing (line numbers at `5d9e8e7`):
- `getp-csrc/engine/gpu_engine.hpp:404–425` — the three-way choice.
- `getp-csrc/kernels/fast_moe_gemm.hpp:412` — the gather inside the kernel is one line: the A
  row address is `gather[r] / topk` instead of `r`.
- `getp-csrc/kernels/fast_moe_gemm.hpp:470–491` — the SwiGLU epilogue; compare it line by line
  with `swiglu_bf16_kernel` (`getp-csrc/kernels/fast_misc.hpp:68`).

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show 5d9e8e7
git diff 5d9e8e7 -- $(git show --name-only --format= 5d9e8e7)   # must print nothing
make runomp
```

Quick check, 20b (one GCD, 64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # ccfc732a4a25 — the same as lesson 10
```

The fingerprint must **not** change: this lesson only removes memory traffic.

Quick check, 120b (a GCD pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 836592181c82 — the same as lesson 10
```

No new test: the bit-for-bit check of the fused kernels is lesson 10's `moe_gemm_test` (its
lines "mlp1 fused vs separate kernels … 0 differ").

Full run:

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-11 --score
```

Expected: about **66,710 tok/s**, fingerprint `496d7ce70b4e` (the same as lesson 10, so the same
scores: METEOR 0.554, BERTScore 0.979).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Fuse SwiGLU (and, without a partner GCD, the row gather) into mlp1"`

## 6. Result

| | before | after |
|---|---:|---:|
| 20b, gather + mlp1 + SwiGLU, one layer, B=512, one GCD | 947 µs | **802 µs** (−15 %) |
| 20b tok/s (8 GCDs, 4,096 prompts) | 64,099 | **66,710** (+4.1 %) |
| 20b fingerprint (full run) | `496d7ce70b4e` | `496d7ce70b4e` (same answers) |
| 20b, 64 prompts × 128, one GCD | 2,509 tok/s | 2,740 tok/s |
| 120b, 64 prompts × 128, a GCD pair | 1,407 tok/s | 1,417 tok/s |

For 120b this lesson alone was not measured on the full input; lesson 13 gives the two together:
32,651 → 35,815 tok/s.

Why 120b gains so little here: it still runs the separate gather (it has a partner), so it only
saves the mlp1 → SwiGLU round trip.

## 7. Decisions and reasons

- **Decided:** fuse SwiGLU into mlp1, and the gather too when there is no partner.
  **Why:** −145 µs per 20b layer (947 → 802), about 3.5 ms per step, with the same bits; the
  47 MB fp32 intermediate per layer disappears.
- **Decided:** keep the separate kernels for shapes the matrix-core kernel does not take
  (`moe_mfma_ok`), and on the exact path.
  **Why:** the exact path must keep reproducing the CPU bit for bit.
- **Not chosen now:** storing the expert weights in reading order at load time, although a
  read-only test showed 855 → 1,296 GB/s. **Why:** it changes the layout the kernel reads, a
  larger change; we waited for the next profile to confirm the weight reads were the main cost.
  It became lesson 15.

## 8. Questions to think about

1. The fused kernel does exactly the same arithmetic as the three separate ones. Where, then,
   does the 15 % come from?
2. SwiGLU needs the gate *and* the up value of the same pair. Why is it convenient that they end
   up in neighbouring lanes, and what would you do if they were 2,880 columns apart?
3. Fusing the gather means the multiply reads A rows from scattered places (by token) instead
   of one contiguous block. Why does that cost almost nothing here?

## 9. Discussion notes

(empty: we write here during the walk-through)
