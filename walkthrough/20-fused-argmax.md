# Lesson 20: Pick the next word inside the last multiplication

| | |
|---|---|
| Reference commit | `9b2edc9` "Pick the greedy next token inside the LM-head multiply", then the docs-only commit `7faed85` |
| Starts from | `fa89592` (lesson 19) |
| To type | 150 lines of program code in 2 files; 122 lines of test (optional); 30 + 1 lines of docs (optional / cherry-picked) |
| Speed after | 20b 75,811 tok/s, 120b 42,113 tok/s (8 GCDs, official input); 20b 84,268 tok/s on 8,192 prompts |

**In plain words:** at the end of every step the program gives all 201,088 possible next words a
score, writes all those scores to memory, and then a second program reads them all back only to
find the best one per request. Now the multiplication that makes the scores keeps just the best
one as it goes, and the scores are never written. Same word chosen every time, including ties.

## 1. Goal

Remove the 412 MB write and re-read of the scores per step (at 512 requests) by doing the greedy
choice inside the LM-head multiply, with exactly the same chosen tokens as before.

## 2. Why this, why now

A full-length profile of the current build (20b, 512 requests, one GCD, 49.6 s of GPU time):
- the LM head (the last multiply) was about **10 %** of the time;
- `argmax_kernel` was **0.9 %** (436 µs per step) — and it existed only to read back the 512 ×
  201,088 fp32 scores = **412 MB** the LM head had just written;
- the fast path uses the scores for nothing else.

## 3. The technique

New words:
- **LM head** (language-model head) — the last multiply: final row (2,880 numbers) × a 201,088 ×
  2,880 table = one score (**logit**) per vocabulary entry.
- **greedy / argmax** — take the entry with the highest score; on a tie, the lowest column (that is
  what `argmax_kernel` and the CPU code do).
- **tile / epilogue** — a block computes one tile of the output (here 128 rows × 128 columns); the
  epilogue is the end of the kernel, where finished results leave the registers.
- **atomic maximum** (`atomicMax`) — a memory operation that replaces a value by a larger one in one
  indivisible step, so many blocks can update the same place without losing an update.
- **64-bit key** — one number that encodes "score, then column" so that a plain integer maximum
  makes the argmax choice.
- **NaN** — "not a number", the result of an invalid operation; `argmax_kernel` never picks it.

```
 before:  LM-head multiply ─► write 512 × 201,088 fp32 (412 MB) ─► argmax_kernel reads 412 MB ─► ids
 after:   LM-head multiply ─► in each tile: best key of each of its 128 rows ─► atomicMax(keys[row])
                           ─► argmax_keys_kernel: key → token id, key reset to 0

 key (64 bits) = [ score bits, made order-preserving | ~column ]
                   high 32 bits                        low 32 bits
   score:   positive x → bits | 0x80000000     negative x → ~bits      (bigger x → bigger integer)
            -0 and +0 → the same key half      NaN → whole key 0 (loses to everything)
   ~column: a lower column gives a bigger low half → on equal scores the lower column wins
```

A maximum gives the same answer in any order, so it does not matter which of the ~1,571 column
tiles of a row finishes first: the result is deterministic.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_gemm.hpp` | +138 | `argmax_key()`; `TileGemm::argmax_rows()` (the epilogue: rows through LDS 16 at a time, each lane scans a quarter of a row, shuffles and LDS combine, one `atomicMax` per row); `gemm_argmax_kernel`; `argmax_keys_kernel`; `launch_argmax`; `gemm_bf16_argmax()` (same tile configuration as `gemm_bf16`) | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +12 −4 | fast path: no logits buffer (memory estimate too), an `amax_keys` array zeroed once; `step_logits` calls `gemm_bf16_argmax`; the exact path keeps logits + `argmax_kernel` | yes |
| 3 | `getp-csrc/tests/lm_head_argmax_test.cpp` | +122 | same token as `gemm_bf16` + `argmax_kernel` on every row, 13 batch sizes, planted ties | optional |
| 4 | `docs/12-small-kernels-and-fusion.md` | +30 | section 5, "The LM head picks the next token" | optional |
| 5 | (`git cherry-pick 7faed85`) `docs/12-small-kernels-and-fusion.md` | +1 −1 | adds the end-to-end gain to the table | nothing to type |

Read while typing (line numbers at `9b2edc9`):
- `getp-csrc/kernels/fast_gemm.hpp:91` — `argmax_key()`: four lines that carry the whole idea.
- `getp-csrc/kernels/fast_gemm.hpp:323` — `argmax_rows()`; `:351` the shuffles that combine the
  four quarters of a row; `:371` the one `atomicMax` per row and tile.
- `getp-csrc/kernels/fast_gemm.hpp:500` — `argmax_keys_kernel` also resets each key to 0, so the
  next step needs no separate clear.
- `getp-csrc/engine/gpu_engine.hpp:535` — the fast path's whole LM head + choice is now one call.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 9b2edc9
git show 9b2edc9 -- getp-csrc/kernels/fast_gemm.hpp       # the change to copy, file by file

git diff 9b2edc9 -- $(git show --name-only --format= 9b2edc9)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 19)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: e205be8f59c8
```

Unit test (one GCD, about a minute):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/lm_head_argmax_test.cpp -o lm_head_argmax_test
HIP_VISIBLE_DEVICES=2 ./lm_head_argmax_test
```

Expected (seen on the reference build): one line per batch size and run, ending with

```
M 2048 run 1: 0 / 2048 rows differ  row 1 -> 777  row 2 -> 3000  row 3 -> 64005  row 0 -> 0
PASS
```

Rows 1–3 hold planted ties (the lower column must win: 777, 3000, 64005); row 0 is all zeros, so
every score is 0 and column 0 must win.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-20 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-20 --score
```

Expected: 20b about 75,800 tok/s, fingerprint `58023f92ead7`; 120b about 42,100 tok/s, fingerprint
`141e0b9dbc17` (both unchanged).

Larger input (optional; the official prompts twice, 1,024 requests per GCD):

```bash
python3 tools/make_input.py evaluation/input.txt 8192 > $W/input_8192.txt
GPTOSS_BATCH=1024 GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i $W/input_8192.txt -g all -t lesson-20-8192
```

Expected about 84,300 tok/s, fingerprint `8ca3004c984a`. (`--score` only works on the official
input file; the first 4,096 answers of this run scored METEOR 0.532, BERTScore 0.978.) While it
runs, `rocm-smi` in another terminal shows every card at its power cap.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Pick the greedy next token inside the LM-head multiply"`

Then copy the docs-only commit (nothing to type): `git cherry-pick 7faed85`

## 6. Result

| One GCD | before (LM head + argmax) | after (fused) |
|---|---:|---:|
| 512 requests | 4,897 + 435 µs | **4,510 + 6 µs** |
| 1,024 requests | 9,719 + 865 µs | **9,223 + 6 µs** |
| Memory for the scores, 512 requests | 412 MB | **4 KB of keys** |

| 8 GCDs (means of 2 alternating runs each) | before | after | change |
|---|---:|---:|---:|
| 20b, 4,096 requests | 74,813 | **75,811** | +1.3 % |
| 120b, 4,096 requests | 41,737 | **42,113** | +0.9 % |
| 20b, 8,192 requests (1,024 per GCD) | 79,504 (lesson 15 build) | **84,268** | +6.0 % over five lessons |

The multiply itself also got faster (4,897 → 4,510 µs), not only the argmax: it no longer writes
412 MB.

**Power cap.** During the timed part of the 8,192-prompt run every card sat at its 560 W limit, and
the GPU clock ran at 1.38–1.58 GHz instead of its 1.7 GHz maximum. With all 8 GCDs busy, the chip
slows itself down to stay within power, so the per-GCD speed on a full machine is lower than on one
GCD alone.

## 7. Decisions and reasons

- **Decided:** fuse the greedy choice into the LM-head multiply.
  **Why:** `argmax_kernel` (0.9 %) existed only to re-read 412 MB the LM head (10 %) had just
  written, and nothing else on the fast path reads the scores.
  **Not chosen:** keeping the scores — only the exact path needs them, and it keeps them.
- **Decided:** encode "score, then lower column" in one 64-bit key and use `atomicMax`.
  **Why:** a maximum is order-independent, so the many tiles of a row can finish in any order and
  the result is still deterministic; the key also reproduces the old tie rule, −0 = +0 and "never
  NaN" exactly. One different token in a rare tie would change every later token of that answer.
- **Decided:** also measure 20b on a larger input (8,192 prompts, 1,024 per GCD).
  **Why:** the expert multiply is more efficient on more rows (79 TFLOP/s at 512 rows, 99 at
  1,024), and 1,024 requests per GCD still fit (bf16 cache 29 GB + 13.7 GB of weights).
  **Not chosen:** 2,048 per GCD — the cache alone would need 58 GB.

## 8. Questions to think about

1. Why does the key put the score in the *high* half and the column in the *low* half, and why is
   the column inverted?
2. Many blocks call `atomicMax` on the same 512 keys. Why is the result the same in every run,
   while a floating-point `atomicAdd` sum would not be?
3. Why does the multiply itself get faster when it stops writing its output, although writing was
   "only" 412 MB per step?

## 9. Discussion notes

(empty: we write here during the walk-through)
