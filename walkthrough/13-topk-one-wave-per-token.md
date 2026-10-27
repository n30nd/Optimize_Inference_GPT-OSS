# Lesson 13: Pick each token's 4 experts with 64 workers per token

| | |
|---|---|
| Reference commit | `85e8426` "Pick each token's top-4 experts with one wavefront per token" |
| Starts from | `0214995` (lesson 12) |
| To type | 64 lines of program code in 2 files (59 in `getp-csrc/kernels/moe.hpp`, 5 in the engine); no tests or docs |
| Speed after | 20b 66,710 → **67,394** tok/s (+1.0 %); 120b (lessons 11 + 13 together) 32,651 → **35,815** (+9.7 %) |

**In plain words:** for every token, a small step picks the 4 experts with the highest scores.
It was done by one worker per token, each looking at all scores one after another, and 512
tokens kept only a handful of the GPU's 104 compute units busy. Now 64 workers share each token:
each looks at a few scores, and they compare notes in a few quick rounds. Same choices, same
weights, four to twelve times faster.

## 1. Goal

Replace `router_topk_exact` (one thread per token) by a kernel that gives the same experts and
the same weights, bit for bit, using one wavefront per token.

## 2. Why this, why now

A profile of a full 20b run after lessons 10–11 showed that, with the big multiplies fast, the
small kernels together were about 10 % of the time:

| small kernel | share of the run |
|---|---:|
| router top-4 (54 µs per call, one thread per row) | 2.3 % |
| expert sum | 1.9 % |
| RMSNorm | 1.7 % |
| RoPE | 1.0 % |
| argmax (436 µs per step) | 0.8 % |
| residual add | 0.7 % |

The top-4 was the largest, and the easiest to fix. For 120b it matters much more: 128 experts
per token instead of 32, 194 µs per layer, 7.0 ms of a ~125 ms step.

## 3. The technique

New words:
- **router** — the small multiply that gives each token one score per expert (32 for 20b, 128
  for 120b).
- **top-k** — picking the k largest scores (here k = 4); then a **softmax** over those 4 turns
  them into weights that add up to 1.
- **tie rule** — when two experts have exactly the same score, the lower expert number wins.
  The CPU code does this; ours must too, or a different expert is picked.
- **shuffle (`__shfl_xor`)** — a lane reads a register of another lane of the same wavefront,
  without going through memory.
- **butterfly reduction** — finding the best of 64 values in 6 shuffle rounds: each lane
  compares with the lane 32 away, then 16, 8, 4, 2, 1; after that every lane holds the best.
- **filling the GPU** — a kernel with 8 small blocks runs on 8 of 104 compute units; the others
  wait. Work must be cut into enough pieces to keep them all busy.

```
 before: router_topk_exact, one thread per token
   512 tokens → 8 blocks of 64 threads (8 of 104 CUs busy)
   thread b: 4 rounds × scan all E scores of row b (32 or 128 loads, one after another)

 after: router_topk_wave_kernel, one wavefront per token, 4 wavefronts per block
   512 tokens → 128 blocks
   lane l holds the scores of experts l, l+64, l+128, l+192   (E ≤ 256)
   round 1..4:
     each lane: best of its own untaken experts (≤ 4 values)
     butterfly over 64 lanes: (score higher) or (score equal and expert lower) wins
     the lane that owns the winner marks it taken
   lane 0: the same softmax as before (fp64 sums, the CPU-exact expf)
```

Same bits: the same 4 experts in the same order, and the weights computed by exactly the same
arithmetic as the old kernel.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe.hpp` | +59 | `TOPK_WAVES`, `router_topk_wave_kernel` | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +5 −1 | `layer_route()`: use the new kernel when E ≤ 256, the old one otherwise | yes |

Read while typing (line numbers at `85e8426`):
- `getp-csrc/kernels/moe.hpp:15–52` — the old kernel, to compare.
- `getp-csrc/kernels/moe.hpp:72–94` — the 4 rounds: own best, then the butterfly at `:82`;
  read the tie condition carefully, it is what keeps the result identical.
- `getp-csrc/kernels/moe.hpp:96–111` — lane 0's softmax: copied from the old kernel (`:38–51`).
- `getp-csrc/engine/gpu_engine.hpp:377–381` — the launch: `(B + 3) / 4` blocks of 256 threads.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show 85e8426
git diff 85e8426 -- $(git show --name-only --format= 85e8426)   # must print nothing
make runomp
```

Quick check, 20b (one GCD, 64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # ccfc732a4a25 — unchanged
```

Quick check, 120b (a GCD pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 836592181c82 — unchanged from lesson 11
```

A changed fingerprint here means a wrong expert somewhere: almost always the tie rule.

No new test file: the fingerprints are the test, on both paths. The exact path uses the same
kernel, so it also keeps its fingerprint (`GPTOSS_EXACT=1`, `ad7fdd092a55`).

Full run:

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-13
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-13-120b
```

Expected: 20b about **67,394 tok/s**, fingerprint `496d7ce70b4e`; 120b about **35,815 tok/s**,
fingerprint `7a6fd77322d9` (both the same as before, so the scores do not change: 20b 0.554 /
0.979, 120b 0.568 / 0.981).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Pick each token's top-4 experts with one wavefront per token"`

## 6. Result

| | before | after |
|---|---:|---:|
| top-4, 20b (32 experts), 512 tokens, per layer | 54 µs | **14.5 µs** |
| top-4, 120b (128 experts), per layer | 194 µs | **15.7 µs** |
| 20b tok/s (8 GCDs) | 66,710 | **67,394** (+1.0 %) |
| 120b tok/s (8 GCDs), lessons 11 + 13 | 32,651 | **35,815** (+9.7 %) |

(120b's "before" is lesson 10's build measured again on the same day: 32,651 instead of 32,485.)

## 7. Decisions and reasons

- **Decided:** speed up the small kernels next, starting with the top-4.
  **Why:** together ~10 % of the run; the top-4 was the largest (2.3 %) and for 120b costs
  7.0 ms of a ~125 ms step.
- **Decided:** one wavefront per token, same tie rule and same softmax code.
  **Why:** 512 tokens give 512 wavefronts (128 blocks) instead of 8 blocks; and identical bits
  mean the fingerprint alone proves the change correct.
- **Kept:** the old kernel for more than 256 experts (each lane holds at most 4 experts).
- **Tried, did not help (same profile):** keeping the whole token loop on the GPU (next tokens,
  outputs and end flags in GPU memory; the CPU queues steps ahead). The profile showed the GPU
  idle 7.5 % of the time, which looked like the CPU making the GPU wait between steps. It was
  not: almost all of that idle time was during warm-up's weight uploads, before the first step.
  During answering the GPU was busy 99.6 % of the time. The change measured +0.2 % (8,710 →
  8,724 tok/s, 512 requests on one GCD) and was reverted to keep the code simple.

## 8. Questions to think about

1. The old kernel used 8 blocks for 512 tokens. Roughly how many of the GPU's 104 compute units
   did it keep busy, and why does that matter more than how fast each thread is?
2. In the butterfly, why must the comparison include "same score and lower expert number"?
   What would go wrong, and how would you notice, if it only compared scores?
3. The GPU-side token loop looked like a good idea from the profile. What would you check in a
   profile before trusting an "idle time" number?

## 9. Discussion notes

(empty: we write here during the walk-through)
