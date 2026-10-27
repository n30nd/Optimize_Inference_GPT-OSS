# Lesson 09: Sort tokens by expert with many workers instead of one

| | |
|---|---|
| Reference commit | `29f59e2` "Group MoE rows with one block per expert instead of one thread" |
| Starts from | `30dd744` (lesson 08) |
| To type | 88 lines of program code in 2 files (+88 −46) |
| Speed after | 20b 7,858 tok/s together with lesson 08 (was 5,741); measured as one step |

**In plain words:** before the experts can work, every token's 4 choices must be sorted into
"expert 0's list, expert 1's list, …". The first version gave each expert a single worker that
read the whole list of choices twice. Now each expert gets 256 workers who read the list
together and agree on where each entry goes. Same sorted result, 22 times faster.

## 1. Goal

Make the grouping step (which rows go to which expert) take microseconds, without changing the
order of the rows — so the answers stay bit for bit the same.

## 2. Why this, why now

- The profile of lesson 04 timed the grouping kernel at **318 µs per layer** at 512 requests: one
  block of 64 threads, each thread (one per expert) scanning all 2,048 (token, choice) pairs twice.
  That is **7.6 ms per 20b step** (24 layers).
- 7.6 ms was nothing next to the 670 ms step of lesson 04. But with the dense multiplies (lesson
  08), attention (lessons 06–07) and soon the expert multiply (lesson 10) fast, a step heads for
  tens of ms — and then 7.6 ms is 10–20 % of it.
- It is a small, self-contained kernel: an easy fix while the big one is still in progress.

## 3. The technique

New words:
- **(token, choice) pair** — one of a token's 4 expert choices; pair number `p = token × 4 + choice`.
  512 tokens → 2,048 pairs.
- **grouping** — reordering the pairs so each expert's rows are one contiguous block: rows
  `offsets[e] .. offsets[e+1]` belong to expert e (lesson 03 explains why the experts want this).
- **prefix sum** — "where does expert e start?" = the sum of the counts of experts 0 .. e−1.
- **shuffle reduction** — the 64 lanes of a wave add up their numbers by passing them between
  lanes (`__shfl_down`), without LDS.
- **ballot** — `__ballot(cond)` gives every lane a 64-bit mask: bit i is set if lane i's
  condition is true. **popcount** (`__popcll`) counts the set bits.

```
before: one block, one thread per expert (20b: 32 experts → 64 threads)
  thread e:  for p in 0..2047: count pairs that chose e        ← every thread reads all 2,048
             (barrier, prefix sum by thread 0)
             for p in 0..2047: if pair p chose e → next row      ← and again

after: kernel 1 (count): one 256-thread block per expert
  block e:   threads take p = tid, tid+256, …   → count, add up in the wave, then the 4 waves
  kernel 2 (place): one 256-thread block per expert
  block e:   start = counts[0] + … + counts[e−1]
             for each chunk of 256 pairs:
                hit  = (pair p chose e)
                mask = __ballot(hit)                       one bit per lane of the wave
                row  = start + (hits in earlier waves) + popcount(mask & lanes below me)
             → the rows come out in pair order, exactly as before
```

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe_ep.hpp` | +81 −42 | `moe_group_ep_kernel` replaced by `moe_count_ep_kernel` (counts per local expert) and `moe_place_ep_kernel` (start of each expert, marks pairs of the partner's experts as −1, then `moe_place_pairs` for own pairs and partner pairs) | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +7 −4 | a `counts` buffer (one int per local expert); `layer_experts` launches the two kernels with one block per local expert | yes |

The same kernels serve 20b (no partner: 32 experts) and 120b (each GCD's 64 experts, rows from
both GCDs of the pair: own pairs first, then the partner's).

Read while typing (line numbers at `29f59e2`):
- `getp-csrc/kernels/moe_ep.hpp:27` `moe_count_ep_kernel` — the wave sum at `:38` (`__shfl_down`).
- `getp-csrc/kernels/moe_ep.hpp:52` `moe_place_pairs()` — `:60` the ballot, `:68` the row of each
  hit: `before + __popcll(mask & ((1ull << lane) - 1))`.
- `getp-csrc/kernels/moe_ep.hpp:82` `moe_place_ep_kernel` — the start of each expert.
- `getp-csrc/engine/gpu_engine.hpp:390-392` — the two launches, `<<<w->e_count, MOE_GROUP_THREADS>>>`.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show 29f59e2                                                  # short enough to read whole
git diff 29f59e2 -- $(git show --name-only --format= 29f59e2)    # must print nothing
make runomp
```

Quick check, 20b (one GCD):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # b2d8ec70d2ff, the same as lesson 08
GPTOSS_EXACT=1 HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64x.txt -n 128
sha256sum $W/out64x.txt | cut -c1-12     # ad7fdd092a55, the exact path of lesson 03
```

The same fingerprints as before this lesson: the rows land in exactly the same order, so nothing
downstream can change.

Quick check, 120b (one pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 5d44468050b5
```

Bit-exact test with a partner (the grouping with own and partner rows):

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
EP=2 tools/gpulock.sh 2,3 -- ./gpu_vs_cpu $M20 4 16
```

Expected last lines (about 25 s):

```
pos  15: logits differing 0 / 804352, max |diff| 0, argmax differing 0
PASS: GPU logits bit-identical to the CPU
```

Full run (20b, 8 GCDs; this is the run that measures lessons 08 + 09, about 10 minutes):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-09 --score
```

Expect about **7,858 tok/s**, METEOR **0.557**, BERTScore **0.980**, fingerprint `a5315bcc37ed`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Group MoE rows with one block per expert instead of one thread"`

## 6. Result

Grouping, one GCD, 512 requests, per layer:

| | before | after |
|---|---:|---:|
| grouping | 318 µs (one 64-thread block) | **14.4 µs** (count 5.1 + place 9.3) |
| per 20b step (24 layers) | 7.6 ms | 0.35 ms |
| output | — | same bits (fast and exact fingerprints unchanged; EP=2 still bit-identical to the CPU) |

End to end (20b, 8 GCDs, 4,096 prompts, **lessons 08 and 09 together**):

| | before | after |
|---|---:|---:|
| tok/s | 5,741 | **7,858** (+37 %) |
| METEOR / BERTScore | 0.549 / 0.979 | 0.557 / 0.980 |

After lessons 08 and 09, one step at 512 requests on one GCD takes 477 ms, of which the simple
expert multiply is **95.6 %** (9.5 ms per call). That is the next lesson.

## 7. Decisions and reasons

- **Decided:** parallelize the grouping now, not after the big multiplies.
  **Why:** 318 µs per layer on a single 64-thread block = 7.6 ms per 20b step. Once the
  multiplies are fast, the step is tens of ms and 7.6 ms is 10–20 % of it; the fix is small.
  **Not chosen:** leaving it until after the GEMMs.
- **Decided:** keep the exact row order of the old kernel (pair order within an expert, own pairs
  before the partner's). **Why:** then the expert multiply sees the same rows in the same places
  and every sum stays the same — the fingerprints prove the change in seconds, no new scores
  needed.
- **Decided:** two kernels (count, then place) with one block per expert. **Why:** the place step
  needs every expert's count to know where it starts; a kernel boundary is the simplest
  "everyone has finished counting". (Lesson 25 merges the two into one kernel.)

## 8. Questions to think about

1. Why does `popcount(mask & lanes below me)` give each hit a distinct row, in pair order, without
   any lane waiting for another?
2. Each block of `moe_place_ep_kernel` adds up the counts of all lower experts by itself. For 32 or
   64 experts that is cheap; when would it not be?
3. The old kernel was slow because each thread read all 2,048 pairs. The new one also reads all
   pairs once per expert block. Why is it still 22× faster?

## 9. Discussion notes

(empty: we write here during the walk-through)
