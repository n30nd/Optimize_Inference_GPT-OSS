# Lesson 07: Store the attention memory in 64-slot blocks

| | |
|---|---|
| Reference commit | `c61cc5d` "Store the fast-path KV cache in 64-slot blocks per key/value head" |
| Starts from | `03e8e9d` (lesson 06) |
| To type | 22 lines of program code in 3 files (+22 −18) |
| Shortcut | `git cherry-pick c61cc5d` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 5,741 tok/s (was 5,697); 120b 2,835 tok/s on the fast path (was 1,188 on the exact path) |

**In plain words:** the stored history was kept in an order where each reader needed a small
slice from every row, so it jumped around memory. We change only *where* each number is stored,
so that each reader's numbers sit together in one block. The arithmetic is untouched. This is
also the first full run of 120b on the fast path.

## 1. Goal

Let the attention kernel of lesson 06 read whole contiguous blocks, so it gets closer to the
memory's full speed.

## 2. Why this, why now

- Lesson 06 measured the new kernel's loads with all the arithmetic removed: on the old layout
  they reach only **~870 GB/s**, while a plain front-to-back read of the same bytes reaches
  **1,312 GB/s**. The kernel itself got 861 GB/s: it was at the ceiling of the layout.
- Lesson 06 already added and tested the new layout (`kv_index_chunk64`,
  `attention_bf16_chunk64`) and measured it: full layer 1,246 → 945 µs, window layer 151 →
  122 µs. What was left was to make the engine write and read it — two lines, plus rounding the
  slot count up.
- We kept it a separate step on purpose: it changes the order of some sums, so it gets its own
  scores.

## 3. The technique

New words:
- **layout** — the rule that says at which address each stored number lives.
- **slot** — the place of one token in the KV cache (the stored keys and values).
- **memory channel / memory page** — GPU memory is split into many channels that work at the
  same time; inside a channel one *page* (about 1–2 KB) is open at a time. Reading more of an
  open page is cheap; jumping between pages costs time.

```
old layout ("row"): one 1 KB row per slot, the 8 key/value heads side by side
 slot 0  │ g0 │ g1 │ g2 │ g3 │ g4 │ g5 │ g6 │ g7 │   each piece 128 bytes
 slot 1  │ g0 │ g1 │ g2 │ g3 │ g4 │ g5 │ g6 │ g7 │
 …         ↑ the waves of head g0 read 128 bytes out of every 1 KB row

new layout ("chunk64"): slots cut into chunks of 64; inside a chunk, each head's 64 slots together
 chunk 0 │ g0: slots 0–63 (8 KB) │ g1: slots 0–63 (8 KB) │ … │ g7: slots 0–63 (8 KB) │
 chunk 1 │ g0: slots 64–127      │ g1: slots 64–127      │ … │ g7: slots 64–127      │
           ↑ the waves of head g0 read one contiguous 8 KB block per chunk

 element (b, slot, g, i) at  b·slots64·512 + (slot/64)·64·512 + g·64·64 + (slot%64)·64 + i
 slots64 = slots rounded up to a multiple of 64
```

The chunk64 entry point also picks the work split by history length: up to 128 positions,
4 splits of 32-position steps (as in lesson 06); beyond, 8 splits of 16-position steps.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_misc.hpp` | +6 −5 | `rope_kv_bf16_kernel` stores each key and value at `kv_index_chunk64(…)` instead of the row address; includes `fast_attention.hpp` | yes |
| 2 | `getp-csrc/kernels/fast_attention.hpp` | +3 −5 | only the comment above `kv_index_chunk64`: the layout is now the one in use | yes |
| 3 | `getp-csrc/engine/gpu_engine.hpp` | +13 −8 | fast-path slot counts rounded up to 64 (`gpu_state_bytes`, `gpu_alloc_state`); `layer_attention` calls `attention_bf16_chunk64` | yes |

Read while typing (line numbers at `c61cc5d`):
- `getp-csrc/kernels/fast_attention.hpp:426` `kv_index_chunk64()` — the formula above.
- `getp-csrc/kernels/fast_attention.hpp:442` `if (n <= 128)` — 4 or 8 history splits.
- `getp-csrc/kernels/fast_misc.hpp:59-64` — the three stores that now use the new address.
- `getp-csrc/engine/gpu_engine.hpp:213` — `g.slots = (g.slots + 63) / 64 * 64;`

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick c61cc5d
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout c61cc5d -- $(git show --name-only --format= c61cc5d)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show c61cc5d                                                  # the whole change is short
git diff c61cc5d -- $(git show --name-only --format= c61cc5d)    # must print nothing
make runomp
```

Quick check, 20b (one GCD):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # 0635c70cdb17, the same as lesson 06
```

Same fingerprint as lesson 06: with at most 128 positions the kernel still uses 4 splits, so
every sum is added in the same order. The full run (1,024 positions) uses 8 splits past 128
positions and gives a different fingerprint.

Quick check, 120b (one pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 59aa33af7244
```

Kernel test and benchmark: `attn_test` from lesson 06 already checks the chunk64 kernel (the
lines marked `chunk`), and `attn_bench variants` times it next to the row layout (build both as in
lesson 06):

```bash
tools/gpulock.sh 2 -- ./attn_test             # ends: largest fp32-output error: 4.19e-05 / ALL PASS
tools/gpulock.sh 2 -- ./attn_bench variants   # every variant tried, next to before / after
```

The lines at 512 requests, full layer, position 1023, from one run:

```
before (simple)                full     512  1023  1024     9529.9      113
after (attention_bf16)         full     512  1023  1024     1243.0      864
proposed: chunk64 layout       full     512  1023  1024      939.4     1143
```

(At this commit the benchmark still calls chunk64 "proposed": its labels were written in lesson 06.)

Full runs (8 GCDs; 20b about 15 minutes, 120b about 25 minutes):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20  -i evaluation/input.txt -g all -t lesson-07 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-07 --score
```

Expect 20b about **5,741 tok/s**, METEOR **0.549**, BERTScore **0.979**, fingerprint
`6a44416e2c28`; 120b about **2,835 tok/s**, METEOR **0.568**, BERTScore **0.982**, fingerprint
`9cc3321a9c5d`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Store the fast-path KV cache in 64-slot blocks per key/value head"`

## 6. Result

Attention kernel, one GCD, 512 requests:

| Layer | before (row layout) | after (chunk64) |
|---|---:|---:|
| full, position 1023 | 1,246 µs (861 GB/s) | **945 µs** (1,136 GB/s) |
| window | 151 µs | **122 µs** |

End to end (8 GCDs, 4,096 prompts):

| | before | after |
|---|---:|---:|
| 20b tok/s | 5,697 | **5,741** (+0.8 %) |
| 20b METEOR / BERTScore | 0.548 / 0.979 | 0.549 / 0.979 |
| 120b tok/s | 1,188 (exact path, lesson 05) | **2,835** (fast path, ×2.4) |
| 120b METEOR / BERTScore | 0.988 / 0.99987 | 0.568 / 0.982 |

The 120b jump is not the layout alone: it is the first 120b run of the fast path (bf16 matrix
inputs and bf16 cache from lesson 04, the attention of lesson 06, this layout). The bf16 cache is
half the size, so all 512 requests of a GCD fit at once — one wave instead of two. The expert and
dense multiplies are still the simple kernels.

Why +0.8 % only: the attention saving is ~0.3 ms per full layer near the end of answers, while a
whole step is still hundreds of ms of simple multiplies (lessons 08–10).

## 7. Decisions and reasons

- **Decided:** adopt the chunk64 layout as its own step, after the new kernel.
  **Why:** the loads-only test showed the row layout capped the kernel at ~870 GB/s; chunk64
  measured 1,246 → 945 µs on a full layer and 151 → 122 µs on a window layer, at every history
  length 1,100–1,200 GB/s.
  **Not chosen:**
  - a plain head-major layout (each head's whole history in one long block): 3× slower at short
    histories, because blocks spaced 128 KB apart pile onto the same memory channels;
  - padding each row to 1,152 bytes: small, inconsistent gains.
- **Decided:** score this step on its own. **Why:** past 128 positions the new entry point splits
  the history 8 ways instead of 4, so its sums are added in another order. 64 prompts × 128
  positions give the same fingerprint with and without chunk64, but the full run's fingerprint
  differs, so only METEOR/BERTScore can accept it.
- **Decided:** 120b moves to the fast path (bf16 cache, 512 requests per GCD).
  **Why:** the bf16 cache halves the memory (21.8 GB for 512 requests), so one wave instead of
  two; the scores stay far above the gates (0.568 ≥ 0.30, 0.982 ≥ 0.90).

## 8. Questions to think about

1. The arithmetic did not change, yet the kernel got 24 % faster. What exactly did the memory
   do differently?
2. Why round each sequence's slot count up to a multiple of 64? What does it cost?
3. The quick check gives the same fingerprint, the full run does not. Is the change "correct"?
   What would you check to be sure?

## 9. Discussion notes

(empty: we write here during the walk-through)
