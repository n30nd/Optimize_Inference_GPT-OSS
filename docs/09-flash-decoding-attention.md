# 09 — Decode attention: grouped-query sharing, FlashAttention and flash-decoding

## The problem in plain words

At every step each request looks back over everything it has written so far. In every layer, each
of its 64 *query heads* compares the newest token with every earlier token, turns the comparison
scores into weights, and mixes what it stored about those tokens using the weights. What is stored
(the *KV cache*) grows by one entry per token, so this work grows with every generated token, and
95 % of our answers run all the way to 1,024 tokens.

The work is mostly *reading*: at position 1,023, one full layer for 512 requests must read
**1.07 GB** of cache. At the 1.25 TB/s our GCD really delivers, that alone takes **0.86 ms**, and it
cannot go faster. The first GPU version read every cached number **8 times** (once per query head
that uses it) and did its arithmetic one number at a time, so it was far slower than that floor.

This document explains how the new kernel reads each cached number once, keeps the arithmetic off
the critical path, and gets as close to the floor as the cache's layout allows: **7.6× faster** on a
full layer at position 1,023 (9.5 ms → 1.25 ms), **9.3×** on a window layer.

## An everyday comparison

Eight students must each write a summary of the same shelf of 1,024 books.

- **Before:** each student walks to the archive, fetches every book, reads it, and brings it back:
  eight trips for the same books.
- **Grouped-query sharing:** one helper fetches each book once and reads it aloud to all eight
  students at the same table.
- **Online softmax (FlashAttention):** nobody keeps a list of all 1,024 opinions; each student keeps
  a running "best so far" and a running total, and corrects the total if a better book turns up.
- **Flash-decoding:** the shelf is long, so four helpers each take every fourth part of it; at the
  end they compare notes and combine their totals.

## The words

- **token** — a word piece; one is produced per request per step.
- **query, key, value** — three lists of 64 numbers each, computed for every token. The newest
  token's *query* asks "what am I looking for?"; every earlier token's *key* says "what do I
  contain?", and its *value* is what it hands over if chosen.
- **head** — one independent copy of this look-back; gpt-oss has 64 query heads per layer.
- **score** — how well the query matches one key: the sum of the 64 pairwise products, divided by 8.
- **softmax** — turns scores into weights that are positive and add up to 1:
  `weight = exp(score − max) / sum of all exp(score − max)`. Subtracting the largest score (*max*)
  first keeps the numbers from overflowing; it does not change the result.
- **attention sink** — one extra learned score per head that joins the softmax but has no value: it
  lets a head give part of its weight to "nothing" (doc 01).
- **sliding window** — even-numbered layers only look at the last 128 tokens (doc 01).
- **grouped-query attention (GQA)** — 8 query heads share one key/value head, so there are only 8
  key/value heads per layer instead of 64 (doc 01).
- **KV cache** — the stored keys and values of all earlier tokens, one *slot* per token, as bf16.
- **bf16 / fp32 / fp64** — 16-, 32- and 64-bit floating-point numbers. bf16 keeps only about 3
  significant decimal digits (8 bits of precision), fp32 about 7 (24 bits), fp64 about 16.
- **memory-bound** — the time is set by how many bytes must be fetched, not by the arithmetic.
- **bandwidth (GB/s)** — bytes fetched per second. Our yardstick is 1.25 TB/s = 1,250 GB/s per GCD.
- **workgroup** — a group of threads that run on the same compute unit and can share LDS (doc 02
  calls it a *block*).
- **wavefront (wave), lane** — 64 threads that execute the same instruction together; each of the
  64 is a *lane*.
- **LDS (local data share)** — the small, fast scratch memory inside each compute unit (64 KB),
  shared by a workgroup.
- **register** — a lane's own private storage, the fastest there is (a few hundred numbers per lane).
- **occupancy** — how many waves fit on one SIMD (one of the 4 calculator groups in a compute unit)
  at once. While one wave waits for memory, another can run, so more waves hide waiting better.
- **MFMA (matrix fused multiply-add)** — one instruction that multiplies two small matrices on the
  matrix cores. The one we use, `v_mfma_f32_16x16x16bf16_1k`, multiplies a 16×16 by a 16×16 bf16
  matrix and adds the 16×16 result into fp32: 4,096 multiply-adds in one instruction.
- **tile** — a small block of the data (here 16 positions) handled as one unit.
- **online softmax** — computing a softmax in one pass, keeping only a running max and a running sum.
- **flash-decoding** — splitting one request's history across several workers and merging their
  partial results.
- **buffer load** — a GPU load instruction that takes a base address and a size; any byte past the
  size reads as 0 instead of faulting.
- **memory channel, memory page** — the GPU's memory is split into many independent *channels*, and
  consecutive addresses are spread over them so all work at once. Inside a channel, the chip opens one
  *page* (a row of about 1–2 KB) at a time: reading more of an open page is cheap, switching pages
  costs time. Reads that jump around, or that all land on a few channels, get less than the full speed.
- **vector instruction** — an ordinary (non-matrix) instruction: one operation per lane.
- **L2 cache** — an 8 MB on-chip copy of recently read memory, shared by the whole GCD.

## How it works, step by step

### 1. What one head computes

For request b, query head h (0..63), at position `pos`:

```
 positions t = t0 .. pos           t0 = 0 on full layers, max(0, pos − 127) on window layers
 score_t   = (q · k_t) / 8          q: 64 fp32 numbers; k_t: 64 bf16 numbers from the cache
 M         = max(sink_h, all score_t)
 weight_t  = exp(score_t − M) / L   L = exp(sink_h − M) + Σ_t exp(score_t − M)
 out       = Σ_t weight_t · v_t     64 numbers, stored as bf16 for the next matrix multiply
```

The cache stores slot `t` (full layers) or `t % 128` (window layers) as
`kc[(b · slots + slot) · 512 + g · 64 + i]`, g = h / 8 the key/value head. A window layer uses all
128 slots once pos ≥ 127, and slots 0..pos before that, so **both layer kinds simply walk slots
0 .. n−1** with n = min(pos + 1, 128) or pos + 1; the order does not matter for a sum.

### 2. Where the time went before

```
 before: one 64-thread workgroup per (query head h, request b)      → 64 × 512 = 32,768 workgroups
   pass 1: thread j computes score_j alone: 64 separate 2-byte loads of k_j, 64 multiply-adds
   pass 2: max and sum over all scores (in LDS)
   pass 3: thread i (dim i) walks all positions: out_i += weight_t · v_t[i]

 the 8 heads of one group (h = 8g .. 8g+7) each read the SAME keys and values  → 8× the bytes
```

At pos 1,023 a full layer with 512 requests read 8 × 1.07 GB = 8.6 GB. In practice it was even
slower than that suggests, because each thread fetched its key 2 bytes at a time (the number in the
table below).

### 3. Grouped-query sharing: one set of waves per key/value head

```
 after: the waves of one (key/value head g, request b) pair     →  8 × 512 = 4,096 pairs
        compute all 8 query heads of group g together           (2 pairs per workgroup, step 5)
        each key and value is fetched from memory ONCE and used by 8 heads
```

The bytes drop 8×, to the 1.07 GB floor. But now every fetched key feeds 8 dot products and every
fetched value feeds 8 weighted sums: 2 × 8 × 64 = 1,024 multiply-adds per position. Done one number
at a time on the ordinary calculators, that is about half of all the arithmetic a GCD can do while
reading at full speed. We measured this "plain" version too (table below): it is much better than
before but stays well short of the floor, so the arithmetic moves to the matrix cores (step 6).

### 4. FlashAttention's online softmax: one pass, no list of scores

The softmax needs the max M before any weight is known. The old kernel therefore computed all scores
first, kept them in LDS, and only then summed values: two passes and a list as long as the history.

*Online softmax* keeps three running things per head and fixes them up as it goes:

```
 m   = largest score seen so far           (starts at the sink score)
 l   = Σ exp(score − m) so far              (the denominator, still missing the sink)
 acc = Σ exp(score − m) · v so far          (64 numbers)

 for each tile of positions:
     s       = scores of the tile
     m_new   = max(m, max(s))
     α       = exp(m − m_new)               old sums were measured against m: shrink them
     l       = α · l + Σ exp(s − m_new)
     acc     = α · acc + Σ exp(s − m_new) · v
     m       = m_new
 end:  out = acc / (l + exp(sink − m))
```

FlashAttention (Dao et al., 2022) uses this to process attention in *tiles* that fit in fast memory:
scores of one tile are made, used and forgotten. Our tile is 16 positions (the MFMA size); each wave
handles 2 tiles (32 positions) per loop turn.

**Lazy rescale.** Multiplying `acc` by α every turn costs instructions even when α = 1, which is
nearly always after the first few tiles. The kernel only raises m when a score exceeds it by more
than 8 (in base-2 units, a factor 256). Until then the weights exp2(s − m) may grow up to 256, which
fp32 handles without any loss, and since the final division uses the same m, the result is the
same. The check is one comparison per wave per turn (`__any(mx > m + 8)`).

### 5. Flash-decoding: split the history, then merge

One request at position 1,023 has 1,024 positions per key/value head. Handing all of them to one
wave would make each wave stream 256 KB while the GPU has room for only a limited number of waves:
the last waves to start would finish long after the others. Flash-decoding (Dao et al., 2023) cuts
the history into pieces handled in parallel and merges the partial results:

```
 4 waves per (request b, key/value head g); positions in 32-position steps, dealt out round-robin:

   step:    0    1    2    3    4    5    6    7   ...  31
   wave:    0    1    2    3    0    1    2    3   ...   3

 each wave w ends with its own (m_w, l_w, acc_w) per head, written to LDS; then
   M   = max(sink, m_0, m_1, m_2, m_3)
   L   = exp(sink − M) + Σ_w l_w · exp(m_w − M)
   out = Σ_w acc_w · exp(m_w − M) / L
```

A wave that got no positions (short histories) has m_w = sink, l_w = 0, acc_w = 0 and drops out of
the sums by itself. The sink is added exactly once, in the merge.

Flash-decoding is usually used *across* workgroups, with a second kernel to merge, because at small
batch sizes there are too few (request, head) pairs to fill the GPU. With 512 requests × 8 heads =
4,096 (request, head) pairs we have plenty, so the split stays *inside* the workgroup and the merge
is a few LDS reads at the end of the same kernel.

**How many splits, and how many heads per workgroup.** We measured every combination (table below).
No split (1 wave per head) is 12 % slower on full layers and 35 % slower on window layers: too few
waves are waiting on memory at once. 16 splits is slower again: most waves get a single step and
the merge costs more than it saves. The winner puts **2 neighbouring key/value heads × 4 splits**
(8 waves) in one workgroup. The two heads' 128-byte slices sit side by side in every cache row,
so the workgroup reads 256 contiguous bytes per row instead of 128. Packing all 8 heads into one
workgroup (which would read whole 1 KB rows) is slower, most likely because only 2 such
workgroups fit on a compute unit at once: 512 of them fill the GPU in about 2.5 rounds, and during
the last half round half of it idles.

### 6. Matrix cores for the two small multiplies

Per 16-position tile and key/value head, the two multiplies are small matrix products:

```
 scores (transposed):  S^T [16 positions × 16 columns] = K [16 positions × 64 dims] · Q^T [64 dims × 16 columns]
                       → 4 MFMAs (64 dims = 4 × 16)
 output:               O [16 rows × 64 dims] += P [16 rows × 16 positions] · V [16 positions × 64 dims]
                       → 4 MFMAs (64 dims = 4 slices of 16)
```

Computing S^T rather than S is deliberate: the MFMA leaves each lane holding the scores for one
column (head) and 4 positions, which is exactly how the second MFMA wants P handed to it. The
weights never leave the registers.

**The hi/lo trick: 16 columns, only 8 heads.** The MFMA always has 16 columns but a group has only
8 heads. Rather than waste half, column c < 8 carries head c's numbers rounded to bf16 (the *hi*
part), and column c + 8 carries the rounding error, also as bf16 (the *lo* part):
`x ≈ hi + lo` with error ≤ 2^-18 · |x|, which is close to fp32. After the score MFMA,
the two columns are added (one cross-lane add, `row_swap8`). The same trick is applied to the
weights P in the rows of the second MFMA, and the lo rows are added to the hi rows once, at the end.
So q and the weights keep ~17 bits of precision instead of bf16's 8, at no extra MFMA cost. The
cache itself is already bf16, so the kernel's remaining error is a few parts in 100,000 of the
output, about 100× smaller than the bf16 rounding of the output itself (unit test below).

Per 32-position step a wave issues 16 MFMAs (~512 cycles of matrix-core time) for 8 KB of cache.
Reading 8 KB at full speed takes ~4,600 SIMD cycles (1.25 TB/s spread over 416 SIMDs), so the matrix
cores are busy about 11 % of the time: the arithmetic is now far from being the limit.

### 7. How the bytes are fetched

```
 one cache row (one slot, request b) = 8 key/value heads × 64 dims × 2 bytes = 1,024 bytes
 ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
 │ g = 0 │ g = 1 │ g = 2 │ g = 3 │ g = 4 │ g = 5 │ g = 6 │ g = 7 │  each 128 bytes
 └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
 the waves of head g read its 128-byte piece of every row; one workgroup holds 2 neighbouring heads
 (256 bytes per row), and the 4 workgroups of request b are next to each other in the launch order,
 so they run at about the same time.

 keys, one 16-position tile:  lane l → row (l % 16), bytes 16·(l/16) .. +16 and 64 + 16·(l/16) .. +16
                              2 load instructions, each reads 16 rows × 64 bytes, every byte used
 values, one tile:            lane l → rows 4·(l/16) + r (r = 0..3), bytes 8·(l%16) .. +8
                              4 load instructions, each reads 4 whole 128-byte rows
```

The MFMA wants each lane's values arranged by position (4 positions of one dim), but memory has
them by dim. Each lane loads 4 dims × 4 positions and swaps them into place with byte-shuffle
instructions (`v_perm`, through `pack_lo` / `pack_hi`).

All loads are *buffer loads* whose range ends after slot n − 1: the addresses are 32-bit offsets
(fewer instructions than 64-bit address arithmetic), and the half-used last tile reads zeros instead
of uninitialised memory, which could otherwise hold NaN (not-a-number), and NaN × 0 is still NaN.

**The layout is now the limit.** We ran the same loads with all the arithmetic removed: on a full
layer they reach only **870 GB/s**, while a plain front-to-back read of the same bytes reaches
1,312 GB/s. The kernel gets 861 GB/s, so it is at the ceiling of this layout. Reading 128 (or 256)
bytes out of every 1 KB row scatters each workgroup's requests over many memory pages. If instead
each key/value head's slots are kept together in blocks of 64 slots (8 KB), the same kernel reaches
1,100–1,200 GB/s (proposal at the end).

## The numbers for our model

Measured with `getp-csrc/bench/attn_bench.cpp` on one GCD of mv-mi250-02 (mean of 30 calls after 5
warm-up calls, 2026-10-25). GB/s counts only the cache bytes a call must read,
B × n × 8 heads × 64 × 2 bytes × 2 (key and value); n = positions looked at.
For reference, a plain front-to-back read of the same bytes reaches **1,312 GB/s** on this GCD.

**512 requests per GCD**

| Layer | pos | n | Before | GB/s | After | GB/s | Faster | Proposed layout (chunk64) | GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| window | 64 | 65 | 787 µs | 87 | 114 µs | 596 | 6.9× | 92 µs | 742 |
| window | 256 | 128 | 1,404 µs | 96 | 151 µs | 887 | 9.3× | 121 µs | 1,110 |
| window | 512 | 128 | 1,405 µs | 96 | 151 µs | 889 | 9.3× | 122 µs | 1,100 |
| window | 1023 | 128 | 1,406 µs | 95 | 151 µs | 887 | 9.3× | 122 µs | 1,101 |
| full | 64 | 65 | 709 µs | 96 | 148 µs | 460 | 4.8× | 104 µs | 656 |
| full | 256 | 257 | 2,452 µs | 110 | 340 µs | 792 | 7.2× | 243 µs | 1,109 |
| full | 512 | 513 | 4,821 µs | 112 | 639 µs | 842 | 7.5× | 465 µs | 1,157 |
| full | 1023 | 1024 | 9,468 µs | 113 | **1,246 µs** | 861 | **7.6×** | 945 µs | 1,136 |

**1,024 requests per GCD**

| Layer | pos | n | Before | GB/s | After | GB/s | Faster | Proposed layout (chunk64) | GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| window | 64 | 65 | 1,575 µs | 87 | 219 µs | 622 | 7.2× | 177 µs | 771 |
| window | 256 | 128 | 2,760 µs | 97 | 290 µs | 927 | 9.5× | 235 µs | 1,144 |
| window | 512 | 128 | 2,760 µs | 97 | 289 µs | 929 | 9.6× | 235 µs | 1,144 |
| window | 1023 | 128 | 2,749 µs | 98 | 289 µs | 929 | 9.5× | 234 µs | 1,149 |
| full | 64 | 65 | 1,405 µs | 97 | 286 µs | 477 | 4.9× | 195 µs | 698 |
| full | 256 | 257 | 4,733 µs | 114 | 662 µs | 814 | 7.1× | 471 µs | 1,144 |
| full | 512 | 513 | 9,287 µs | 116 | 1,253 µs | 859 | 7.4× | 895 µs | 1,202 |
| full | 1023 | 1024 | 17,952 µs | 120 | **2,446 µs** | 878 | **7.3×** | 1,795 µs | 1,196 |

- At pos 64 the GB/s are lower because only 68 MB of cache are read, while the queries (8 MB), the
  output (4 MB) and starting the 4,096 workgroups cost about the same as at any position.
- Over one whole answer (positions 0 to 1,023; 12 full + 12 window layers), attention costs on average
  about **74 ms per step before and 10 ms after** at 512 requests, 130 ms → 17 ms at the last step.
- Yardstick: PyTorch's `scaled_dot_product_attention` (`getp-csrc/bench/attn_torch_yardstick.py`) has
  no grouped-query decode kernel for this GPU: 9,908 µs on the full layer at pos 1023 (108 GB/s),
  even with the friendlier head-major layout and no sink. The plain math version takes 60,669 µs.

**Every idea we measured** (512 requests, µs per call; from four benchmark runs on the same kind of
GCD, so differences below ~3 % are noise):

| Idea | Full layer, pos 1023 | Window layer | Kept? | Why |
|---|---:|---:|---|---|
| Before: one workgroup per query head | 9,468 | 1,406 | — | reads every key/value 8× |
| Grouped-query sharing with plain vector math, exact two-pass softmax | 3,825 | 284 | no | 1,024 multiply-adds per position at one number per lane: arithmetic-bound (281 GB/s) |
| + matrix cores, online softmax, 4 splits (first version) | 1,321 | 170 | yes | arithmetic leaves the critical path |
| + buffer loads, lazy rescale, masking only in the last step | 1,303 | 166 | yes (−1.4 %) | 257 → 207 vector instructions per step; small gain, so the arithmetic was no longer the limit |
| hi/lo parts *truncated* instead of rounded | not timed | | no | errors up to 1e-4; rounding to nearest: 3.6e-5 |
| no split (1 wave per head) | 1,391 | 203 | no | too little in flight per head |
| 16 splits | 1,298 | 244 | no | most waves get one step; merge overhead |
| 64 positions per wave step instead of 32 | 1,283 | 167 | no | no gain, more registers (3 instead of 4 waves per SIMD) |
| 8 key/value heads per workgroup (reads whole 1 KB rows) | 1,282 | 206 | no | 512 large workgroups: poor tail, fewer waves in flight |
| **2 key/value heads × 4 splits per workgroup (final)** | **1,245** | **151** | **yes** | 256 contiguous bytes per row; same arithmetic per head, so bit-identical results |
| The same loads with no arithmetic, current layout | 1,236 | 128 | — | ceiling of the current layout |
| Head-major layout (b, g, slot, 64) | 1,023 | 122 | no | full layers at short histories 437 µs (3× slower): 128 KB-spaced blocks pile onto the same memory channels |
| Padded rows (1,152 bytes per slot) | 1,256 | 137 | no | small, inconsistent gain (worse with 2 heads per workgroup) |
| **chunk64 layout (proposal)** | **945** | **122** | proposed | contiguous 8 KB blocks at every history length: 1,100–1,200 GB/s |
| Plain read of the same bytes | 818 | | — | memory ceiling |

## Where it is in our code

All in `getp-csrc/kernels/fast_attention.hpp` unless noted:

- `:1-43` — the design in short (work split, matrix-core layout, memory access).
- `:74` `bf16_piece()` — the hi/lo split, rounded to nearest; `:96` `row_swap8()` adds the two columns.
- `:91` `make_rsrc()` — buffer descriptor whose range ends at slot n − 1.
- `:113` `KVLayout` — where a slot lives; the engine's layout is `{slots·512, 512, 64, 30, 0}`.
- `:124` `attention_gqa8_kernel` — the kernel. `:147` queries as the matrix-core operand; `:179` the
  loop over this wave's steps; `:189` key loads; `:209` scores (hi + lo); `:211` masking of the last
  step; `:233` lazy rescale; `:257` weights as the operand of the second multiply; `:273` per-wave
  sums; `:291` merge of the splits and the sink.
- `:395` `attention_bf16()` — the entry point the engine calls; `:409` picks 2 heads × 4 splits.
- `:332` `attention_bf16_simple_kernel` — the previous kernel, kept as the fallback for other head
  counts (`-DGETP_ATTN_SIMPLE` forces it, to build a "before" binary).
- `:430` `kv_index_chunk64()`, `:438` `attention_bf16_chunk64()` — the proposed layout, ready to use.
- Called from `getp-csrc/engine/gpu_engine.hpp:323` (`gpu_step_fast()`); the cache is written by
  `rope_kv_bf16_kernel` (`getp-csrc/kernels/fast_misc.hpp:47`).
- Unit test: `getp-csrc/tests/attn_test.cpp`. Benchmark: `getp-csrc/bench/attn_bench.cpp` (`variants`
  for every idea above; `:80` the vector-only version, `:158` the loads-only version, `:63` the plain
  read). PyTorch yardstick: `getp-csrc/bench/attn_torch_yardstick.py`.

## What it gave

**Correctness of the kernel** (`tests/attn_test.cpp`, random inputs against an fp64 CPU reference;
both layer kinds; pos 0, 1, 127, 128, 129, 1023; 37 requests, so no tiling divides evenly; plus 512
requests at pos 1023; cache slots not in use filled with NaN to catch any stray read):

- largest error of the fp32 output (before the final bf16 rounding): **4.2e-5** for outputs up to
  2 in size (3.6e-5 or less in all but the "peaky" cases with 4× larger queries). The bf16
  output's error is 3.9e-3 at most, the same as the previous kernel's: it is the bf16 rounding itself.
- `attention_bf16` gives bit-identical output to the 1-head-per-workgroup variant (checked in the test).

**End to end** (`./run … -m getp`, gpt-oss-20b, one GCD, `tools/ref_match.py` against the
reference answers):

| Run | Before (previous kernel) | After |
|---|---|---|
| 64 prompts × 128 tokens | 43 / 64 lines identical, 80.86 % of tokens before the first difference | 44 / 64 lines, 83.19 % |
| 16 prompts × 1,024 tokens | 0 / 16 lines, 30.73 % | 0 / 16 lines, 36.50 % |

Long answers drift away from the reference with either kernel: the fast path stores the cache and
every matrix input in bf16, so small differences pile up over 1,000 tokens. The new kernel is
slightly *closer* to the reference, not further. The real pass/fail test is METEOR/BERTScore on the
full input, run by the lead.

## Pitfalls

- **Matrix-core lane layouts are easy to get wrong and still produce plausible numbers.** Every
  mapping (which lane holds which row, column and k) must match between the two operands. The unit
  test against an fp64 reference, at sizes that do not divide evenly, is what proves them.
- **Unused cache slots can hold NaN.** The cache is never cleared, and NaN × 0 = NaN, so setting a
  weight to 0 is not enough: the loads themselves must not pick up those slots. Buffer loads with an
  exact range do that for free. The test fills unused slots with NaN on purpose.
- **bf16 queries lose precision.** Rounding q to bf16 for the matrix cores adds an error as large as
  the cache's own. The hi/lo split restores it at no cost. Truncating the parts instead of rounding
  them was 3× less accurate.
- **The window layer's slots are rotated** (position t in slot t % 128). Our sum does not care about
  order. A kernel that needs to know positions inside a tile (a causal mask while reading a prompt in
  one go) would.
- **Rescaling the output every step is not free.** The compiler keeps matrix results in special
  accumulator registers and copies them out for ordinary arithmetic (`v_accvgpr_read`/`write`): in
  the first version that was about 50 of the 257 vector instructions per step. Lazy rescaling skips
  it in almost every step.
- **More instructions saved ≠ faster.** Cutting 20 % of the vector instructions gained 1.4 %: at
  that point the time was set by the loads. Check *what* limits the kernel (loads-only version,
  plain-read yardstick) before optimizing.
- **Benchmark bigger than the cache.** The L2 cache holds 8 MB; every shape here reads ≥ 68 MB per
  call, so the numbers are memory numbers, not cache numbers.
- **Shared GPUs.** Other jobs (including one on the other die of the same card) move timings by a
  few %. Each before/after pair in the two big tables comes from one run.

## Changes proposed outside this file

1. **Store the cache in chunks of 64 slots per key/value head ("chunk64").** Element
   (b, slot, g, i) at `b·slots64·512 + (slot/64)·64·512 + g·64·64 + (slot%64)·64 + i`, with
   `slots64` = slots rounded up to a multiple of 64. Measured: full layer at pos 1023 1,246 → 945 µs
   (−24 %), window layer 151 → 122 µs (−19 %), 1,100–1,200 GB/s at every history length. Unlike a
   plain head-major layout it never degrades at short histories. Ready to use:
   `kv_index_chunk64()` and `attention_bf16_chunk64()` at the end of `fast_attention.hpp` (tested in
   `attn_test.cpp`). The engine changes are two lines: the store index in `rope_kv_bf16_kernel`
   (`fast_misc.hpp:47`), and calling `attention_bf16_chunk64` (with `slots` rounded up to 64) in
   `gpu_step_fast()`.
2. **Per-request lengths, for continuous batching later.** Today all requests share `pos`, so one `n`
   serves the whole batch. With continuous batching the kernel needs `n` per request (an int array
   indexed by `b`); the kernel already handles any `n` per workgroup.
3. **A smaller cache (int8 or fp8 keys/values)** would halve the bytes again, and the time with them.
   It changes the numbers, so it needs the METEOR/BERTScore gate; two of the reference teams used an
   int8 cache.
4. **Fusing RoPE and the cache write into the attention kernel** would save one launch per layer
   (about 10 µs): small, so not proposed now.

## Further reading

- Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: fast and memory-efficient exact attention with
  IO-awareness* (2022) — tiles and online softmax. https://arxiv.org/abs/2205.14135
- Dao, *FlashAttention-2* (2023) — better work split. https://arxiv.org/abs/2307.08691
- Dao, Haziza, Massa, Sizov, *Flash-Decoding for long-context inference* (2023) — splitting the
  history for decode. https://pytorch.org/blog/flash-decoding/
- Milakov, Gimelshein, *Online normalizer calculation for softmax* (2018) — the running max/sum.
  https://arxiv.org/abs/1805.02867
- Ainslie et al., *GQA: training generalized multi-query transformer models* (2023).
  https://arxiv.org/abs/2305.13245
- Xiao et al., *Efficient streaming language models with attention sinks* (2023).
  https://arxiv.org/abs/2309.17453
- AMD, *AMD matrix instruction calculator* (lane layouts of every MFMA) —
  https://github.com/ROCm/amd_matrix_instruction_calculator ; *CDNA2 ISA* (buffer loads, DPP).
