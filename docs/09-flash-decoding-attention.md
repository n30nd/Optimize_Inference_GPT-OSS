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

- `:1-52` — the design in short (work split, matrix-core layout, memory access, int8 V).
- `:80` `bf16_piece()` — the hi/lo split, rounded to nearest; `:128` `row_swap8()` adds the two columns.
- `:97` `make_rsrc()` — buffer descriptor whose range ends at slot n − 1.
- `:104` `sbyte_to_f32()`, `:123` `sbytes_to_bf16x4()` — int8 codes to the matrix operand (SDWA, `v_perm`).
- `:145` `KVLayout` — where a slot lives; chunk64 is `{slots·512, 64, 64·64, 6, 64·512}`.
- `:161` `attention_gqa8_kernel` — the kernel (`Q8 = true`: int8 V). `:186` queries as the
  matrix-core operand; `:211`, `:214` int8 V code and scale buffers; `:226` the loop over this wave's
  steps; `:237` key loads; `:240` int8 V loads; `:270` scores (hi + lo); `:272` masking of the last
  step; `:294` lazy rescale; `:317` weights (times the scale for int8 V); `:320` weights as the
  operand of the second multiply; `:321` the int8 V multiply; `:345` per-wave sums; `:365` merge of
  the splits and the sink.
- `:391` `attention_gqa8()`, `:401` `attention_gqa8_q8v()` — launch helpers for any split.
- `:478` `attention_bf16()` — row layout entry point; `:415` `attention_bf16_simple_kernel` — the
  first kernel, kept as the fallback for other head counts (`-DGETP_ATTN_SIMPLE` forces it).
- `:511` `kv_index_chunk64()`, `:519` `attention_bf16_chunk64()` — chunk64 with bf16 V.
- `:544` `vq_index_chunk64()`, `:550` `vs_index_chunk64()`, `:556` `attention_q8v_chunk64()` — int8 V,
  what the engine calls; `:564` the split per case.
- Called from `layer_attention()` in `getp-csrc/engine/gpu_engine.hpp:396`; the cache is written by
  `rope_kv_bf16_kernel` (`getp-csrc/kernels/fast_misc.hpp:39`, int8 V from `:66`), allocated in
  `gpu_alloc_state()` (`gpu_engine.hpp:259`).
- Unit test: `getp-csrc/tests/attn_test.cpp` (int8 V: `run_case_q8` `:243`, writer `check_writer`
  `:384`); value-vector statistics on the real model: `getp-csrc/tests/attn_vstats.cpp`. Benchmark: `getp-csrc/bench/attn_bench.cpp` (`variants` for every idea above, `q8` for the
  int8 V splits; `:83` the vector-only version, `:161` the loads-only version, `:196` its int8 V
  version, `:66` the plain read). PyTorch yardstick: `getp-csrc/bench/attn_torch_yardstick.py`.

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

## A smaller cache: the values stored as int8

### The idea in plain words

Attention spends its time *reading* the cache, and half of what it reads are the values (V). Each
value is a bf16 number, 2 bytes. We now store each value in **1 byte**, as a whole number between
−127 and 127, plus one "unit size" per group of 64 values that says what a step of 1 is worth. The
keys (K) stay in bf16. Per slot and key/value head the kernel now reads 128 + 64 + 4 = **196 bytes
instead of 256** (−23 %), and the full layers get faster by about that much.

The stored values are now slightly rounded, so the answers change a little. That is a
quality trade, not a free speed-up, so it was accepted only after the two answer-quality scores
(METEOR and BERTScore) were measured on the full input for both models.

### An everyday comparison

A shop writes down 64 prices for every customer. Before, each price was written with three
significant digits ("2.54", "0.0312", "−1.20"). Now each list starts with one line, "one unit =
2 cents", and every price is written as a whole number of units between −127 and 127 ("127", "2",
"−60"). The list is half as long. A price can be off by at most half a unit (1 cent). The unit is
chosen per list, from that list's largest price, so a list of small prices gets small units.

### The new words

- **int8** — a whole number stored in 8 bits (1 byte), from −128 to 127. We use −127..127.
- **quantize** — replace a number by the nearest step on a coarse grid; here the grid is
  `scale × (−127, …, 127)`. **dequantize**: turn the stored whole number back into `scale × code`.
- **code** — the stored whole number. **scale** — the size of one step, one fp32 number per 64 values.
- **outlier** — one value much larger than the others in the same group of 64. It sets the scale, so
  the other 63 values get coarser steps.
- **SDWA (sub-dword addressing)** — a way for one gfx90a instruction to read just one byte (or half)
  of a 32-bit register. **`v_perm`** — an instruction that builds a 32-bit word from any 4 bytes of
  two registers.
- **METEOR / BERTScore** — the two answer-quality scores of the course (section 3 of the project
  brief): word overlap with the reference answers, and similarity of meaning. Pass marks 0.30 / 0.90.

### How a value vector is stored

```
 one slot, one key/value head: 64 values (fp32, just computed by the QKV matrix multiply)
     v =   0.31   -1.20   0.05   ...   2.54   ...   -0.48        largest |v| = 2.54
 scale s = 2.54 / 127 = 0.0200                                   (1 fp32, 4 bytes)
 codes c = round(v / s)
         =   16     -60      2   ...    127   ...     -24        (64 int8, 64 bytes)
 read back s × c
         = 0.320  -1.200  0.040  ...  2.540   ...  -0.480        error at most s / 2 = 0.01
```

Before: 64 × 2 = 128 bytes. Now: 64 + 4 = 68 bytes. The cache writer (`rope_kv_bf16_kernel`) does
this as it stores each new token: one wave holds one head's 64 values, finds the largest |v| with 6
lane exchanges (`__shfl_xor`), and writes the codes and the scale.

The chunk64 idea stays: each head's 64 slots are one block (now 4 KB of codes instead of 8 KB), so
the kernel still reads long contiguous runs. The scales sit in their own array, 64 per block (256
bytes). Inside each block of 16 slots (1 KB) the codes are in the order the lanes want them:

```
 16 slots = 4 groups of 4 slots (256 bytes each)
 one group = 16 pieces of 16 bytes; piece d holds dims 4d .. 4d+3 of the group's 4 slots:
     [dim 4d: slot 0 1 2 3] [dim 4d+1: slot 0 1 2 3] [dim 4d+2: ...] [dim 4d+3: ...]
 lane (grp, col) reads piece col of group grp: ONE 16-byte load; the 64 lanes read the 1 KB in order
 (bf16 V needed 4 loads of 8 bytes per lane for the same tile)
```

Each 4-byte word of a lane is then exactly what one matrix-core multiply wants: the codes of 4
consecutive positions for one dim.

### How attention uses the codes

```
 out = Σ_t p_t · v_t / L  ≈  Σ_t p_t · (s_t · c_t) / L  =  Σ_t (p_t · s_t) · c_t / L
       weights · values                                    new weight   · code
 L   = exp(sink − M) + Σ_t p_t          (unchanged: the scale does not belong in the denominator)
```

So the kernel multiplies each position's weight by that position's scale (one multiply per
weight, then the same hi + lo split as before), and gives the second matrix multiply the codes
themselves. A code is a whole number with |c| ≤ 127, and bf16 holds every such number exactly, so the
matrix multiply adds **no** rounding of its own: the only new error is the int8 rounding, made once
when the token is stored.

Turning 4 codes into the matrix operand: each code becomes an fp32 number with one instruction
(`v_cvt_f32_i32` reading one byte, with SDWA), and the upper halves of two fp32 numbers, which are
their exact bf16 bits, go into one register with one `v_perm`. That is 24 instructions per 16-position
tile. Written in plain C++ (`(float)(int8_t)(w >> 8 * r)`), the compiler emits a shift, a bit-extract
and the conversion per code, about 60 instructions per tile. Timed, both are equally fast on long
histories, and the hand-written one is 2–3 % faster at position 64: the reads set the time.

The positions past n are handled as before: the scale buffer ends exactly after slot n − 1, so later
slots read scale 0 (the unused part of the cache can hold NaN). The codes of those positions are
read as part of their 16-slot tile, but a code is always a finite whole number, and its weight is 0.

### The numbers

Kernel time on one GCD of mv-mi250-03 (`attn_bench`, mean of 30 calls, 2026-10-25; both rows of a
pair from the same run). GB/s counts the bytes each version must read: 256 per slot and head for
bf16 V, 196 for int8 V. The plain read of the same memory reaches 1,273 GB/s.

**512 requests per GCD**

| Layer | pos | n | bf16 V | GB/s | int8 V | GB/s | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| window | 64 | 65 | 93.0 µs | 733 | 82.2 µs | 635 | −12 % |
| window | 256 | 128 | 122.9 µs | 1,092 | 107.1 µs | 960 | −13 % |
| window | 512 | 128 | 122.7 µs | 1,094 | 107.6 µs | 955 | −12 % |
| window | 1023 | 128 | 122.9 µs | 1,092 | 107.1 µs | 959 | −13 % |
| full | 64 | 65 | 105.0 µs | 649 | 100.3 µs | 520 | −4 % |
| full | 256 | 257 | 240.8 µs | 1,119 | 205.0 µs | 1,007 | −15 % |
| full | 512 | 513 | 459.7 µs | 1,170 | 357.5 µs | 1,152 | −22 % |
| full | 1023 | 1024 | **935.1 µs** | 1,148 | **695.6 µs** | 1,182 | **−26 %** |

**1,024 requests per GCD**

| Layer | pos | n | bf16 V | GB/s | int8 V | GB/s | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| window | 64 | 65 | 177.3 µs | 769 | 152.1 µs | 686 | −14 % |
| window | 256 | 128 | 235.0 µs | 1,142 | 200.3 µs | 1,026 | −15 % |
| window | 512 | 128 | 235.4 µs | 1,140 | 200.2 µs | 1,027 | −15 % |
| window | 1023 | 128 | 235.6 µs | 1,139 | 200.3 µs | 1,026 | −15 % |
| full | 64 | 65 | 197.2 µs | 691 | 189.2 µs | 552 | −4 % |
| full | 256 | 257 | 463.6 µs | 1,162 | 395.6 µs | 1,043 | −15 % |
| full | 512 | 513 | 884.6 µs | 1,216 | 692.0 µs | 1,190 | −22 % |
| full | 1023 | 1024 | **1,776.5 µs** | 1,209 | **1,362.2 µs** | 1,207 | **−23 %** |

- Full layers with long histories keep the same GB/s on fewer bytes, so the time falls with the bytes.
  Window layers read only 128 positions per request: each wave does just 2 steps, and the fixed
  work (queries, merge, launch) weighs more; the loads alone take 77 µs there, the kernel 107 µs.
- Over one whole answer at 512 requests, attention costs on average about **7.2 ms per step with
  bf16 V and 5.7 ms with int8 V** (12 full + 12 window layers, interpolated between the positions
  above).
- Memory: the cache of 512 requests at 1,024 positions takes 14.5 → 11.1 GB for 20b and
  21.7 → 16.6 GB for 120b.
- Split per case (`attention_q8v_chunk64`): window layers 2 waves per head (107 µs; 4 waves:
  109 µs), full layers 4 waves up to 128 positions and 8 waves above (695 µs at pos 1023; 4 waves:
  761 µs). All splits were measured (`attn_bench q8`).

**Accuracy of the kernel** (`tests/attn_test.cpp`, second part; the same cases as above, plus value
vectors with one outlier 8× larger than the rest):

| Compared with an fp64 reference on … | int8 V kernel | bf16 V kernel |
|---|---:|---|
| … the dequantized values `s × c` (the kernel's own arithmetic), fp32 output | **4.3e-5** at most | 4.2e-5 on its bf16 values |
| … the original fp32 values, bf16 output, largest error | 1.16e-2 | 7.75e-3 |
| … the original fp32 values, root-mean-square error, full layer pos 1023 | 2.45e-4 | 1.30e-4 |
| same, window layer pos 1023 | 6.6e-4 | 3.5e-4 |
| same, with one 8× outlier per vector, window pos 1023 | 2.9e-3 | 4.9e-4 |

So the kernel's own arithmetic is as exact as before, and the int8 rounding roughly doubles the
typical error of the output against the original values (6× with strong outliers). The cache writer
gives exactly the codes and scales of the CPU rule (56,832 codes checked, including all-zero and
outlier vectors).

**End to end** (`tools/bench.py`, `evaluation/input.txt`, 4,096 requests, 8 GCDs of mv-mi250-02):

| Model | bf16 V (commit cdca2c5) | int8 V |
|---|---|---|
| gpt-oss-20b | 72,258 tok/s, METEOR 0.554, BERTScore 0.979 | **73,475 tok/s (+1.7 %)**, METEOR **0.533**, BERTScore **0.978** |
| gpt-oss-120b | 40,164 tok/s, METEOR 0.576, BERTScore 0.982 | **40,949 tok/s (+2.0 %)**, METEOR **0.562**, BERTScore **0.981** |

Both stay far above the pass marks (0.30 / 0.90). The meaning score (BERTScore) moves by 0.001; the
word-overlap score (METEOR) drops by 0.021 (20b) and 0.014 (120b), more than the ±0.005 that bf16
versions of the program differ among themselves (0.548–0.557 for 20b), so the drop is real. It is
the size the other team saw (0.534 → 0.519). The speed gain is much smaller than the kernel gain because
attention is only about 14 % of a step: the kernels save about 1.5 ms of a 55 ms step (20b, 512
requests per GCD, estimate from the table above), and the whole timed run went from 56.4 to 55.5 s
(20b) and from 100.7 to 98.9 s (120b).

Smoke test (20b, 64 prompts × 128 tokens, one GCD): 45 → 43 of 64 answers identical to the
reference, 84.5 % → 82.0 % of tokens before the first difference.

**Why the rounding cannot easily be made smaller.** `tests/attn_vstats.cpp` runs 20b on 64 real
prompts for 400 steps and measures the stored value vectors: their largest |v| is only **2.6–2.9×
their root mean square** in every layer (random numbers give about 2.5), so no few large values
waste the 255 steps. The int8 rounding noise is then about **0.6 % of a typical value**, in every
layer (bf16 rounding: about 0.1 %). Taking out the bias or evening out the dims first would
lower it only to 0.57–0.60 %. Doing better needs more bits per value, which is what we wanted to
avoid.

### Why the keys stay bf16

Another team on the same course measured both halves: int8 V moved their METEOR from 0.534 to
0.519, int8 K dropped it to 0.342. An error in a key changes the *score*, which goes through
`exp()`: a small error in a large score becomes a large error in a weight, and it can move weight
from one position to another. An error in a value only shifts the output a little, in proportion to
the weight. So only V is stored as int8.

### Ideas measured or considered, and not used

| Idea | Result | Why not |
|---|---|---|
| int8 K as well | METEOR 0.342 in the other team's run | quality |
| Multiply the scale into V (decode to bf16 `s × c`) instead of into the weight | not built | adds bf16 rounding (2^-9) on top of the int8 rounding, and costs a multiply per value instead of per weight |
| Codes converted by the compiler (shift + extract + convert) | same speed at long histories, +2–3 % at pos 64 | kept the one-instruction SDWA version |
| fp16 matrix multiply, codes decoded with one `v_perm` per 2 codes (fp16 bits 0x64 then the byte u = the number 1,024 + u) | not built | weights would be fp16: anything below 6e-5 loses precision or becomes 0, and the conversion is not the limit (row above) |
| Subtract the value bias before rounding (add it back after attention), or divide each dim by a fixed size first ("smoothing") | rounding noise 0.61 % → 0.60 % (bias), 0.58 % (smoothing), 0.57 % (both) | not worth the work: the real value vectors are not peaky (below) |
| 4 waves per head on window layers | 109 vs 107 µs | slower |
| 8 waves per head on full layers up to 128 positions | 108 vs 100 µs at pos 64 | slower |

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
- **int8 V: the scale goes into the weight, not into the denominator.** `L` sums the weights `p`;
  the output sums `p · s · c`. Putting `p · s` into `L` gives plausible-looking but wrong outputs.
- **int8 V: unused scales can be NaN.** A weight of 0 times a NaN scale is NaN. The scale buffer's
  range ends right after slot n − 1, so the kernel reads 0 there. The test fills unused scales with
  NaN and unused codes with random bytes.
- **int8 V: one large value per vector costs the other 63 their precision.** The scale is set by the
  largest |v|. The test's outlier cases show 6× the typical error; the quality scores on the real
  model are the test that matters.

## Changes proposed outside this file

1. **Store the cache in chunks of 64 slots per key/value head ("chunk64").** Element
   (b, slot, g, i) at `b·slots64·512 + (slot/64)·64·512 + g·64·64 + (slot%64)·64 + i`, with
   `slots64` = slots rounded up to a multiple of 64. Measured: full layer at pos 1023 1,246 → 945 µs
   (−24 %), window layer 151 → 122 µs (−19 %), 1,100–1,200 GB/s at every history length. Unlike a
   plain head-major layout it never degrades at short histories. Ready to use:
   `kv_index_chunk64()` and `attention_bf16_chunk64()` at the end of `fast_attention.hpp` (tested in
   `attn_test.cpp`). The engine changes are two lines: the store index in `rope_kv_bf16_kernel`
   and calling `attention_bf16_chunk64` (with `slots` rounded up to 64). **Done** (commit c61cc5d).
2. **Per-request lengths, for continuous batching later.** Today all requests share `pos`, so one `n`
   serves the whole batch. With continuous batching the kernel needs `n` per request (an int array
   indexed by `b`); the kernel already handles any `n` per workgroup.
3. **A smaller cache (int8 or fp8 keys/values)** would halve the bytes again, and the time with them.
   It changes the numbers, so it needs the METEOR/BERTScore gate; two of the reference teams used an
   int8 cache. **Done for V** (section "A smaller cache" above); K stays bf16 for quality.
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
