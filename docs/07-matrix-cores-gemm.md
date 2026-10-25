# 07 — Dense matrix multiplies on the matrix cores

## The problem in plain words

Most of the model's arithmetic is "multiply a table of numbers by another table": every request's
current word (a row of 2,880 numbers) is multiplied by a big table of learned weights. For 512 requests
at once this is a **matrix multiply** (also called **GEMM**, general matrix multiply): 512 rows times a
weight table of, for example, 5,120 × 2,880 numbers.

Our first GPU version did this one multiply-and-add at a time on the ordinary calculators and reached
about 6 TFLOP/s (6 million million multiply-or-add operations per second). The chip also has **matrix
cores**: special units that multiply a small block of numbers by another small block in one
instruction. Their peak is 181 TFLOP/s on one GCD. This document explains how we feed them, and how
close we get to AMD's own library.

## An everyday comparison

Filling a stadium's seats one by one with a single usher versus with a team that carries whole rows of
people at a time. The team is much faster, but only if someone keeps bringing it full rows without
gaps. Most of the work below is about the "bringing": having the next rows ready at the right place
exactly when the team is free, and giving every team the same amount of work so none stands idle at
the end.

## The words

- **matrix multiply / GEMM** — C = A · Wᵀ: every number of the result is a sum of 2,880 (or 4,096)
  products of a row of A with a row of W.
- **M, N, K** — the three sizes of a GEMM: M rows of A (the requests in the batch), N rows of W (the
  outputs per request), K numbers summed for each output.
- **TFLOP/s** — a speed: million million arithmetic operations per second (a multiply-add counts as 2).
- **GCD** — one of the two chips on an MI250 card; the software sees each as its own GPU (doc 02).
- **CU (compute unit)** — one of the GCD's 104 groups of calculators, each with its own LDS (below).
- **hipBLASLt** — AMD's own matrix-multiply library, our speed yardstick (we may not use it in `./run`).
- **bf16** — a short number format (2 bytes): the inputs. **fp32** — the normal 4-byte format: the
  sums and the result. Multiplying two bf16 numbers gives an exact fp32 number, so only the adding
  rounds.
- **wavefront (wave)** — a group of 64 threads that run in lockstep (doc 02).
- **SIMD** — one of the 4 lanes of calculators inside a compute unit; each runs one wave at a time and
  owns one matrix core.
- **MFMA (matrix fused multiply-add)** — the matrix-core instruction. We use
  `v_mfma_f32_32x32x8bf16_1k`: one wave multiplies a 32 × 8 block of A by an 8 × 32 block of Wᵀ and
  adds the 32 × 32 result to 32 × 32 running sums kept in its registers. It takes 64 clock ticks.
- **register** — a calculator's own tiny storage, the fastest memory there is.
- **LDS (local data share)** — the small fast scratch memory (64 KB per compute unit) shared by the
  threads of one block (doc 02).
- **tile** — the piece of the result one block of threads computes, e.g. 128 × 128 numbers.
- **K step** — one slice of the sum: the block handles 32 of the K values at a time.
- **barrier** — a point in the program where all threads of a block wait for each other.
- **double buffering** — two copies of a scratch area: one is being used while the other is filled.
- **prefetch** — asking memory for data before it is needed, so it has arrived by then.
- **bank conflict** — LDS is split into 32 banks; two threads asking the same bank at once wait in turn.
- **occupancy** — how many waves can sit on one SIMD at once; when one waits, another can work.
- **L2 cache** — an 8 MB memory between the compute units and the main memory (HBM): data read
  recently by one block is found there by the next one.
- **round (of tiles)** — the GPU runs as many blocks at once as fit (here 104 compute units × 2 blocks);
  if there are more tiles, the rest run in a next round.
- **split K** — letting several blocks share one tile, each summing part of K, and adding their parts
  at the end.
- **wave quantization** — the time lost when the number of tiles does not fill the GPU's rounds
  evenly: the last round is only partly full, so some compute units finish early and wait. ("Wave"
  here means one round of tiles, not a wavefront.)
- **persistent kernel** — a kernel that starts exactly as many blocks as the GPU runs at once (here one
  per compute unit) and lets each block work through a list of jobs, instead of one block per tile.
- **stream-K** — a persistent kernel that shares out the K steps of all tiles evenly between its
  blocks, so a block may do the end of one tile and the start of the next.
- **accumulator** — the running fp32 sums of a tile, kept in registers while K steps are added.
- **atomic add** — an addition that memory carries out as one indivisible step, so when several
  blocks add 1 to the same counter at the same moment, every one of them is counted.

## How it works

### 1. What one MFMA wants

```
 A operand (32 rows x 8 k)          W operand (32 rows of W x 8 k)        32x32 sums (16 per lane)
 lane  0..31 : row lane,  k 0..3    lane  0..31 : row lane,  k 0..3       lane l holds column l%32,
 lane 32..63 : row lane-32, k 4..7  lane 32..63 : row lane-32, k 4..7     rows 8*(v/4)+4*(l/32)+v%4
```

Both A and W are stored "k-contiguous" (each row's 2,880 numbers side by side), which is exactly
this layout: every lane needs a few neighbouring k values of one row. The k slots do not have to be
the real k order, only the same for A and W, so each lane reads **8** neighbouring values (16 bytes,
one load) and spends them on **two** MFMAs.

### 2. Tiles, LDS and the waves of a block

```
                 K (2880) ───────────────────────────────►
   A rows   ┌──────────┬──────────┬──────────┬───
   m0..     │ step 0   │ step 1   │ step 2   │ ...      each K step: 32 k values
   m0+128   └──────────┴──────────┴──────────┴───
                  │ copy to LDS (one block = 256 threads = 4 waves)
                  ▼
   LDS: sA[2][128 x 32]   sW[2][128 x 32]    (two buffers: double buffering, 32 KB)
                  │ read 16-byte pieces ("fragments")
                  ▼
   wave (0,0) ─ 64x64 of C   wave (0,1) ─ 64x64 of C     128 x 128 tile, 4 waves,
   wave (1,0) ─ 64x64 of C   wave (1,1) ─ 64x64 of C     each wave 2x2 MFMA blocks of 32x32
```

A block copies a slice of A (128 rows × 32 k) and of W (128 rows × 32 k) into LDS once; then each of
its waves reads the parts it needs. Every number copied is used by two waves, and every fragment a wave
reads feeds two MFMAs. Rows past the end of A or W are replaced by the last real row (computed, never
written), so the main loop has no special cases. With 32 KB of LDS per block, two blocks fit on one
compute unit: two waves per SIMD, so one can work while the other waits.

The version we finally use splits the same 128 × 128 tile over **8 waves** (2 × 4, each 64 × 32):

```
   cols:  0-31   32-63  64-95  96-127
   rows   ┌──────┬──────┬──────┬──────┐
   0-63   │ w0   │ w2   │ w4   │ w6   │    same 32 KB of LDS per block, but 8 waves:
   64-127 │ w1   │ w3   │ w5   │ w7   │    two blocks per compute unit = 4 waves per SIMD
          └──────┴──────┴──────┴──────┘
```

Each wave does half the MFMAs per step, but when one wave waits (for the barrier or for memory) three
others on the same SIMD can use the matrix core. That alone was worth 3–9 %.

### 3. No waiting in LDS: the XOR layout

A 16-byte read by 8 neighbouring lanes is served at once only if they hit 8 different groups of banks.
With 64-byte rows, rows r and r + 2 start on the same bank, so the 8 lanes reading rows 0..7 collided.
We store piece c of row r at position `c XOR ((r / 2) % 4)` of that row: the 8 lanes now land on 8
different groups. The profiler confirms it: `SQ_LDS_BANK_CONFLICT` = 0.

### 4. The pipeline of one K step

One K step = two "chunks" of 16 k values; each chunk is 8 MFMAs per wave.

```
 time ──────────────────────────────────────────────────────────────────────────►
 matrix core : [ MFMA MFMA MFMA MFMA MFMA MFMA MFMA MFMA ][ MFMA ×8 (chunk 1) ......... ]
 LDS reads   :     r    r    r    r   (chunk 1 operands)  |  r  r  r  r (chunk 0 of next step)
 LDS writes  :                         w    w    w    w   |  (next step → other buffer)
 HBM/L2 loads:                         g    g    g    g   |  (the step after next)
 barrier     :                                           ▲ one per step
```

- While the matrix core works on 16 k values, the operands of the next 16 are read from LDS into a
  second set of registers.
- The next step's data has been in registers since the previous step (prefetch). It is written into the
  *other* LDS buffer at the start of the step, while the current one is still being read (double
  buffering), and the registers are refilled from memory with the step after.
- One barrier per step makes the new buffer visible to every wave. Because the LDS writes went out at
  the start of the step, they are finished long before the barrier, so nobody waits there for them.
- The compiler is told the exact order "one MFMA, one memory instruction, one MFMA, …"
  (`__builtin_amdgcn_sched_group_barrier`). Without it, it bunched the four LDS writes right before the
  barrier and read the next operands only after the barrier, with only 4 MFMAs left to overlap them.
  Moving the copy to the start of the step gave 4–13 %; the ordering hints alone 1–9 %.

### 5. One tile size does not fit all shapes

The GPU runs a fixed number of blocks at once (104 compute units × 1 or 2 blocks). How many tiles a
multiply has decides which tile size wins:

```
 qkv, 512 requests:  512 x 5120 outputs
   128 x 128 tiles → 4 x 40 = 160 tiles, 2 blocks per CU fit → all run at once      156 us
   256 x 256 tiles → 2 x 20 =  40 tiles → 64 of 104 CUs idle                        278 us
 attention out, 512 requests: 512 x 2880 outputs, K = 4096
   128 x 128 x 64  → 4 x 23 =  92 tiles, one per CU, half as many barriers         121 us
 LM head, 2048 requests: 2048 x 201088 outputs
   256 x 256 tiles → 6,288 tiles: plenty; big tiles need half the L2 traffic       19.2 ms
```

`gemm_pick_config` holds the fastest choice for each of our shapes at 64 … 2048 requests (measured with
`bench/gemm_bench.cpp --sweep`, which times every configuration). Section 7 shows that qkv's 160 tiles
still do not fit the compute units well, and what we do about it.

**Split K for small batches.** At 64 requests there are too few tiles for 104 compute units. Then the
same tile is computed by 2 blocks, each summing half of K; the second waits for a flag saying the
first has written its half, then adds its own (fixed order: same bits every run, checked by the test).
Attention out at 64 requests: 55.7 → 44.9 µs.

### 6. Small outputs: the router

The router has only 32 or 128 outputs per row. Up to 512 requests we use 32 × 32 tiles and split K in
4 parts: part w takes every 4th slice of k (slices w, w + 4, …), reading straight from memory, and the
parts are added in a fixed order (same result every run). From 1,024 requests on, the 64 × 64 tiles
with 2 K slices are faster.

At first one block (4 waves, one per part) did a whole tile. For 20b at 512 requests that is only 16
tiles, so 16 of the 104 compute units did all the reading, and a compute unit can only have so many
loads on their way at once: 20 µs alone, 28 µs between two big multiplies (as in the engine, where the
router's input has been pushed out of the L2 cache). Loading further ahead did not help, which showed
that the limit was the compute unit, not the wave. Now each part is its own one-wave block, so a tile's
loads come from 4 compute units; each part writes its sums to a scratch area, and the last part of a
tile to finish (found with an atomic counter) adds the 4 in the same fixed order as before: same bits.

### 7. When the tiles do not fit the compute units: wave quantization and stream-K

**The problem.** Each tile is done by one block on one compute unit. qkv at 512 requests has
4 × 40 = 160 tiles of 128 × 128. Two blocks fit on a compute unit, so all 160 start at once, but 56
compute units get two tiles and 48 get one. A compute unit with two tiles runs each at about half
speed, so the multiply lasts as long as two tiles, and the 48 lightly loaded units sit idle for the
second half. Measured with the same kernel and more or fewer columns:

```
 tiles of 128 x 128 (512 rows)    104      160      208
 time                             94 us    156 us   184 us
```

208 tiles is 30 % more work than 160 but only 18 % more time: about a fifth of the GPU is wasted at
160. This is wave quantization.

**Stream-K: share out K steps, not tiles.** With K = 2880 in steps of 64, a tile is 45 steps; all of
qkv is 160 × 45 = 7,200 steps. A persistent kernel starts one block per compute unit (104 "workers")
and gives each 7,200 / 104 = 69.2 steps instead of up to 90. A worker's share usually starts in the
middle of one tile and ends in the middle of another:

```
 K steps of one row of tiles:  0          45         90         135        180
                               | tile 0   | tile 1   | tile 2   | tile 3   | ...
 worker 0 (steps 0-68)         [  full    ][ head ]
 worker 1 (steps 69-137)                          [tail][  full    ][h]
 worker 2 (steps 138-206)                                           [ tail ][ head ...
```

**Keeping every bit.** The usual stream-K lets each worker sum its part of a tile from zero and adds
the parts at the end. That adds the numbers in a different order, so the last bits change. We keep
the order instead: the worker that does the start ("head") of a tile writes its accumulators, unrounded,
to a scratch area and sets a flag; the worker that does the rest ("tail") waits for the flag, loads
those sums and keeps adding to them, with the same instructions in the same order. The result is the
one a single block would have given, bit for bit (the test checks this for every shape). To make sure
nobody waits, each worker does its head first, then its whole tiles, then its tail: by the time a
worker reaches its tail, the previous worker's head is long finished.

**Keeping the L2 cache useful.** The earlier stream-K attempt (see Pitfalls) let every worker walk its
own run of tiles, so the GPU read 208 different weight tiles at once. Here the workers are grouped by
row of tiles: for qkv, 26 workers for each of the 4 rows, and workers with the same position in their
row read the same weight tile at the same K step at the same time, as in the plain kernel.

**The limit: one tile's steps run one after another.** Keeping the order means a tile's 45 (or 64)
steps can never be done in parallel. If a share is shorter than a tile, some tile is split three ways
and the third worker must wait for the second. Attention out at 512 requests (92 tiles of 64 steps,
55 steps per worker) hits exactly this: 125 µs instead of 121 µs. Smaller tiles (128 × 64 or 64 × 128)
avoid the three-way split but each step does less work per byte loaded: 126–128 µs. So attention out
at 512 keeps the plain kernel; `gemm_pick_config` uses stream-K only where every worker gets at least
one tile's worth of steps (`stream_fits`).

| Multiply | requests | plain kernel | stream-K | change |
|---|---:|---:|---:|---:|
| qkv 5120 × 2880 | 384 | 152 | 122 | −20 % |
| | **512** | **155** | **141** | **−9 %** |
| | 768 | 288 | 200 | −31 % |
| | 1,024 | 299 | 264 | −11 % |
| attention out 2880 × 4096 | 384 | 115 | 104 (64 × 128 tiles) | −10 % |
| | 512 | **121** | 125 (not used) | +3 % |
| | 768 | 217 | 164 | −24 % |
| | 1,024 | 229 | 214 | −6 % |

(µs on one GCD of mi250-03, mean of 2 runs of 30 calls; at 2,048 requests there are enough tiles and
the plain kernel stays faster.)

**Where the rest of the time goes** (qkv at 512, timed inside the kernel with the GPU's clock): 69
steps × 1.74 µs = 120 µs of matrix work. Each piece costs about 2 µs more to start (the first loads of
a piece have to arrive before its pipeline runs) and 2 µs to write out; a tail also waits about 2 µs
for its flag and the 64 KB of handed-over sums. At the very end all 104 workers write their last 64 KB
tile at the same moment: 6.8 MB at the memory's write speed of about 900 GB/s, about 7 µs.

## The numbers for our model

One GCD of mv-mi250-02 (HIP 0, all GPUs otherwise idle), median of 3 runs of 20 calls each, times in
µs. "Before" is the first fast-path kernel (fp32 multiply-adds, `bench/gemm_old_fma.hpp`); "library"
is AMD's hipBLASLt through PyTorch (`yardsticks/gemm_bench.json`, bf16 output; ours writes fp32, twice
the bytes). "% of library" = library time ÷ our time.

| Multiply (N × K) | requests (M) | before | after | library | % of library |
|---|---:|---:|---:|---:|---:|
| qkv 5120 × 2880 | 256 | 1,573 | 87.2 | 82.3 | 94 % |
| | **512** | 2,353 | **156** | 161 | **103 %** |
| | 1,024 | 3,919 | 300 | 289 | 96 % |
| | 2,048 | 7,107 | 550 | 513 | 93 % |
| attention out 2880 × 4096 | 256 | 1,085 | 77.3 | 73.5 | 95 % |
| | **512** | 2,195 | **121** | 125 | **103 %** |
| | 1,024 | 3,336 | 232 | 211 | 91 % |
| | 2,048 | 5,620 | 424 | 410 | 97 % |
| router 20b 32 × 2880 | **512** | 460 | **21.1** | 28.2 | 134 % |
| | 2,048 | 643 | 30.6 | 29.0 | 95 % |
| router 120b 128 × 2880 | **512** | 529 | **23.3** | 29.0 | 124 % |
| | 2,048 | 755 | 33.8 | 33.1 | 98 % |
| LM head 201088 × 2880 | 256 | 35,704 | 2,544 | 2,912 | 114 % |
| | **512** | 71,085 | **4,894** | 7,191 | **147 %** |
| | 1,024 | 141,869 | 9,761 | 9,526 | 98 % |
| | 2,048 | 283,417 | 19,247 | 18,503 | 96 % |

At 64 and 128 requests: 92–106 % of the library (qkv 46.7 / 57.6 µs, attention out 45.1 / 53.5 µs, LM
head 1,384 / 1,569 µs). The best speed reached is 121–123 TFLOP/s (LM head), about 68 % of the
matrix cores' peak.

All dense multiplies of one decode step of gpt-oss-20b at 512 requests on one GCD (24 layers × (qkv +
attention out + router) + LM head): 191 ms before, **12.1 ms** after.

The table is the first version. Sections 6 and 7 changed these rows since (mi250-03, HIP 0, mean of
2 runs of 30 calls; "between" = timed between two attention-out multiplies, `gemm_bench --interleave`):

| Multiply | requests | first version | now | library |
|---|---:|---:|---:|---:|
| qkv | 384 / **512** / 768 / 1,024 | 152 / **155** / 288 / 299 | 122 / **141** / 200 / 264 | — / 161 / — / 289 |
| attention out | 384 / 768 / 1,024 | 115 / 217 / 229 | 104 / 164 / 214 | — / — / 211 |
| router 20b | **512** (alone / between) | 21.6 / 30.0 | **15.2 / 18.6** | 28.2 |
| router 120b | **512** (alone / between) | 22.9 / 30.4 | 24.2 / **25.3** | 29.0 |

Everything else is unchanged (same kernels). The router of 120b is slower alone but faster between
other kernels, which is how the engine runs it.

## Where it is in the code

- `getp-csrc/kernels/fast_gemm.hpp` — everything:
  - `:71` `lds_slot` — the XOR layout of section 3.
  - `:127` `TileGemm` — one tile; `:153` `mainloop` (the pipeline of section 4, the loop at `:272`,
    the instruction-order hints at `:239` and `:257`), `:295` `load_bias`, `:307` `save_acc` /
    `load_acc` (the hand-over of section 7), `:338` `store_c`.
  - `:439` `gemm_mfma_kernel` — one block per tile, split K with the flags of `:78` `wait_flag` /
    `:87` `set_flag`.
  - `:500` `gemm_stream_kernel` — stream-K of section 7; `:759` `launch_stream`.
  - `:555` `router_part`, `:614` `gemm_splitk_kernel`, `:648` `gemm_router_kernel` — the router of
    section 6.
  - `:796` the list of configurations, `:828` `stream_fits`, `:838` `gemm_pick_config` (sections 5
    and 7), `:892` `gemm_bf16`.
- Called from `getp-csrc/engine/gpu_engine.hpp:408` (qkv) and `:414` (attention out) in
  `layer_attention()`, `:434` (router) in `layer_route()`, `:535` (LM head, `gemm_bf16_argmax`) in
  `step_logits()`.
- `getp-csrc/tests/gemm_test.cpp` — the check against an fp64 CPU sum (`--all-configs` checks every
  configuration, with and without split K); every run without split K must also give exactly the
  bits of the 128 × 128 × 32 kernel.
- `getp-csrc/bench/gemm_bench.cpp` — the timings above (`--old` adds the first kernel, `--sweep` every
  configuration, `--interleave` times each call between two big multiplies, as in the engine);
  `getp-csrc/bench/gemm_torch.py` — the library numbers.

## What it gave

- 7–24× faster than the first kernel (13–18× for qkv and LM head from 256 requests up); at 91–147 % of
  AMD's library everywhere (goal was 80 %), above it at our main size of 512 requests for every
  multiply.
- Same numbers as before up to fp32 rounding: largest error 8.0 × 10⁻⁶ of the typical output size over
  1,214 test cases (odd sizes such as 7 × 1000 included); split K gives the same bits on every run.
- End to end, gpt-oss-20b, 64 prompts × 128 tokens on one GCD: 205 → 262 tokens/s; 46 of 64 answers
  identical to the reference (86.5 % of tokens before the first difference; the old fast path: 43 of 64,
  80.9 %).
- Stream-K for qkv and the spread router (sections 6 and 7) change no bit: the full official run gives
  the same output hash as before. Whole program on 8 GCDs: 20b 75,911 → 76,149 tokens/s (+0.3 %),
  120b 42,129 → 42,175 (+0.1 %), two runs each. Inside the engine the gain is about half of the
  standalone one (profiler, 1 GCD: qkv 156 → 148 µs, router 28.5 → 23 µs per call), and these
  multiplies are a small part of a step next to the expert multiplies.

## Pitfalls we hit

- **The compiler reorders what you wrote.** The source put the LDS copy at the start of the step; the
  scheduler moved it back to just before the barrier. Only the assembly (`hipcc -S`) showed it; the fix
  is `__builtin_amdgcn_sched_group_barrier`.
- **Register budget decides occupancy.** One refactoring raised a kernel from 206 to 262 registers per
  thread, which silently halved the waves per SIMD. `__attribute__((amdgpu_waves_per_eu(n)))` tells the
  compiler the budget; `-Rpass-analysis=kernel-resource-usage` prints the result.
- **Stream-K looked right on paper and lost, the first time.** We built it (each of the 208 resident
  blocks gets the same share of all (tile, K step) work; partial tiles meet in a scratch area, added in
  a fixed order). It was correct, but each block walks its own run of tiles, so at any moment the GPU
  touches 208 different weight tiles and the L2 cache stops helping: LM head at 2,048 requests
  19.2 → 51 ms. The gains elsewhere were ≤ 3 µs, so it was removed. The version of section 7 keeps the
  workers of different tile rows in step (L2 reuse as in the plain kernel), hands sums over instead of
  adding parts (same bits), and is used only for qkv and attention out where it measures faster.
- **A branch in a loop can cost a full memory wait.** The compiler counts outstanding loads to wait
  for exactly the one an instruction needs; after an `if` that may or may not have issued loads or
  stores, it can no longer count and waits for everything. In the router's prefetch loop an
  `if (chunk < count)` made every MFMA wait for all loads; the fix was a branch-free main loop plus a
  short tail loop. For the same reason a stream-K version that ran all pieces of a worker through one
  main loop (loading the next piece while writing out the last) was 1–7 % slower than running the
  pieces one after another, so the simpler form stayed.
- **Things that did not help**: loading 2 K steps ahead instead of 1 (±1 %); tile widths that divide N
  exactly (128 × 192, 128 × 96: 11–119 % slower, because the bigger LDS use leaves one wave per SIMD).
- **Split K needs one stream per GCD.** A stream is one queue of GPU work; the engine uses one per GCD.
  The flags are shared by all calls on a GCD, so two split-K multiplies running at the same time from
  two streams of one GCD would mix them up.
- **Timings move a few percent between runs** (one attention-out run gave 247 instead of 226 µs). Take
  the median of several runs before calling a 2 % change a win.

## Further reading

- AMD, *AMD Instinct MI200 instruction set architecture* (CDNA2 ISA), chapter "Matrix arithmetic
  instructions" — operand layouts of every MFMA.
- AMD, *Matrix instruction calculator* — https://github.com/ROCm/amd_matrix_instruction_calculator —
  prints which lane holds which element for any MFMA.
- AMD ROCm blog, *AMD matrix cores* — https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html
- LLVM AMDGPU docs, `__builtin_amdgcn_sched_group_barrier` — telling the compiler how to interleave
  instruction types.
- Volkov, *Better performance at lower occupancy* (GTC 2010) — why registers and prefetching can beat
  more threads.
- Osama et al., *Stream-K: work-centric parallel decomposition for dense matrix multiplication* (2023)
  — the even split of (tile, K step) work we tried (see Pitfalls); their hybrid version keeps whole
  tiles for full rounds, which avoids the cache problem we hit.
