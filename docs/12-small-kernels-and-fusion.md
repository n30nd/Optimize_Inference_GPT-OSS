# 12 — Small kernels: merging them, giving them enough workers, skipping dead work

## The problem in plain words

Once the big matrix multiplies run on the matrix cores, a decode step of gpt-oss-20b takes about
52 ms for 512 requests on one GCD. The many small steps around them (normalizing, adding results
back, choosing experts, sorting tokens by expert) then add up to about 10 % of that time. Each one
is cheap alone, but each reads and writes the whole batch's numbers from memory once more, and some
were written so that only a handful of the GPU's calculators had anything to do.

Four simple ideas fixed most of it:

1. **Merge kernels that touch the same data** (kernel fusion): do two passes' work in one pass.
2. **Give a kernel enough workers**: split each request's work across a wavefront (64 threads)
   instead of one thread.
3. **Sort in parallel**: one block per expert instead of one thread per expert.
4. **Skip work nobody needs**: requests that already ended do not need their experts computed.
5. **Keep only what the next step reads**: the LM head's 201,088 scores per request are only
   used to pick the largest one, so the multiply picks it itself and never writes the scores.

## An everyday comparison

A clerk who walks to the archive to fetch a file for "add the totals", walks back, then walks again
for "check the totals", wastes the walk. Merging the two jobs means one walk. And a task split
among a whole team finishes sooner than one person doing it while the team watches.

## The words

- **kernel fusion** — merging two small GPU programs into one, so the data is read from memory once.
- **wavefront** — 64 threads that run in lockstep on AMD GPUs.
- **reduction** — combining many values into one (a sum, or the largest value); within a wavefront
  this is done with *shuffles*, threads reading each other's registers directly.
- **residual stream** — the 2,880 numbers per request that every layer adds its result into.
- **LM head** (language-model head) — the last multiply: it gives every word of the vocabulary
  (201,088 of them) a score; greedy decoding then takes the word with the highest score (*argmax*).
- **epilogue** — the end of a matrix-multiply kernel, where the finished results leave the registers.
- **atomic maximum** (`atomicMax`) — a memory operation that replaces a value with a larger one in one
  indivisible step, so many blocks can update the same place without losing an update.

## How it works

**1. Fusion.** Each pair below used to be two kernels; now it is one, with exactly the same
arithmetic in the same order, so the output bits do not change:

```
before                                       after
attention-out + bias ─► x += out ─► norm(x)   attention-out + bias ─► [x += out; t = norm(x)]
expert sum ─► x += sum ─► next layer norm(x)  [x += sum; t = norm(x) for the next layer]
mlp1 ─► write 47 MB fp32 ─► SwiGLU ─► bf16    [mlp1 with SwiGLU in its last step] ─► bf16
```

The trick that keeps the bits identical: each thread handles the same elements (thread t takes
t, t + 256, t + 512, …) and adds them in the same order as the separate kernels did.

**2. Top-4 with a wavefront per request.** Choosing each token's 4 experts used one thread per
token: 512 tokens filled 8 small blocks, and for 120b each thread scanned 128 experts 4 times. Now 64
threads share a token: each holds some experts, the wavefront finds the best one with shuffles, marks
it taken, and repeats 4 times; one thread then applies the same weighting as the CPU code.

**3. Grouping in parallel.** Sorting the (token, choice) pairs by expert used one block with one
thread per expert, each scanning all 2,048 pairs twice. Now one 256-thread block per expert counts
its pairs, then places them, using wavefront "ballots" (each thread votes yes/no, and a bit count
tells every thread how many yes votes came before it). Same order as before.

**4. Finished requests get no expert.** In a wave of 512 requests, a few end early (their answer
hit the end token). Their rows are still in the batch, but the router now gives them expert −1,
"none", so the expert multiplies, which are half of each step, skip them. 3.1 % of the row-steps of
a 4,096-request run belong to finished requests.

**5. The LM head picks the next token.** Before, the LM head wrote 512 × 201,088 scores in fp32
(412 MB) per step, and a second kernel read them all back only to find each row's largest. Now the
epilogue of the LM-head multiply does the search:

```
before:  multiply ─► write 412 MB of scores ─► argmax reads 412 MB ─► 512 token ids
after:   multiply ─► [each tile: best score of each of its rows] ─► atomicMax on 512 keys ─► ids
```

Each block's tile covers 128 rows and 128 columns. It puts its scores through shared memory 16 rows
at a time, so that every thread can scan a quarter of one row; lane shuffles and shared memory then
combine the quarters and the waves. The block ends with one atomic maximum per row on a 64-bit *key*.
The key holds the score in its high half, rearranged so that bigger scores give bigger integers,
and the column with its bits flipped in its low half. A bigger key is then exactly "higher score,
or the same score in a lower column": the same choice argmax_kernel makes. A maximum gives the same
answer in any order, so the result does not depend on which block finishes first.

## The numbers for our model

| Change | Kernel time | End to end (8 GCDs, 4,096 requests) |
|---|---|---|
| mlp1 + SwiGLU (+ gather for 20b) fused | 947 → 802 µs per layer (20b, B=512) | 20b +4.1 % |
| Top-4, wavefront per request | 54 → 14.5 µs (32 experts), 194 → 15.7 µs (128 experts) | 20b +1.0 %, 120b +9.7 % (with the fusion above) |
| Grouping, block per expert | 318 → 14.4 µs per layer | (part of a +37 % row) |
| Finished requests skip the experts | — | 20b +0.8 %, 120b +1.2 % |
| Adds fused with the norms | 2 passes over the residual stream fewer per layer | 20b +0.9 %, 120b +0.3 % |
| LM head picks the token | 4,897 + 435 → 4,510 + 6 µs (B=512); 9,719 + 865 → 9,223 + 6 µs (B=1024) | 20b +1.3 %, 120b +0.9 % |

Every row above was checked to leave the output bits unchanged (same output hash).

## Where it is in our code

- `getp-csrc/kernels/fast_misc.hpp` — `add_rmsnorm_bf16_kernel`, `rms_inv_row`, `rmsnorm_store_bf16`.
- `getp-csrc/kernels/moe_ep.hpp` — `moe_count_ep_kernel`, `moe_place_ep_kernel`,
  `moe_combine_rmsnorm_kernel`, `moe_combine_partial_rmsnorm_kernel`.
- `getp-csrc/kernels/moe.hpp` — `router_topk_wave_kernel` (expert −1 for finished requests).
- `getp-csrc/kernels/fast_moe_gemm.hpp` — `gemm_grouped_mxfp4_swiglu`, `gemm_grouped_mxfp4_gather_swiglu`.
- `getp-csrc/kernels/fast_gemm.hpp` — `argmax_key`, `TileGemm::argmax_rows`, `gemm_argmax_kernel`,
  `gemm_bf16_argmax`; checked by `getp-csrc/tests/lm_head_argmax_test.cpp` (same token as the
  unfused pair on every row, planted ties included).
- `getp-csrc/engine/gpu_engine.hpp` — `layer_attention`, `layer_route`, `layer_combine`, and the
  done flags in `gpu_run_wave`.

## Pitfalls

- A merged kernel must keep each thread's element order, or the sums change in their last bit.
- A fused argmax must break ties exactly like the old one (lowest column), treat -0 and +0 as equal
  and never pick a NaN, or a rare tie changes a token and every token after it.
- Our first guess at "the GPU is idle 7.5 % of the time" was wrong: the idle time was the weight
  upload before inference, which is not timed. Measure only the timed part.

## Round two: less waiting

### The problem in plain words

After the first round, a profile of the whole run still showed the small kernels taking 1.5 to 3
times the time their bytes need. A copy of 21 MB takes about 17 µs on one GCD; `add_rmsnorm` moves
20 MB and took 26 µs, `rope_kv` moves 20 MB and took 22 µs, the expert combine moves 38 MB and took
51 µs. Grouping (count + place) and top-4 move almost nothing and still took 15 and 16 µs.

The time went into *waiting*, not into work. Five fixes, all with the same arithmetic as before, so
the output bits do not change:

1. **Ask for all the numbers first.** The old loops did "load, wait, add, store, next". Now each
   thread asks for all of its numbers at once, then computes, then stores.
2. **Keep the code short.** A small kernel's code is no longer in the instruction cache when it
   starts (a whole layer of other kernels ran since its last call), so long code costs real time.
3. **Rotate the query where it is read.** The attention kernel reads the query anyway; it now
   applies RoPE itself, so `rope_kv` only handles keys and values (2 MB instead of 20 MB).
4. **Count and place in one pass** when grouping the (token, expert) pairs.
5. **Let the attention kernel's first loads go out before it reads the query**, and each step's
   next loads go out while it computes (software pipelining).

For gpt-oss-120b, whose experts are split between two GCDs, one more: **send the activations to
the partner once instead of letting it fetch them row by row.**

### An everyday comparison

A cook who fetches one ingredient, cooks it, then walks back for the next, spends most of the
evening walking. Writing the whole shopping list first and fetching everything in one trip is the
first fix. The second: a recipe that fills ten pages takes longer to *read* than a one-page recipe,
even if the cooking is the same.

### The words

- **latency** — how long one request to memory takes before the answer arrives (about 1 to 2 µs
  here). Many requests can be on their way at the same time; a thread that waits for each answer
  before asking the next one pays the full latency every time.
- **memory-level parallelism** — having many memory requests on their way at once, so their waits
  overlap.
- **instruction cache** — a small fast memory next to the compute units that holds the code being
  run. When a kernel's code is not in it ("cold"), every 64 bytes of code must first be fetched
  from main memory.
- **scratch memory** — per-thread memory the compiler uses when an array cannot stay in registers
  (for example, when it is indexed by a number only known at run time). Every use is a slow memory
  access.
- **software pipelining** — starting the loads for the next round of a loop before the current
  round's work, so the loading and the computing overlap.
- **scan (prefix sum)** — giving each thread the sum of the values of all threads before it; here,
  "how many pairs of this expert come before mine".
- **link** — the connection between the two GCDs of one MI250 card (about 150-180 GB/s, ten times
  less than a GCD's own memory).
- **RoPE** (rotary position embedding) — the step that turns pairs of numbers of each query and key
  by an angle that depends on the token's position, so attention can tell near words from far ones.
- **shared memory** (LDS, local data share) — a small fast memory that the threads of one block
  share; used here to pass partial sums between the block's wavefronts.
- **buffer descriptor** — a small record (start address and size) that a load goes through; a load
  outside the size returns zero instead of touching memory.
- **rocprofv3** — AMD's profiler; it records when every kernel of a run starts and ends.

### How it works

**1. All loads first.** On this GPU a thread's wait for one load also waits for every store it
issued before, so a loop "load, store, load, store" waits for each store as well. The new kernels
load everything a thread needs into registers (12 numbers of x, 12 of y, 12 weights for a row of
2,880 handled by 256 threads), then compute, then store. To make the compiler issue the loads
together, the loop counts are compile-time constants (one version for the 2,880 of gpt-oss, a
general one for other sizes) and loads past the end of a row read the last element again instead of
being skipped. The sum of squares of a row still goes through the same halving tree of 256 values:
the first two levels through shared memory, the last six inside a wavefront with shuffles, which
add the same pairs in the same order.

**2. Short code.** We found this one by accident: a kernel that did the router multiply and the
top-4 in one go took 24 µs in a test that called it again and again, but 45 µs inside the real run.
The test kept its 17 KB of code in the instruction cache; the real run, with a whole layer of other
kernels between two calls, did not. A test option (`--evict`) that runs other kernels between calls
reproduced it. So the top-4 and grouping kernels are written with short loops and no scratch
memory, and the merged router + top-4 was dropped (it saved a kernel start but its long code cost
as much).

**3. The query is rotated inside the attention kernel.** RoPE turns each pair of numbers
(i, i + 32) of a 64-wide head by an angle that depends on the position. Each attention lane already
loads dims d and d + 32 of the query, exactly the pairs RoPE needs, so it rotates them right after
loading, with the same two rounding steps as before (`rope_pair`). `rope_kv` now reads only the
keys and values (2 MB instead of 10.5 MB, and writes nothing back to the query).

**4. Grouping in one pass.** Each of the 32 (or 64) blocks, one per expert, reads all the pairs
once, 8 or 16 per thread with 16-byte loads. It counts the pairs of lower experts (that sum is the
expert's first row) and marks its own pairs; a scan over the 256 threads gives each of its pairs a
row, in pair order, exactly as the two old kernels did.

**5. Attention loads early.** Each attention wave now asks for its first keys and values before it
reads the query, then, inside the loop, asks for the next step's keys right after the scores and
the next values at the end of the step. Positions past the end read zeros (the buffer descriptors
already guarantee that), so the extra last request is harmless.

**6. (120b) Activations sent once.** The expert multiply of one GCD needs the normalized
activations of the partner's tokens that chose its experts. It used to read them across the link
row by row: about 1,000 rows (5.9 MB) per layer, most tokens twice (two of their four experts are
on each side). Now the kernel that computes the activations (`add_rmsnorm`) also writes them once
(2.9 MB) into the partner's memory, and the partner reads them locally. The existing signals
between the two GCDs already guarantee the order (written before "routed", rewritten only after the
partner's "experts done").

### The numbers for our model

Per call on one GCD inside the real run (rocprofv3), 512 requests, 1,024 steps (20b) or one pair of
GCDs, 128 steps (120b):

| Kernel | 20b before → after | 120b before → after |
|---|---|---|
| add + RMSNorm | 25.8 → 20.0 µs | 26.8 → 20.8 µs (30.7 with the copy to the partner) |
| rope_kv (keys and values only now) | 22.3 → 9.4 µs | 22.8 → 10.7 µs |
| attention, full layers past 128 positions | 419.0 → 402.7 µs | — |
| top-4 | 16.2 → 12.9 µs | 18.8 → 14.5 µs |
| count + place → one kernel | 5.6 + 9.8 → 9.1 µs | 14.9 + 18.9 → 14.4 µs |
| gather of the expert inputs | (inside mlp1) | 42.8 → 18.6 µs |
| partial sums for the partner | — | 23.0 → 18.5 µs |
| combine + RMSNorm | 51.4 → 35.6 µs | 70.7 → 60.6 µs |
| GPU busy time | 46.5 → 45.0 s (−3.2 %) | 10.62 → 10.41 s (−2.0 %) |

End to end (official input, 8 GCDs, old and new builds alternated twice): gpt-oss-20b
78,773 → 80,059 tok/s (+1.6 %), gpt-oss-120b 44,200 → 45,210 tok/s (+2.3 %). In the full runs
every package is at its power limit, so the small kernels' share is smaller than on one GCD.
Every change leaves the output hash unchanged; `getp-csrc/tests/small_kernels_test.cpp` compares
each new kernel with the one it replaced, bit for bit.

### What did not help

- **Router multiply + top-4 in one kernel.** Same bits, 5 µs faster in a warm test, but slower in
  the run (cold code, and a long serial tail in the last block of each tile).
- **Pushing the 120b partial sums to the partner too.** The 5.9 MB still cross the link; the cost
  only moved from the partner's combine to the signal that follows (it must flush them).
  Fine-grained memory for these buffers made the writes wait for the link inside the kernels.
- **A shallower load ring in the router multiply** (shorter code): no measurable change.

### Where it is

- `getp-csrc/kernels/fast_misc.hpp` — `add_rmsnorm_bf16` (with `load_row256`, `block_sum256`,
  `rms_inv_row`, `rmsnorm_store_bf16`), `rope_kv_bf16`.
- `getp-csrc/kernels/fast_common.hpp` — `rope_pair`, the one place the rotation is written.
- `getp-csrc/kernels/fast_attention.hpp` — `attention_gqa8_kernel`: `ROT`, `load_k` / `load_v`.
- `getp-csrc/kernels/moe.hpp` — `router_topk_wave_kernel`.
- `getp-csrc/kernels/moe_ep.hpp` — `moe_group_ep`, `moe_gather_ep_bf16`, `moe_partial_ep`,
  `moe_combine_rmsnorm`, `moe_combine_partial_rmsnorm`.
- `getp-csrc/engine/gpu_engine.hpp` — `t16_peer` and the calls above.
- `getp-csrc/tests/small_kernels_test.cpp` (bit-for-bit test against
  `getp-csrc/tests/small_kernels_ref.hpp`, the kernels before) and
  `getp-csrc/bench/small_kernels_bench.cpp` (timing, `--evict`).

### Pitfalls

- A loop guarded by a run-time bound (`if (j < H)`) splits the code into small pieces, and the
  compiler then waits for all outstanding memory operations at each piece. Use compile-time counts
  and clamp addresses instead of skipping loads.
- Array indexed by a run-time variable → scratch memory. Unroll so every index is a constant, or
  keep values in a different form (one bit per element in a word, one value per lane).
- Measure small kernels in the real run or with other kernels between calls: a test that calls one
  kernel in a loop keeps its code and data in the caches and can rank two versions the wrong way.

## Further reading

- NVIDIA developer blog, *Using shared memory and warp shuffles for reductions* (the same ideas on
  AMD wavefronts).
- AMD ROCm docs, *HIP cross-lane intrinsics* (`__shfl_xor`, `__ballot`).
