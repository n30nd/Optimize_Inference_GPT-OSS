# Lesson 25: Small kernels, round two — ask for everything first, wait less

| | |
|---|---|
| Reference commit | `9eb944f` "Small kernels: loads first, count and place in one kernel, query rotated in attention" |
| Starts from | `25d09b2` (lesson 24) |
| To type | 648 lines of program code in 6 files (232 removed) — typed in 5 parts below; 1,365 lines of tests and benchmarks and 160 of docs (optional) |
| Speed after | 20b 80,000 tok/s (+1.4 %), 120b 45,200 tok/s (+2.3 %) |

**In plain words:** between the big multiplications run a dozen small jobs: add two lists of
numbers, normalise them, turn the query, pick each word's 4 experts, sort the words by expert.
Each moves only a few megabytes, yet took 1.5 to 3 times as long as its bytes need. The time went
into *waiting*: a worker asked for one number, waited, used it, asked for the next. Now each
worker asks for all its numbers at once. Same arithmetic, same answers, a little faster.

## 1. Goal

Make the small kernels before and after the expert multiply take about the time their bytes
need, without changing a single output bit:
- add + RMSNorm, the expert combine + RMSNorm, the 120b gather and partial sums: all loads first;
- RoPE: rotate the query inside the attention kernel, so `rope_kv` only handles keys and values;
- grouping: count and place in one kernel; top-4: no scratch memory, short code;
- attention: issue the next keys and values while computing the current step;
- 120b: the norm writes the expert input straight into the partner GCD's memory.

## 2. Why this, why now

A profile of the build after lesson 24 (one GCD, 512 requests, 20b; one pair for 120b):
- 20b per layer: router 23.5 + top-4 16.3 + count 5.7 + place 9.9 + rope 22.6 + add_rmsnorm
  25.9 = 104 µs of 1.90 ms (5.5 %); 120b 137 µs of 2.46 ms (5.6 %).
- After the experts: 20b combine + RMSNorm 51.4 µs per layer (2.7 %), although its ~35 MB would
  take ~30 µs; 120b gather 42.6 + partial sums 22.7 + combine 70.6 + signal 14 + wait 52.4 =
  202 µs per layer (8.3 %), much of it reads over the link between the two GCDs.
- Their data is small (router scores 64 KB, rope 10.5 MB, add_rmsnorm ~20 MB). A copy of 21 MB
  takes about 17 µs, so by bytes they could take about half their time.
- The expert multiply (54–57 % of the time) had just had two rounds of work (lessons 19 and 24).

## 3. The technique

New words:
- **latency** — how long one memory request takes before the answer arrives (about 1–2 µs here).
  Many requests can be on their way at once; a thread that waits for each answer before asking
  the next pays the full latency every time.
- **memory-level parallelism** — having many memory requests on their way at once, so their waits
  overlap.
- **instruction cache** — a small fast memory near the compute units holding the code being run.
  A small kernel's code is usually not in it ("cold": a whole layer of other kernels ran since its
  last call), so every 64 bytes of code is first fetched from main memory.
- **scratch memory** — per-thread memory in HBM the compiler uses when an array cannot stay in
  registers (for example, indexed by a number known only at run time).
- **software pipelining** — starting the loads of the next loop round before the current round's
  work, so loading and computing overlap.
- **scan (prefix sum)** — giving each thread the sum of the values of all threads before it; here,
  "how many pairs of this expert come before mine".
- **RoPE** (rotary position embedding) — turning pairs of numbers of each query and key by an
  angle that depends on the token's position.
- **push / pull** — the GCD that produces data writes it into the partner's memory (push), instead
  of the partner reading it across the link (pull).

```
 before (one thread of add_rmsnorm, 12 elements):
   load x0 ─wait─ load y0 ─wait─ add, store ─ load x1 ─wait─ … ─ 12 × (2 latencies)
   (on this GPU a wait for a load also waits for every earlier store)

 after:
   load x0..x11, y0..y11, w0..w11  ─── one wait ───  add all, square, sum, store all
```

The sum of squares still goes through the same halving tree of 256 values (first levels through
the LDS, the last six inside a wavefront with shuffles that add the same pairs in the same order),
so the bits stay. Loads past the end of a row read the last element again instead of being
skipped: a skipped load is a branch, and a branch makes the compiler wait for everything.

The query is rotated where it is read:

```
 before:  rope_kv: read q,k,v (10.5 MB) → rotate q,k → write q back, write k,v to the cache
          attention: read q …
 after:   rope_kv: read k,v (2 MB) → rotate k → write k,v to the cache
          attention: each lane loads dims d and d+32 of q (exactly a RoPE pair) → rope_pair → use
```

`rope_pair` (one shared function) does the rotation with the same two roundings as before, so q
gets the same bits.

## 4. The change, file by file (typing order)

Type it in five parts. After each part, check just that part:
`git diff 9eb944f -- <files of the part>` must print nothing. The program builds only after
part E (the engine calls the new signatures).

| Part | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| A | `getp-csrc/kernels/fast_common.hpp` | +9 −1 | `rope_pair`: the one place the rotation is written | yes |
| A | `getp-csrc/kernels/fast_misc.hpp` | +176 −69 | `block_sum256`, `load_row256`, `rms_inv_row`, `rmsnorm_store_bf16`, `add_rmsnorm_bf16_kernel` (loads first, optional second copy `out2`); `rope_kv_bf16` for keys and values only | yes |
| B | `getp-csrc/kernels/fast_attention.hpp` | +92 −43 | `attention_gqa8_kernel` with `ROT`: rotate the query as it is loaded; `load_k` / `load_v` of the next step issued early | yes |
| C | `getp-csrc/kernels/moe.hpp` | +38 −25 | `router_topk_wave_kernel`: choice j kept by lane j (no scratch arrays), short loops | yes |
| D | `getp-csrc/kernels/moe_ep.hpp` | +296 −65 | `moe_group_ep_kernel` (count and place in one kernel, with a scan); `moe_gather_ep_bf16` (16-byte copies); `moe_partial_ep`, `moe_combine_rmsnorm`, `moe_combine_partial_rmsnorm` with loads first | yes |
| E | `getp-csrc/engine/gpu_engine.hpp` | +37 −27 | the new calls; `t16_peer`, the buffer the partner's norm writes into | yes |
| — | `getp-csrc/tests/small_kernels_ref.hpp` | +326 | the kernels as they were before, to compare against | optional |
| — | `getp-csrc/tests/small_kernels_test.cpp` | +427 | every new kernel against the old one, bit for bit | optional |
| — | `getp-csrc/tests/peer_write_test.cpp` | +78 | a GCD sees what its partner pushed into its memory, even with old lines in its cache | optional |
| — | `getp-csrc/tests/attn_test.cpp` | +1 −1 | calls `rope_kv_bf16` the new way | optional |
| — | `getp-csrc/bench/small_kernels_bench.cpp` | +533 | old against new timing, in the engine's order; `--evict` makes the code cold | optional |
| — | `docs/12-small-kernels-and-fusion.md` | +160 −1 | "Round two: less waiting" | optional |

Read while typing (line numbers at `9eb944f`):
- Part A: `getp-csrc/kernels/fast_misc.hpp:175` `block_sum256` (same additions as the old tree),
  `:198` `load_row256` (the clamped index instead of a skipped load), `:238-258`
  `add_rmsnorm_bf16_kernel` (three `load_row256`, then compute, then stores).
- Part B: `getp-csrc/kernels/fast_attention.hpp:247-248` (first K and V loads before the query is
  read), `:263-274` (the query rotated with `rope_pair`), `:310` and `:380` (next step's loads).
- Part C: `getp-csrc/kernels/moe.hpp:64` `router_topk_wave_kernel`.
- Part D: `getp-csrc/kernels/moe_ep.hpp:135-146` (the comment explains the one-pass grouping and
  why its code is kept short), `:146` `moe_group_ep_kernel`.
- Part E: `getp-csrc/engine/gpu_engine.hpp:500-504` — the norm also writes into the partner's
  `t16_peer`; `:564` the gather reads that local copy.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 9eb944f
git show 9eb944f -- getp-csrc/kernels/fast_misc.hpp        # part A, and so on

git diff 9eb944f -- $(git show --name-only --format= 9eb944f)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 24)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 10e37a089467
```

Every new kernel against the one it replaced (one GCD, about 30 s):

```bash
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/tests/small_kernels_test.cpp -o small_kernels_test
HIP_VISIBLE_DEVICES=2 ./small_kernels_test
```

Expected (seen on the reference build): 101 checks, each ending in `identical`, then

```
router_topk_wave_kernel B=33 E=16 k=8 done rows                  identical
all identical
```

Pushed writes are seen by the partner (two GCDs of **one card**, under a second):

```bash
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/tests/peer_write_test.cpp -o peer_write_test
HIP_VISIBLE_DEVICES=4,5 ./peer_write_test
```

Expected: `iters 3000, 262144 ints (1024 KB): stale values seen 0`.

Attention test (one GCD, about 40 s), still passing with the new `rope_kv_bf16`:

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/attn_test.cpp -o attn_test
HIP_VISIBLE_DEVICES=2 ./attn_test             # last line: ALL PASS
```

Old against new timing, with cold code (one GCD, under 10 s):

```bash
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/small_kernels_bench.cpp -o small_bench
HIP_VISIBLE_DEVICES=2 ./small_bench --evict
```

Seen on the reference build (µs per call, "after pre" = right after the kernel that precedes it in
the engine; each line is preceded by `bit-identical`):

```
add_rmsnorm                  after pre: old   27.89  new   27.28 us ( -2.2%)
rope_kv                      after pre: old   27.90  new   24.33 us (-12.8%)
  top-k alone                after pre: old   22.71  new   12.79 us (-43.7%)
group 20b (32 experts)       after pre: old   17.05  new    8.60 us (-49.6%)
combine_rmsnorm              after pre: old   51.17  new   36.58 us (-28.5%)
```

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-25 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-25 --score
```

Expected: 20b about 80,000 tok/s, fingerprint `58023f92ead7`; 120b about 45,200 tok/s,
fingerprint `973e97c40a85` (both unchanged).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Small kernels: loads first, count and place in one kernel, query rotated in attention"`

## 6. Result

Per call inside the real run (profiler; 20b: one GCD, 512 requests; 120b: one pair), µs:

| Kernel | 20b before → after | 120b before → after |
|---|---|---|
| add + RMSNorm | 25.8 → 20.0 | 26.8 → 20.8 (30.7 with the copy to the partner) |
| rope_kv (keys and values only now) | 22.3 → 9.4 | 22.8 → 10.7 |
| attention, full layers past 128 positions | 419.0 → 402.7 | — |
| top-4 | 16.2 → 12.9 | 18.8 → 14.5 |
| count + place → one kernel | 5.6 + 9.8 → 9.1 | 14.9 + 18.9 → 14.4 |
| gather of the expert inputs | (inside mlp1) | 42.8 → 18.6 |
| partial sums for the partner | — | 23.0 → 18.5 |
| combine + RMSNorm | 51.4 → 35.6 | 70.7 → 60.6 |

Whole program, official input, 8 GCDs, old and new builds alternated twice:

| | before | after |
|---|---:|---:|
| 20b | 78,857 tok/s | **80,000 tok/s (+1.4 %)** |
| 120b | 44,163 tok/s | **45,200 tok/s (+2.3 %)** |
| Output | | same fingerprints; every new kernel bit-identical to the old one |

In the full runs every card sits at its power limit, so the small kernels' share is smaller than
on one GCD alone, and the gain end to end is smaller than the per-kernel numbers suggest.

## 7. Decisions and reasons

- **Decided:** the small kernels before the experts in this round, not the experts again.
  **Why:** they were 5.5 % (20b) / 5.6 % (120b) of a step and could lose about half of it by
  their bytes; the expert multiply had just had two rounds. **Not chosen:** int8 keys in the cache
  (would change the answers; quality cost measured by another team); attention (its full-window
  kernel already reads at ~1.1 of the ~1.25 TB/s possible).
- **Decided:** keep the same arithmetic in the same order everywhere. **Why:** then a bit-for-bit
  test against the old kernels proves each change, and the full-run fingerprint stays.
- **Decided:** short code for top-4 and grouping. **Why:** a router multiply + top-4 merged into
  one kernel took 24 µs in a test loop but 45 µs in the real run: the test kept its 17 KB of code
  in the instruction cache, the real run did not. That merged kernel was dropped.
- **Decided (120b):** the norm pushes the expert input (2.9 MB) into the partner once, instead of
  the partner pulling about 1,000 rows (5.9 MB, most tokens twice) across the link. The existing
  signals already order the writes and the reads. `tests/peer_write_test.cpp` checks that a pushed
  write is never hidden by an old copy in the reader's cache.
- **Not chosen:** pushing the 120b partial sums too (the same 5.9 MB still cross the link; the cost
  only moved into the signal that follows); fine-grained memory for these buffers (the writes then
  waited for the link inside the kernels); a shallower load ring in the router multiply (no
  measurable change).

## 8. Questions to think about

1. Why does "ask for all 36 numbers, then wait once" beat "ask, wait, use" by so much, when the
   number of bytes read is the same?
2. A test that calls one kernel in a loop ranked two versions the wrong way round. What does such
   a test keep warm that the real run does not, and how does `--evict` fix it?
3. Rotating the query inside attention cuts `rope_kv`'s reads from 10.5 to 2 MB and its query
   write-back to nothing, but adds work to the attention kernel. Why is that a good trade here?

## 9. Discussion notes

(empty: we write here during the walk-through)
