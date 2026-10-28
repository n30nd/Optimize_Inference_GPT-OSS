# Lesson 21: Where the expert multiply's time goes (measurement only)

| | |
|---|---|
| Reference commit | `64f5f16` "docs/08: where the expert GEMM's time goes; bench replays real routing" |
| Starts from | `9b2edc9` + `7faed85` (lesson 20) |
| To type | no program code; 26 lines of benchmark (optional); 67 lines of docs (optional) |
| Shortcut | `git cherry-pick 64f5f16` — copies the finished commit instead of typing (section 5) |
| Speed after | unchanged: 20b 75,811 tok/s, 120b 42,113 tok/s |

**In plain words:** the expert multiply, the biggest single cost, ran at about half of what the
chip can do on paper. Before trying yet another fix, we split the missing half into measured
pieces: a slower clock, matrix units that sit idle part of the time, and work done on padding that
nobody needs. Then we tried the obvious fixes for the small pieces, and they did not pay. This
lesson changes no program code; it changes what we work on next.

## 1. Goal

Explain the measured rate of the 20b mlp1 expert multiply (95 of 181 TFLOP/s) as a product of
measured factors, find which part is worth attacking, and make the benchmark replay real 120b
routing instead of an invented one.

## 2. Why this, why now

- After lessons 10, 11, 15 and 19 the expert multiplies were still about half of every step.
- Each further guess had to be checked anyway (lesson 19's first guess was wrong). A full account
  of the missing half tells us in advance which guesses can pay at all.
- The benchmark's 120b routing was **synthetic** (made up with the same unevenness as the measured
  20b routing). Real 120b routing turned out much more uneven, and unevenness is exactly what
  creates padding and a long queue tail.

## 3. The technique

New words:
- **peak** — 181 TFLOP/s: every matrix unit of the GCD finishing one multiply instruction (MFMA)
  every cycle at the maximum clock of 1.7 GHz.
- **clock / cycle** — the GPU's heartbeat; one cycle is one tick. Under full load the clock drops
  below its maximum (lesson 20: the power cap).
- **busy %** — share of cycles in which the matrix units have work issued, read from hardware
  counters (rocprofv3, AMD's profiler). It counts padding work as busy.
- **padding** — multiplies on zeros, done because the work comes in fixed-size pieces: rows in
  tiles of 16, columns in tiles of 256, k in steps of 128.
- **queue tail** — the end of a persistent kernel's queue (lesson 10), when some blocks have
  finished and wait for the last items.
- **routing dump** — a file with, for every layer and step of a real run, how many rows each expert
  got (the `offsets` table). **Replay** = feed those exact shapes to the benchmark.

```
 20b mlp1, 512 requests: useful work 2 × 2,048 × 5,760 × 2,880 = 68 GFLOP in 715 µs = 95 TFLOP/s

 181 TFLOP/s   peak (1.7 GHz, matrix units busy every cycle)
   × 0.86      clock under load: 1.05 M cycles in 720 µs = 1.46 GHz
   × 0.70      matrix units busy: 9.62 M MFMAs × 32 cycles / (416 SIMDs × cycles) = 70.3 %
   × 0.86      useful share of the issued work: 67.95 of 78.8 GFLOP
                 = rows 0.895 (16-row tiles) × columns 0.978 (22.5 tiles of 256) × k 0.978 (22.5 steps)
 =  94 TFLOP/s  ≈ the 95 measured

 the 30 % idle time, per block, 20b mlp1 / mlp2:
   queue tail 3.6–3.7 % / 7.3–8.4 %   gaps between items 1.4 / 2.4 %   inside items 93.8 / 88.3 %
   → most of it is stalls INSIDE items (~24 %): waiting for weights (small items), LDS reads (big)
```

How the benchmark replays real 120b routing (`MOE_DUMP120`): each dump line is one GCD's 64
experts at one layer and step (`rows e_begin layer offsets[0..64]`, 72 lines per step = 36 layers ×
2 GCDs). The bench skips the first 16 positions (the prompts begin with the same shared text, so
every request routes the same way there) and takes every 37th line after that, so consecutive calls
mix layers and both GCDs — as in a real step. Without the file it falls back to the synthetic
routing.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/bench/moe_gemm_bench.cpp` | +26 −3 | `MOE_DUMP120=<path with %d for B>`: read real 120b routing lines, skip the first 16 positions, keep every 37th line | optional |
| 2 | `docs/08-moe-grouped-gemm-mxfp4.md` | +67 −1 | section 9 "Where the other half of the peak goes"; three more rejected ideas in the table | optional |

Read while typing (line numbers at `64f5f16`):
- `getp-csrc/bench/moe_gemm_bench.cpp:76` `read_dump()` — the 20b dump format, `B pos layer
  offsets[0..32]`.
- `getp-csrc/bench/moe_gemm_bench.cpp:98` `synth_offsets()` and `:188` — the synthetic routing:
  expert popularity from a gamma distribution, shape 0.45, chosen to match the measured 20b
  unevenness.
- `getp-csrc/bench/moe_gemm_bench.cpp:215` — `n++ >= 16 * 72 && n % 37 == 0`: skip 16 positions,
  then every 37th line; `:221` the synthetic fallback.
- `docs/08-moe-grouped-gemm-mxfp4.md:446` — section 9.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick 64f5f16
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout 64f5f16 -- $(git show --name-only --format= 64f5f16)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 64f5f16
git show 64f5f16 -- getp-csrc/bench/moe_gemm_bench.cpp

git diff 64f5f16 -- $(git show --name-only --format= 64f5f16)   # must print nothing
make runomp                                                    # nothing in the program changed
```

Quick checks: the program is unchanged, so the fingerprints are lesson 20's: 20b `446027378dd3`,
120b on a GCD pair `e205be8f59c8`.

The benchmark (one GCD) needs routing dumps, which are **not** in the repository (they were written
by a temporary print in the engine). They are kept on this machine:
- 20b: `/home/namvu/claude_working/claude_job/gpt-oss/final-project/results/moe_routing_20b/`
  `off_b512.txt`, `off_b1024.txt`, `off_b2048.txt` (`B pos layer offsets[0..32]` per line);
- 120b, for `MOE_DUMP120`: `/home/namvu/claude_working/claude_job/gpt-oss/final-project/results/moe_routing_120b/`
  `off120_b512.txt`, `off120_b1024.txt`.

```bash
R=/home/namvu/claude_working/claude_job/gpt-oss/final-project/results
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/moe_gemm_bench.cpp -o moe_gemm_bench
D20="$R/moe_routing_20b/off_b512.txt $R/moe_routing_20b/off_b1024.txt $R/moe_routing_20b/off_b2048.txt"
MOE_ONLY="20b mlp1 512" HIP_VISIBLE_DEVICES=2 ./moe_gemm_bench $D20
MOE_ONLY="120b mlp1 512" MOE_DUMP120=$R/moe_routing_120b/off120_b%d.txt \
  HIP_VISIBLE_DEVICES=2 ./moe_gemm_bench $D20
```

Seen on the reference build (one GCD, other GCDs busy):

```
routing unevenness (std/mean of rows per expert): real 20b B=512 1.29, synthetic 20b B=512 1.22
model shape     B |    old us    new us   new min |   TFLOP/s    GB/s speedup |   lib est   lib avg |  vs lib
20b   mlp1    512 |   13052.4     707.9     666.6 |      96.0     429   18.4x |    1433.3    1555.2 |   2.02x
120b  mlp1    512 |   16316.0     834.6     662.1 |      78.0     597   19.5x |    2358.5    2877.2 |   2.83x
```

"old" is the first, simple version of lesson 04; "lib est" is the vendor library's estimated time
for the same expert shapes (bf16 weights, one call per expert). The 20b line is the 96 TFLOP/s this
lesson takes apart.

Save: `git add -A -- . ':!walkthrough' && git commit -m "docs/08: where the expert GEMM's time goes; bench replays real routing"`

## 6. Result

No speed change; what the measurement found:

| Where the other half of the peak goes (20b mlp1, 512 requests) | factor |
|---|---:|
| clock under load 1.46 GHz instead of 1.7 | × 0.86 |
| matrix units busy 70.3 % of the cycles | × 0.70 |
| useful share of the issued work (padding of rows, columns, k) | × 0.86 |
| product: 181 → 94 TFLOP/s (measured: 95) | |

| Row padding (16-row tiles) | 512 requests | 1,024 requests |
|---|---:|---:|
| 20b | 10.5 % | 5.6 % |
| 120b split (real routing) | 19.2 % | 10.6 % |

## 7. Decisions and reasons

- **Decided:** rows stay padded to 16. **Why:** a smaller multiply shape would add the products in
  another order, so the output bits would change; the padding is the price of bit-identical
  results.
- **Not chosen — skipping the column padding** (the wavefronts past N only copy A instead of
  multiplying zeros): fused mlp1 −0.2 to −0.9 %, but plain mlp2 +1.3 % and uniform 48/64-row mlp2
  +12 / +9 %. **Why:** the time was not waiting on those MFMAs.
- **Not chosen — splitting small experts' items to shorten the queue tail:** as two halves on 4
  wavefronts each, +0.2 to +2.4 %; with 8 wavefronts of one 16-column piece each, the tail did
  shrink (20b mlp1 3.6 → 2.2 %) but calls ended −0.6 to +1.1 %. **Why:** each extra item repeats its
  A copy, queue ticket, scale loads and epilogue (about 12 µs).
- **Not attempted — the gaps between items** (1.4–2.4 %): taking a ticket earlier would make the
  queue tail longer.
- **Decided:** the next work on this kernel goes to the ~24 % of stalls *inside* items (lesson 24
  returns to the kernel), not to padding or the queue.
- **Decided:** replay real 120b routing in the benchmark. **Why:** the real routing is much more
  uneven than the synthetic one, and unevenness drives both the row padding (19.2 % on real 120b
  routing) and the queue tail, so synthetic numbers would mislead 120b decisions.

## 8. Questions to think about

1. "Busy 70 %" counts padding MFMAs as busy. Why is it still useful to split *busy* and *useful*
   into two separate factors?
2. Why does row padding fall from 10.5 % to 5.6 % when the batch doubles from 512 to 1,024?
3. Shortening the queue tail made the tail smaller but the call not faster. What does that say
   about measuring a part of a kernel versus the whole kernel?

## 9. Discussion notes

(empty: we write here during the walk-through)
