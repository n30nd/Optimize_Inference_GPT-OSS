# Lesson 19: Let small expert jobs fetch first

| | |
|---|---|
| Reference commit | `fa89592` "Raise the wave priority of small expert items; zero-pad the mlp1 input" |
| Starts from | `027549a` + `6b315b5` (lesson 18) |
| To type | 30 lines of program code in 2 files; 142 lines of benchmark (optional); 184 lines of docs (optional) |
| Shortcut | `git cherry-pick fa89592` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 74,746 tok/s, 120b 41,737 tok/s (8 GCDs, official input) |

**In plain words:** we suspected one kind of memory read in the expert multiply was wasteful, and
measured it: the suspicion was wrong. Measuring piece by piece showed instead that experts with
few tokens spend most of their time waiting for their weights to arrive. So now, inside each
compute unit, the workers that are about to ask for weights go first. Same results bit for bit; a
small gain on 120b, none measurable on 20b. The lesson is as much about checking a guess as about
the fix.

## 1. Goal

Find out what still limits the expert multiply after the repacking of lesson 15, and act only on
what the measurements support.

## 2. Why this, why now

- The expert multiplies were still about **53 %** of a step.
- A test build without the loads of A (the token rows) was about 10 % faster, as was one without
  the weight loads. And each 64-byte line of A is read by two 32-byte load instructions (two
  neighbouring lanes share a line). The plan: load whole lines with one instruction, halving the
  requests.
- That plan was tested first — and section 3 shows why it was dropped.

## 3. The technique

New words:
- **work item** — one piece of the expert multiply: up to 64 rows of one expert × 256 output
  columns. A persistent kernel (lesson 10: blocks that stay alive and take items from a queue)
  works through all items.
- **wavefront / SIMD** — 64 threads that run together / one of the 4 instruction units of a compute
  unit. Each SIMD here holds 4 wavefronts and issues one instruction at a time, normally from the
  oldest wavefront that is ready.
- **wave priority** (`s_setprio`) — a number 0–3 a wavefront sets for itself. When several are ready,
  the SIMD serves the highest priority first.
- **L1 / L2 cache** — small fast memories in front of HBM: L1 (16 KB) in each compute unit, L2
  shared by the whole GCD. Two reads of the same 64-byte line close together are served by one L2
  request.
- **ablation** — a test build with one part removed (for example, no weight loads). Its results are
  wrong, but its time shows what that part costs.
- **weight stream** — the flow of expert weights from HBM into the compute units.

```
 1. the guess:   "two 32-byte loads per 64-byte line of A = two L2 requests"
    test:        whole-line loads → L1 lookups −34 %, but L2 requests 11.89 M → 11.88 M (unchanged)
    verdict:     the L1 had already merged the two halves; time 717 → 719–724 µs. Dropped.

 2. ablations, 20b, every expert 16 rows:  kernel 314 µs; loads alone (no unpack, no MFMA) 276 µs
    verdict:     small items wait on the weight stream (88 % of their time), not on arithmetic

 3. the fix, items of ≤ 32 rows, per step of each wavefront:
       issue next step's loads  │  unpack + MFMAs  │  store A, barrier
       priority 2               │  priority 1      │  priority 0
    a wave that just passed the barrier gets its loads out before the others continue
```

Items of 48 and 64 rows keep the default priority: with priorities they got slower (48 rows up to
12 % on mlp2, 64 rows 1 %), because they are limited by their multiply loop, not by the stream.

The commit also fixes a latent bug: the fused gather + mlp1 (lesson 11) reads its last 128-k step
whole, so the last token row reads up to 64 values past the end of `t16`. Those values meet zero
weights, but if the memory there held a NaN ("not a number"), NaN × 0 is still NaN. The buffer now
has 64 extra values, set to zero once.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_moe_gemm.hpp` | +25 | `MoePrio<MSA>` (priorities by row-tile count), three `s_setprio` calls in `moe_step`, a `sched_barrier` in `moe_item` that keeps the scale loads ahead of step 0 | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +5 −1 | `t16` gets 64 extra values, zeroed | yes |
| 3 | `getp-csrc/bench/moe_stream_bench.cpp` | +142 | the weight stream alone, as the kernel reads it: the ceiling for small items | optional |
| 4 | `docs/08-moe-grouped-gemm-mxfp4.md` | +184 −18 | section 8: what limits the kernel now, and what did not help | optional |

Read while typing (line numbers at `fa89592`):
- `getp-csrc/kernels/fast_moe_gemm.hpp:315-326` — the comment and `MoePrio`: `MSA` is the number
  of 16-row tiles of the item (≤ 2 means ≤ 32 rows).
- `getp-csrc/kernels/fast_moe_gemm.hpp:359`, `:370`, `:402` — priority 2 before the loads, 1
  before the MFMA loop, 0 before the A store and barrier.
- `getp-csrc/kernels/fast_moe_gemm.hpp:469-474` — why the order of the loads at the start of an
  item matters for every turn of the loop (the compiler's wait counts).
- `getp-csrc/engine/gpu_engine.hpp:288-292` — the zero padding of `t16`.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick fa89592
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout fa89592 -- $(git show --name-only --format= fa89592)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat fa89592
git show fa89592 -- getp-csrc/kernels/fast_moe_gemm.hpp   # the change to copy, file by file

git diff fa89592 -- $(git show --name-only --format= fa89592)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 18)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: e205be8f59c8
```

Priorities only change *when* instructions issue, never their operands, so the bits cannot change.

Weight-stream ceiling (optional, one GCD):

```bash
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/moe_stream_bench.cpp -o moe_stream_bench
HIP_VISIBLE_DEVICES=2 ./moe_stream_bench
```

Seen on the reference build (one GCD, while other GCDs were busy), first and last lines:

```
736 items x 360 KB = 271 MB of codes
grid 104  depth 1  barrier 1  tile-major    203.4 us   1334 GB/s
...
grid 208  depth 1  barrier 1  tile-major    242.3 us   1120 GB/s     <- what the kernel does
grid 208  depth 2  barrier 1  tile-major    237.7 us   1142 GB/s     <- loading further ahead: ~same
grid 208  depth 1  barrier 1  step-major    210.0 us   1292 GB/s
```

The kernel's own setup (208 blocks, one step ahead) streams about 1,120 GB/s; loading further ahead
barely moves it.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-19 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-19 --score
```

Expected: 20b about 74,750 tok/s, fingerprint `58023f92ead7`; 120b about 41,740 tok/s, fingerprint
`141e0b9dbc17` (both unchanged from lesson 18).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Raise the wave priority of small expert items; zero-pad the mlp1 input"`

## 6. Result

| µs per call, one GCD | before | after | change |
|---|---:|---:|---:|
| 20b mlp1 / mlp2, 512 requests | 720 / 404 | **715 / 401** | −0.7 / −0.8 % |
| 120b split mlp1 / mlp2, 512 requests | 848 / 477 | **843 / 471** | −0.6 / −1.2 % |
| 20b, every expert 16 rows, mlp1 / mlp2 | 315.9 / 174.8 | **301.6 / 164.8** | −4.5 / −5.7 % |
| 20b, every expert 32 rows, mlp1 / mlp2 | 403.1 / 251.6 | **398.6 / 224.2** | −1.1 / −10.9 % |

| 8 GCDs, 4,096 requests (means of 2 alternating runs each) | before | after | change |
|---|---:|---:|---:|
| 20b | 74,850 | **74,746** | −0.1 % (noise) |
| 120b | 41,604 | **41,737** | +0.3 % |

Why 120b gains and 20b does not: a 120b GCD holds 64 experts and its experts average about 32 rows
per step, while 20b's average about 64 — exactly the size the priorities do not touch. (The quick
check with 64 prompts, where every expert has only ~8 rows, showed +2.2 % / +1.6 %; that is why
small runs must never be used to judge speed.)

## 7. Decisions and reasons

- **Decided:** test the "fewer, wider A loads" idea before building on it.
  **Why it looked right:** an ablation without A loads saved ~10 %, and each 64-byte line of A was
  read by two 32-byte loads.
  **What the measurement said:** requests reaching the L2 did not change (11.89 → 11.88 million per
  call): the L1 had already merged the halves. Time 717 → 719–724 µs. **Dropped** (the whole-line
  version was +0.3 to +3.7 % slower on the shapes tried).
- **Decided:** wave priorities for items of at most 32 rows.
  **Why:** ablations showed small items are bound by the weight stream (276 of 314 µs are loads
  at 16 rows), so getting their loads out earlier helps them: −0.3 to −1.2 % per real call, −4.5 to
  −10.9 % for 16- and 32-row experts, identical bits.
- **Not chosen** (all measured, see `docs/08` "What did not help"):
  - priorities also for 48- and 64-row items — up to 12 % and 1 % slower;
  - loading the weights two steps ahead — no gain: the stream does not get faster with more in
    flight (`moe_stream_bench` 1,116 → 1,148 GB/s);
  - a "step-major" weight order — the plain stream got 15 % faster, the kernel did not;
  - more registers per thread (fewer wavefronts) — 11–15 % slower.
- **Decided:** keep the change although 20b shows no end-to-end gain. **Why:** it is bit-identical,
  helps 120b and the small-expert cases, and costs a few lines.

## 8. Questions to think about

1. The ablation "no A loads" saved 10 %, yet halving the A load instructions saved nothing. How can
   both be true?
2. Why does a priority help an item that waits on memory but hurt an item that waits on its own
   multiply loop?
3. The quick check (64 prompts) showed +2.2 % for 20b, the full run −0.1 %. What is different about
   the experts in the two runs?

## 9. Discussion notes

(empty: we write here during the walk-through)
