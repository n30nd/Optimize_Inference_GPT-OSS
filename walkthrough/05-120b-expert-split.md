# Lesson 05: 120b does not fit one GCD — split its experts over a card's two GCDs

| | |
|---|---|
| Reference commit | `d28c548` "Split gpt-oss-120b's experts between the two GCDs of each card" |
| Starts from | `3163e03` (lesson 04) |
| To type | 562 lines of program code in 2 files (+421 −179 in the engine, 141 new); 36 lines of tests (optional); 91 lines of docs (optional) |
| Shortcut | `git cherry-pick d28c548` — copies the finished commit instead of typing (section 5) |
| Speed after | 120b 1,188 tok/s — the first GPU run of 120b, measured on the **exact** path; 20b unchanged (5,217) |

**In plain words:** the big model is slightly larger than one GPU die's memory. Most of it is the
"experts", and each word only uses 4 of the 128. So the two dies of one card each keep half of the
experts and everything else in full. Each die serves its own questions; only in the expert step
do the two dies hand work to each other.

## 1. Goal

Run gpt-oss-120b on the GPUs at all, with the same exact numbers as the CPU, using all 8 GCDs.

## 2. Why this, why now

- 120b stored compactly (experts in MXFP4) is **65 GB**; one GCD has **64 GB**. It does not fit,
  even before any room for the requests' memory.
- 20b already ran on the GPUs (lessons 03–04); 120b was still the CPU at 0.40 tok/s.
- We split as little as possible: 95 % of the bytes are experts, and the rest (attention,
  router, embeddings: 4.2 GB) is small enough to copy to both GCDs.

## 3. The technique

New words:
- **expert parallelism (EP)** — splitting the experts of each layer over several GPUs. Here
  EP = 2: GCD A holds experts 0–63, GCD B experts 64–127, of every layer.
- **data parallelism** — copies working side by side on different requests, never talking. The
  8 GCDs form 4 *groups* (pairs); groups are data parallel, the two GCDs of a pair are not.
- **package (card)** — one MI250 card: two GCDs joined by their fastest link. HIP devices
  (0,1), (2,3), (4,5), (6,7) are the pairs on our machines.
- **peer access** — one GPU reading another GPU's memory directly over the link
  (`hipDeviceEnablePeerAccess`).
- **stream** — a queue of GPU work; the GPU runs it in order. The engine uses one per GCD.
- **event** — a marker put into a stream (`hipEventRecord`). Another stream can be told "wait
  until that marker is passed" (`hipStreamWaitEvent`); the GPU itself waits, the host does not.
- **KV cache** — the stored keys and values of all earlier tokens of each request (lesson 06
  explains them); it grows with every token.

One layer on a pair, for one step:

```
        GCD A (experts 0–63)                         GCD B (experts 64–127)
        its own 512 requests                         its own 512 requests
 attention, router for its rows               attention, router for its rows
        │ topk_A, t_A ready ── event ──┐  ┌── event ── topk_B, t_B ready │
        ▼                              ▼  ▼                              ▼
 group the rows that chose 0–63           group the rows that chose 64–127
 from BOTH A's and B's tokens              from BOTH A's and B's tokens
 (reads topk_B, t_B in B's memory)         (reads topk_A, t_A in A's memory)
 run experts 0–63 → Z_A                    run experts 64–127 → Z_B
        │ Z_A ready ── event ──┐        ┌── event ── Z_B ready               │
        ▼                      ▼        ▼                                    ▼
 for each own token: add its 4 expert     same on B
 outputs, reading rows from Z_A or Z_B
```

Two waits per layer, both on the GPU. One host thread drives both GCDs of a pair, so it always
queues the marker before the wait that refers to it. One copy of each shared buffer is enough:
a GCD overwrites `t`, `topk` or `Z` only in the next layer, after waiting for a marker its
partner sets once it has finished reading them.

Each GCD **reads** (pulls) its partner's buffers instead of the partner writing (pushing) into
it: another team saw a receiving GPU's cache keep stale contents after pushed copies.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe_ep.hpp` | +141 (new) | `moe_group_ep_kernel` (group own and partner rows by local expert), `moe_gather_ep_kernel<T>` (copy each row's input from own or partner memory), `moe_combine_ep_kernel<kExact>` (add each own token's 4 outputs, some read from the partner) | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +421 −179 | `GpuWorker` gains `e_begin`/`e_count`, `peer`, two events; new `GpuGroup`; `gpu_pair()`; the step cut into `step_embed`, `layer_attention`, `layer_route`, `layer_experts`, `layer_combine`, `step_logits`; `gpu_step_group()`; waves served per group (`WaveSlot`, `gpu_run_wave`); `gpu_group_size()`; groups built in `gpu_warm_up()` | yes |
| 3 | `getp-csrc/tests/gpu_vs_cpu.cpp` | +36 −51 | `EP=2` splits the experts over GPUs 0 and 1 in the bit-exact test | optional |
| 4 | `docs/11-expert-parallelism.md` | +91 | this lesson's technique for newcomers | optional |

Read while typing (line numbers at `d28c548`):
- `getp-csrc/kernels/moe_ep.hpp:7-8` — a row refers to an own pair as `p` and a partner pair as
  `-(p + 1)`; the sign tells the kernels whose memory to read.
- `getp-csrc/engine/gpu_engine.hpp:302` `gpu_pair()` — peer access switched on in both directions.
- `getp-csrc/engine/gpu_engine.hpp:448` `gpu_step_group()` — the three loops per layer; the
  markers at lines 462 and 471, the waits at 468 and 477.
- `getp-csrc/engine/gpu_engine.hpp:602` `gpu_group_size()` — 2 when the weights would fill more
  than 60 % of a GCD; `GPTOSS_EP` overrides.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick d28c548
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout d28c548 -- $(git show --name-only --format= d28c548)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat d28c548
git show d28c548 -- getp-csrc/kernels/moe_ep.hpp
git diff d28c548 -- $(git show --name-only --format= d28c548)   # must print nothing
make runomp
```

Quick check, 20b (one GCD; 20b is not split, so nothing may change):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # b4c73f55a422, the same as lesson 04
```

Quick check, 120b (one pair = two GCDs of one card; this runs the default fast path):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 78d2c8fd9f66
```

The startup line says `fast path, 2 GPU(s) in 1 group(s) of 2 …`.

Bit-exact test with the experts split (build and run commands from the test's first lines):

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
EP=2 tools/gpulock.sh 2,3 -- ./gpu_vs_cpu $M20 4 16      # 20b, 4 prompts x 16 steps
EP=2 tools/gpulock.sh 2,3 -- ./gpu_vs_cpu $M120 2 12     # 120b, 2 prompts x 12 steps
```

Expected: one line per step with `logits differing 0`, then the last line (about 35 s for 20b,
about 1 minute for 120b, most of it the CPU side):

```
pos  15: logits differing 0 / 804352, max |diff| 0, argmax differing 0
PASS: GPU logits bit-identical to the CPU
```

Full run, 120b. **The first 120b number was taken on the exact path** (fp32 cache, so only 256
requests per GCD fit: two waves of 256), hence `GPTOSS_EXACT=1` in front. At 1,188 tok/s it takes
about an hour:

```bash
GPTOSS_EXACT=1 GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-05 --score
```

Expect about **1,188 tok/s**, METEOR **0.988**, BERTScore **0.99987**, fingerprint `33c21d6e726b`.
The 20b full run is unchanged (lesson 04: 5,217 tok/s).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Split gpt-oss-120b's experts between the two GCDs of each card"`

## 6. Result

| 120b | before | after |
|---|---|---|
| Where it runs | CPU | 8 GCDs = 4 pairs |
| tok/s (official 4,096 prompts) | 0.40 | **1,188** (×3,000) |
| METEOR / BERTScore | — | 0.988 / 0.99987 |
| Answers identical to the reference | — | 3,992 of 4,096 lines |
| Logits vs the CPU, experts split over 2 GCDs | — | bit-identical (20b 4 × 16, 120b 2 × 12) |

Memory of one 120b GCD:

| | per GCD |
|---|---:|
| all dense weights (bf16) | 4.2 GB |
| half of the experts (MXFP4) | 30.4 GB |
| KV cache, 512 requests, bf16 (fast path) | 21.8 GB |
| read from the partner per layer, on average: inputs (bf16) / expert outputs (fp32) | ≈ 5.9 MB / ≈ 11.8 MB |

**This first 120b number is the exact path.** The exact path keeps its KV cache in fp32 (twice
the bytes), so only 256 requests per GCD fit: the 1,188 run served each GCD's 512 requests in two
waves of 256. The fast path also runs at this commit (the quick check above), but its first
full-run number comes in lesson 07: 2,835 tok/s.

## 7. Decisions and reasons

- **Decided:** split each layer's experts between the two GCDs of one card (EP = 2), four such
  pairs side by side; each GCD keeps all dense weights and serves its own requests.
  **Why:** half the experts plus all dense weights is 34.6 GB per GCD, leaving room for 512
  requests' bf16 KV cache (21.8 GB). The two GCDs of a card have the fastest link. Each GCD reads
  only ~6 MB + ~12 MB of its partner's data per layer.
  **Not chosen:**
  - spreading the experts over all 8 GCDs (EP = 8): every token's work crosses slower links to
    many GPUs, and every layer waits for 8 of them; another team measured 63 % of the time spent
    waiting on that exchange;
  - **tensor parallelism** (cutting every matrix of every layer in two): it needs an exchange
    in attention too, every layer.
- **Decided:** the first 120b run uses the exact path. **Why:** the same rule as lesson 03 —
  correct first. With bit-identical logits, any later difference is a real bug, not rounding.
- **Decided:** one host thread per pair, GPU-side waits (events). **Why:** with one thread per
  GCD, one thread could queue "wait for the partner's marker" before the partner's thread queued
  the marker, and the wait would pass at once.
- Measured later: splitting 20b the same way (so each expert sees twice the rows) was **4 %
  slower** — the bigger multiplies did not pay for reading the partner's data. 20b stays one
  model copy per GCD.

## 8. Questions to think about

1. Why split the experts and not the attention? Count what each GCD would have to send to its
   partner in each case.
2. A GCD reads its partner's `t` buffer during the expert step. What stops the partner from
   overwriting `t` with the next layer's values while it is still being read?
3. Each GCD runs its 64 experts on rows from 1,024 tokens (its own and its partner's). Why is that
   better for the expert multiply than 512 tokens on 128 experts?

## 9. Discussion notes

(empty: we write here during the walk-through)
