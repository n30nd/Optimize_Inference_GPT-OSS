# Lesson 22: Share the work evenly over all 104 compute units

| | |
|---|---|
| Reference commit | `1131a42` "Stream-K for qkv and attention-out; router K parts on more CUs" |
| Starts from | `64f5f16` (lesson 21) |
| To type | 319 lines of program code in 1 file (62 removed); 54 lines of tests and benchmarks and 145 of docs (optional) |
| Shortcut | `git cherry-pick 1131a42` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 76,149 tok/s (+0.3 %), 120b 42,176 tok/s (+0.1 %) |

**In plain words:** a GPU die has 104 workers. When a job is cut into 160 equal pieces, 56 workers
get two pieces and 48 get one, so for half of the time almost half of the workers stand idle. We
now cut the job into many small steps and give every worker the same number of steps. The
answers stay exactly the same.

## 1. Goal

Make the two dense multiplies of attention (qkv, and attention-out at some batch sizes) and the
router multiply use all 104 compute units evenly, without changing a single output bit.

## 2. Why this, why now

A profile of the 20b run (one GCD, 512 requests) after lesson 20:

- qkv took 7.7 % of the time at 96.8 TFLOP/s, attention-out 6.1 % at 99.6 TFLOP/s, while the LM
  head (the last, biggest multiply) reached 121 TFLOP/s with the same kernel. Same kernel, slower
  rate: the shapes, not the code, were the problem.
- qkv at 512 requests has 160 tiles of 128 × 128 outputs for 104 compute units: 56 units get two
  tiles, 48 get one. Attention-out has 92 tiles: 12 units get none.
- The router multiply (32 or 128 outputs per row) ran as 16 tiles for 20b at 512 requests, so only
  16 of the 104 compute units did all its reading.

## 3. The technique

New words:
- **compute unit (CU)** — one of the 104 independent workers of a GCD; each runs a few blocks of
  threads at once.
- **tile** — the block of outputs (here 128 × 128) one block of threads computes, adding over all
  of K (the 2,880 or 4,096 numbers each output sums over).
- **K step** — one slice of 64 of those numbers; a qkv tile is 45 K steps.
- **wave quantization** — the time lost when the tile count does not fill the GPU's rounds evenly:
  the last round is only partly full, so some CUs finish early and wait.
- **persistent kernel** — a kernel that starts exactly one block per CU and lets each block work
  through a list of jobs, instead of one block per tile.
- **stream-K** — a persistent kernel that shares out the K steps of all tiles evenly, so a block
  may do the end of one tile and the start of the next.
- **accumulator** — the running fp32 sums of a tile, kept in registers while K steps are added.
- **flag** — a number in GPU memory one block sets and another block waits for.
- **atomic add** — an addition memory does as one indivisible step, so when several blocks add 1
  to the same counter at once, every one is counted.

Stream-K for qkv at 512 requests: 160 tiles × 45 steps = 7,200 steps, 104 workers → 69.2 each.

```
 K steps of one row of tiles:  0          45         90         135        180
                               | tile 0   | tile 1   | tile 2   | tile 3   | ...
 worker 0 (steps 0-68)         [  full    ][ head ]
 worker 1 (steps 69-137)                          [tail][  full    ][h]
 worker 2 (steps 138-206)                                           [ tail ][ head ...

 head: start of a tile  → write the raw fp32 sums to scratch, set a flag
 tail: rest of the tile → wait for the flag, load those sums, keep adding
```

Keeping every bit: the usual stream-K sums each part of a tile from zero and adds the parts at
the end, which changes the order of the additions. Here the tail *continues* the head's sums with
the same instructions in the same order, so the result is what one block alone would give. Each
worker does its head first, its whole tiles next and its tail last, so by the time it reaches its
tail the previous worker's head has long finished.

The limit: a tile's steps still run one after another. If a worker's share is shorter than a
tile, some tile is split three ways and the third worker waits for the second. `stream_fits`
picks stream-K only where every worker gets at least one tile's worth of steps.

The router: each of the 4 K parts of a 32 × 32 tile is now its own one-wavefront block (64
blocks for 20b instead of 16), so a tile's reads come from 4 CUs. Each part writes its sums to
scratch; the last part to finish (found with an atomic counter) adds the 4 in the old fixed
order: same bits.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_gemm.hpp` | +319 −62 | `TileGemm::mainloop` can continue from sums handed in (`keep_acc`); `save_acc` / `load_acc`; `gemm_stream_kernel` and `launch_stream`; `gemm_router_kernel` (one wave per K part, last block adds); `stream_fits` and the new rows of `gemm_pick_config` | yes |
| 2 | `getp-csrc/tests/gemm_test.cpp` | +21 −8 | every run without split K must give exactly the bits of configuration 0; stream configurations also run with 3 workers per CU | optional |
| 3 | `getp-csrc/bench/gemm_bench.cpp` | +33 −4 | `--interleave`: time a call between two attention-out multiplies, as in the engine | optional |
| 4 | `docs/07-matrix-cores-gemm.md` | +145 −21 | sections 6 (router) and 7 (wave quantization and stream-K) | optional |

Read while typing (line numbers at `1131a42`):
- `getp-csrc/kernels/fast_gemm.hpp:150-182` — `mainloop` with `keep_acc`: the only change to the
  tile loop is "do not zero the sums when continuing".
- `:307` `save_acc` and `:319` `load_acc` — the raw accumulators, in a layout only these two use.
- `:500-547` `gemm_stream_kernel` — how a worker finds its steps `[b, e)`, its head, whole tiles
  and tail; `wait_flag` / `set_flag` at `:534` and `:543`.
- `:648` `gemm_router_kernel` — the atomic counter and the fixed-order sum of the 4 parts; `:828`
  `stream_fits`, `:838` `gemm_pick_config`.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick 1131a42
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout 1131a42 -- $(git show --name-only --format= 1131a42)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 1131a42
git show 1131a42 -- getp-csrc/kernels/fast_gemm.hpp       # the change to copy

git diff 1131a42 -- $(git show --name-only --format= 1131a42)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 21)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: e205be8f59c8
```

Unit test (one GCD, about 15 s; `--all-configs` about 20 s):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/gemm_test.cpp -o gemm_test
HIP_VISIBLE_DEVICES=2 ./gemm_test
HIP_VISIBLE_DEVICES=2 ./gemm_test --all-configs
```

Expected (seen on the reference build): the qkv row at 512 requests says
`stream 128x128x64 2x4 ks1 rel err 5.52e-06 ok (same bits)`, and the last lines are

```
80 cases, 0 failed, worst rel err 7.99e-06 -> PASS
1490 cases, 0 failed, worst rel err 7.99e-06 -> PASS          (with --all-configs)
```

"same bits" = the output is identical, bit for bit, to the plain tiled kernel's.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-22 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-22 --score
```

Expected: 20b about 76,100 tok/s, fingerprint `58023f92ead7`; 120b about 42,200 tok/s,
fingerprint `141e0b9dbc17` (both unchanged, so the scores are unchanged too).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Stream-K for qkv and attention-out; router K parts on more CUs"`

## 6. Result

Per call on one GCD, µs (from `docs/07` section 7 and the commit message):

| Multiply | requests | before | after |
|---|---:|---:|---:|
| qkv | 384 | 152 | 122 |
| qkv | **512** | **155** | **141** |
| qkv | 768 / 1,024 | 288 / 299 | 200 / 264 |
| attention-out | 384 / 768 / 1,024 | 115 / 217 / 229 | 104 / 164 / 214 |
| attention-out | 512 (the official run) | 121 | 121 (stream-K not used: 125) |
| router 20b, 512, between two big multiplies | | 30.0 | 18.6 |
| router 120b, 512, between two big multiplies | | 30.4 | 25.3 |

Whole program, official input, 8 GCDs, old and new builds alternated twice:

| | before | after |
|---|---:|---:|
| 20b | 75,911 tok/s | **76,149 tok/s (+0.3 %)** |
| 120b | 42,129 tok/s | **42,176 tok/s (+0.1 %, noise level)** |
| Output | | same fingerprints (`58023f92ead7`, `141e0b9dbc17`) |

Inside the engine the gain is about half the standalone one (qkv 156 → 148 µs per call), and
these multiplies are a small part of a step next to the expert multiplies.

## 7. Decisions and reasons

- **Decided:** stream-K that hands the raw sums on, instead of the usual "add the parts at the
  end". **Why:** the output keeps every bit, so a full-run fingerprint is enough to show nothing
  broke. **Not chosen:** the usual form: the order of additions changes, so every run would need
  new quality scores.
- **Decided:** use stream-K only where `stream_fits` says every worker gets a whole tile of steps.
  **Why:** attention-out at 512 requests (92 tiles of 64 steps, 55 steps per worker) splits tiles
  three ways and measured 125 µs against 121 for the plain kernel; smaller tiles (128 × 64,
  64 × 128) measured 126–128 µs. So it keeps the plain kernel there.
- **Decided:** group the workers by row of tiles, so workers at the same place in their row read
  the same weight tile at the same time. **Why:** an earlier stream-K version let every worker walk
  its own run of tiles; the GPU then touched 208 weight tiles at once, the L2 cache (the GPU's
  shared on-chip copy of recently read memory) stopped helping, and the LM head at 2,048 requests
  went from 19.2 to 51 ms. It was removed.
- **Not chosen:** two workers per CU (208): each gets fewer steps than a tile, three-way splits
  again, and the registers spill: qkv 155 → 169–179 µs. Chaining all pieces of a worker through
  one loop (loading the next piece while writing the last): 1–7 % slower.
- **Kept although small:** the end-to-end gain is +0.3 % / +0.1 %. It costs no accuracy and no
  bits, and the router change helps exactly where the engine runs it (between big kernels).

## 8. Questions to think about

1. 208 tiles is 30 % more work than 160 but took only 18 % more time (184 against 156 µs). What
   does that tell you about how much of the GPU was idle at 160 tiles?
2. Why does "hand the raw sums to the next worker" keep the bits, while "each part sums from zero,
   add the parts at the end" does not, although both add the same numbers?
3. The standalone qkv gain was 155 → 141 µs, inside the engine only 156 → 148, and the whole run
   +0.3 %. Where did the rest of the gain go?

## 9. Discussion notes

(empty: we write here during the walk-through)
