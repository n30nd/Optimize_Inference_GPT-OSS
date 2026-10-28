# Lesson 08: The ordinary multiplications on the matrix units

| | |
|---|---|
| Reference commit | `30dd744` "Run the dense matrix multiplies on the matrix cores" |
| Starts from | `c61cc5d` (lesson 07) |
| To type | 511 lines of program code in 1 file (+511 −47); 558 lines of tests and benchmarks (optional); 267 lines of docs (optional) |
| Shortcut | `git cherry-pick 30dd744` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 7,858 tok/s together with lesson 09 (was 5,741); measured as one step |

**In plain words:** most of the model's arithmetic is "multiply a table of numbers by another
table". Our first version did it one multiply-and-add at a time. The chip has special units that
multiply whole small blocks in one instruction, 30 times faster on paper. This lesson feeds them:
keep the next pieces ready exactly when the units are free, and give every unit the same amount
of work.

## 1. Goal

Run the four "dense" multiplies of every step — qkv, attention-out, router and the final LM head —
on the matrix cores, at least at 80 % of AMD's own library speed.

## 2. Why this, why now

- The time profile of lesson 04 (one step, 512 requests, one GCD: 670 ms) put the dense multiplies
  at **29 %** of a step, second after the expert multiplies (68 %).
- The first kernel reached about 6 TFLOP/s; the matrix cores' peak is 181 TFLOP/s on one GCD.
- The yardstick of lesson 02 says AMD's library reaches 105–128 TFLOP/s on our shapes, so the gap
  was known to be about 20×.

## 3. The technique

New words:
- **matrix multiply (GEMM)** — C = A · Wᵀ: each output is the sum of K products of a row of A with
  a row of W. **M, N, K** are its sizes: M rows of A (the requests), N rows of W (outputs per
  request), K numbers summed per output.
- **CU (compute unit)** — one of the GCD's 104 groups of calculators, each with its own LDS (the
  64 KB fast scratch memory) and 4 **SIMDs** (each runs one wave at a time and has one matrix core).
- **MFMA** — the matrix-core instruction; here `v_mfma_f32_32x32x8bf16_1k`: one wave multiplies a
  32 × 8 block of A by an 8 × 32 block of Wᵀ and adds the 32 × 32 result into fp32 sums kept in
  its **registers** (each lane's own tiny, fastest storage).
- **tile** — the piece of the result one block of threads computes, e.g. 128 × 128 outputs.
- **K step** — one slice of the sum; a block handles 32 (or 64) k values at a time.
- **barrier** — a point where all threads of a block wait for each other (`__syncthreads`).
- **double buffering** — two copies of a scratch area: one is used while the other is filled.
- **prefetch** — asking memory for data before it is needed, so it has arrived by then.
- **bank conflict** — LDS is split into 32 banks; two lanes asking the same bank wait in turn.
- **split K** — several blocks share one tile, each summing part of K, then add their parts in a
  fixed order (so the bits are the same every run).
- **hipBLASLt** — AMD's own matrix library: our yardstick, never part of `./run`.

```
                 K (2880) ───────────────────────────────►
   A rows   ┌──────────┬──────────┬──────────┬───
   m0..     │ step 0   │ step 1   │ step 2   │ ...      each K step: 32 k values
   m0+128   └──────────┴──────────┴──────────┴───
                  │ copy to LDS (two buffers: while one is read, the next step fills the other)
                  ▼
   LDS: sA[2][128 x 32]   sW[2][128 x 32]
                  │ each wave reads its 16-byte pieces
                  ▼
   cols:  0-31   32-63  64-95  96-127
   rows   ┌──────┬──────┬──────┬──────┐   128 × 128 tile, 8 waves, each 64 × 32:
   0-63   │ w0   │ w2   │ w4   │ w6   │   two blocks per CU = 4 waves per SIMD, so while one
   64-127 │ w1   │ w3   │ w5   │ w7   │   wave waits, others keep the matrix core busy
          └──────┴──────┴──────┴──────┘

 one K step in time:
 matrix core : [ MFMA ×8 (chunk 0) ][ MFMA ×8 (chunk 1) ]
 LDS reads   :   operands of chunk 1  |  chunk 0 of the next step
 LDS writes  :   next step → other buffer
 memory loads:   the step after next (prefetch)
 barrier     :                       ▲ one per step
```

Three more details:
- **XOR layout in LDS:** piece c of row r is stored at `c XOR ((r / 2) % 4)`, so the 8 lanes
  reading 8 rows hit 8 different bank groups — no bank conflicts.
- **Instruction order hints:** the compiler moved the LDS copy to just before the barrier;
  `__builtin_amdgcn_sched_group_barrier` tells it "one MFMA, one memory instruction, …".
- **A tile size per shape:** how many tiles a multiply has decides the winner (qkv at 512 rows:
  128 × 128 → 160 tiles, all run at once; 256 × 256 → 40 tiles, 64 of 104 CUs idle).
  `gemm_pick_config` holds the measured best per shape and batch. The narrow router (32 or 128
  outputs) gets its own split-K kernel.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_gemm.hpp` | +511 −47 | `lds_slot` (XOR layout), `wait_flag`/`set_flag` (split-K order), `TileGemm` with `mainloop` (the pipeline), `load_bias`, `store_c`; `gemm_mfma_kernel` (one block per tile, optional split K); `gemm_splitk_kernel` (router); `kGemmConfigs` and `gemm_pick_config`; `gemm_bf16()` keeps its old signature | yes |
| 2 | `getp-csrc/tests/gemm_test.cpp` | +238 | every gpt-oss shape and odd sizes against an fp64 CPU sum; split-K runs repeated must give the same bits | optional |
| 3 | `getp-csrc/bench/gemm_bench.cpp` | +198 | times every shape; `--old` adds the first kernel, `--sweep` every configuration | optional |
| 4 | `getp-csrc/bench/gemm_old_fma.hpp` | +65 | the first kernel, kept for `--old` | optional |
| 5 | `getp-csrc/bench/gemm_torch.py` | +57 | the library numbers | optional |
| 6 | `docs/07-matrix-cores-gemm.md` | +267 | the full story | optional |

The engine does not change: it already called `gemm_bf16()`.

Read while typing (line numbers at `30dd744`, file `getp-csrc/kernels/fast_gemm.hpp`):
- `:9-18` — which lane gives which row and k to the MFMA (the comment is the key to the code).
- `:64` `lds_slot()` — the XOR layout.
- `:131` `mainloop()` — the pipeline; `:221-238` the instruction-order hints.
- `:474` `gemm_pick_config()` and `:524` `gemm_bf16()` — the choice per shape.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick 30dd744
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout 30dd744 -- $(git show --name-only --format= 30dd744)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 30dd744
git show 30dd744 -- getp-csrc/kernels/fast_gemm.hpp
git diff 30dd744 -- $(git show --name-only --format= 30dd744)   # must print nothing
make runomp
```

Quick check, 20b (one GCD):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # b2d8ec70d2ff
```

The fingerprint changes from lesson 07: the sums are now added in another order (fp32 sums in
tiles instead of one long sum).

Quick check, 120b (one pair; 120b uses the same dense multiplies):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 5d44468050b5
```

Kernel test and benchmark (commands from the files' first lines):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/gemm_test.cpp -o gemm_test
tools/gpulock.sh 2 -- ./gemm_test
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/gemm_bench.cpp -o gemm_bench
tools/gpulock.sh 2 -- ./gemm_bench --old --ms 512
```

`gemm_test` checks 80 cases by default (every shape at every batch, odd sizes, with and without
bias) and ends with:

```
80 cases, 0 failed, worst rel err 7.99e-06 -> PASS
```

`--all-configs` checks every kernel configuration too, with and without split K (about 50 s):
`1214 cases, 0 failed, worst rel err 7.99e-06 -> PASS`.
`gemm_bench --old --ms 512` prints, among others (µs per call, TFLOP/s):

```
qkv            512    5120  2880  old fp32 FMA                   2336.7      6.5
qkv            512    5120  2880  128x128x32 2x4 ks1              154.0     98.1
lm_head        512  201088  2880  old fp32 FMA                  70684.1      8.4
lm_head        512  201088  2880  128x128x32 2x4 ks1             4912.2    120.7
```

Full run (20b, 8 GCDs, includes lesson 09; about 10 minutes):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-08 --score
```

The logged number (7,858 tok/s, METEOR 0.557, BERTScore 0.980, fingerprint `a5315bcc37ed`) was
measured with lessons 08 and 09 together, so run it after lesson 09 to compare.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Run the dense matrix multiplies on the matrix cores"`

## 6. Result

One GCD, 512 requests, µs per call (library = hipBLASLt through PyTorch):

| Multiply (N × K) | before | after | library | % of library |
|---|---:|---:|---:|---:|
| qkv 5120 × 2880 | 2,353 | **156** | 161 | 103 % |
| attention-out 2880 × 4096 | 2,195 | **121** | 125 | 103 % |
| router 20b 32 × 2880 | 460 | **21.1** | 28.2 | 134 % |
| LM head 201088 × 2880 | 71,085 | **4,894** | 7,191 | 147 % |

- All dense multiplies of one 20b step at 512 requests: **191 → 12.1 ms**.
- 91–147 % of the library at every shape and batch from 64 to 2,048 rows; best 121–123 TFLOP/s
  (LM head), about 68 % of the matrix cores' peak.
- Accuracy: 1,214 test cases, largest error 8.0 × 10⁻⁶ of the typical output size; split K gives
  the same bits on every run.

End to end (20b, 8 GCDs, 4,096 prompts, **lessons 08 and 09 together**):

| | before | after |
|---|---:|---:|
| tok/s | 5,741 | **7,858** (+37 %) |
| METEOR / BERTScore | 0.549 / 0.979 | 0.557 / 0.980 |

Why only +37 % when the dense multiplies got 16× faster: they were 29 % of the step. The
remaining step is now almost all the simple expert multiply (lesson 10).

## 7. Decisions and reasons

- **Decided:** rebuild the dense multiplies on the matrix cores, alongside attention and the
  expert multiply. **Why:** the profile gave them 29 % of a step; the expert multiply (68 %) is
  the bigger part and comes in lesson 10. **Not chosen:** tuning the small kernels first — together
  they were under 3 %.
- **Decided:** 8 waves per 128 × 128 tile, a measured tile size per shape, split K only for small
  batches. **Why:** 8 waves alone gave 3–9 % (more waves to hide waiting); the right tile count
  for 104 CUs matters more than any single trick; split K helps only when there are too few tiles
  (attention-out at 64 rows: 55.7 → 44.9 µs).
- **Not chosen:**
  - **Stream-K** (every block gets an equal share of all (tile, K step) work): correct, but each
    block walked its own tiles, so the L2 cache (8 MB of recently read data shared by the GCD)
    stopped helping: LM head at 2,048 rows 19.2 → 51 ms. It comes back, in a narrower form, in
    lesson 22.
  - loading 2 K steps ahead instead of 1: ±1 %;
  - tile widths that divide N exactly (128 × 192, 128 × 96): 11–119 % slower, because the bigger
    LDS use leaves one wave per SIMD.
- **Rule that follows:** split-K flags are one array per GCD, so two split-K multiplies must never
  run at the same time on one GCD. The engine uses one stream per GCD, which guarantees it.

## 8. Questions to think about

1. A 128 × 128 tile reads 128 rows of A and 128 rows of W per K step and does 128 × 128 × 32
   multiply-adds. Why does a bigger tile need fewer bytes per multiply-add — and why is bigger
   still not always faster?
2. qkv at 512 rows has 160 tiles of 128 × 128 for 104 CUs, two blocks per CU. Draw which CUs
   get two tiles and which get one. What does that mean for the time? (Lesson 22 comes back to it.)
3. Split K must add its parts "in a fixed order". What would go wrong with the fingerprint check
   if the order depended on which block finished first?

## 9. Discussion notes

(empty: we write here during the walk-through)
