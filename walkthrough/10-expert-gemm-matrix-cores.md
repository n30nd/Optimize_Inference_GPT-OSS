# Lesson 10: The expert multiplications on the matrix units (the ×8 step)

| | |
|---|---|
| Reference commit | `9813b7a` "Run the MoE expert multiplies on the matrix cores, reading MXFP4" |
| Starts from | `29f59e2` (lesson 09) |
| To type | 658 lines of program code in 1 file (`getp-csrc/kernels/fast_moe_gemm.hpp`, 14 lines removed); 869 lines of tests and benchmarks (optional); 370 lines of docs (optional) |
| Speed after | 20b 7,858 → **64,099** tok/s (×8.2); 120b 2,835 → **32,485** tok/s (with lessons 08–09) |

**In plain words:** after lessons 08 and 09, almost all of the time (96 %) went into one job:
each expert multiplying the rows of the tokens that chose it by its own big table of numbers.
We rewrite that one job so it uses the GPU's fastest calculators, unpacks the compressed
numbers right next to them, and hands out the uneven amounts of work fairly. The whole program
becomes eight times faster.

## 1. Goal

Make `gemm_grouped_mxfp4` — the multiply that runs every expert on the rows that chose it — run
on the matrix cores, reading the 4-bit weights as stored, with no work wasted on experts that
got few or no rows. Same entry point, so the engine does not change.

## 2. Why this, why now

- Profile after lessons 08–09 (20b, 512 requests, one GCD): one step took **477 ms**, and the
  simple expert multiply was **95.6 %** of it (about 9.5 ms per call, two calls per layer, 24
  layers). Dense multiplies 2.5 %, attention 0.3 %, everything else under 2 %.
- So nothing else was worth touching until this changed.
- **Why the ×8 comes from this and not from grouping:** tokens have been sorted by expert since
  lesson 03 (`moe_group_kernel`, `docs/06`), so each expert's weights were already read only
  once per step. Lesson 09 only made that sorting itself fast. What stayed slow was the
  multiply that runs *after* the sorting:
  - it used the ordinary calculators, one multiply-add at a time, not the matrix cores;
  - it unpacked the 4-bit weights to fp32 in shared memory first;
  - it launched one block for every (column tile, possible row tile, expert), sized for an
    expert that takes all 512 tokens. An average 20b expert gets 64 rows (512 × 4 ÷ 32), so most
    blocks found nothing to do and stopped at once.
  - Result: about 5 TFLOP/s on the biggest call (20b mlp1, 13,168 µs; computed from the
    doc's numbers), where the matrix cores can do 181.

## 3. The technique

New words:
- **matrix core / MFMA** — a calculator unit that multiplies small blocks of numbers in one
  instruction. `v_mfma_f32_16x16x16bf16_1k` multiplies a 16×16 bf16 block by another 16×16 block
  and adds the result into 16×16 fp32 totals. It is about 8× faster than the ordinary
  calculators (181 vs 22.6 TFLOP/s on one GCD).
- **GEMM** — "rows times a table of weights"; a **grouped GEMM** does many independent ones
  (one per expert) in one kernel.
- **mlp1 / mlp2** — the two multiplies of an expert: mlp1 (2,880 → 5,760 numbers per row),
  then SwiGLU (an activation that combines pairs of mlp1's outputs), then mlp2 (2,880 → 2,880).
- **wavefront / block / CU** — 64 threads that move in lockstep are a wavefront; a block is a
  few wavefronts on the same compute unit (CU); one GCD has 104 CUs.
- **register** — the tiny, fastest storage of each thread (4 bytes each, at most 128 per thread
  here).
- **LDS** — 64 KB of fast scratch memory per CU, shared by the threads of a block.
- **L2 cache** — an 8 MB store between the CUs and HBM, split into 16 channels; each answers
  about one request (for one 64-byte piece, a cache line) per clock tick.
- **persistent kernel** — a kernel launched with only as many blocks as fit on the GPU at once;
  each block loops, taking the next piece of work until none is left.
- **atomic add** — an add to a memory word done for one thread at a time, so every block can
  take a unique ticket from one counter.

The idea in one picture:

```
 before (simple): one block per (64-column tile, row tile 0..7, expert 0..31),
                  most of them empty (an expert averages 64 rows = 1 row tile)
                  MXFP4 → fp32 in LDS → fp32 multiply-adds, 64 × 64 tiles

 after (this lesson):
   1. every block counts each expert's 64-row tiles (offsets are on the GPU),
      ranks experts biggest first, builds the list of work items:
         expert (biggest first) → 256-column tile → 64-row tile
   2. 208 blocks (2 per CU) loop:
         ticket = atomicAdd(queue, 1)        one counter per GCD
         ticket → (expert, column tile, row tile)
         for each step of 128 k (23 steps for K = 2,880):
             A slice (64 rows × 128 k, bf16)  → LDS, shared by the block's 8 wavefronts
             16 weight bytes per lane          → registers, unpacked to bf16 with v_perm
             8 × MFMA 16x16x16 per wavefront   → fp32 totals in registers
         add bias, store (or SwiGLU, then store bf16)
   3. the last block to finish resets the queue for the next call

 work item = 64 rows × 256 columns:
                 columns n0 … n0+255
               ┌──────┬──────┬─── … ───┬──────┐
 rows m0 …     │ wf 0 │ wf 1 │         │ wf 7 │   each wavefront: 64 rows × 32 columns
 m0+63         └──────┴──────┴─── … ───┴──────┘   = 4 × 2 MFMA tiles of 16 × 16
```

Four details carry most of the speed:
- **Unpack in registers.** A 4-bit code has 8 possible sizes (0, 0.5, 1, 1.5, 2, 3, 4, 6) times
  the group's scale 2^(s−127): always an exact bf16. Two 8-byte tables (high and low byte of
  each bf16) are built per group, and `v_perm_b32` looks up 4 codes at once: about 2
  instructions per weight, and the weights travel from memory as 4 bits, not 16.
- **Every lane reads 16 neighbouring bytes.** The matrix core does not care in which order the
  k values come, as long as A and W use the same order. The kernel uses that freedom so each
  lane reads one whole group of 32 codes (16 bytes) per step.
- **Scales once per item.** The 256 columns' scale bytes sit next to each other; the block
  copies them once into LDS, then into 6 registers per lane, instead of one tiny load per step.
- **Fewer L2 requests.** The first matrix-core draft asked the L2 for 30.3 million lines per call
  in 1.88 million ticks (16 per tick: the L2's limit) and kept the matrix cores busy only 38 % of
  the time. This layout: 14.2 million, 59 % busy.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_moe_gemm.hpp` | +658 −14 | the new kernel and its entry points (parts A–D below) | yes |
| 2 | `getp-csrc/tests/moe_gemm_test.cpp` | 416 | unit test against a CPU reference in double precision | optional |
| 3 | `getp-csrc/bench/moe_gemm_bench.cpp` | 351 | speed per call on recorded routings | optional |
| 4 | `getp-csrc/bench/moe_torch_lib.py` | 102 | the library yardstick: one library call per expert | optional |
| 5 | `docs/08-moe-grouped-gemm-mxfp4.md` | 370 | the technique for newcomers | optional (read it) |

The engine is not touched: it already calls `gemm_grouped_mxfp4` (`getp-csrc/engine/gpu_engine.hpp:406`
and `:410` at `29f59e2`). Type the kernel file in four parts. Line numbers are those of the file
at `9813b7a` (`git show 9813b7a:getp-csrc/kernels/fast_moe_gemm.hpp | less -N`). Type parts A–C
*above* the old simple kernel; the old code keeps working until part D.

| Part | Lines | What | Stop and check |
|---|---|---|---|
| A — unpacking | 1–145 | the long header comment (read it slowly: it is the design), constants `MOE_MT`…`MOE_MAX_E`, `Fp4Tables`, `fp4_tables` (:104), `fp4x8_to_bf16` (:124), `as_bf16x4` | `make runomp` builds; quick check still `b2d8ec70d2ff` (nothing calls the new code yet) |
| B — the work list | 147–235 | `MoeSmem` (:151, the 32 KB of LDS), the queue counter `g_moe_queue` (:160), `MoeTiles`, `moe_tile_prefix` (:176, rank experts biggest first, prefix sum of row tiles), `moe_find_rank` (:227) | builds |
| C — one work item | 237–492 | `moe_store_a` (A into LDS in the "even k first" order), buffer loads `moe_rsrc`/`moe_load16`, `MoeLanes`, `moe_step` (:284, one step of 128 k: loads, unpacking, MFMAs at :339, the other LDS buffer), `moe_item` (:357, scales :370–400, addresses :402, step loop, bias and store, the SwiGLU epilogue) | builds |
| D — the kernel and the entry points | 493–725 | `gemm_grouped_mxfp4_mfma_kernel` (:494; ticket :512, `readfirstlane` :518, queue reset at the end), the old kernel renamed `gemm_grouped_mxfp4_simple` (:630), `moe_mfma_grid`, `moe_mfma_ok` (:661), `gemm_grouped_mxfp4` (:684) choosing between them, and the two fused mlp1 entry points (not called yet; lesson 11 uses them) | quick check gives `ccfc732a4a25` |

Spots worth reading twice while typing:
- `:128–135` — the whole MXFP4 → bf16 unpack is 8 `__builtin_amdgcn_perm` calls.
- `:176–223` — ranking experts by row count inside one wavefront, with `readlane` and
  `__shfl_up`, no sorting library.
- `:316` and `:346` — `__builtin_amdgcn_sched_barrier(0)`: stops the compiler from moving the
  loads to the end of the step (where nothing hides their wait).
- `:514–518` — why the ticket goes through `readfirstlane`: otherwise the compiler treats it as
  different per lane and every load becomes a small loop.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 9813b7a
git show 9813b7a -- getp-csrc/kernels/fast_moe_gemm.hpp
git checkout 9813b7a -- getp-csrc/tests/moe_gemm_test.cpp getp-csrc/bench docs   # if you skip typing them

git diff 9813b7a -- $(git show --name-only --format= 9813b7a)   # must print nothing
make runomp
```

Quick check, 20b (one GCD, 64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # ccfc732a4a25
```

The fingerprint changes from lesson 09's `b2d8ec70d2ff`: the new kernel adds the products in
another order, so a few answers drift later (as every bf16 change does; lesson 04). Its tok/s
jumps from about 265 to about 2,650 on this tiny input (for orientation only).

Quick check, 120b (a GCD pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 836592181c82
```

Unit test (header of `getp-csrc/tests/moe_gemm_test.cpp`; about 90 s):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/moe_gemm_test.cpp -o moe_gemm_test
tools/gpulock.sh 2 -- ./moe_gemm_test
```

Expected (first and last lines):

```
decode (all codes, scales 2..252)  0 of 737280 weights differ  ok
mlp1 fused vs separate kernels     E= 32 rows= 2048  swiglu: 0 differ, gather+swiglu: 0 differ of 5898240 …  ok
…
N=64 K=64 single row               E=  1 N=  64 K=  64 rows=    1 … max err/sum|AW| 3.95e-08 …  ok
all passed
```

The benchmark `moe_gemm_bench` replays routings recorded from real 20b runs. Those files are not
in the repo; they are kept in
`/home/namvu/claude_working/claude_job/gpt-oss/final-project/results/moe_routing_20b/`:

```bash
R=/home/namvu/claude_working/claude_job/gpt-oss/final-project/results/moe_routing_20b
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/moe_gemm_bench.cpp -o moe_gemm_bench
MOE_ONLY="20b mlp1 512" tools/gpulock.sh 2 -- ./moe_gemm_bench $R/off_b512.txt $R/off_b1024.txt $R/off_b2048.txt
```

Seen on the reference build (old = lesson 04's kernel, new = this one; µs per call):

```
model shape     B |    old us    new us   new min |   TFLOP/s    GB/s speedup |   lib est   lib avg |  vs lib
20b   mlp1    512 |   13004.6     854.1     779.0 |      79.6     355   15.2x |    1433.3    1555.2 |   1.68x
```

Full run:

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-10 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-10-120b --score
```

Expected: 20b about **64,099 tok/s**, METEOR 0.554, BERTScore 0.979, fingerprint `496d7ce70b4e`;
120b about **32,485 tok/s**, METEOR 0.568, BERTScore 0.981, fingerprint `7a6fd77322d9`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Run the MoE expert multiplies on the matrix cores, reading MXFP4"`

## 6. Result

End to end (8 GCDs, 4,096 prompts):

| | before | after |
|---|---:|---:|
| 20b tok/s | 7,858 | **64,099** (×8.2) |
| 120b tok/s (lessons 08–10 together) | 2,835 | **32,485** (×11.5) |
| 20b METEOR / BERTScore | 0.557 / 0.980 | 0.554 / 0.979 |

One call, one GCD, 512 requests (µs):

| shape | simple | first matrix-core draft | **this lesson** | library, one call per expert (bf16 weights) |
|---|---:|---:|---:|---:|
| 20b mlp1 | 13,168 | 1,227 | **857** (79 TFLOP/s) | 1,691 |
| 20b mlp2 | 6,745 | 616 | **472** | 1,098 |
| 120b mlp1 (one GCD of a pair) | 17,771 | 1,718 | **1,116** | 2,870 |
| 120b mlp2 | 9,065 | 953 | **602** | 1,979 |

15× faster than the simple kernel, 1.5–3.3× faster than the library on every shape and batch
size measured. In the 477 ms step of section 2, the expert multiplies go from about 456 ms to
about 32 ms.

## 7. Decisions and reasons

- **Decided:** the expert multiply is the only priority until it lands.
  **Why:** 95.6 % of a step (section 2). **Not chosen:** tuning anything else first — at most
  4 % of the step to win.
- **Decided:** one persistent kernel with one shared queue, biggest experts first.
  **Why:** how many rows each expert got is known only on the GPU; asking the CPU would stop the
  GPU every layer. Routings are very uneven (in real 20b steps the busiest expert got up to 473 of
  512 tokens, and 10 % of the experts got none). A queue gives each block the next item the
  moment it is free; handing out the big experts first leaves the small, cheap items for the
  end, so all blocks finish at about the same time. Experts with no rows have no items at all.
- **Decided:** unpack the 4-bit weights in registers, right before the matrix core.
  **Why:** the weights then cross memory at 4.25 bits per value instead of 16 (3.8× fewer bytes);
  the unpacking costs about 3 % of the kernel (measured by a test build without it).
  **Not chosen:** a bf16 copy of the expert weights in memory: 3.8× more bytes to read every step.
- **Decided:** 64-row × 256-column items, 8 wavefronts, scales once per item.
  **Why:** the first matrix-core draft was limited by L2 requests (16 per tick), not by the
  arithmetic; this shape needs 0.32 cache lines per 1,000 multiply-adds instead of 0.73.
- **Not chosen (measured, slower):** 64 × 256 on 4 wavefronts (192 registers needed, spills:
  1,040 → 5,032 µs); weights prefetched 3 steps ahead (848 → 1,022 µs, spills); 3 rotating weight
  registers (34 spills). At 4 wavefronts per SIMD every thread has 128 registers; every idea that
  needed more lost 20–40 %.
- **Kept for later:** the fused gather + mlp1 + SwiGLU entry points are in this file but unused
  (lesson 11 switches them on); storing the weights in reading order (a read-only test: 855 →
  1,296 GB/s) became lesson 15.

## 8. Questions to think about

1. The simple kernel launched a block for every row tile an expert *could* have. Why is that
   so wasteful here, and why can't the CPU simply launch the right number of blocks?
2. Why does handing out the biggest experts first make the whole call finish earlier? What
   would happen if the biggest expert were handed out last?
3. The draft was limited by requests to the L2 cache, not by the matrix cores. How can loading
   one scale byte per step cost as much as loading the weights themselves?

## 9. Discussion notes

(empty: we write here during the walk-through)
