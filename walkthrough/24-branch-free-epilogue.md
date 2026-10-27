# Lesson 24: Expert multiply — store the results without branches or hidden waits

| | |
|---|---|
| Reference commit | `25d09b2` "Expert GEMM: branch-free epilogue through buffer descriptors" |
| Starts from | `a9b23f7` (lesson 23) |
| To type | 79 lines of program code in 1 file (50 removed); 34 lines of benchmark and 221 of docs (optional) |
| Speed after | 20b 78,566 tok/s (+3.0 %), 120b 44,164 tok/s (+2.8 %) |

**In plain words:** the expert multiply spent 13 % of its time writing its results, when writing
them should take almost nothing. A stopwatch inside the kernel showed why: the kind of "write"
instruction the compiler had picked made every later wait also wait for all earlier writes to
reach memory. We write the results another way, and give the kernel's to-do list less room in
the fast registers. The results are exactly the same; the program got about 3 % faster.

## 1. Goal

Make the end of each expert work item (adding the bias, applying SwiGLU for mlp1, storing the
results) take a few µs instead of about 16, and stop the kernel from spilling registers to slow
memory, without changing any output bit.

## 2. Why this, why now

- A 120b profile after lesson 23: the expert multiply was 58.7 % of the time. Per GCD and layer it
  streams about 796 MB of weights at 0.61 TB/s (its loads alone, without the math, reach 0.87),
  while the weight-stream benchmark reaches 1.12–1.34 TB/s. So something other than reading
  weights was costing time.
- A **timeline build** (below) of 20b mlp1 at 512 requests split a block's time: 7–9 % starting an
  item, 74–76 % in the 23 steps, **13 % (14.8–16.3 µs per item) in the epilogue**, 3–4 % idle at
  the end. An item writes at most 64 × 256 numbers: that should take 1–2 µs.
- The compiler also spilled 9–29 registers of this kernel to scratch memory.

## 3. The technique

New words:
- **work item** — one piece of the expert multiply: up to 64 rows of one expert × 256 output
  columns, done by one block (lesson 10).
- **prologue / epilogue** — the start of a work item (taking its ticket, first loads, first
  barrier) and its end (bias, SwiGLU, stores).
- **SwiGLU** — the step after the first expert multiply: it combines each pair of neighbouring
  output columns (gate, up) into one value (lesson 11 moved it into this kernel).
- **flat store / buffer store** — two kinds of store instruction. A *flat* store takes a full
  64-bit address and works for any kind of memory, so the GPU counts it both as a memory store and
  as an LDS (the block's small shared on-chip memory) operation. A *buffer* store goes through a
  buffer descriptor and counts only as a memory store.
- **buffer descriptor** — a small record (start address, size) held in scalar registers; each lane
  adds a 32-bit offset. A load past the size returns 0; a store past it is simply dropped.
- **waitcnt** (`s_waitcnt`) — the instruction a wavefront uses to wait until at most N of its memory
  (or LDS) operations are still in flight. The counters are in order: waiting for one operation
  means waiting for every one of that kind issued before it.
- **DPP** (data-parallel primitives) — a way for an ordinary instruction to read a register of a
  neighbouring lane (here lane l reads lane l xor 1) without going through the LDS.
- **register spill / scratch** — when a kernel needs more than the 128 registers it may use, the
  compiler parks values in per-thread memory in HBM (scratch); every reload is a slow memory read.
- **timeline build** — a test build in which thread 0 of each block writes a clock value at chosen
  points of each item, so the time of each part can be summed.

What went wrong, and the fix:

```
 before: output pointer may be nullptr  →  compiler cannot prove "GPU memory"  →  flat_store
   lane: store r0 (flat) … store r31 (flat)
         SwiGLU needs the neighbour lane's value: __shfl_xor → LDS  → s_waitcnt lgkmcnt(0)
                                                   └─ also waits for every flat store to finish
   = 32 stores, one memory round trip each, one after another          ≈ 16 µs per item

 after:  descriptor over rows m0 .. m0+mv-1 of the item (size = mv rows)
   lane: store r0 … r31 (buffer)       rows past mv / columns past N → offset past the end → dropped
         neighbour value by a DPP move  → no LDS, no wait
   = stores leave without waiting, no branch                            ≈ 4 µs per item
```

Fewer registers for the work list: every wavefront keeps a list "expert of each rank, row tiles
before it" for every group of 64 experts. With room for 256 experts that was 8 registers held
through the whole step loop. Now one register per group holds both (expert in the low 8 bits),
at most 128 experts are supported, and the kernel is compiled twice, for 1 and 2 groups.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_moe_gemm.hpp` | +79 −50 | `MoeTiles<NG>`: the work list in one register per 64 experts; `moe_item`'s epilogue through buffer descriptors (`raw_buffer_store`), the DPP neighbour move, the output columns computed after the step loop; `gemm_grouped_mxfp4_mfma_kernel<1>` / `<2>` and the choice between them | yes |
| 2 | `getp-csrc/bench/moe_stream_bench.cpp` | +34 −13 | the weight stream alone, now also with one block of 16 wavefronts per CU and with 1 KB per load | optional |
| 3 | `docs/08-moe-grouped-gemm-mxfp4.md` | +221 −30 | section 10 and its pitfalls | optional |

Read while typing (line numbers at `25d09b2`):
- `getp-csrc/kernels/fast_moe_gemm.hpp:194-206` — `MoeTiles`: why every register outside the
  step loop matters.
- `:289-296` `moe_rsrc` — making a buffer descriptor (`__builtin_amdgcn_make_buffer_rsrc`).
- `:507-571` — the new epilogue: the bias read through a descriptor that returns 0 past N, the
  plain store at `:540` (`off = 0x80000000` for a lane that must not store), the SwiGLU with the
  DPP move at `:564` and the bf16 store at `:568`.
- `:592-600` — the ticket made uniform with `readfirstlane` (a value the compiler cannot prove is
  the same in every lane turns each buffer load into a loop).

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 25d09b2
git show 25d09b2 -- getp-csrc/kernels/fast_moe_gemm.hpp   # the change to copy

git diff 25d09b2 -- $(git show --name-only --format= 25d09b2)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 23)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 10e37a089467
```

Unit test of the expert multiply (unchanged file, still the gate; one GCD, about 2 minutes):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a \
  getp-csrc/tests/moe_gemm_test.cpp -o moe_gemm_test
HIP_VISIBLE_DEVICES=2 ./moe_gemm_test
```

Expected (seen on the reference build), the last lines:

```
N=64 K=64 single row               E=  1 N=  64 K=  64 rows=    1 max_rows=   1 empty=  0  max err/sum|AW| 3.95e-08  max rel err 2.05e-07  ok
all passed
```

The weight stream alone (one GCD, under a second):

```bash
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/moe_stream_bench.cpp -o moe_stream_bench
HIP_VISIBLE_DEVICES=2 ./moe_stream_bench
```

Seen on the reference build (one line per variant; the kernel runs 208 blocks of 8 waves):

```
grid 104 x  8 waves  depth 1  barrier 1  tile-major   4 x 256 B   200.6 us   1353 GB/s
grid 208 x  8 waves  depth 1  barrier 1  tile-major   4 x 256 B   246.6 us   1100 GB/s
grid 104 x 16 waves  depth 1  barrier 1  tile-major   4 x 256 B   262.9 us    987 GB/s
grid 104 x 16 waves  depth 1  barrier 1  tile-major   1 KB        220.2 us   1178 GB/s
```

Fewer wavefronts reading at once per CU stream faster; this is why the 16-wavefront idea below
did not pay.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-24 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-24 --score
```

Expected: 20b about 78,600 tok/s, fingerprint `58023f92ead7`; 120b about 44,200 tok/s,
fingerprint `973e97c40a85` (both unchanged).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Expert GEMM: branch-free epilogue through buffer descriptors"`

## 6. Result

Per work item (timeline build): epilogue 14.8–16.3 → 3.9–4.2 µs; prologue 9.2–9.7 → 7.2–7.9 µs
(its barrier no longer waits for a slow epilogue). Register spills: 9–29 → 0.

Per call on one GCD, real routings, µs (from `docs/08` section 10 and the log):

| Model | Call | B | before | after | change |
|---|---|---:|---:|---:|---:|
| 20b | mlp1 | 512 | 714.4 | **652.1** | −8.7 % |
| 20b | mlp2 | 512 | 401.4 | **355.9** | −11.3 % |
| 20b | gather + mlp1 + SwiGLU (the engine's call) | 512 | 707.2 | 665.5 | −5.9 % |
| 120b | mlp1 | 512 | 830.0 | **756.5** | −8.9 % |
| 120b | mlp2 | 512 | 465.0 | **412.4** | −11.3 % |

Whole program, official input, 8 GCDs, old and new builds alternated twice:

| | before | after |
|---|---:|---:|
| 20b | 76,302 tok/s | **78,566 tok/s (+3.0 %)** |
| 120b | 42,967 tok/s | **44,164 tok/s (+2.8 %)** |
| Output | | same fingerprints (`58023f92ead7`, `973e97c40a85`); old and new kernel gave the same bits on 2.73 billion outputs |

## 7. Decisions and reasons

- **Decided:** stores through buffer descriptors, the range check doing the "row past the end?"
  test. **Why:** no branch, no 64-bit address arithmetic, and buffer stores do not count as LDS
  operations, so the SwiGLU's neighbour exchange no longer waits for them.
- **Decided:** the neighbour lane's value by a DPP move instead of `__shfl_xor`. **Why:**
  `__shfl_xor` goes through the LDS and then waits on the LDS counter; DPP reads the register
  directly.
- **Decided:** one register per 64 experts, at most 128 experts, two compiled variants. **Why:**
  every register held across the step loop is one the loop cannot use; room for 128 experts in
  every call (one variant) was 0.3–1.4 % slower for 32 / 64 experts.
- **Not chosen** (all bit-identical, measured on the same real routings):
  - one block of 16 wavefronts per CU with 512-column items: −0.2 to −0.5 %, no gain (the weight
    stream gets slower with 16 wavefronts per CU, see the benchmark above);
  - the two blocks of a CU taking items from the two ends of the list (largest and smallest
    experts): 120b mlp1 −1.8 to −3.2 %, but 20b +1 to +5 %;
  - issuing the next item's ticket and first loads during this item's epilogue: +2 to +7 %: the
    compiler spilled 21–122 registers, and every reload waited, in order, for the loads just issued.

## 8. Questions to think about

1. Why does a store instruction that is "more general" (works for any memory) end up slower here,
   although each store moves the same bytes?
2. A store "dropped by the range check" replaces an `if (m < mv)`. Why can a branch around a store
   cost more than the store itself on a GPU?
3. The work list moved from 8 registers to 1–2, and the kernel got faster although the list's
   content did not change. What does that say about where registers matter?

## 9. Discussion notes

(empty: we write here during the walk-through)
