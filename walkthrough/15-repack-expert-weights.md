# Lesson 15: Reorder the expert weights once, so reads are in order

| | |
|---|---|
| Reference commit | `cdca2c5` "Repack the MXFP4 expert weights at load time for contiguous reads" |
| Starts from | `d68a11d` (lesson 14) |
| To type | 212 lines of program code in 2 files (kernel +176 −79, engine +36 −6); 94 lines of tests and benchmark (optional); 216 lines of docs (optional) |
| Shortcut | `git cherry-pick cdca2c5` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 67,394 → **72,258** tok/s (+7.2 %); 120b 36,132 → **40,164** (+11.2 %) |

**In plain words:** the expert weights were stored in the order the model file uses: one long
row per output column. But the expert multiply reads them in a different order: in each step a
small piece from each of 256 rows, far apart. Memory is fastest when neighbouring bytes are read
together. So, once, while the model loads, we copy the weights into exactly the order the
multiply reads them. The multiply gets the same numbers in the same places of its registers, so
the answers do not change at all; it just reads them faster.

## 1. Goal

At load time, rearrange every expert's MXFP4 codes and scales so that one step of one work item
(256 columns × 128 k) is one contiguous 16 KB piece, in the order the lanes load it; and make
the kernel read that layout. Same bits.

## 2. Why this, why now

- A profile of a full 20b run after lesson 11: the expert multiplies were still **55.6 %** of
  the time.
- Test builds of the lesson-10 kernel that skip one part (wrong results, but their time shows
  that part's cost): without the weight loads it was **17 %** faster (42 % when every expert has
  only 16 rows). The weight reads were the main cost left.
- A read-only test of the 32 mlp1 experts (265 MB): **855 GB/s** in the kernel's read pattern on
  the checkpoint layout, **1,296 GB/s** reading the same bytes contiguously. Padding the rows to a
  multiple of 64 bytes did not help (309 vs 311 µs).

## 3. The technique

New words:
- **layout** — the order in which numbers sit in memory.
- **contiguous** — right next to each other, with no gaps.
- **repack** — copy the numbers once into a new order. Here it happens in `warm_up`, which is not
  timed, so it costs nothing when answering.
- **staging buffer** — a temporary piece of GPU memory the raw checkpoint bytes pass through
  before they are rearranged into their final place.
- **buffer descriptor** — a small record (start address and length) that a GPU load instruction
  uses; a load asking for bytes past the length gets zeros and touches no memory.

```
 checkpoint layout, one expert (mlp1: 5,760 columns × 1,440 bytes of codes)
   column n0     [ step0 | step1 | … | step22 ]   1,440 bytes
   column n0+1   [ step0 | step1 | … | step22 ]
   …
   column n0+255 [ step0 | step1 | … | step22 ]
   one step of a block = the 64-byte piece "stepP" of each of 256 rows:
   256 small reads spread over 368 KB; 1,440 is not a multiple of 64, so every other
   piece straddles two 64-byte cache lines

 packed layout (moe_pack_mxfp4, once at load):
   tile t: [ step 0: 16 KB ][ step 1: 16 KB ] … [ step 22: 8 KB ]
   one step, ordered by lane group kq, then 16-column piece c, then column j:
     kq=0 [ c0 | c1 | … | c15 ]  kq=1 [ … ]  kq=2 [ … ]  kq=3 [ … ]     each [ c ] = 256 bytes
   lane (j, kq) still gets the same 16 bytes as before; one load instruction of a
   wavefront = 4 runs of 256 bytes; one step of the block = one contiguous 16 KB

 scales, per 16 columns: 1,536 bytes; each lane's 24 bytes (one per step) side by side
   → 2 direct loads (16 + 8 bytes) per lane, no more copy through LDS
```

Same bits: only where each byte sits changes; every lane ends up with the same bytes in the same
registers, so every product and every addition is the same.

Two more changes come with it in the kernel: the weights of the next step are loaded one step
ahead instead of two (with contiguous reads one is enough, and it frees registers), and in the
last half step (2,880 = 22.5 × 128) the lanes past the end simply read past the tile's buffer
descriptor and get zero codes.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_moe_gemm.hpp` | +176 −79 | the packed layout (comment at the top), `MOE_SPIECE`, `moe_packed_blocks_bytes` / `moe_packed_scales_bytes`, the item and step code reading the packed layout, the packing kernels `moe_pack_blocks_kernel`, `moe_pack_scales_kernel` and `moe_pack_mxfp4` | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +36 −6 | `gpu_upload_weights()`: the `experts` helper uploads each layer's checkpoint bytes into a staging buffer, packs them into the final buffers, frees the staging buffer at the end; the exact path and shapes the fast kernel does not take keep the checkpoint layout | yes |
| 3 | `getp-csrc/tests/moe_gemm_test.cpp` | +80 −2 | `pack_test` (packer against a host version of the layout, byte by byte) and `pack_on_gpu` (all other checks now run on packed weights) | optional |
| 4 | `getp-csrc/bench/moe_gemm_bench.cpp` | +14 −4 | packs the weights before timing | optional |
| 5 | `docs/08-moe-grouped-gemm-mxfp4.md` | +216 −46 | section 7, "weights stored in reading order" | optional (read it) |

The kernel diff edits code from lesson 10 in many places (16 hunks); type it hunk by hunk from
`git show cdca2c5 -- getp-csrc/kernels/fast_moe_gemm.hpp`.

Read while typing (line numbers at `cdca2c5`):
- `getp-csrc/kernels/fast_moe_gemm.hpp:39–60` — the layout, as a formula: the offset of the 16
  code bytes of column `256t + 16c + j`, group `4p + kq`.
- `getp-csrc/kernels/fast_moe_gemm.hpp:604–624` — `moe_pack_blocks_kernel`: one thread per 16
  bytes; it computes that same offset and copies.
- `getp-csrc/kernels/fast_moe_gemm.hpp:651` — `moe_pack_mxfp4`, the entry point the engine calls.
- `getp-csrc/engine/gpu_engine.hpp:153–178` — the `experts` helper; `:196` and `:198` use it for
  mlp1 and mlp2.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick cdca2c5
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout cdca2c5 -- $(git show --name-only --format= cdca2c5)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat cdca2c5
git show cdca2c5 -- getp-csrc/kernels/fast_moe_gemm.hpp
git diff cdca2c5 -- $(git show --name-only --format= cdca2c5)   # must print nothing
make runomp
```

Quick check, 20b (one GCD, 64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # ccfc732a4a25 — unchanged
```

Quick check, 120b (a GCD pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 4ed6e262b73a — unchanged from lesson 14
```

Unit test (header of `getp-csrc/tests/moe_gemm_test.cpp`; about 90 s):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/moe_gemm_test.cpp -o moe_gemm_test
tools/gpulock.sh 2 -- ./moe_gemm_test
```

Expected — three new lines at the top, then lesson 10's checks, now on packed weights:

```
pack E=2 N=5760 K=2880             0 of 17694720 packed bytes differ  ok
pack E=3 N=200 K=192               0 of 119808 packed bytes differ  ok
pack E=2 N=520 K=256               0 of 236544 packed bytes differ  ok
decode (all codes, scales 2..252)  0 of 737280 weights differ  ok
…
N=200 K=192 half step              E=  3 N= 200 K= 192 rows=   75 … max err/sum|AW| 2.12e-07 …  ok
N=64 K=64 single row               E=  1 N=  64 K=  64 rows=    1 … max err/sum|AW| 3.95e-08 …  ok
all passed
```

Full run:

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-15
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-15-120b
```

Expected: 20b about **72,258 tok/s**, fingerprint `496d7ce70b4e`; 120b about **40,164 tok/s**,
fingerprint `e723ce53480f` — both unchanged, so the scores stay (20b 0.554 / 0.979, 120b 0.576
/ 0.982).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Repack the MXFP4 expert weights at load time for contiguous reads"`

## 6. Result

One call, one GCD, B = 512 (µs):

| shape | checkpoint layout | packed | change |
|---|---:|---:|---:|
| 20b mlp1 | 852 | **717** | −16 % |
| 20b mlp2 | 471 | **405** | −14 % |
| 120b mlp1 (one GCD of a pair) | 1,122 | **845** | −25 % |
| 120b mlp2 | 598 | **475** | −21 % |
| every expert with only 16 rows | 487 | **313** | −36 % |

Hardware counters (20b mlp1): requests to the L2 cache 14.3 → 9.3 million per call; matrix cores
busy 53 → 70 % of the time.

End to end (8 GCDs):

| | before | after |
|---|---:|---:|
| 20b, 4,096 prompts | 67,394 | **72,258** (+7.2 %) |
| 120b, 4,096 prompts | 36,132 | **40,164** (+11.2 %) |
| 20b, 8,192 prompts (lesson 12's input, `GPTOSS_BATCH=1024`) | 75,640 | **79,504** (+5.1 %) |

120b gains more: its experts get fewer rows each (64 experts per GCD), so each row tile does
less arithmetic per weight byte and the weight stream matters more.

Warm-up did not get measurably longer: it is dominated by reading the model files. The packed scales take 96 bytes per column instead of 90 (0.4 % more for
a whole expert); 120b still fits 512 requests per GCD.

## 7. Decisions and reasons

- **Decided:** repack the expert weights once at load, so each 256-column × 128-k block is one
  contiguous piece in reading order.
  **Why:** the expert multiplies were 55.6 % of the run; skipping the weight loads made the
  kernel 17 % faster; a read-only test showed 855 → 1,296 GB/s for contiguous reads of the same
  bytes.
- **Decided:** the exact path keeps the checkpoint layout.
  **Why:** its kernels (and the simple fallback) read the checkpoint layout; the engine packs
  exactly when `moe_mfma_ok` holds, the same condition under which the fast entry points expect
  packed weights. Handing checkpoint bytes to the fast kernel would give plausible-looking
  garbage, not a crash.
- **Not chosen:** padding the rows to a multiple of 64 bytes (no gain: 309 vs 311 µs);
  one load instruction = 1 KB contiguous (749 → 755 µs, no gain and 2.2 % more memory);
  keeping the two-steps-ahead weight loads (1.5–2.8 % slower once reads are contiguous).

## 8. Questions to think about

1. The packed layout holds exactly the same bytes, and the kernel does exactly the same
   arithmetic. Why can it be 16–36 % faster?
2. Why does the repack happen in `warm_up` and on the GPU, rather than once on disk as a new
   model file? Think about the rules of lesson 00 and about the exact path.
3. Experts with only 16 rows gained the most (−36 %). Why do small experts depend more on how
   the weights are read than big ones?

## 9. Discussion notes

(empty: we write here during the walk-through)
