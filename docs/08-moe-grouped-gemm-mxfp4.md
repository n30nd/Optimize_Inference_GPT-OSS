# 08 — The expert matrix multiply on the matrix cores, straight from 4-bit weights

## The problem in plain words

Every layer of gpt-oss sends each token to 4 of its experts (doc 06). After the tokens are sorted
by expert, each expert multiplies "its" rows of numbers by its own big table of weights. With the
first, simple version of this multiply, it was the single largest cost of a decode step (one step
makes one new token for every request): **68 %** of the step at first, and 96 % once the other big
multiplies were made fast.

Three things made the simple version slow:

1. It did the arithmetic one number at a time on the ordinary calculators, not on the matrix
   cores, which are about 8 times faster at exactly this job.
2. The weights are stored in a 4-bit format (MXFP4). The matrix cores cannot read that format, so
   the weights must be unpacked on the way, and unpacking costs time too.
3. The experts get very different numbers of rows: in real steps of gpt-oss-20b with 512 tokens, the
   busiest expert got up to 473 of the 512 tokens and 10 % of the experts got none.

The new kernel feeds the matrix cores straight from the 4-bit weights and shares the work fairly
among the GPU's compute units. It is **15× faster than the simple version** and 1.5–3.3× faster
than calling the vendor's matrix library once per expert (with 4× larger 16-bit weights).

A second step (section 7) stores the 4-bit weights once, while the model loads, in the order the
kernel reads them. That made each call another **7–25 % faster** (most for gpt-oss-120b, whose
experts get few rows), with exactly the same results.

A third round of measurements (section 8) found what limits the kernel now. The suspected cause, the
way the rows of A are loaded, turned out not to matter. Experts with few rows wait on the stream of
weights from memory, not on the unpacking. Telling the GPU which wavefronts to serve first (wave
priorities) makes those experts **5–11 % faster** and every real call **0.3–1.2 % faster**, with
the same results (whole program +1.6 to +2.2 %).

A fourth round (section 10) timed each part of a work item. Writing an item's results took 16 µs,
about as long as 4 of its 23 steps, because the compiler had picked a kind of store instruction that
made every later wait also wait for all earlier stores. Writing them another way, and keeping the
work list in fewer registers, made every call **3–13 % faster** (whole program **+3.3 % for 20b,
+2.8 % for 120b**), again with exactly the same results.

## An everyday comparison

A print shop with very fast presses (the matrix cores). The paper arrives vacuum-packed to save
truck space (the 4-bit weights). The old shop had no presses: clerks copied every page by hand. The
new shop unpacks each bundle right next to the press, just before it is needed. Orders are uneven:
one customer wants 400 copies, another none. Each press takes the next job from one shared queue
the moment it is free, and the biggest orders are handed out first so no press is left with a big
job at the end.

It turned out the presses were not the limit at first: the **corridor to the warehouse** was. Each
press sent a runner for every small bundle, and the corridor lets only so many runners through per
second. The big win came from sending fewer runners with bigger loads.

Then the warehouse itself was re-shelved. The bundles were stored in the supplier's order, so for
one step of work a runner had to visit 256 different shelves and take a small packet from each.
Now, once, before the shop opens, a clerk puts the 256 packets that one step needs side by side on
one shelf, in the order the press uses them. The runner grabs one armful.

Last, the shop set a rule for who goes first at the shared door. Before, the door was served
oldest-first. Now a worker who must send a runner to the warehouse goes before one who is only
operating a press, and both go before one who is just tidying up. Small orders wait mostly for
paper, so their runners now leave a little earlier.

Then someone timed a whole order with a stopwatch. Printing was fine; the slow part was putting the
finished pages on the shelf. Each clerk had to sign a receipt for every single page before passing the
next page to a colleague, and waited at the counter for each receipt. Now the pages are put on the
shelf without waiting, and a page that does not belong to the order is simply not accepted by the
shelf (no clerk has to check it). And the clerks used to carry a notebook big enough for 256
customers; with a pocket notebook for the 32 or 64 customers they actually have, their hands are free
for printing.

## The words

- **kernel** — a small program that runs on the GPU.
- **thread / wavefront / block** — a thread is one calculator's job; the GPU runs threads in groups
  of 64 that move in lockstep (a *wavefront*); a *block* is a few wavefronts that run on the same
  compute unit and can share fast memory. Here a block is 8 wavefronts (512 threads).
- **CU (compute unit)** — a group of calculators with its own fast scratch memory; one GCD (one of
  the two halves of an MI250 card, seen by the software as one GPU) has 104. Each CU has 4 **SIMDs**,
  the actual calculator rows; each SIMD takes turns among the wavefronts that live on it.
- **register** — the tiny, fastest storage inside each thread (4 bytes each). A SIMD has 512 of them
  for each of its 64 thread positions; if 4 wavefronts share a SIMD, each thread may use at most 128.
- **spill** — when a kernel needs more registers than it may use, the compiler parks some values in
  slow memory and fetches them back; inside a hot loop that is very costly.
- **LDS (local data share)** — 64 KB of fast scratch memory per CU, shared by a block.
- **HBM (high-bandwidth memory)** — the GPU's main memory: big (64 GB) but far away (about a
  microsecond per trip).
- **L2 cache** — an 8 MB store between the compute units and HBM that keeps recently read data. It is
  split into 16 channels; each channel answers about one request per clock tick, and a request
  brings one 64-byte piece (a **cache line**).
- **MFMA (matrix fused multiply-add)** — one matrix-core instruction. The one we use,
  `v_mfma_f32_16x16x16bf16_1k`, multiplies a 16×16 block of numbers by another 16×16 block and adds
  the 16×16 result into 32-bit totals, in 32 clock ticks.
- **bf16 (brain float 16)** — a 16-bit number format: the same range as a normal 32-bit float but
  fewer digits. The matrix cores take bf16 inputs.
- **MXFP4** — the 4-bit weight format of the checkpoint (explained below).
- **GEMM (general matrix multiply)** — the "rows times table of weights" operation; a **grouped
  GEMM** does many independent ones (one per expert) in one kernel.
- **TFLOP/s** — trillions of arithmetic operations per second (a multiply and an add count as 2).
- **prefetch** — asking memory for data a while before it is needed, so it has arrived by then.
- **barrier** — a point where every wavefront of a block waits until all of them have arrived.
- **atomic add** — an add to a memory word that the hardware does for one thread at a time, so many
  threads can each take a unique ticket from one counter.
- **layout** — the order in which numbers sit in memory.
- **contiguous** — sitting right next to each other in memory, with no gaps.
- **repack** — copy numbers once into a new order. Here it happens while the model loads
  (`warm_up`, not timed), so it costs nothing when answering.
- **buffer descriptor** — a small record (a start address and a length) that a GPU load
  instruction uses to find its data; a load that asks for bytes past the length gets zeros back
  and never touches memory.
- **staging buffer** — a temporary piece of GPU memory that the raw bytes pass through before
  they are rearranged.
- **bit-identical** — exactly the same bits: not even the last digit of any number changes.
- **ablation** — a test build with one part removed (for example, no weight loads). Its results are
  wrong, but its time shows how much that part costs.
- **wave priority** (`s_setprio`) — a number 0–3 that a wavefront sets for itself. When several
  wavefronts of a SIMD are ready, the SIMD issues the next instruction of the one with the highest
  priority (among equals, the oldest).
- **L1 cache** — the small cache (16 KB) of each compute unit, in front of the L2. Two loads of the
  same 64-byte line close together are served by one L2 request.
- **prologue / epilogue** — the start of a work item (taking its ticket, looking it up, its first
  loads and first barrier) and its end, where its results (plus the bias, and for mlp1 the SwiGLU)
  are written to memory.
- **flat store / buffer store** — two kinds of store instruction. A *flat* store takes a full 64-bit
  address and works for any kind of memory, so the GPU counts it both as a memory store and as an LDS
  operation. A *buffer* store takes a buffer descriptor and a 32-bit offset, and counts only as a
  memory store.
- **waitcnt** (`s_waitcnt`) — the instruction with which a wavefront waits until at most a given
  number of its memory (or LDS) operations are still in flight. The counters are in order: waiting
  for one operation means waiting for every operation of that kind issued before it.
- **DPP (data-parallel primitives)** — a way for an ordinary instruction to read a register of a
  neighbouring lane (here lane l reads lane l xor 1), without going through the LDS.
- **scratch** — per-thread memory in HBM; spilled registers go there.
- **timeline build** — a test build in which thread 0 of each block writes the time
  (`s_memrealtime`, a clock that ticks 25 million times per second) at chosen points of each work
  item, so that the time of each part can be summed over a call.

## How it works

### 1. The MXFP4 format

Each weight row is cut into groups of 32 numbers. Every group stores 32 four-bit codes (16
bytes, two codes per byte, **low half of the byte first**) and one shared scale byte `s`.

```
 one group of 32 weights = 16 bytes of codes + 1 scale byte
 byte j:  [ code 2j+1 | code 2j ]          code = sign | 2 exponent bits | 1 mantissa bit
            high 4 bits  low 4 bits

 code (low 3 bits) : 0    1    2    3    4    5    6    7
 magnitude         : 0   0.5   1   1.5   2    3    4    6        (bit 3 = minus sign)

 weight = magnitude[code] × 2^(s − 127)
```

A weight costs 4.25 bits instead of 16 in bf16: **3.8× fewer bytes** to read. Every MXFP4 value
times its scale is exactly a bf16 number (it has at most 2 significant bits), so unpacking loses
nothing.

For our shapes, one expert's mlp1 weight (5,760 rows × 2,880) is 8.3 MB of codes + 0.5 MB of scales;
mlp2 (2,880 × 2,880) is half that.

### 2. Unpacking 4 codes with one instruction

A bf16 number is two bytes. For one group we build two small tables of 8 bytes each in registers:
the high byte and the low byte of `magnitude[i] × 2^(s−127)` for i = 0..7. The instruction
`v_perm_b32` picks 4 bytes out of an 8-byte table by 4 small indices at once, which is exactly a
4-entry table lookup:

```
 4 codes ──mask──► 4 indices (0..7) ──v_perm_b32──► 4 high bytes ──OR sign bit──┐
                                    └─v_perm_b32──► 4 low bytes ────────────────┴─v_perm_b32 ×2──► 4 bf16
```

About 14 instructions per 8 weights, all on registers (`fp4_tables`, `fp4x8_to_bf16`). The tables
are built once per group of 32 weights (9 instructions).

### 3. Feeding the matrix core

One MFMA multiplies a 16 (rows of A) × 16 (k) block by a 16 (k) × 16 (columns) block. Each of the 64
threads of the wavefront supplies 4 numbers of each:

```
 thread l supplies:  A[row = l % 16][4 k values]     W[column = l % 16][the same 4 k values]
 threads  0–15 → their 4 k come from "slot 0",  16–31 → slot 1,  32–47 → slot 2,  48–63 → slot 3
```

The sum over k does not care in which order the k values arrive, as long as A and W use the same
order. We use that freedom so that each thread reads **16 contiguous weight bytes** (one whole group
of 32 codes) per step of 128 k:

```
 a column's 128 k of one step = 64 bytes of codes:
   bytes  0–15 → threads 0–15 (k 0–31)     bytes 16–31 → threads 16–31 (k 32–63)
   bytes 32–47 → threads 32–47 (k 64–95)   bytes 48–63 → threads 48–63 (k 96–127)
 each thread's 32 k feed 8 MFMAs, 4 k each; the decoder gives "even k first, then odd k",
 and A is placed in LDS in exactly that order
```

### 4. One work item: 64 rows × 256 columns

```
                 columns n0 … n0+255  (weights, unpacked in registers by each wavefront)
               ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
 rows m0 …     │ wf 0 │ wf 1 │ wf 2 │ wf 3 │ wf 4 │ wf 5 │ wf 6 │ wf 7 │   each wavefront (wf):
 m0+63         │32 col│      │      │      │      │      │      │      │   64 rows × 32 columns
 (A, in LDS,   │      │      │      │      │      │      │      │      │   = 4 × 2 MFMA tiles
  shared)      └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

K (2,880) is walked in 23 steps of 128. One step, for one wavefront:

```
 time ───────────────────────────────────────────────────────────────────────────►
 issue loads: A of step p+1 (→ registers), weights of step p+1 (→ registers)
             unpack weights of step p ─┬─ 8 × (read A from LDS, 8 MFMAs) ─┬─ A of step p+1 → LDS ─ barrier
                                       └── loads travel meanwhile ────────┘
```

Two LDS buffers (16 KB each) alternate: the MFMAs read one while the next A slice is written into
the other. The weights of the next step are loaded while the current one is unpacked (the first
version loaded them two steps early, in two register sets; with the packed weights of section 7
one step is enough and 1.5–3 % faster). 32 KB of LDS per block means 2 blocks (16 wavefronts, 4
per SIMD) fit on one CU.

### 5. The scales: once per item, not once per step

Each thread needs one scale byte per step for each of its two columns: 23 bytes per column for
K = 2,880. It loads them all at the start of an item into 6 registers (4 bytes each) and shifts one
byte out per step. With the packed layout of section 7 these 24 bytes already sit next to each
other in memory, so a thread fetches them with 2 loads (16 + 8 bytes). The first version read the
checkpoint's 23 KB of scale bytes for the item's 256 columns into LDS and let every thread pick out
its bytes, with two barriers; the direct loads are 1–4 % faster. (Before either, every step loaded
each scale byte on its own: one cache-line request for 1 useful byte, as many requests as for the
weights themselves.)

### 6. Uneven experts: one queue

The CPU does not know how many rows each expert got — that number lives on the GPU — so the
kernel decides on the GPU:

```
 offsets (on GPU) ──► every wavefront counts each expert's 64-row tiles (a running sum)
                      work items, in order: expert (biggest first) → 256-column tile → 64-row tile
 a fixed set of blocks (as many as fit on the GPU at once: 208) loops:
        ticket = atomic_add(queue, 1)  →  find its (expert, column tile, row tile)  →  compute
 the last block to finish resets the queue for the next call
```

- An expert with 0 rows has no items at all.
- A tile with fewer than 64 real rows skips the 16-row MFMA tiles that are completely empty: an
  expert with 20 rows pays for 32 rows of arithmetic, not 64.
- The row tiles of one expert that use the same weights are next to each other in the queue, so
  they start at the same time and the second one finds the weights in the L2 cache.
- Biggest experts first: the last items handed out are small, so blocks finish at nearly the same time.

### 7. Weights stored in reading order (the packed layout)

**The problem.** The checkpoint stores the weights of one output column (2,880 numbers, 1,440 bytes
of codes) as one long row. But a block does not read a column from start to end. In each step it
needs the next 64 bytes of each of its 256 columns, and in the checkpoint order those 256 pieces
lie 1,440 bytes apart:

```
 checkpoint order, one expert (mlp1: 5,760 columns x 1,440 bytes = 8.3 MB of codes)

   column n0     [ step0 | step1 | step2 | ... | step22 ]      1,440 bytes
   column n0+1   [ step0 | step1 | step2 | ... | step22 ]      the next 1,440 bytes
   ...
   column n0+255 [ step0 | step1 | step2 | ... | step22 ]
                     ^
   one step of a block = this 64-byte piece of each of the 256 rows:
   256 small reads spread over 368 KB. 1,440 is not a multiple of 64, so for every
   other column the piece starts in the middle of a 64-byte cache line: 2 lines for 1.
```

Memory chips are fastest when neighbouring bytes are asked for together (they open a stretch of
1–2 KB at a time); scattered 64-byte reads waste most of each opened stretch. A test that only
reads the 32 mlp1 experts (265 MB) measured 855 GB/s in this pattern and 1,296 GB/s when the same
bytes are read contiguously.

**The fix.** While the model loads, a small GPU kernel (`moe_pack_mxfp4`) copies every expert's
codes into a new order: for each 256-column tile and each step of 128 k, the 16 KB that the block
will read are put side by side, in exactly the order the threads ask for them.

```
 packed order, one expert, tile t (256 columns = 16 pieces of 16 columns), K = 2,880

   tile t: [ step 0: 16 KB ][ step 1: 16 KB ] ... [ step 21: 16 KB ][ step 22: 8 KB ]

   one step (16 KB), ordered by thread group kq, then piece c, then column j:
     kq=0 [ c0 | c1 | ... | c15 ]  kq=1 [ c0 | ... | c15 ]  kq=2 [ ... ]  kq=3 [ ... ]
     each [ c ] = 16 columns x 16 bytes = 256 bytes

   thread (j, kq) of the wavefront that owns piece c gets 16 bytes = the 32 codes of
   column 256t + 16c + j, k = 128p + 32kq ... 128p + 32kq + 31   (the same bytes as before)
   one load instruction of a wavefront = 4 full runs of 256 bytes; one step of the
   block = one contiguous 16 KB
```

The packed codes take the same space as the checkpoint's (the column count is only rounded up to a
multiple of 16, which 5,760 and 2,880 already are); the packed scales take 96 bytes per column
instead of 90, 0.4 % more for a whole expert. The 120b split still fits 512 requests per GCD.

**The last half step.** 2,880 is 22.5 steps of 128. Because kq comes first inside a step, a half
step is simply the first half (kq 0 and 1, 8 KB). The threads with kq 2 and 3 ask for bytes past
the end of the tile; the buffer descriptor of each work item is made to end exactly there, so those
loads return zero codes (zero weights) without memory traffic. The first version had to replace
those codes by zeros with an extra instruction per register.

**The scales too.** Each thread's 24 scale bytes per column (one per step, section 5) are stored
together:

```
 packed scales, per piece of 16 columns: 1,536 bytes
   bytes    0 .. 1,023: thread l's steps 0..15  at 16*l .. 16*l+15    (one 16-byte load)
   bytes 1,024 .. 1,535: thread l's steps 16..23 at 1,024 + 8*l .. +7  (one 8-byte load)
   (steps past K hold 0)
```

**The same numbers, bit for bit.** Only the place of each byte in memory changes. Every thread
ends up with exactly the same bytes in the same registers as with the checkpoint layout, so every
product and every addition is the same. Checked three ways: the packer against a host version of
the layout, byte by byte; the old kernel on the checkpoint layout against the new kernel on the
packed layout, 1.45 billion outputs of real and synthetic routings, none different; and the
end-to-end output hashes of `./run` (20b `ccfc732a4a25`, 120b `836592181c82`, unchanged).

**In the engine.** `gpu_upload_weights` (`getp-csrc/engine/gpu_engine.hpp:152`, `:195`, `:197`)
copies each layer's checkpoint bytes into one staging buffer on the GPU, packs them into the final
buffers with `moe_pack_mxfp4` and frees the staging buffer at the end. The exact path, and any
shape the fast kernel does not take (`moe_mfma_ok`), keep the checkpoint layout. Warm-up time did
not change measurably (20b about 5 s, 120b 26–34 s, both dominated by reading the files).

**Results.** µs per call, mean of 40 calls (two rounds agree within 1 %), one GCD of mv-mi250-02,
same routings as below; "before" = the first matrix-core kernel on the checkpoint layout:

| model | shape | B | before | packed | change | TFLOP/s after |
|---|---|---:|---:|---:|---:|---:|
| 20b | mlp1 | 512 | 852 | **717** | −15.9 % | 95 |
| 20b | mlp2 | 512 | 471 | **405** | −14.1 % | 84 |
| 20b | mlp1 | 1024 | 1,378 | **1,235** | −10.4 % | 110 |
| 20b | mlp2 | 1024 | 749 | **685** | −8.6 % | 99 |
| 20b | mlp1 | 2048 | 2,477 | **2,281** | −7.9 % | 119 |
| 20b | mlp2 | 2048 | 1,327 | **1,232** | −7.1 % | 110 |
| 120b | mlp1 | 512 | 1,122 | **845** | −24.7 % | 78 |
| 120b | mlp2 | 512 | 598 | **475** | −20.6 % | 70 |
| 120b | mlp1 | 1024 | 1,661 | **1,372** | −17.4 % | 98 |
| 120b | mlp2 | 1024 | 892 | **753** | −15.6 % | 89 |
| 120b | mlp1 | 2048 | 2,838 | **2,504** | −11.8 % | 113 |
| 120b | mlp2 | 2048 | 1,504 | **1,340** | −10.9 % | 105 |

The fused mlp1 entry points the engine calls: 20b gather + mlp1 + SwiGLU 801 → 718 µs (B=512) and
1,358 → 1,256 µs (B=1024); 120b mlp1 + SwiGLU 1,046 → 828 µs (B=512) and 1,607 → 1,370 µs
(B=1024). Every expert with only 16 rows (where the weights matter most): 487 → 313 µs (−36 %).

Where the gain comes from (20b mlp1, B=512; each version timed back to back with the one before it
on the same GPU):

| change | µs per call |
|---|---:|
| first matrix-core kernel, checkpoint layout | 862 |
| codes packed (scales still through LDS) | 764 |
| scales packed, 2 direct loads per thread | 750 |
| weights loaded one step ahead instead of two | 734 |
| no forced register copy before the unpacking (see Pitfalls) | 722 |

Hardware counters (rocprofv3, same call): requests to the L2 cache 14.3 → 9.3 million per call,
matrix cores busy 53 → 70 % of the time (120b mlp1 B=512: 17.6 → 10.5 million, 48 → 64 %).

Whole program, 64 prompts × 128 tokens (`./run … -n 128`, mv-mi250-02): 20b on one GCD 2,743 →
**3,315 tok/s** (+21 %), 120b on a GCD pair 1,433 → **1,755 tok/s** (+22 %), same output hashes.

### 8. What limits it now, and wave priorities

**The starting guess was wrong.** After section 7, a test build without the A loads was 10 % faster,
and each 64-byte line of A is read by two load instructions of 32 bytes (`moe_item`, thread
`(arow, apart)`: lanes 2i and 2i+1 share a line). So the guess was: every A line costs two requests
to the L2 cache. A version where 4 neighbouring lanes read the 4 quarters of one line in a single
instruction cut the lookups in the L1 cache by 34 % (19.8 → 13.1 million per call). But the
requests that reach the L2 did not change (11.89 → 11.88 million; 20b mlp1, B=512). The L1 had
already merged the two halves of each line. The time did not improve either: 20b mlp1 B=512
717 → 719–724 µs, and 64-row tiles 643 → 667 µs. A first version also had 3 times more LDS bank
conflicts; an XOR of the LDS row index removed them, still with no gain. So the A loads stay as
they were.

**What each part costs** (ablations, µs per call, one GCD, mean of 20 calls, two rounds within 1 %):

| removed from the kernel | 20b mlp1 B=512 | 120b mlp1 B=512 | 16 rows per expert | 64 rows per expert |
|---|---:|---:|---:|---:|
| nothing (the kernel) | 713 | 843 | 314 | 642 |
| A loads | 666 (−7 %) | 803 (−5 %) | 337 (+7 %) | 604 (−6 %) |
| weight loads | 666 (−7 %) | 753 (−11 %) | 249 (−21 %) | 629 (−2 %) |
| both | 618 (−13 %) | 699 (−17 %) | 230 (−27 %) | 585 (−9 %) |
| unpacking | 693 (−3 %) | 826 (−2 %) | 309 (−2 %) | 657 (+2 %) |
| unpacking and MFMAs (loads only) | 425 | 613 | 276 | 388 |

("16 / 64 rows per expert": every one of the 32 experts gets exactly that many rows, `MOE_UNIFORM`.)
Read the last line: with 16 rows per expert, the loads alone take 276 of the 314 µs (88 %). Without
the unpacking it is only 2 % faster. So small experts are limited by the **stream of weights from
memory**, not by the unpacking. Even with nothing to load, 64-row tiles reach only about 75 % of the
matrix cores' peak (585 µs). There the register file is full (128 per thread), so the compiler reads
each A operand from LDS just before its MFMA and waits for it (`ds_read` then
`s_waitcnt lgkmcnt`, about 16 times per step).

**How fast can that stream be?** `getp-csrc/bench/moe_stream_bench.cpp` reads the weights exactly as
the kernel does (same queue, same 208 blocks, 16 KB per block and step) and does nothing else:

| setup | GB/s |
|---|---:|
| 208 blocks, loads 1 step ahead (the kernel) | 1,116–1,122 |
| 208 blocks, 2 or 4 steps ahead | 1,136–1,148 |
| 104 blocks (one per CU) | 1,342–1,344 |
| 208 blocks, "step-major" order (step p of all 23 tiles of an expert side by side) | 1,286–1,293 |

So loading further ahead gives nothing at this grid; what helps the memory is fewer separate
streams, or putting the pieces that run at the same time next to each other. Neither carried over to
the kernel: loading the weights two steps ahead (possible for 16- and 32-row tiles, which hold fewer
accumulators) changed nothing (−1 to +2 %), and the step-major order gave −2.4 % at 16 rows per
expert and ±1 % elsewhere (see "What did not help").

**What did help: wave priorities.** A SIMD has 4 wavefronts (2 from each block of its CU). Each
step of a wavefront has three phases: issue the loads of the next step, the unpacking and MFMA loop,
then store A and wait at the barrier. Normally the SIMD serves the oldest ready wavefront first.
For tiles of at most 32 rows, the kernel now sets a priority per phase (`s_setprio`):

```
 phase:     issue next loads  │  unpack + MFMAs  │  store A, barrier
 priority:        2           │        1         │        0
```

A wavefront that has just passed the barrier gets its loads out at once, before the others in their
MFMA loop issue their next instruction. A wavefront in its MFMA loop goes before one that only
stores A and waits. These tiles wait on the weight stream, and their loads now leave earlier.

Measured, µs per call (mean of 3 rounds × 40 calls on each of mv-mi250-02 and mv-mi250-03; the
real 20b routing and the synthetic 120b routing of the next section):

| model | call | B | before | after | change |
|---|---|---:|---:|---:|---:|
| 20b | mlp1 | 512 | 720.4 | **715.2** | −0.7 % |
| 20b | mlp2 | 512 | 403.9 | **400.6** | −0.8 % |
| 20b | gather + mlp1 + SwiGLU (engine) | 512 | 718.2 | **711.6** | −0.9 % |
| 20b | mlp1 | 1024 | 1,239.2 | **1,231.5** | −0.6 % |
| 20b | mlp2 | 1024 | 683.0 | **680.8** | −0.3 % |
| 20b | gather + mlp1 + SwiGLU (engine) | 1024 | 1,254.2 | **1,244.9** | −0.7 % |
| 120b | mlp1 | 512 | 848.0 | **842.6** | −0.6 % |
| 120b | mlp2 | 512 | 477.0 | **471.1** | −1.2 % |
| 120b | mlp1 + SwiGLU (engine) | 512 | 826.7 | **821.4** | −0.6 % |
| 120b | mlp1 | 1024 | 1,374.7 | **1,365.6** | −0.7 % |
| 120b | mlp2 | 1024 | 751.2 | **747.6** | −0.5 % |
| 120b | mlp1 + SwiGLU (engine) | 1024 | 1,369.2 | **1,360.7** | −0.6 % |
| 20b, 16 rows per expert | mlp1 / mlp2 | — | 315.9 / 174.8 | **301.6 / 164.8** | −4.5 / −5.7 % |
| 20b, 32 rows per expert | mlp1 / mlp2 | — | 403.1 / 251.6 | **398.6 / 224.2** | −1.1 / −10.9 % |
| 20b, 48 and 64 rows per expert | mlp1 / mlp2 | — | | | within ±0.5 % |

Hardware counters (rocprofv3, one call): requests to the L2 unchanged (20b mlp1 B=512 9.31 M, 120b
mlp1 B=512 10.47 M); matrix cores busy 69.9 → 70.4 % (20b mlp1 B=512), 64.1 → 64.7 % (120b mlp1
B=512), 61.4 → 62.0 % (20b mlp2 B=512), 56.8 → 57.6 % (120b mlp2 B=512).

Whole program, 64 prompts × 128 tokens, mv-mi250-02, 3 runs each: 20b on one GCD 3,438 →
**3,513 tok/s** (+2.2 %), 120b on a GCD pair 2,188 → **2,222 tok/s** (+1.6 %); output hashes
unchanged (`ccfc732a4a25`, `4ed6e262b73a`). At 64 prompts every expert has about 8 rows, the case
the priorities help most.

**Why only up to 32 rows.** With the same priorities, 48-row tiles were 12 % slower on uniform
mlp2 (287 → 323 µs, both machines), and 64-row tiles 1 % slower. Any priority in their MFMA loop
caused it; a priority for their loads alone still cost 6.5 %. On the real routings it made no
difference (±0.5 %), so those tiles keep the default (`MoePrio`, `fast_moe_gemm.hpp:324`).

**One small fix found on the way.** At the start of an item, the compiler had issued some scale
loads after the weight loads of step 0. Its wait counting then made the top of every loop turn wait
for all but one load in flight (`s_waitcnt vmcnt(1)` in the 16- and 48-row loops). A
`sched_barrier` after the scale loads (`:474`) keeps them first, and every loop turn now waits only
for the loads its step needs. The time did not change measurably, but the test of loading two steps
ahead only meant something after this fix.

### 9. Where the other half of the peak goes

20b mlp1 at B=512 does 2 × 2,048 × 5,760 × 2,880 = 68 GFLOP of useful work in 715 µs: 95 TFLOP/s,
52 % of the 181 TFLOP/s peak. The three factors below multiply to it (counters of one call,
rocprofv3, mv-mi250-03):

| factor | measured | × |
|---|---|---:|
| clock: 1.05 million cycles in 720 µs | 1.46 GHz, not the 1.7 GHz of the peak | 0.86 |
| matrix cores busy: MFMAs issued × 32 cycles / (416 SIMDs × cycles) | 9.62 M MFMAs → 70.3 % | 0.70 |
| of the issued MFMA work, the useful part: 67.95 of 78.8 GFLOP | rows 0.895 × columns 0.978 × k 0.978 | 0.86 |
| product: 181 × 0.86 × 0.70 × 0.86 | 94 TFLOP/s | |

"Busy" counts every issued MFMA, also the padding ones, so it is a measure of issue, not of
useful work. The padding comes from three places:

- **Rows** (−10.5 %): an item computes its rows in 16-row MFMA tiles, so an expert with 20 rows
  pays for 32. Measured on real routings: 20b 10.5 % (B=512) and 5.6 % (B=1,024); the gpt-oss-120b
  split 19.2 % and 10.6 % (its real routing, dumped from `./run`, positions from 16 on, is much more
  uneven than the synthetic one of the next section). Rows are already rounded up to 16, not 64; a
  smaller MFMA shape would add the products in another order, so this part stays as long as the
  output must not change by a bit.
- **Columns** (−2.2 % for mlp1, −6.25 % for mlp2): N = 5,760 is 22.5 tiles of 256, and N = 2,880
  is 11.25. In the last tile the wavefronts whose pieces lie past N (4 of 8 for mlp1, 6 of 8 for
  mlp2) load zero weights and still issue all their MFMAs.
- **k** (−2.2 %): K = 2,880 is 22.5 steps of 128; in the last step half of each MFMA's 16 k slots
  hold zeros. Packing the last 64 k into fewer MFMAs would also change the order of the additions.

The 30 % of time with idle matrix cores, measured per block with a timeline build (each block
records when it starts, when it ends, and the time it spends inside items):

| call, B=512 | idle at the end (queue tail) | gaps between items (ticket, lookup) | in items | items per block |
|---|---:|---:|---:|---:|
| 20b mlp1 | 3.6–3.7 % | 1.4 % | 93.8 % | 5.5 |
| 20b mlp2 | 7.3–8.4 % | 2.4 % | 88.3 % | 2.9 |
| 120b mlp1 / mlp2 (real routing) | 3.0–3.1 / 5.6–5.7 % | 1.5 / 1.5 % | 94.4 / 91.1 % | 7.8 / 4.1 |

(The rest, about 1 %, is the start, the last ticket and the exit.)

At B=1,024 the tail is 2.2 / 5.5 % (20b) and 1.9 / 3.5 % (120b). The rest, about 24 % of the time
inside items, are the stalls of section 8: the weight stream for small items, and LDS waits for
the big ones.

**What was tried against these parts** (all bit-identical; µs per call on the real routings, the
120b ones with `MOE_DUMP120`, 3 rounds):

- *Column padding:* the wavefronts without a column took a path that only copies A and joins the
  barriers. The fused mlp1 calls got 0.2–0.9 % faster, plain mlp2 at B=512 1.3 % slower (20b
  404 → 409, 120b 467 → 473), uniform 48/64-row mlp2 12 / 9 % slower. Those MFMAs on zero
  weights were not what the time waited for.
- *Queue tail:* the items of the smallest experts (≤ 16 rows) were split into two items of 128
  columns. As two halves on 4 wavefronts each (the others only copy A), the time grew 0.2–2.4 %:
  the busy wavefronts did as much as before. As 8 wavefronts with one 16-column piece each (half
  the work per wavefront), the tail did shrink (20b mlp1 B=512 3.6 → 2.2 %, mlp2 7.3 → 5.4 %, 120b
  mlp2 5.7 → 3.2 %). But each extra item repeats its A copy, ticket, scale loads and epilogue
  (about 12 µs per extra item), and the calls ended 0.6 % faster to 1.1 % slower.
- The gaps between items (ticket, expert lookup) are 1.4–2.4 % of a call: not attempted, as a
  ticket taken early would make the queue tail longer.

So the part worth a next attack is the 24 % inside items (section 8, "What could come next"), not
the padding or the queue.

### 10. Writing the results, and the registers of the work list

**What the stopwatch showed.** A timeline build (see the words) wrote the time at the start of each
item, after its first barrier, after its step loop and after its last store. Summed over a real call
(20b mlp1, B=512; 120b mlp1 B=512 alike), a block spent its time like this:

| part of an item | µs per item | share of the block's time |
|---|---:|---:|
| ticket, item lookup, scale and step-0 loads, first barrier | 9.2–9.7 | 7–9 % |
| the 23 steps | 81–98 | 74–76 % |
| **epilogue (bias, stores, SwiGLU)** | **14.8–16.3** | **13 %** |
| idle at the end of the call | | 3–4 % |

The epilogue writes 64 × 256 numbers at most; that should take 1–2 µs, not 16.

**Why it was slow.** The machine code showed `flat_store` instructions. The pointer to the output
could be `nullptr` (`C ? C + ... : nullptr`), so the compiler no longer knew it pointed to GPU memory
and used the general *flat* stores. A flat store counts as a memory operation and also as an LDS
operation. In the SwiGLU epilogue every output needs the value of the neighbouring lane, which
`__shfl_xor` fetches through the LDS (`ds_bpermute`), and then waits for its LDS counter to reach
zero: that wait also waited until every earlier store had reached memory. So the 32 stores of a lane
went out one at a time, each a full round trip to memory. The plain epilogue (mlp2) had the same flat
stores, each behind a branch (`if (m < mv)`).

**The fix** (`fast_moe_gemm.hpp:508`). The results now go out through buffer descriptors, like the
loads:

- the output descriptor of an item starts at its first row and ends after its last valid row
  (`mv`), so the hardware simply drops the stores to the rows past `mv` (no branch);
- a lane whose column is past N (or, for SwiGLU, an odd lane) gets an offset past the end, so its
  stores are dropped too;
- the bias is read through a descriptor that returns 0 past N or without a bias;
- the neighbouring lane's value comes from a DPP move (`__builtin_amdgcn_mov_dpp`, `:564`), not
  through the LDS.

The epilogue fell from 14.8–16.3 µs to 3.9–4.2 µs per item, and the prologue from 9.2–9.7 to
7.2–7.9 µs (its barrier no longer waits for a slow epilogue). The stored values come from the same
instructions as before, so the results are the same bits.

**Fewer registers for the work list** (`:203`, `:576`). Every wavefront keeps the work list in
registers: for each group of 64 experts, the expert of each rank and the number of row tiles before
it. With room for 256 experts that was 8 registers, held during the whole step loop, and the step
loop needs all 128. The compiler then parked some values in scratch (29 spilled registers in the
first version). Now one register per group holds both numbers (expert in the low 8 bits), at most
128 experts are supported (gpt-oss has 32 or 128; one GCD of the 120b split has 64), and the kernel
is compiled twice, for 1 and for 2 groups; calls of up to 64 experts use the first. No register is
spilled any more. (Two pitfalls on the way, see Pitfalls: as a template the kernel stopped inlining
`moe_item`, and the 2-group version once took 132 registers.) Each item also computes its 2 output
columns after the step loop instead of before it, which frees 2 more registers during the loop.

**Results.** µs per call, mean of 3 rounds × 40 calls, one GCD of mv-mi250-03, real routings (the
120b ones from `MOE_DUMP120`); "engine call" = the variant the engine uses for mlp1 (20b: gather +
mlp1 + SwiGLU; 120b: mlp1 + SwiGLU):

| model | call | B | before | after | change |
|---|---|---:|---:|---:|---:|
| 20b | mlp1 | 512 | 714.4 | **652.1** | −8.7 % |
| 20b | mlp2 | 512 | 401.4 | **355.9** | −11.3 % |
| 20b | gather + mlp1 + SwiGLU (engine call) | 512 | 707.2 | **665.5** | −5.9 % |
| 20b | mlp1 | 1024 | 1,224.9 | **1,157.0** | −5.5 % |
| 20b | mlp2 | 1024 | 678.1 | **618.2** | −8.8 % |
| 20b | gather + mlp1 + SwiGLU (engine call) | 1024 | 1,238.6 | **1,203.2** | −2.9 % |
| 120b | mlp1 | 512 | 830.0 | **756.5** | −8.9 % |
| 120b | mlp2 | 512 | 465.0 | **412.4** | −11.3 % |
| 120b | mlp1 + SwiGLU (engine call) | 512 | 807.2 | **766.6** | −5.0 % |
| 120b | mlp1 | 1024 | 1,343.9 | **1,255.4** | −6.6 % |
| 120b | mlp2 | 1024 | 736.9 | **674.5** | −8.5 % |
| 120b | mlp1 + SwiGLU (engine call) | 1024 | 1,337.4 | **1,287.9** | −3.7 % |
| 20b, 16 / 32 / 64 rows per expert | mlp1 | — | 299.3 / 398.5 / 641.8 | **267.5 / 353.8 / 600.9** | −10.6 / −11.2 / −6.4 % |
| 20b, 16 / 32 / 64 rows per expert | mlp2 | — | 164.7 / 224.6 / 348.8 | **142.8 / 216.0 / 319.7** | −13.3 / −3.8 / −8.3 % |

The two parts, measured one after the other: the new epilogue gave −3.5 to −8.1 % (engine calls
−1.9 to −3.5 %), the work-list registers and the columns after the loop another −0.8 to −3.0 % (of
which the columns −0.1 to −1 %).

The fused mlp1 calls gain less than plain mlp1 (−5.9 against −8.7 % for 20b at B=512). Part of
their epilogue is the SwiGLU arithmetic (an exponential and a division per output, the same
instructions as `swiglu_bf16_kernel`, which the bits must match), and that part did not change.

Whole program, official input (4,096 requests), all 8 GCDs of mv-mi250-02, old and new builds
alternated twice: 20b 76,202 → **78,680 tok/s** (+3.3 %), 120b 42,966 → **44,154 tok/s** (+2.8 %);
output hashes unchanged (20b `58023f92ead7`, 120b `973e97c40a85`). The new epilogue alone had
given +1.5 % (20b) and +1.6 % (120b).

**What else was tried in this round** (all bit-identical; µs per call on the same real routings):

- *One block of 16 wavefronts per CU, 512 columns per item* (fewer, longer weight streams; half the
  A traffic per multiply-add): −0.2 to −0.5 %, i.e. no gain. The stream benchmark explains it: the
  plain weight stream with 16 wavefronts per CU and 32 KB per step reaches only 0.97–1.01 TB/s
  (1.12–1.18 TB/s when one load instruction reads 1 KB in one piece), against 1.10 TB/s for 2 blocks
  of 8 wavefronts (1.20–1.26 with 1 KB pieces) and 1.34 TB/s for 1 block of 8 (either way). What
  the stream likes is fewer wavefronts per CU reading at the same time, not longer steps.
- *The two blocks of a CU take items from the two ends of the list* (one the largest experts, one
  the smallest; each block finds its CU in the `HW_ID` register, which tells a wavefront where it
  runs). Small experts wait on memory and large ones on the matrix cores, so a CU would run one of
  each. 120b mlp1 B=512 −1.8 to −3.2 %, but 20b +1 to +5 % and 120b mlp2 ±1 %. The timeline showed why: next to a small item a 64-row item took 165 µs
  instead of 151 (the small item's wave priorities win the SIMD), and the call ended on medium items
  (blocks finished up to 130 µs apart instead of 50). Measured alone, the small experts of a 120b
  call take 250 µs and the large ones 604 µs, the whole call 820 µs: they overlap by only ~35 µs,
  but small items are not purely memory-bound (their 16-row MFMAs, 63 % padding, still need a third
  of the matrix cores' time).
- *The next item's ticket, scales and step-0 loads issued during the current item's epilogue*, and
  *only the ticket and the bias taken during the last step*: +2 to +7 % and +2 to +5 %. Both needed
  more values alive across the step loop; the compiler spilled 21–122 registers, and every reload
  from scratch waited (in order) for the loads just issued.
- *The last step's (unused) loads of a "next step" sent past the end of the buffer* (no memory
  traffic): +0.3 to +0.7 % (a select in the loop).
- *Step-0 A loads before the scale loads* (so the first barrier does not wait for the scales):
  ±0.3 %, and every loop turn got the extra wait that the comment at `:469` warns about.

## What was limiting it: requests to the L2 cache

The previous draft of this kernel (64 × 128 items, 4 wavefronts, steps of 64 k, one scale byte loaded
per step) already used the matrix cores, but a hardware-counter profile (rocprofv3, 20b mlp1, B=512)
showed the matrix cores busy only 38 % of the time. The telling number:

```
 L2 requests per call: 30.3 million in 1.88 million clock ticks = 16.1 per tick
 the L2 has 16 channels, each answering about 1 request per tick  →  it was at its limit
```

Per item it asked for 64 rows of A (5,760 lines), 128 columns of weights at 32 bytes per line and
step (5,760 lines), and a whole line for each 1-byte scale (5,760 lines). The new layout needs fewer:

| per item | draft (64 × 128) | new (64 × 256) | packed (section 7) |
|---|---:|---:|---:|
| A lines | 5,760 | 5,760 (but for twice the columns) | 5,760 |
| weight lines | 5,760 | 8,832 (for twice the columns: 16 bytes per thread; odd columns straddle two lines) | 5,760 (every line read whole) |
| scale lines | 5,760 | 360 | 384 |
| lines per 1,000 multiply-adds | 0.73 | 0.32 | 0.25 |

Measured: **14.2 million requests per call (11.4 per tick), matrix cores busy 59 %**; with the
packed weights 9.3 million and 70 %.

Two compiler details mattered as much (both visible only in the generated machine code, kept in
the build with `-save-temps`):

- **The item number must be known to be the same in every thread** (`__builtin_amdgcn_readfirstlane`).
  It comes from LDS, so the compiler assumed it could differ per thread; everything computed from it
  then lived in per-thread registers, and every A load turned into a small loop over all distinct
  values, which also made the compiler wait for *all* outstanding loads at the top of each step —
  prefetch included.
- **The compiler moved the loads to the end of the step** (to free registers), where nothing hides
  their latency, and pulled the A reordering right after its own load. Two
  `__builtin_amdgcn_sched_barrier(0)` markers keep the loads at the start and the reordering at the end.

What was left before section 7 (test builds that skip one part, so their results are wrong but their
time tells the cost of that part): without the barrier and A copy the time does not drop (871 µs vs
845–857 µs); without the unpacking it drops 3 %; without the weight loads, 17 % (and 42 % when every
expert has only 16 rows). With the packed weights (before the last two changes of section 7):
without the weight loads −11 % (−17 % at 16 rows per expert), without the A loads −10 % (no change
at 16 rows per expert).
So the weight traffic is still the main cost, and the register file is full (128 per thread), which
is why deeper prefetch did not help (see below).

## The numbers for our model

One GCD of mv-mi250-03, `getp-csrc/bench/moe_gemm_bench.cpp`, mean of 40 calls (the simple kernel: 4
calls), each call on the routing of another real layer and step (20b, dumped from `./run`). 120b: one
GCD of a pair holds 64 of the 128 experts and gets the rows of both GCDs' tokens (≈ 4 × B rows),
synthetic routing with the same unevenness. "Library" = the vendor library (hipBLASLt through
PyTorch, bf16 weights, so 4× the weight bytes) called once per expert on the same routing, all calls
of one grouped multiply captured in one HIP graph so launch costs are small
(`getp-csrc/bench/moe_torch_lib.py`, mv-mi250-02); "library est." = the sum over experts of the
single-call library time for that expert's row count (`yardsticks/gemm_bench.json`). µs per call,
before the packed layout (section 7 has the current numbers):

| model | shape | B | simple | draft | **new** | new TFLOP/s | library | library est. | new vs library |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20b | mlp1 5760×2880 | 512 | 13,168 | 1,227 | **857** | 79 | 1,691 | 1,433 | 1.97× |
| 20b | mlp2 2880×2880 | 512 | 6,745 | 616 | **472** | 72 | 1,098 | 988 | 2.33× |
| 20b | mlp1 | 1024 | 20,815 | 1,851 | **1,377** | 99 | 2,339 | 1,901 | 1.70× |
| 20b | mlp2 | 1024 | 10,546 | 958 | **746** | 91 | 1,457 | 1,244 | 1.95× |
| 20b | mlp1 | 2048 | 37,340 | 3,161 | **2,476** | 110 | 3,645 | 2,990 | 1.47× |
| 20b | mlp2 | 2048 | 18,424 | 1,666 | **1,324** | 103 | 2,350 | 1,760 | 1.78× |
| 120b | mlp1 | 512 | 17,771 | 1,718 | **1,116** | 60 | 2,870 | 2,477 | 2.57× |
| 120b | mlp2 | 512 | 9,065 | 953 | **602** | 55 | 1,979 | 1,729 | 3.29× |
| 120b | mlp1 | 1024 | 25,194 | 2,427 | **1,670** | 80 | 3,517 | 2,947 | 2.11× |
| 120b | mlp2 | 1024 | 12,706 | 1,285 | **891** | 75 | 2,281 | 2,035 | 2.56× |
| 120b | mlp1 | 2048 | 43,561 | 3,786 | **2,819** | 100 | 4,820 | 3,998 | 1.71× |
| 120b | mlp2 | 2048 | 21,599 | 1,997 | **1,501** | 94 | 2,994 | 2,577 | 1.99× |

The goal set before the work was "beat 32 × the library time at 64 rows": 1,555 µs (mlp1) and
1,040 µs (mlp2) at B=512. Reached: 857 and 472 µs.

Peak for comparison: the matrix cores do 181 TFLOP/s at 1.7 GHz, but this kernel runs at about
1.5 GHz (measured from the profiler's clock counter), so 160 TFLOP/s is the real ceiling; the new
kernel reaches 34–69 % of it. Memory: the weight bytes of a 20b mlp1 call (at most 282 MB) at
1.25 TB/s would take 225 µs, so at B=512 the kernel is neither purely memory- nor purely
arithmetic-limited.

## What it gave

Per call: 15× faster than the simple kernel, 1.5–3.3× faster than the library, and 1.3–1.6× faster
than the first matrix-core draft, on every shape.

Whole program, gpt-oss-20b, 512 prompts × 128 new tokens, one GCD (an engine where the dense
multiplies were still the simple version): 662 tok/s (simple) → 1,710 (draft) → **1,797** (new).
In the lead's profile after the dense-GEMM change (477 ms per step, 96 % in this kernel at
≈ 9.5 ms per call), the new kernel's ≈ 1.3 ms per mlp1 + mlp2 pair brings the expert multiplies from
≈ 456 ms to ≈ 32 ms per step.

**One more step, now used by the engine:** `gemm_grouped_mxfp4_gather_swiglu` does the
gather (copying each token's row into expert order), mlp1 and SwiGLU in one kernel, so the
fp32 mlp1 output (47 MB per layer at B=512) is never written and read back. Measured (mlp1, same
setup): 20b B=512 947 → 802 µs, B=1024 1,560 → 1,342 µs, B=2048 2,823 → 2,471 µs, 120b B=512
1,208 → 1,036 µs (−12 to −15 %), bit-identical to the three separate kernels
(`MOE_FUSED=1 ./moe_gemm_bench …`).

**Weights stored in reading order** (section 7): another 7–25 % per call; 64 prompts × 128 tokens
20b 2,743 → 3,315 tok/s, 120b 1,433 → 1,755 tok/s.

**Wave priorities** (section 8): another 0.3–1.2 % per call, 5–11 % for experts of 16–32 rows; 64
prompts × 128 tokens 20b 3,438 → 3,513 tok/s, 120b 2,188 → 2,222 tok/s.

**Results written with buffer stores, a smaller work list** (section 10): another 3–13 % per call
(engine calls 3–6 %); whole program, official input, 8 GCDs: 20b 76,202 → 78,680 tok/s (+3.3 %),
120b 42,966 → 44,154 tok/s (+2.8 %).

**What could come next.** After section 10 a block spends about 84 % of its time in the step loop,
7–9 % between items (ticket, lookup, the first loads and barrier; 9–10 µs per item) and 3–6 % idle
at the end of the call. Ablations (test builds without one part, 20b mlp1 B=512, µs per call before
section 10): without the in-loop loads 616 instead of 710, also without LDS reads 614, also without
unpacking 584, also without barriers 586 (MFMAs alone: 81 % busy, the rest is item boundaries and
the queue's end). So:

- The loads cost about 13 % (120b: 17 %), the LDS reads and the unpacking almost nothing while
  loads are in flight. The A operands read just before their MFMA, which section 8 suspected, are
  not the limit: removing the LDS reads entirely saved 0–2.4 %.
- For 120b the small experts (≤ 16 rows, half of all items) are the memory-bound part: alone they
  take 250 of the call's 820 µs, at about 1.0 TB/s. The weight stream itself is faster with fewer
  readers per CU (1.34 TB/s with one block of 8 wavefronts per CU, 1.10 with two) and with 1 KB read
  in one piece per load instruction (1.20–1.26 TB/s with two blocks); no kernel version so far turned
  that into a gain.
- Moving work across item boundaries (the next item's loads during this item's epilogue) failed only
  on register pressure (section 10). A version with fewer registers in the step loop would make it
  possible.
- 19 % of 120b's issued MFMA work is row padding (rows rounded up to 16). A smaller MFMA shape would
  change the order of the additions (no longer bit-identical), so it needs scored runs.

## Where it is in our code

All in `getp-csrc/kernels/fast_moe_gemm.hpp`:

- `:39` the packed layout, described at the top of the file; `:122` `moe_packed_blocks_bytes`,
  `:125` `moe_packed_scales_bytes` — its size per expert.
- `:654` `moe_pack_blocks_kernel`, `:676` `moe_pack_scales_kernel`, `:701` `moe_pack_mxfp4` —
  the repacking, run once at load time.
- `:138` `fp4_tables`, `:158` `fp4x8_to_bf16` — unpacking MXFP4 to bf16 with `v_perm_b32`.
- `:184` `MoeSmem` — the 32 KB of LDS (two A buffers).
- `:203` `MoeTiles` — the work list, one register per group of 64 experts (section 10);
  `:211` `moe_tile_prefix`, `:261` `moe_find_rank` — counting each expert's row tiles, biggest
  first, and finding the item of a ticket.
- `:274` `moe_store_a` — A into LDS in the "even k first" order.
- `:325` `MoePrio` — the wave priorities of section 8 (2 / 1 / 0 for items of at most 32 rows).
- `:336` `moe_step` — one step of 128 k: priority for the loads (`:360`), loads of the next step
  (`:365` the weights, `:369` keeps them at the start), priority for the MFMA loop (`:371`),
  unpacking, 64 MFMAs, priority back to 0 (`:403`), A into the other buffer, barrier.
- `:416` `moe_item` — one work item: the tile's packed buffer descriptor (`:441`), per-thread
  offsets, the scale loads (`:453`) kept before the step-0 loads (`:474`), the step loop
  (`:500`), the epilogue through buffer descriptors (`:508`; stores `:531`, SwiGLU with the DPP
  move `:564`).
- `:576` `gemm_grouped_mxfp4_mfma_kernel` — the persistent loop, compiled for 1 and 2 groups of 64
  experts: ticket (`:594`), made uniform (`:600`), the 1–4 row-tile variants (`:624`), queue reset
  (`:643`); `:821` picks the variant.
- `:834` `gemm_grouped_mxfp4` — the entry point for mlp2 (`getp-csrc/engine/gpu_engine.hpp:569`);
  `:773` the simple version, still used (on the checkpoint layout) for shapes the fast kernel does
  not take (`:806` `moe_mfma_ok`).
- `:852` `gemm_grouped_mxfp4_swiglu` (120b split, `gpu_engine.hpp:559`), `:867`
  `gemm_grouped_mxfp4_gather_swiglu` (20b, `gpu_engine.hpp:551`) — the fused mlp1 versions, equal
  bit for bit to the separate kernels.
- `getp-csrc/engine/gpu_engine.hpp:217` — the `experts` helper of `gpu_upload_weights` that stages
  and packs each layer's expert weights (`:259` mlp1, `:261` mlp2).

Tests and measurements: `getp-csrc/tests/moe_gemm_test.cpp` (unit test),
`getp-csrc/bench/moe_gemm_bench.cpp` (speed, `MOE_UNIFORM=64` for even routing, `MOE_ONLY=…` for
one configuration, `MOE_DUMP120=…/off120_b%d.txt` for the real routing of the 120b split instead of
the synthetic one), `getp-csrc/bench/moe_stream_bench.cpp` (the weight stream alone, sections 8 and
10, also with one block of 16 wavefronts per CU), `getp-csrc/bench/moe_torch_lib.py` (library
yardstick, separate script).

## Correctness

- Unit test, against a CPU reference in double precision of the same bf16 A and the decoded
  weights: 24 cases — real routing (first and last dumped layer), 32 / 64 / 128 experts, skewed
  random routing, every token on the same 4 experts, one expert with all 2,048 rows, row counts
  around 16 and 64 (1, 15, 16, 17, 63, 64, 65, …), empty experts, small odd shapes. Worst error
  divided by Σ|A·W| (the size fp32 rounding grows with): 8.0e-7; worst relative error 2.3e-5.
- Unpacking: one-hot rows of A turn the output into the weights themselves: all 16 codes × scale
  bytes 2..252 are bit-exact.
- The fused SwiGLU variants: bit-identical to the separate kernels on 5.9 million outputs.
- The packer: every byte of the packed copy where the layout says, against a host version (mlp1
  shape and two odd shapes with a half step and N not a multiple of 16 or 256).
- Packed against checkpoint layout: the kernel before section 7 (checkpoint layout) and the current
  one (packed) give the same bits on 1.45 billion outputs — real routings of 11 layer-steps each at
  B=512 and 1,024, synthetic 120b routings at B=512 to 2,048, 16 rows per expert, tile-edge row
  counts, 128 experts, small odd shapes — for the plain multiply and both fused versions.
- End-to-end output hashes unchanged by section 7: 20b `ccfc732a4a25` (one GCD), 120b
  `836592181c82` (GCD pair), 64 prompts × 128 tokens.
- Section 8 (wave priorities, scale-load order): every instruction gets the same operands in the
  same order, so the bits cannot change; checked anyway. The kernel before section 8 and after it,
  on the same packed weights, give the same bits on 1.45 billion outputs (the same routings as
  above, plain multiply and both fused versions). The unit test passes. The output hashes of `./run`
  (64 prompts × 128 tokens) are unchanged on one GCD (20b) and on a GCD pair (120b). The inner loops of the generated code have no `s_waitcnt vmcnt(0)` and no register
  copies (checked in the `.s` file).
- Section 10 (epilogue through buffer descriptors, the work list in one register per group): the
  stored values come from the same instructions in the same order. The kernel before section 10 and
  after it give the same bits on 2.73 billion outputs: the real 20b routings at B=512 and 1,024, the
  real 120b-split routings at B=512 and 1,024 (`MOE_DUMP120` files), synthetic 120b routings at
  B=512 to 2,048, 16 rows per expert, tile-edge row counts, 128 experts (the 2-group variant), small
  odd shapes; plain multiply and both fused versions. The unit test passes (both variants). Output
  hashes of the full official run on 8 GCDs unchanged: 20b `58023f92ead7`, 120b `973e97c40a85`. The
  inner loops still have no `s_waitcnt vmcnt(0)` and no register copies; no register is spilled.
- End to end, 64 prompts × 128 tokens: 49 of 64 answers identical to the reference, 88.6 % of tokens
  before the first difference (simple kernel: 43 / 64, 80.9 %). The order of additions changed, so
  some answers drift later, as with every bf16 change (doc 10).

## What did not help

| idea | result (20b mlp1 B=512 unless noted) | why |
|---|---|---|
| draft with 64 × 256 items on 4 wavefronts | 1,040 → 5,032 µs (fastest call) | 192 registers needed: spills, 2 wavefronts per SIMD |
| draft with 2 instead of 4 wavefronts per SIMD | 1,207 → 1,070 µs | better, but the request limit stayed; superseded |
| draft: no branches around the prefetch loads; A prefetched 2 steps ahead | no change (1,247 → 1,262 / 1,272 µs) | the limit was the L2 requests, not latency |
| scales from a fixed byte (steps unrolled by 4) | 856 → 901 µs | 9 % fewer instructions but twice the code |
| zero codes in the last half step by an out-of-range address instead of a select | 845 → 864 µs | more instructions in the end (compiler) |
| weights prefetched 3 steps ahead | 848 → 1,022 µs | 13 spills inside the loop (128-register limit) |
| 3 rotating weight registers, reload right after unpacking | ≈ 850 → 1,199 µs | 34 spills |
| skip the arithmetic of wavefronts whose columns are all past N | within noise | the branch made the compiler wait for all loads at every loop turn |
| packed, but one load instruction = 1 KB contiguous (piece first, kq second; K padded to whole steps) | 749 → 755 µs; 120b 878 → 886 µs; 16 rows 329 → 324 µs | no gain over 4 runs of 256 bytes, and 2.2 % more memory for the padding |
| packed, weights still loaded two steps ahead (two register sets) | 734 → 751 µs (all 8 shapes 1.5–2.8 % slower) | one step is enough once the weights stream contiguously; the second set only takes registers |
| packed, the decode reads a forced copy of the loaded registers | 722 → 734 µs | 8 more instructions per step; without it the compiler alternates two register sets by itself |
| A loaded in whole lines: 4 neighbouring lanes read the 4 quarters of one 64-byte line in one instruction (section 8) | 717 → 719–724 µs; 64 rows 643 → 667 µs | L1 lookups −34 %, but L2 requests unchanged (11.89 → 11.88 M): the L1 had already merged the two halves; the new pattern first tripled LDS bank conflicts (3.4 → 10.4 M cycles; fixed with an XOR of the LDS row) |
| A rows past the expert's last row read as zeros (no memory traffic) instead of repeating the last row | within noise (722–724 µs) | the repeated rows hit the same line in the same instruction, so they cost almost nothing before either |
| weights two steps ahead for tiles of ≤ 32 rows (whole step unpacked first, which their fewer accumulators allow) | 721 → 726 µs; 120b 848 → 861 µs; 16 rows 319 → 315 µs; with 48-row tiles too 752 / 893 µs | the stream of weights does not get faster with more in flight (`moe_stream_bench`: 1,116 → 1,148 GB/s), and unpacking everything first loses the mixing of unpacking and MFMAs |
| "step-major" weight order (step p of all tiles of an expert side by side) | 16 rows 313 → 305 µs; real routings −0.3 to +0.9 %; 64 rows +1.4 % | the plain stream gets 15 % faster (1,116 → 1,290 GB/s), but in the kernel the stream is not the only wait |
| scale bytes shifted out once per two steps instead of once per step | same machine code | the compiler had already merged the two one-byte shifts of a loop turn |
| first quarter of the next step unpacked before the barrier | +3 to +6 % on every shape | the barrier then waits for every wave's weight loads (`s_waitcnt vmcnt(0)` before it) |
| k-steps 2d and 2d+1 of a row side by side in LDS (one 16-byte LDS read for both) | +5 to +16 % | its A mapping put the two halves of a line in different halves of the wavefront: L2 requests 9.31 → 9.83 M |
| MFMAs ordered row tile first | ±0.5 % | the compiler schedules both orders the same way |
| more registers per thread (2 or 3 wavefronts per SIMD, 177 / 163 registers, one block per CU) | +11 to +15 % | half the wavefronts to hide waits with |
| items handed out in fixed round-robin order instead of the queue | 717 → 823 µs (+15 %) | the biggest-first queue matters; without the per-step barrier on top: 821 µs, 16 rows 325 → 344 µs (the barrier is not what limits small tiles) |
| wave priorities also for 48-row tiles | uniform 48-row mlp2 287 → 323 µs (+12 %); real routings ±0.5 % | see section 8; any MFMA-loop priority did it, a load priority alone +6.5 % |
| wave priorities also for 64-row tiles | +1 % | these tiles are bound by the MFMA loop itself |
| wavefronts without a column (last tile) only copy A instead of multiplying zero weights (section 9) | fused mlp1 −0.2..−0.9 %; mlp2 B=512 +1.3 % (20b and 120b); uniform 48 / 64-row mlp2 +12 / +9 % | the time did not wait on those MFMAs |
| items of experts with ≤ 16 rows split into two 128-column items on 4 wavefronts each | +0.2..+2.4 % | the busy wavefronts did the same work as before; the other 4 only copied A |
| the same split, 8 wavefronts with one 16-column piece each | queue tail 3.6 → 2.2 % (20b mlp1 B=512), 7.3 → 5.4 % (mlp2); calls −0.6..+1.1 % | each extra item repeats its A copy, ticket, scale loads and epilogue (about 12 µs) |
| one block of 16 wavefronts per CU, items of 64 × 512 (section 10) | −0.2..−0.5 % on all 8 real-routing calls | the weight stream gets slower with 16 wavefronts per CU (0.97–1.01 TB/s with 32 KB steps), which eats the halved A traffic |
| the two blocks of a CU take items from the two ends of the list (largest / smallest experts), CU from `HW_ID` | 120b mlp1 B=512 −1.8..−3.2 %; 20b +1..+5 %; 120b mlp2 ±1 % | a 64-row item next to a small one took 165 instead of 151 µs, and the call ended on medium items (blocks finished 130 instead of 50 µs apart) |
| the next item's ticket, scales and step-0 loads during the current item's epilogue | +2..+7 % | 32–122 spilled registers; each reload from scratch waited for the loads just issued (the wait counters are in order) |
| ticket and bias taken during the item's last step (which then loads nothing) | +2..+5 % | 21 spilled registers outside the loop |
| last step's unused loads sent past the end of the buffer | +0.3..+0.7 % | a select in the loop body |
| step-0 A loads before the scale loads | ±0.3 % | every loop turn gets one more wait (`:469`) |
| room for 128 experts in every call (one variant, 2 registers per group) | 0.3–1.4 % slower than the 1-group variant for 32 / 64 experts | two registers held through the step loop, and 2 spilled ones |

## Pitfalls

- **A and W must use the same k order.** The matrix core happily multiplies mismatched pairs and the
  result still looks like plausible numbers. The one-hot test turns the output into the decoded
  weights, so any mix-up shows up as a wrong weight, not as a small error.
- **The smallest scale bytes do not fit.** For scale byte 1, 0.5 × 2^(1−127) is below the smallest
  normal bf16; we do not build that pattern. Scale bytes 2..252 are exact; the checkpoint uses 115..136.
- **The last half step.** 2,880 is not a multiple of 128: in the last step the threads whose 32 k lie
  past the end ask for bytes past the end of the tile's buffer descriptor and get zero codes. The A
  values they meet belong to the next row (finite numbers), so they add exactly 0. This relies on
  the hardware's range check counting the per-step offset that sits in a scalar register, not only
  the per-thread offset; a small test confirmed it on gfx90a (voffset + soffset ≥ length → 0).
- **The engine and the kernel must agree on the layout.** The fast entry points take packed
  weights exactly when `moe_mfma_ok` holds; the engine packs under the same condition and never on
  the exact path, whose kernels read the checkpoint layout. Handing checkpoint bytes to the fast
  kernel gives plausible-looking garbage, not a crash.
- **Register moves at the end of the loop.** Whether the loaded weights land in the registers the
  next step reads is up to the compiler. In one version it loaded into other registers and copied
  them back at the end of every loop turn, after waiting for all loads in flight (`s_waitcnt
  vmcnt(0)` followed by 8 `v_mov_b32`). Check the inner loop of the `.s` file after every change.
- **Look at the machine code.** Both big slowdowns found here (a per-load loop from a value the
  compiler thought could differ per thread, and loads moved to the end of the step) were invisible in
  the source and in the results; they show in the `.s` file and in the `s_waitcnt` (wait for memory)
  instructions inside the loop.
- **Registers are the wall.** At 4 wavefronts per SIMD each thread has 128 registers; every idea that
  needed 8 more spilled and lost 20–40 %.
- **One queue counter per GPU.** Two calls of this kernel on two streams of the same GPU at the same
  time would share the counter and mix up their work. The engine uses one stream per GPU.
- **Rows past an expert's end** are loaded (copies of its last row) and computed, but never stored;
  columns past N load zeros (an address past the end of the buffer reads 0 without memory traffic).
- **The order of the loads at an item's start matters for every loop turn.** The compiler counts
  loads in flight across the loop's back edge. When a load issued before the loop (here a scale
  load) is still counted as pending, the top of every loop turn waits for all but one load
  (`vmcnt(1)`), including the next step's weights. Keep such loads first (`sched_barrier`) and look
  at the `s_waitcnt` at the loop top.
- **Ablation builds need the same check.** One "loads only" test build got an `s_waitcnt vmcnt(0)`
  at the end of its loop, so its time said nothing about loading further ahead. Read its `.s` before
  trusting its time.
- **Microbenchmarks change with compile-time constants.** The same stream test measured 1,148 or
  1,344 GB/s at 104 blocks depending on whether K was a constant or a kernel argument (the compiler
  builds a different loop). Pass sizes at run time, as the kernel gets them.
- **A pointer that may be `nullptr` makes flat stores.** `C ? C + x : nullptr` loses the
  knowledge that the pointer is GPU memory, and the compiler falls back to `flat_store`, which also
  counts as an LDS operation. Every later LDS wait (`s_waitcnt lgkmcnt`) then waits for the stores
  to reach memory. Search the `.s` file for `flat_`; this kernel now has none except the item slot.
- **Templates change inlining.** Once the kernel became a template (two instances), the compiler
  stopped inlining `moe_item`: real calls, 552 bytes of scratch per thread, 3 times slower. It is
  `__forceinline__` now. Look for `s_swappc` in the `.s` file.
- **Uniform values must look uniform.** A value the compiler cannot prove is the same in every lane
  (one read from memory through a struct field, a total made with `__shfl`, a select on a boolean)
  puts a buffer descriptor into vector registers, and every load becomes a loop
  (`v_readfirstlane` + `s_cbranch_execnz`). `__builtin_amdgcn_readfirstlane` fixes it. And
  `atomicAdd` in one thread is rewritten by the compiler into a wave-wide version that waits for the
  result at once, wherever the result is used.
- **Registers outside the step loop count too.** Every value held across the step loop (the work
  list, the output columns, a next item's loads) takes a register the loop needs; at 128 the
  compiler spills, and a reload waits, in order, for all loads issued before it.
- **Uniform routings can mislead.** With every expert at 48 rows, mlp2 has 384 items for 208 blocks
  (less than two rounds), and the time depends on how the last round is spread. A change can look
  12 % worse there and neutral on real routings. Always measure both.

## Further reading

- OCP, *Microscaling (MX) formats specification v1.0* (2023) — the MXFP4 format.
- AMD, *CDNA2 instruction set architecture* — `v_mfma_f32_16x16x16bf16_1k` operand layout, `v_perm_b32`,
  buffer loads. AMD's matrix instruction calculator (github.com/ROCm/amd_matrix_instruction_calculator)
  prints which thread holds which element.
- Doc 07 (matrix cores for the dense multiplies) — the same instructions on 16-bit weights.
- Frantar et al., *MARLIN: mixed-precision auto-regressive parallel inference on large language
  models* (2024) — 4-bit weights unpacked in registers right before the matrix units, on NVIDIA.
- Gale et al., *MegaBlocks* (2022) — grouped GEMM for mixture-of-experts.
