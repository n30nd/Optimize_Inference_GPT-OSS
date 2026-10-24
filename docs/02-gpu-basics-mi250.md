# 02 — GPU basics on the AMD MI250

## The problem in plain words

A GPU is thousands of small calculators that all run the same short program at once, each on its own
piece of data. It is very fast at arithmetic, but it can only work on numbers once they have been
fetched from its memory, and fetching is often the slow part. Nearly every optimization later in
this project is about one of two things: keeping the calculators busy, or fetching fewer bytes.

## An everyday comparison

A huge kitchen with hundreds of cooks (calculators) and one corridor to the pantry (memory). Fast
cooks do not help if the corridor is jammed; a wider corridor does not help if every cook is already
busy. Which one limits you depends on the recipe: how much cooking each ingredient needs.

## The words

- **GCD (graphics compute die)** — one MI250 card holds two; the software shows each as its own GPU,
  so our machines show 8 "GPUs".
- **kernel** — a small program that runs on the chip; we launch one per step of the model.
- **thread / wavefront / block** — a thread is one calculator's job; AMD runs threads in groups of 64
  that move in lockstep (a *wavefront*); a *block* is a few wavefronts that can share fast memory.
- **CU (compute unit)** — a group of calculators with its own fast scratch memory; one GCD has 104.
- **HBM (high-bandwidth memory)** — the GPU's main memory: big (64 GB per GCD) but far away.
- **LDS (local data share)** — the small, fast scratch memory inside each CU (64 KB), shared by a
  block. Copying a tile of data into LDS once and reusing it many times is the core GEMM trick.
- **matrix cores (MFMA)** — special units that multiply small matrices in one instruction, ~8× faster
  than doing it number by number.
- **memory-bound / compute-bound** — limited by fetching bytes / by doing arithmetic.
- **arithmetic intensity** — arithmetic done per byte fetched. Low → memory-bound; high → compute-bound.

## How it works

```
 one MI250 card
 ┌───────────────────────────────┬───────────────────────────────┐
 │ GCD 0                         │ GCD 1                         │
 │  104 CUs, each:               │                               │
 │   4 SIMD units + matrix cores │        (same)                 │
 │   64 KB LDS                   │                               │
 │  8 MB L2 cache                │                               │
 │  64 GB HBM2e  ~1.6 TB/s       │  64 GB HBM2e                  │
 └───────────────────────────────┴───────────────────────────────┘
        └──── xGMI links between GCDs (and to the other cards) ────┘
```

A kernel is launched as a **grid** of blocks; the GPU spreads blocks over the CUs. Each thread works
out from its block and thread number which piece of the data it owns.

## The numbers for our model (one GCD)

| What | Number | Source |
|------|-------:|--------|
| Peak bf16 on matrix cores | 181 TFLOP/s | 104 CUs × 1,024 per clock × 1.7 GHz |
| Peak fp32 / fp64 vector | ~23 TFLOP/s each | MI250 data sheet ÷ 2 |
| HBM bandwidth on paper / measured copy | 1.6 TB/s / **1.25 TB/s** | `tools/yardstick_gemm.py` |
| AMD's library on our largest matrix sizes | 105–128 TFLOP/s | same |

**The decisive ratio.** At 1.25 TB/s and ~120 TFLOP/s, the chip can do ~100 multiply-adds in the
time it fetches one byte. A matrix multiply for *one* token reads each weight once and uses it for
one multiply-add: intensity ≈ 1 per 2 bytes (bf16) — **~200× too little**; the calculators idle.
With a **batch** of B tokens, each weight fetched is used B times. Around B ≈ 200–300 the multiply
stops waiting for memory. That is why batching (doc 03) is the first big win.

Measured with the library (expert mlp1, 5760 × 2880):

| Tokens in the batch | Time | TFLOP/s | What limits it |
|---:|---:|---:|---|
| 1 | 42 µs | 0.8 | fetching the 33 MB of weights |
| 64 | 49 µs | 44 | still mostly fetching |
| 256 | 80 µs | 106 | arithmetic |
| 1,024 | 302 µs | 112 | arithmetic |

64× more tokens costs only 1.2× more time: the extra tokens ride along on weights already fetched.

## Where it is in our code

- Launching kernels: `getp-csrc/engine/gpu_engine.hpp:228` (`gpu_step()`).
- The LDS tile trick: `getp-csrc/kernels/gemm.hpp:48` (`gemm_tile_exact`): each block copies a
  64 × 32 slice of activations and of weights into LDS, then every thread reuses them 64 times.
- Which GPU is free: `tools/gpus.py` (HIP numbering differs from `rocm-smi` numbering on mi250-02).

## Pitfalls we hit

- HIP device numbers and `rocm-smi` numbers are not the same on mv-mi250-02 (HIP 0 = rocm-smi 2,
  …). Always pick GPUs with `tools/gpus.py`.
- The machines are shared; other people's jobs change our timings. `tools/gpulock.sh` only keeps
  *our* jobs apart.

## Further reading

- AMD, *CDNA2 instruction set architecture* and *MI250 data sheet* — https://www.amd.com/en/products/accelerators/instinct/mi200.html
- AMD ROCm docs, *HIP programming guide* — https://rocm.docs.amd.com/projects/HIP/en/latest/
- Williams et al., *Roofline: an insightful visual performance model* (2009)
