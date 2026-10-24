# 04 — Reproducing the CPU bit for bit on the GPU

## The problem in plain words

Computers round every result to a fixed number of digits. Two programs that compute "the same"
formula in a different order, or with a different rounding step, get answers that differ in the last
digit. Those tiny differences can change which token wins when two scores are nearly tied, and from
then on the two answers go their separate ways. So "the GPU answer differs from the reference" can
mean a bug, or only rounding, and you cannot tell which.

Our first GPU version removes the doubt: it repeats the CPU's arithmetic **exactly**, so any
difference at all is a bug. Later, faster versions are then compared against a GPU baseline we know
is right.

## An everyday comparison

Two accountants add the same 2,880 receipts, rounding to the cent after every addition. If one adds
from the top and the other from the bottom, the totals can differ by a cent. To check one against
the other you make them add in the same order and round at the same moments.

## The words

- **fp32 / fp64** — numbers stored in 4 / 8 bytes; fp64 keeps about twice as many digits.
- **rounding** — cutting a result to the digits the format can hold.
- **FMA (fused multiply-add)** — `a × b + c` computed with a single rounding at the end, instead of
  rounding after the multiply and again after the add. Same formula, possibly different last digit.
- **correctly rounded** — the result is the exact answer rounded once; the best a computer can do.
- **bit-identical** — every bit of every number is the same.

## How it works

For each operation in `run.cpp`'s `forward()` we read the machine code the compiler produced
(`objdump -d run`) and copy its exact rounding steps:

| CPU code | What the compiled CPU code does | GPU copy |
|---|---|---|
| `matmul`: `val += w*x` | fp32 product, rounded; added into an fp64 sum, in order of k | `__dadd_rn(acc, (double)__fmul_rn(a, w))`, k in order |
| attention scores | same as matmul | same |
| attention `tb += a*v` | FMA in fp32 | `__fmaf_rn(a, v, tb)` |
| RoPE `x1*c - x2*s` | `x2*s` rounded, then FMA | `__fmaf_rn(x1, c, -__fmul_rn(x2, s))` |
| expert sum `e_agg += tb2*w` | FMA in fp32 | `__fmaf_rn` |
| softmax, rmsnorm | fp64 sums in order, one `sqrtf`, one `expf` per entry | same order, one thread does the sum |
| `expf` | glibc 2.39's FMA version | our port of the same algorithm |

Only additions need a fixed order. Products, maxima and per-element work can still run in parallel.

## What we found

| Check | Result |
|---|---|
| Our GPU `expf` vs the CPU's, all 2^32 inputs | 0 differences |
| HIP `sqrtf` vs the CPU, all 2^32 inputs | 0 differences |
| **HIP `__fsqrt_rn` vs the CPU, all 2^32 inputs** | **323,287,622 differences (7.5 %)** |
| fp64 square root rounded to fp32 (our `sqrt_cpu`) | 0 differences |
| HIP `__fdiv_rn(1, x)` vs the CPU | 0 differences |
| Whole model, GPU vs CPU, 4 prompts × 32 steps, 24 layers | all 201,088 logits bit-identical at every step |
| `getp`, 64 prompts × 128 steps, vs the official reference answers | 64 / 64 identical |

`__fsqrt_rn` promises "round to nearest", but on gfx90a it is wrong in the last bit for 7.5 % of
inputs. It made rmsnorm's scale differ now and then, which moved every logit slightly. We found it by
cutting the model to 1 layer (layer 0 correct at steps 0 and 1, wrong at step 2), then comparing each
stage of that step, then testing `sqrt` on every possible input.

## The cost

Copying the CPU this closely is slow: every product is followed by a conversion and an fp64 add, and
no matrix cores are used. At 512 requests a step takes 1.08 s on one GCD. The fast kernels come
next, and each one is accepted on two measures: METEOR/BERTScore, and how many tokens still match
this exact baseline.

## Where it is in our code

- `getp-csrc/kernels/exact_math.hpp` — `expf_cpu`, `sqrt_cpu`, `div_cpu`.
- `getp-csrc/kernels/{gemm,attention,moe,misc}.hpp` — the `*_exact` kernels.
- Tests: `getp-csrc/tests/gpu_vs_cpu.cpp` (whole model; `LAYERS=n` cuts the model to n layers),
  `stage_check.cpp` (one kernel at a time), `expf_exhaustive.cpp`, `fp32_ops_exhaustive.cpp`.

## Further reading

- Goldberg, *What every computer scientist should know about floating-point arithmetic* (1991).
- Figueroa, *When is double rounding innocuous?* (1995) — why fp64-then-fp32 gives a correctly
  rounded fp32 square root.
- ARM optimized-routines, `math/expf.c` — the `expf` algorithm glibc uses.
