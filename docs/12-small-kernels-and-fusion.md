# 12 — Small kernels: merging them, giving them enough workers, skipping dead work

## The problem in plain words

Once the big matrix multiplies run on the matrix cores, a decode step of gpt-oss-20b takes about
52 ms for 512 requests on one GCD. The many small steps around them (normalizing, adding results
back, choosing experts, sorting tokens by expert) then add up to about 10 % of that time. Each one
is cheap alone, but each reads and writes the whole batch's numbers from memory once more, and some
were written so that only a handful of the GPU's calculators had anything to do.

Four simple ideas fixed most of it:

1. **Merge kernels that touch the same data** (kernel fusion): do two passes' work in one pass.
2. **Give a kernel enough workers**: split each request's work across a wavefront (64 threads)
   instead of one thread.
3. **Sort in parallel**: one block per expert instead of one thread per expert.
4. **Skip work nobody needs**: requests that already ended do not need their experts computed.

## An everyday comparison

A clerk who walks to the archive to fetch a file for "add the totals", walks back, then walks again
for "check the totals", wastes the walk. Merging the two jobs means one walk. And a task split
among a whole team finishes sooner than one person doing it while the team watches.

## The words

- **kernel fusion** — merging two small GPU programs into one, so the data is read from memory once.
- **wavefront** — 64 threads that run in lockstep on AMD GPUs.
- **reduction** — combining many values into one (a sum, or the largest value); within a wavefront
  this is done with *shuffles*, threads reading each other's registers directly.
- **residual stream** — the 2,880 numbers per request that every layer adds its result into.

## How it works

**1. Fusion.** Each pair below used to be two kernels; now it is one, with exactly the same
arithmetic in the same order, so the output bits do not change:

```
before                                       after
attention-out + bias ─► x += out ─► norm(x)   attention-out + bias ─► [x += out; t = norm(x)]
expert sum ─► x += sum ─► next layer norm(x)  [x += sum; t = norm(x) for the next layer]
mlp1 ─► write 47 MB fp32 ─► SwiGLU ─► bf16    [mlp1 with SwiGLU in its last step] ─► bf16
```

The trick that keeps the bits identical: each thread handles the same elements (thread t takes
t, t + 256, t + 512, …) and adds them in the same order as the separate kernels did.

**2. Top-4 with a wavefront per request.** Choosing each token's 4 experts used one thread per
token: 512 tokens filled 8 small blocks, and for 120b each thread scanned 128 experts 4 times. Now 64
threads share a token: each holds some experts, the wavefront finds the best one with shuffles, marks
it taken, and repeats 4 times; one thread then applies the same weighting as the CPU code.

**3. Grouping in parallel.** Sorting the (token, choice) pairs by expert used one block with one
thread per expert, each scanning all 2,048 pairs twice. Now one 256-thread block per expert counts
its pairs, then places them, using wavefront "ballots" (each thread votes yes/no, and a bit count
tells every thread how many yes votes came before it). Same order as before.

**4. Finished requests get no expert.** In a wave of 512 requests, a few end early (their answer
hit the end token). Their rows are still in the batch, but the router now gives them expert −1,
"none", so the expert multiplies, which are half of each step, skip them. 3.1 % of the row-steps of
a 4,096-request run belong to finished requests.

## The numbers for our model

| Change | Kernel time | End to end (8 GCDs, 4,096 requests) |
|---|---|---|
| mlp1 + SwiGLU (+ gather for 20b) fused | 947 → 802 µs per layer (20b, B=512) | 20b +4.1 % |
| Top-4, wavefront per request | 54 → 14.5 µs (32 experts), 194 → 15.7 µs (128 experts) | 20b +1.0 %, 120b +9.7 % (with the fusion above) |
| Grouping, block per expert | 318 → 14.4 µs per layer | (part of a +37 % row) |
| Finished requests skip the experts | — | 20b +0.8 %, 120b +1.2 % |
| Adds fused with the norms | 2 passes over the residual stream fewer per layer | 20b +0.9 %, 120b +0.3 % |

Every row above was checked to leave the output bits unchanged (same output hash).

## Where it is in our code

- `getp-csrc/kernels/fast_misc.hpp` — `add_rmsnorm_bf16_kernel`, `rmsnorm_row_bf16`.
- `getp-csrc/kernels/moe_ep.hpp` — `moe_count_ep_kernel`, `moe_place_ep_kernel`,
  `moe_combine_rmsnorm_kernel`, `moe_combine_partial_rmsnorm_kernel`.
- `getp-csrc/kernels/moe.hpp` — `router_topk_wave_kernel` (expert −1 for finished requests).
- `getp-csrc/kernels/fast_moe_gemm.hpp` — `gemm_grouped_mxfp4_swiglu`, `gemm_grouped_mxfp4_gather_swiglu`.
- `getp-csrc/engine/gpu_engine.hpp` — `layer_attention`, `layer_route`, `layer_combine`, and the
  done flags in `gpu_run_wave`.

## Pitfalls

- A merged kernel must keep each thread's element order, or the sums change in their last bit.
- Our first guess at "the GPU is idle 7.5 % of the time" was wrong: the idle time was the weight
  upload before inference, which is not timed. Measure only the timed part.

## Further reading

- NVIDIA developer blog, *Using shared memory and warp shuffles for reductions* (the same ideas on
  AMD wavefronts).
- AMD ROCm docs, *HIP cross-lane intrinsics* (`__shfl_xor`, `__ballot`).
