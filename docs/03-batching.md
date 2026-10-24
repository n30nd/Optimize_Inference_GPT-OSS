# 03 — Batching (static batching)

## The problem in plain words

To produce one token for one request, the GPU must read every weight that token needs, about
3.6 billion numbers for gpt-oss-20b, and do one multiply-add with each. Reading is slow; the
arithmetic is cheap (doc 02). If we serve requests one after another, almost all the time goes to
reading the same weights again and again.

**Batching** means producing the next token for many requests in one pass: each weight is read once
and used for every request in the batch.

## An everyday comparison

A bus instead of taxis. Driving the route (reading the weights) costs about the same whether 1 or 50
passengers ride. Taking all passengers on each trip divides the cost by the number of riders.

## How it works

```
one request at a time                 batch of B requests
for each request:                     for each step:
  for each step:                        one pass over the weights
    one pass over the weights           → next token for all B requests
    → one token
weights read: requests × steps        weights read: steps
```

In the code the activations become a matrix: instead of one row of 2,880 numbers we carry B rows,
and every matrix-vector multiply becomes a matrix-matrix multiply (GEMM) with B rows.

**Static batching**, what Phase 1 uses: the requests of a *wave* all start together at position 0
and move forward one position per step. A request that finishes (end token, or the 1,024 limit) stays
in the batch until the wave ends; its rows are still computed, but the results are ignored.

```
step:      0  1  2  3 ... 1023
request 0  ■  ■  ■  ■ ...  ■      (prompt tokens are fed in, then the model's own)
request 1  ■  ■  ■  □ ...  □      □ = finished, still computed, ignored
request 2  ■  ■  ■  ■ ...  ■
```

The waste (□) is small for us: 95 % of the answers run to the limit anyway. *Continuous batching*,
which refills finished slots with new requests, is a later document.

## The numbers for our model

Same exact kernels, one GCD, gpt-oss-20b (`tools/bench.py`, 2026-10-25):

| Requests per pass | Time per step | Tokens per second |
|---:|---:|---:|
| 1 (CPU, the given code) | 0.30 s | 3.3 |
| 64 (GPU) | 0.44 s | 130 |
| 512 (GPU) | 1.08 s | 357 |

- 8× more requests (64 → 512) cost only 2.5× more time per step.
- The GPU numbers are low in absolute terms because Phase 1 deliberately copies the CPU's
  arithmetic (fp64 sums, no matrix cores; doc 04). Batching is what later lets the matrix cores
  pay off.

Memory sets the batch limit: each request needs its own KV cache (56.6 MB in fp32 for 20b at 1,024
positions), so 512 requests take 29 GB of the GCD's 64 GB, next to 13.7 GB of weights.

## Where it is in our code

- `getp-csrc/engine/gpu_engine.hpp:278` (`gpu_run_wave()`): one wave; the loop at line 300 runs one
  step for all requests, then decides each request's next token (prompt token or model token).
- `getp-csrc/engine/gpu_engine.hpp:228` (`gpu_step()`): every kernel takes the batch size `B`.
- `GPTOSS_BATCH` (default 512) sets the requests per wave; `gpu_create_worker()` lowers it if the
  KV cache would not fit.

## Pitfalls

- A request's prompt is fed one token per step, like the original code. With prompts of ~20 tokens
  that costs ~2 % of the steps; processing the whole prompt in one step (batched prefill) is a later
  optimization.

## Further reading

- Yu et al., *Orca: a distributed serving system for transformer-based generative models* (OSDI
  2022) — introduces iteration-level (continuous) batching.
- Anyscale blog, *How continuous batching enables 23x throughput in LLM inference*.
