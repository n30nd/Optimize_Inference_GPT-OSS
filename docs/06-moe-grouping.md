# 06 — Running the experts: grouping tokens by expert

## The problem in plain words

In every layer each token is sent to 4 of the 32 experts (128 in the 120b model). Each expert is a
pair of large weight matrices. Done naively — for every token, run its 4 experts — every expert's
weights are read once per token that chose it, and a batch of 512 tokens makes 2,048 such runs.

Instead we first sort the tokens by expert, so each expert sees all of its tokens at once and reads
its weights once per step.

## An everyday comparison

A post office with 32 counters. Letting each customer queue at a counter, get served, then join the
next queue makes clerks fetch the same forms over and over. Sorting everyone's errands by counter
first means each clerk fetches the forms once and serves a whole line.

## How it works

```
router: token → 4 experts          sort by expert (counting sort)        one multiply per expert
 t0 → 5 13 15 16                    expert 2:  t2                         W1[e] · rows of expert e
 t1 → 26 25  4 28        ───►       expert 4:  t1 t3                ───►  (a "grouped GEMM": one kernel,
 t2 → 13  2 21 17                   expert 5:  t0                          one slice of rows per expert)
 t3 → 16  4  2 26                   ...
                                    offsets[e] = where expert e's rows start
```

1. `router_topk_exact` picks each token's 4 experts and their weights.
2. `moe_group_kernel` counts tokens per expert, turns the counts into start offsets, and writes the
   (token, choice) pairs expert by expert: a counting sort.
3. `moe_gather_kernel` copies each pair's input row into that order.
4. `gemm_grouped_mxfp4_exact` launches one kernel whose third grid dimension is the expert: block
   (n, m, e) multiplies rows `offsets[e]…offsets[e+1]` by expert e's weights.
5. SwiGLU, the second grouped GEMM, then `moe_combine_exact` adds each token's 4 expert outputs back,
   weighted, in increasing expert order (the CPU's order).

## The numbers for our model

gpt-oss-20b, 512 tokens per step on one GCD; expert weights in MXFP4 (13.2 MB per expert: 8.8 MB for W1, 4.4 MB for W2):

| | Weight bytes read per layer |
|---|---:|
| Naive: one expert run per (token, choice) | 2,048 × 13.2 MB = 27 GB |
| Grouped: one run per expert | 32 × 13.2 MB = 0.42 GB |

That is 64× less weight traffic. At 1.25 TB/s, 27 GB per layer would cost 22 ms per layer, about
0.5 s per step just for reading.

With 512 tokens and 32 experts, each expert gets 64 rows on average (512 × 4 / 32), but routing is
uneven, so some experts get many more rows than others. The grid is sized for the worst case (all 512
tokens on one expert); blocks with no rows exit at once.

## Where it is in our code

- `getp-csrc/kernels/moe.hpp:15` `router_topk_exact`, `:58` `moe_group_kernel`, `:91`
  `moe_gather_kernel`, `:102` `swiglu_exact`, `:126` `moe_combine_exact`.
- `getp-csrc/kernels/gemm.hpp:108` `gemm_grouped_mxfp4_exact_kernel`.
- Called from `getp-csrc/engine/gpu_engine.hpp:228` (`gpu_step()`).

## Pitfalls

- The grouping must not change any number. Each token's 4 expert outputs land in separate rows and
  are added in a fixed order afterwards, not with atomic adds (whose order varies from run to run).
- Uneven routing: the busiest expert decides how long the grouped GEMM takes. For the 120b model,
  where experts are spread over GPUs, this becomes a load-balancing problem.

## Further reading

- Shazeer et al., *Outrageously large neural networks: the sparsely-gated mixture-of-experts layer*
  (2017).
- Gale et al., *MegaBlocks: efficient sparse training with mixture-of-experts* (2022) — grouped GEMM
  for MoE.
