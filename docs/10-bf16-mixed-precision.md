# 10 — Mixed precision: bf16 for the matrix multiplies, fp32 for the rest

## The problem in plain words

The exact GPU version (doc 04) keeps every number in 4 bytes and adds in 8 bytes. That is faithful
but slow, and it cannot use the GPU's matrix cores, which only accept 2-byte numbers. Mixed precision
stores the numbers that go *into* a matrix multiply in a 2-byte format (bf16), while every sum and the
running state of the model stay in 4 bytes (fp32).

## An everyday comparison

Measuring ingredients with a kitchen scale that shows grams instead of milligrams. Each single
measurement is a little rougher, but you still add up the totals on a precise calculator, so the
dish comes out the same, maybe with a slightly different garnish.

## The words

- **bf16 (bfloat16)** — a 2-byte number: the same range as fp32 (8 exponent bits) but only 8 bits
  of precision (about 3 decimal digits) instead of 24.
- **fp32** — a 4-byte number, about 7 decimal digits.
- **accumulate** — add up many products; we always do this in fp32.
- **residual stream** — the 2,880 numbers `x` that every layer adds its result into; kept in fp32.

## How it works

```
                 fp32 x  (residual stream, never rounded to bf16)
                   │
     RMSNorm ──► bf16 t ──► matrix multiply (bf16 × bf16, fp32 sums) ──► fp32 result
                                                        │
     attention reads the bf16 KV cache, computes in fp32 ─► bf16 output ─► next multiply
                                                        │
                 fp32 x += fp32 result
```

What becomes bf16, and why:

| Value | Format | Why |
|---|---|---|
| Weights | as stored: bf16, experts 4-bit MXFP4 | already in the checkpoint, nothing lost |
| Inputs to every matrix multiply | bf16 | matrix cores need it; half the bytes to move |
| KV cache | bf16 | half the memory per request (56.6 → 28.3 MB for 20b), so bigger batches fit |
| Sums inside multiplies, softmax, norms | fp32 | rounding errors would otherwise pile up over 2,880 terms |
| Residual stream `x` | fp32 | it carries the model's state across 24–36 layers |

## The numbers for our model

gpt-oss-20b, `evaluation/input.txt`, 8 GCDs, simple kernels in both cases:

| | Exact (fp32 / fp64) | Mixed precision (bf16 inputs) |
|---|---:|---:|
| tok/s | 3,348 | **5,217** (+56 %) |
| METEOR (pass ≥ 0.30) | 0.9967 | 0.552 |
| BERTScore F1 (pass ≥ 0.90) | 0.99998 | 0.979 |
| Answers identical to the reference | 99.2 % | 8.4 % |
| Median first differing token | — | 138 |

- The +56 % comes only from moving half the bytes; the matrix cores are not used yet (that is
  docs 07–09).
- Only 8 % of answers stay word for word identical, yet the scores stay high: when two next tokens
  are nearly tied, rounding picks the other one, and the answer continues differently but just as
  fluently. BERTScore 0.979 says the meaning is nearly the same.

## Where it is in our code

- `getp-csrc/kernels/fast_common.hpp` — `f32_to_bf16()` (round to nearest even), `bf16_to_f32()`.
- `getp-csrc/kernels/fast_misc.hpp` — RMSNorm writing bf16, RoPE writing the bf16 KV cache, SwiGLU
  writing bf16.
- `getp-csrc/engine/gpu_engine.hpp` — `gpu_step_fast()`; `GPTOSS_EXACT=1` switches back to the exact
  path.

## Pitfalls

- Rounding to bf16 must be round-to-nearest-even; simply cutting off the low 16 bits biases every
  value towards zero.
- Never round the residual stream: it would lose precision at every one of the 24–36 layers.

## Further reading

- Kalamkar et al., *A study of BFLOAT16 for deep learning training* (2019).
- Micikevicius et al., *Mixed precision training* (ICLR 2018).
