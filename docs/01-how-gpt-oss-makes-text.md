# 01 — How gpt-oss makes text

Every later document assumes the picture on this page. It explains what the program computes,
not how to make it fast.

## The problem in plain words

A language model writes one word piece at a time. To choose the next piece it reads everything
written so far, passes it through a long stack of identical processing blocks, and ends with a
score for every piece it knows. We take the piece with the highest score, append it, and repeat.

## An everyday comparison

Think of a relay of 24 editors (36 for the big model) sitting in a row. A note describing "the text
so far, focused on the newest word" goes down the row; each editor reads it, consults their own
notes about earlier words, adds a correction, and passes it on. The last editor turns the note into
a vote over all 201,088 possible next pieces.

## The words

- **token** — a word piece, about ¾ of an English word. "Hello world" is 2 tokens: `13225 2375`.
- **embedding** — the list of 2,880 numbers the model uses to stand for one token.
- **hidden state (x)** — the 2,880 numbers that travel down the stack for the newest token.
- **layer / block** — one processing stage of the stack (24 in 20b, 36 in 120b).
- **attention** — the step where the newest token looks back at earlier tokens.
- **KV cache** — the notes each layer keeps about earlier tokens (one *key* and one *value* list per
  token), so they are not recomputed at every step.
- **MoE (mixture of experts)** — instead of one big sub-network per layer, many smaller "experts";
  a small **router** picks the 4 that handle each token.
- **logits** — the final scores, one per possible next token.
- **prefill / decode** — reading the prompt / producing the answer one token at a time.

## How it works, step by step

```
token id ──► embedding (2,880 numbers) = x
             │
             ▼   repeated for each of the 24 (or 36) layers
   ┌──────────────────────────────────────────────────────────────┐
   │ RMSNorm(x)                      rescale x to a standard size │
   │ q, k, v = Wqkv · x + b          64 query heads, 8 key/value  │
   │ RoPE(q, k)                      mix in the token's position  │
   │ store k, v in the KV cache                                   │
   │ attention: q against every cached k → softmax → mix of v's   │
   │ x = x + Wo · (mix) + b          "residual": add, don't replace│
   │ RMSNorm(x)                                                   │
   │ router = Wr · x + b → pick 4 best of 32 (128) experts        │
   │ for each chosen expert: W2 · SwiGLU(W1 · x + b1) + b2        │
   │ x = x + weighted sum of the 4 expert outputs                 │
   └──────────────────────────────────────────────────────────────┘
             │
             ▼
   RMSNorm(x) → logits = Wout · x  (201,088 scores) → take the largest → next token
```

Three gpt-oss specifics, all in `run.cpp`'s `forward()`:

- **Sliding window** — on even-numbered layers a token only looks back at the last 128 tokens
  (`run.cpp:617`). Odd layers see the whole history.
- **Attention sink** — each head has one extra learned score that takes part in the softmax but has
  no value attached (`run.cpp:624`); it lets a head "look at nothing".
- **Grouped-query attention** — 64 query heads share 8 key/value heads (8 queries per key/value),
  so the KV cache is 8× smaller than it would be otherwise.

## The numbers for our models

|                                  | 20b | 120b |
|----------------------------------|----:|-----:|
| Layers                           | 24 | 36 |
| Experts per layer, used per token| 32, 4 | 128, 4 |
| Numbers in the whole model       | 20.9 billion | 116.8 billion |
| Numbers used for one token       | ≈ 3.6 billion | ≈ 5.1 billion |
| Multiply-adds per generated token| ≈ 3.6 billion | ≈ 5.1 billion |
| KV cache per request at 1,024 tokens (fp32 / bf16) | 56.6 MB / 28.3 MB | 85 MB / 42.5 MB |

The experts are 95 % of the model but each token only touches 4 of them. That is why gpt-oss-20b
"costs" like a 3.6-billion-number model per token while storing 20.9 billion numbers.

Our workload (`evaluation/input.txt`): 4,096 short prompts; 95 % of the answers run to the
1,024-token limit, so a run produces about 4.06 million tokens, almost all in the decode phase.

## Where it is in our code

- CPU reference: `run.cpp:528-756` (`forward()`), and `getp-csrc/engine/cpu_forward.hpp`, the same
  code reading the weights straight from the Hugging Face files.
- GPU: `getp-csrc/engine/gpu_engine.hpp:228` (`gpu_step()`), one kernel launch per box in the
  drawing above.

## Further reading

- OpenAI, *Introducing gpt-oss* — https://openai.com/index/introducing-gpt-oss/
- Karpathy, *llama2.c* — https://github.com/karpathy/llama2.c (the program ours grew from)
- Jay Alammar, *The Illustrated Transformer* — https://jalammar.github.io/illustrated-transformer/
