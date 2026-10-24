# 05 — Using all 8 GPUs: data parallelism

## The problem in plain words

One machine has 8 GCDs. The simplest way to use them all: give each one a full copy of the model and
an equal share of the requests. They never need to talk to each other while working; at the end
their answers are collected. This is **data parallelism (DP)**: the data (requests) is split, the
model is copied.

## An everyday comparison

Eight identical kitchens, each with the full recipe book, each cooking an eighth of the orders.
Nothing is passed between kitchens, so nobody waits for anybody — as long as each kitchen gets the
same amount of work.

## How it works

```
evaluation/input.txt: 4,096 prompts
        │ split into 8 equal, contiguous shares
        ▼
 GCD 0: prompts    0 –  511  ─┐
 GCD 1: prompts  512 – 1023  ─┤  each: full model copy (13.7 GB),
 ...                          ├─ own KV cache, own host thread,
 GCD 7: prompts 3584 – 4095  ─┘  runs its share in waves of up to 512
        │
        ▼ every thread writes its answers straight into the shared output table
```

- `warm_up()` starts one host thread per GCD; each uploads its own copy of the weights. Warm-up is not
  timed, so copying 13.7 GB eight times costs nothing in the score.
- `inference()` starts one host thread per GCD again; each serves its share and returns its token
  count; the counts are added up.

## The numbers for our model

| | Value |
|---|---|
| Model copy per GCD (bf16 dense + MXFP4 experts) | 13.7 GB |
| KV cache per GCD, 512 requests, fp32 | 29 GB |
| Requests per GCD with the official input | 512 |
| Full run, exact kernels, 8 GCDs | 3,348 tok/s, 1,214 s. The GCDs never wait for each other, so this should be close to 8× one GCD serving 512 requests (one GCD alone was not timed over a full run) |

**Its limit.** DP never makes one GCD's batch bigger: with 4,096 requests, 8 GCDs means 512 requests
each, however fast the kernels get. Bigger batches per GCD (better use of the matrix cores, doc 02)
need either more requests (repeating the input, allowed for speed measurements) or splitting the
*model* across GCDs so several GCDs' requests share one expert's weights: tensor or expert
parallelism, a later document. The 120b model needs that anyway: it does not fit on one GCD.

## Where it is in our code

- `getp-csrc/engine/gpu_engine.hpp:341` `gpu_warm_up()` — one worker per visible GCD.
- `getp-csrc/engine/gpu_engine.hpp:358` `gpu_inference()` — the split and the threads.
- Which GCDs: whatever `HIP_VISIBLE_DEVICES` shows; `tools/gpulock.sh` sets it.

## Pitfalls

- Equal shares only balance if requests are equally long. For our input they nearly are (95 % run to
  the limit); otherwise the slowest GCD sets the time.
- The host threads only launch work and wait; with 8 GCDs that is 8 CPU threads, well within the 96
  available.

## Further reading

- Hugging Face, *The ultra-scale playbook*, chapter "Data parallelism" —
  https://huggingface.co/spaces/nanotron/ultrascale-playbook
