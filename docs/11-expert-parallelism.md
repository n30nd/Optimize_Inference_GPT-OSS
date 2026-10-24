# 11 — Expert parallelism: sharing gpt-oss-120b between two GPUs

## The problem in plain words

gpt-oss-120b stored compactly (experts in 4-bit MXFP4) is 65 GB, and one GCD has 64 GB, before
any room for the requests' KV cache. The model must be split. But 95 % of it is experts, and each
token only uses 4 of the 128 experts per layer. So we split **the experts**: each of two GCDs keeps
half of every layer's experts plus a full copy of everything else (attention, router, embeddings:
4.2 GB). Each GCD serves its own requests as before; only in the expert step do the two GCDs help
each other.

## An everyday comparison

Two doctors' offices share one building. Each has its own receptionist and waiting room (attention,
its own patients), but the specialists are split: cardiology on floor 1, neurology on floor 2. A
patient's file is walked to whichever floor has the specialist, and the report is walked back.

## The words

- **expert parallelism (EP)** — splitting the experts of each layer across GPUs.
- **peer access** — one GPU reading another GPU's memory directly over the link between them.
- **event** — a marker a GPU sets in its queue of work; another GPU can be told "wait until that
  marker is passed" before it continues.
- **package** — one MI250 card: two GCDs joined by their fastest link.

## How it works

```
         GCD A (experts 0–63)                     GCD B (experts 64–127)
         its own 512 requests                     its own 512 requests
 attention, router for its rows             attention, router for its rows
         │ topk_A, t_A ready ── event ──┐  ┌── event ── topk_B, t_B ready │
         ▼                              ▼  ▼                              ▼
 group the rows that chose experts 0–63    group the rows that chose 64–127
 from BOTH A's and B's tokens               from BOTH A's and B's tokens
 (reads topk_B, t_B from B's memory)        (reads topk_A, t_A from A's memory)
 run experts 0–63 → Z_A                     run experts 64–127 → Z_B
         │ Z_A ready ── event ──┐        ┌── event ── Z_B ready            │
         ▼                      ▼        ▼                                 ▼
 for each own token: add its 4 expert     same on B
 outputs, reading rows from Z_A or Z_B
```

Two waits per layer, both done by the GPUs themselves (`hipStreamWaitEvent`); the host does not
stop. One host thread drives both GCDs of a pair, so it can always queue the event before the wait
that refers to it.

**Why one copy of each shared buffer is enough.** A GCD overwrites `t`, `topk` or `Z` only in the
*next* layer, and by then it has already waited for an event its partner sets *after* it finished
reading them. So the partner can never read half-overwritten data.

**Pull, not push.** Each GCD *reads* its partner's buffers. The other team found that when a copy
engine *wrote* into a GPU's memory, the receiving GPU's cache could still hold the old contents;
reading directly avoided that. Our bit-exact test (below) would show any such stale read.

## The numbers for our model

| | gpt-oss-120b per GCD |
|---|---:|
| All dense weights (bf16) | 4.2 GB |
| Half of the experts (MXFP4) | 30.4 GB |
| KV cache, 512 requests, bf16 | 21.8 GB |
| Read from the partner per layer, on average: activations (bf16) for the ~1,024 of its pairs that chose this GCD's experts / expert outputs (fp32) for the ~1,024 of own pairs computed there | ≈ 5.9 MB / ≈ 11.8 MB |

8 GCDs = 4 pairs; the 4,096 requests are split 1,024 per pair, 512 per GCD.

Correctness: with the exact kernels, 120b split over two GCDs gives **bit-identical logits to the
CPU** (2 prompts × 12 steps, `EP=2 tests/gpu_vs_cpu.cpp`), and 20b split the same way too. Full-run
speed and scores: see the optimization log, rows 3 and 4.

## Where it is in our code

- `getp-csrc/engine/gpu_engine.hpp`: `GpuGroup`, `gpu_pair()` (peer access), `gpu_step_group()` (the
  four parts per layer and the two waits), `gpu_group_size()` (2 when the weights would fill more
  than 60 % of a GCD; `GPTOSS_EP` overrides).
- `getp-csrc/kernels/moe_ep.hpp`: grouping rows from both GCDs, gathering them, combining outputs from
  both GCDs.
- GCD pairs: HIP devices (0,1), (2,3), (4,5), (6,7) are the two dies of one card on our machines.

## Pitfalls

- Peer access must be switched on in both directions (`hipDeviceEnablePeerAccess`).
- With per-GCD host threads, one thread could queue "wait for the partner's marker" before the
  partner's thread has queued the marker itself, and the wait would pass at once. One thread per pair
  avoids that.

## Further reading

- Lepikhin et al., *GShard: scaling giant models with conditional computation and automatic
  sharding* (2020) — expert parallelism.
- AMD ROCm docs, *HIP peer-to-peer and multi-GPU programming*.
