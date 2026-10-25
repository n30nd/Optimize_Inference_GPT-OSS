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
- **signal** — our own cheaper version of an event: a counter in GPU memory that one GCD raises and
  the other GCD watches (section "Cheaper waits" below).
- **package** — one MI250 card: two GCDs joined by their fastest link.

## How it works

```
         GCD A (64 of the experts)                GCD B (the other 64)
         its own 512 requests                     its own 512 requests
 attention, router for its rows             attention, router for its rows
         │ topk_A, t_A ready ── signal ─┐  ┌─ signal ── topk_B, t_B ready │
         ▼                              ▼  ▼                              ▼
 group the rows that chose A's experts     group the rows that chose B's experts
 from BOTH A's and B's tokens               from BOTH A's and B's tokens
 (reads topk_B, t_B from B's memory)        (reads topk_A, t_A from A's memory)
 run A's experts → Z_A                      run B's experts → Z_B
         │ Z_A ready ── signal ─┐        ┌─ signal ── Z_B ready            │
         ▼                      ▼        ▼                                 ▼
 for each own token: add its 4 expert     same on B
 outputs, reading rows from Z_A or Z_B
```

Two waits per layer, both done by the GPUs themselves (first with `hipStreamWaitEvent`, now with
our signals); the host does not stop. One host thread drives both GCDs of a pair, so it can always
queue the signal before the wait that refers to it. Which 64 experts each GCD holds is decided per
layer by a table, so that both get the same amount of work (section "Giving both GCDs the same
amount of expert work" below).

**Why one copy of each shared buffer is enough.** A GCD overwrites `t`, `topk` or `Z` only in the
*next* layer, and by then it has already waited for a signal its partner sends *after* it finished
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

## Sending one finished row per token instead of each expert's row

The first version let each GCD read, for every one of its tokens, the outputs of the experts its
partner ran: about 2 rows of 2,880 fp32 numbers per token. Now the GCD that ran the experts first
adds them up, weighted, for each of its partner's tokens (`moe_partial_ep_kernel`), and the partner
reads **one** row per token (`moe_combine_partial_kernel`).

```
before: B reads A's expert rows  r1, r2 ─────────────► B adds w1·r1 + w2·r2 + its own
after : A computes s = w1·r1 + w2·r2 for B's token ──► B reads s, adds s + its own
```

Half the bytes over the link, and one contiguous row per token. It changes the order of the
additions, so the output changes slightly (checked with METEOR / BERTScore: 0.576 / 0.982). 120b
on 8 GCDs: 35,815 → 36,132 tok/s (+0.9 %).

## Giving both GCDs the same amount of expert work

### The problem in plain words

The two GCDs of a pair wait for each other twice in every layer. If one of them has more expert
work, the other one sits idle until it is done. And the work is not even: the router (the small
layer that picks each token's 4 experts) prefers some experts strongly. With 1,024 tokens per step
the average expert gets 32 rows, but on the evaluation prompts one expert of one layer gets 527 on
average, and in some steps an expert gets all 1,024 tokens while others get none. With experts
0–63 on GCD A and 64–127 on GCD B, the GCD that happens to hold more of the popular experts of a
layer is slower in that layer in almost every step.

Measured on 1,024 prompts on one pair: GCD A spent 3.8 s and GCD B 4.6 s of a 95 s run waiting
for the partner's expert results. Summed over the 36 layers, the two GCDs' expert times differed
by 6.7 ms per step (a step is about 93 ms), and 5.6 ms of that was the same every step: a fixed
lean of each layer towards one GCD, not chance.

### An everyday comparison

Two cashiers share the customers of a shop, split by the first letter of the last name: A–M to
the first, N–Z to the second. Some letters are much more common than others, so one queue is
always longer. Better: count a few days of customers once, then split the letters so that both
cashiers expect the same number of people.

### The words

- **load balancing** — sharing work so that every worker finishes at about the same time and
  nobody waits.
- **calibration** — measuring once on sample data to choose a setting that is then kept fixed.
- **slot** — the place of an expert in one GCD's weight memory. The router now writes slot
  numbers instead of the checkpoint's expert numbers.
- **work item** — one piece of the expert matrix multiply: up to 64 rows of one expert times 256
  output columns, computed by one block of the GPU (doc 08).

### How it works

The weights of an expert cannot move during a run, so the split is chosen once, before, and
stored as a table.

1. **Record the load.** With `GPTOSS_EXPERT_LOAD=<file>`, `getp` writes after every step how many
   rows every expert got in every layer (the counts the grouping step computes anyway).
2. **Predict the work.** An expert that got n rows in a step costs about
   `3.0 · (n > 0) + 6.1 · ceil(n / 64) + 3.3 · ceil(n / 16) + 0.14 · n` microseconds (ceil(n / 64)
   is n / 64 rounded up): its weights are read once from memory, every 64-row work item unpacks
   them again, the matrix cores work in 16-row tiles, and gathering and adding up grow with the
   rows. The four numbers come from a profile (a recording of how long every GPU kernel ran) of a
   run whose loads were recorded too (`tools/ep_split.py fit`). The predicted difference between
   the two GCDs follows the measured one closely: correlation 0.99 per step and layer, that is,
   they rise and fall together almost exactly.
3. **Choose the split.** For each layer, `tools/ep_split.py make` puts the experts on the two GCDs
   from the most to the least work, each to the GCD that has less so far, and then keeps swapping
   one expert of A with one of B while that lowers the average difference over all recorded steps.
4. **Use it.** The table is `getp-csrc/engine/ep_split_120b.hpp`. At load time each GCD uploads the
   64 experts of its slots. The router turns each chosen expert number into its slot number
   (`slot_of`) as it writes `topk_e`, so grouping, multiplying and combining work unchanged on
   slots 0–63 and 64–127. The exact path's combine still adds the experts in checkpoint order
   (`phys_of`), so it stays bit-identical to the CPU.

**Calibration data.** The recordings come from our own prompts (`tools/make_calib_prompts.py`), never
from `evaluation/`. Which experts are popular depends a lot on the kind of text: the per-expert
loads of code prompts and of story prompts agree only weakly (correlation 0.33). A table made from
one kind balances that kind well and others badly. So the table is made from four kinds at once
(stories and everyday writing, explanations and essays, code, math; 1,024 prompts each), which
gives a split that balances each of them (predicted difference 1.1–1.9 ms per step instead of
3.9–6.2 ms with the halves).

### The numbers

1,024 evaluation prompts on one pair, profiled; full runs: official input on 8 GCDs, the builds
alternated, 2 runs each.

| | experts in halves | experts by the table |
|---|---:|---:|
| difference in expert time between the two GCDs, summed over 36 layers, mean per step | 6.7 ms | 2.4 ms |
| of that, the part that is the same in every step | 5.6 ms | 1.8 ms |
| waiting before the combine, GCD A / GCD B | 3.8 / 4.6 s of 95 s | 1.7 / 2.1 s of 91 s (other pair) |
| full run, 8 GCDs | 42,449 tok/s | **42,904 tok/s (+1.1 %)** |
| METEOR / BERTScore | 0.562 / 0.981 | 0.552 / 0.981 |

Applied to the loads recorded during the evaluation run, the work model gives 2.4 ms with the
table and 6.5 ms with the halves, as measured. A table fitted to the evaluation prompts themselves
would give 1.0 ms,
so the evaluation text still has its own favourite experts; we do not ship such a table, because the
split must not be tuned to the test set. What is left after any fixed split (about 1 ms per step on
our own prompts) changes from step to step and could only be removed by deciding per step, with
some experts stored on both GCDs; at most about 0.5 ms per step (0.5 %) to gain, so we did not.

### Pitfalls

- The table is only for gpt-oss-120b (36 layers, 128 experts); other models keep the halves.
  `GPTOSS_EP_SPLIT=0` also keeps the halves.
- Changing which GCD adds which expert changes the order of some fp32 additions, so the output
  changes slightly. Check the scores again after making a new table.
- After changing the expert kernels, refit the work model (`fit`) and remake the table.

## Cheaper waits: a counter instead of an event

### The problem in plain words

Every wait for the partner (`hipStreamWaitEvent` on an event of the other GCD) cost about 18
microseconds, even when the partner had finished long before. Two waits per layer, 36 layers:
72 waits, 1.3 ms of every 93 ms step.

### How it works

Each GCD has two counters in fine-grained memory (memory that both GCDs always see up to date,
without either one keeping a stale copy). After its router, a GCD runs a one-thread kernel
(`ep_signal_kernel`) that first writes its cache back to memory, so the partner can read the fresh
data, and then raises its "routed" counter; after its experts it raises its "experts" counter.
Instead of the event wait, the partner runs a one-thread kernel (`ep_wait_kernel`) that reads the
counter until it reaches the expected value and then drops the partner's data it might still
have in its own cache. The host knows the expected value: it counts the signals it has queued.

### The numbers

| | event (`hipStreamWaitEvent`) | signal |
|---|---:|---:|
| one exchange, seen in a profile | about 18 µs of waiting | about 6 µs wait kernel + 7 µs signal kernel |
| full run, 8 GCDs (2 runs each, alternated) | 42,904 tok/s | **43,099 tok/s (+0.45 %)** |

The output is the same with both (same hash), and the exact path is still bit-identical to the CPU.
Most of what is left is the cache work itself: before the partner may read, a GCD must write its
L2 cache back to memory, and the reader must drop its stale copies.

## Where it is in our code

- `getp-csrc/engine/gpu_engine.hpp`: `GpuGroup`, `gpu_pair()` (peer access), `gpu_step_group()` (the
  four parts per layer and the two waits), `gpu_group_size()` (2 when the weights would fill more
  than 60 % of a GCD; `GPTOSS_EP` overrides).
- `getp-csrc/kernels/moe_ep.hpp`: grouping rows from both GCDs, gathering them, combining outputs from
  both GCDs; `ep_signal_kernel` / `ep_wait_kernel`.
- `getp-csrc/engine/ep_split_120b.hpp` (made by `tools/ep_split.py`): which experts each GCD holds;
  `gpu_expert_order()` and the expert upload in `gpu_upload_weights()` use it;
  `tools/make_calib_prompts.py` writes the calibration prompts.
- GCD pairs: HIP devices (0,1), (2,3), (4,5), (6,7) are the two dies of one card on our machines.

## Pitfalls

- Peer access must be switched on in both directions (`hipDeviceEnablePeerAccess`).
- With per-GCD host threads, one thread could queue "wait for the partner's marker" before the
  partner's thread has queued the marker itself, and the wait would pass at once. One thread per pair
  avoids that.
- A signal must be sent in every layer even when a GCD has no requests, or its partner waits forever.

## Further reading

- Lepikhin et al., *GShard: scaling giant models with conditional computation and automatic
  sharding* (2020) — expert parallelism.
- AMD ROCm docs, *HIP peer-to-peer and multi-GPU programming*.
