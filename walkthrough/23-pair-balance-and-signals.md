# Lesson 23: 120b — give both GCDs of a card the same expert work

| | |
|---|---|
| Reference commit | `a9b23f7` "120b: balance the expert work of a GCD pair; signals instead of events" |
| Starts from | `1131a42` (lesson 22) |
| To type | 251 lines of program code in 4 files (42 removed), of which 53 are a generated table; 294 lines of tools; 316 lines of prompt lists and 142 of docs (optional) |
| Shortcut | `git cherry-pick a9b23f7` — copies the finished commit instead of typing (section 5) |
| Speed after | 120b 42,981 tok/s (+1.7 %); 20b 76,461 tok/s (unchanged path) |

**In plain words:** for 120b the two GPU dies of one card share the experts, and in every layer
each waits for the other. One die held more of the popular experts, so the other one often stood
waiting. We counted, once and on our own practice questions, how busy each expert gets, and
dealt the experts out so both dies get about the same work. We also replaced the slow "I am
done" message between the two dies by a cheaper one.

## 1. Goal

Two changes for the 120b pair (lesson 05 split each layer's 128 experts over the two GCDs of one
card):
- choose, per layer, which 64 experts each GCD holds, so both have about the same expert work;
- make the two waits per layer cheaper.

## 2. Why this, why now

A profile of 120b on one pair, 1,024 prompts:
- each GCD was idle 6.4 % / 5.1 % of the time, 4.65 / 3.68 s of it waiting for the partner's
  experts;
- per layer the two GCDs' expert times differed by up to 483 µs (layer 26: 1,193 against
  1,676 µs); summed over the 36 layers, 6.66 ms per step (a step is about 93 ms);
- 5.54 ms of those 6.66 ms was the same every step: a fixed lean of each layer towards one GCD
  (experts 0–63 on one, 64–127 on the other), not chance.

And every wait (`hipStreamWaitEvent` on the partner's event) cost about 18 µs even when the
partner had long finished: 72 waits per step, about 1.3 ms of 93.

## 3. The technique

New words:
- **load balancing** — sharing work so every worker finishes at about the same time.
- **calibration** — measuring once on sample data to choose a setting that is then kept fixed.
- **slot** — the place of an expert in one GCD's weight memory. The router now writes slot
  numbers instead of the checkpoint's expert numbers.
- **work model** — a formula that predicts an expert's time from its row count n:
  `3.0·(n > 0) + 6.1·ceil(n/64) + 3.3·ceil(n/16) + 0.14·n` µs (weights read once, every 64-row
  work item decodes them again, 16-row matrix-core tiles, gathering and adding per row).
- **event** — HIP's way for one stream to wait until another stream (here on the other GCD)
  reaches a point (`hipEventRecord` / `hipStreamWaitEvent`).
- **fine-grained memory** — memory both GCDs always see up to date, without either keeping a
  stale copy in its cache.
- **release / acquire** — a release store first makes all earlier writes visible, then raises the
  counter; an acquire, after seeing the counter, drops stale cached copies before reading.

The split, chosen once before any run:

```
 1. record   GPTOSS_EXPERT_LOAD=load_code ./run $M120 -m getp -i calib_code.txt …
             → rows per expert, per step and layer      (4 kinds of our own prompts:
                                                          prose, essays, code, math)
 2. choose   tools/ep_split.py make load_*.0
             per layer: experts from most to least predicted work, each to the GCD
             with less so far; then swap pairs while the average difference drops
 3. use      getp-csrc/engine/ep_split_120b.hpp   (36 layers × 128 slots)
             upload: GCD 0 holds slots 0-63, GCD 1 slots 64-127
             router: writes slot_of[expert] → grouping and multiplies unchanged
```

The cheaper wait, twice per layer:

```
 GCD A                                   GCD B
 router done ── ep_signal_kernel ──►  counter_A = n     (1 thread, release store)
                                      ep_wait_kernel: spin until counter_A ≥ n,
                                                      then acquire fence
                                      experts …
```

The host counts the signals it has queued, so it knows the value each wait must see. One host
thread drives both GCDs and always queues a signal before the wait that refers to it.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe_ep.hpp` | +32 −7 | `ep_signal_kernel` and `ep_wait_kernel`; the exact path's combine adds experts in checkpoint order through `phys_of` | yes |
| 2 | `getp-csrc/kernels/moe.hpp` | +5 −2 | `router_topk_wave_kernel` writes `slot_of[expert]` | yes |
| 3 | `getp-csrc/engine/ep_split_120b.hpp` | +53 | the table: for each layer, the checkpoint expert in each of the 128 slots | copy it (`git checkout a9b23f7 -- …`): generated numbers |
| 4 | `getp-csrc/engine/gpu_engine.hpp` | +161 −33 | `gpu_expert_order` (table or plain halves, `GPTOSS_EP_SPLIT`); experts uploaded by slot; `slot_of` / `phys_of`; the two counters in fine-grained memory; `ep_signal` / `ep_wait`; `GPTOSS_EXPERT_LOAD` recording | yes |
| 5 | `tools/ep_split.py` | +294 | `fit` the work model from a profile, `make` the table, `check` a table on a recording | yes |
| 6 | `tools/make_calib_prompts.py` | +316 | writes our own calibration prompts (4 kinds); mostly word lists | optional |
| 7 | `docs/11-expert-parallelism.md` | +142 −10 | "Giving both GCDs the same amount of expert work", "Cheaper waits" | optional |

Read while typing (line numbers at `a9b23f7`):
- `getp-csrc/kernels/moe_ep.hpp:29-37` — the whole signal and wait: one release store; a loop of
  relaxed loads with a short sleep, then an acquire fence.
- `getp-csrc/engine/gpu_engine.hpp:411` — the counters allocated with `hipDeviceMallocFinegrained`.
- `getp-csrc/engine/gpu_engine.hpp:620-629` `ep_signal` / `ep_wait`, and `:652-667` the order in
  `gpu_step_group`: routed signal → wait → experts → experts signal → wait → combine.
- `getp-csrc/engine/gpu_engine.hpp:146` `gpu_expert_order` and `getp-csrc/kernels/moe.hpp:121`,
  where the router renumbers its choice.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick a9b23f7
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout a9b23f7 -- $(git show --name-only --format= a9b23f7)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat a9b23f7
git show a9b23f7 -- getp-csrc/engine/gpu_engine.hpp      # the change to copy
git checkout a9b23f7 -- getp-csrc/engine/ep_split_120b.hpp tools/make_calib_prompts.py

git diff a9b23f7 -- $(git show --name-only --format= a9b23f7)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (20b has no partner, so nothing changes for it):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 22)
```

Quick check, 120b on a GCD pair (the fingerprint changes here, see section 6):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 10e37a089467
```

Bit-exact test on the split (two GCDs of one card, 120b, 4 prompts × 16 steps, about 70 s): the
exact path must still give the CPU's logits, bit for bit, with the experts placed by the table.

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
EP=2 HIP_VISIBLE_DEVICES=6,7 ./gpu_vs_cpu $M120 4 16
```

Expected (seen on the reference build):

```
pos  15: logits differing 0 / 804352, max |diff| 0, argmax differing 0
PASS: GPU logits bit-identical to the CPU
```

Optional, to see the imbalance yourself (not run here: it needs full 120b runs): record a run
with `GPTOSS_EXPERT_LOAD=$W/load ./run $M120 …`, then
`python3 tools/ep_split.py check $W/load.0 --halves` and
`python3 tools/ep_split.py check $W/load.0 --table getp-csrc/engine/ep_split_120b.hpp`.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-23 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-23 --score
```

Expected: 120b about 43,000 tok/s, fingerprint `973e97c40a85`, METEOR 0.552, BERTScore 0.981;
20b about 76,500 tok/s, fingerprint `58023f92ead7` (unchanged).

Save: `git add -A -- . ':!walkthrough' && git commit -m "120b: balance the expert work of a GCD pair; signals instead of events"`

## 6. Result

The balance, 1,024 evaluation prompts on one pair (from `docs/11`):

| | experts in halves | experts by the table |
|---|---:|---:|
| expert-time difference between the GCDs, per step | 6.7 ms | 2.4 ms |
| of that, the part that is the same every step | 5.6 ms | 1.8 ms |
| waiting before the combine, GCD A / GCD B | 3.8 / 4.6 s of 95 s | 1.7 / 2.1 s of 91 s |

The waits: about 18 µs per event wait → about 6 µs wait kernel + 7 µs signal kernel.

Whole program, official input, 8 GCDs, old and new builds alternated twice:

| | before | after |
|---|---:|---:|
| 120b | 42,251 tok/s | **42,981 tok/s (+1.7 %)** |
| 120b difference in expert time per step (full run) | 6.70 ms | 2.55 ms |
| 120b idle per GCD (of ~93 s) | 5.0 / 5.8 s | 2.6 / 3.6 s |
| 120b METEOR / BERTScore | 0.562 / 0.981 | 0.552 / 0.981 |
| 20b | — | 76,461 tok/s, same fingerprint |

**The 120b answers change slightly.** A different expert placement changes which GCD adds which
expert's output, so some fp32 additions happen in another order and the last bits move. The scores
stay far above the gates. The exact path still adds in checkpoint order (`phys_of`) and stays
bit-identical to the CPU (the test above).

## 7. Decisions and reasons

- **Decided:** a fixed split per layer, chosen from recorded loads with a work model.
  **Why:** 5.54 of the 6.66 ms imbalance was the same every step, so a fixed table can remove
  most of it. The work model follows the measured difference closely (correlation 0.99 per step
  and layer). **Not chosen:** moving expert weights during the run: the link between the GCDs is
  much slower than a GCD's own memory, so moving weights costs more than the waiting.
- **Decided:** calibrate on our own prompts of four kinds (prose, essays, code, math), never on
  `evaluation/`. **Why:** which experts are popular depends on the kind of text (code and story
  loads agree only weakly, correlation 0.33). One mixed set balanced the evaluation run only to
  3.21 ms; three kinds 3.27 ms; four kinds 2.4 ms and 43,099 against 42,749 tok/s. No evaluation
  topic or prompt is in `tools/make_calib_prompts.py`. Honest note: the fourth kind was added after
  seeing that three kinds left the evaluation run unbalanced, so the *choice of kinds* was made
  while looking at the evaluation imbalance; the table itself uses no evaluation data.
  **Not chosen:** a table fitted on the evaluation prompts (predicts 1.0 ms): that would tune the
  program to the test set.
- **Decided:** a counter in fine-grained memory with release/acquire instead of HIP events.
  **Why:** about 13 µs per exchange instead of about 18; same output.
- **Not chosen: splitting 120b's experts over 4 or 8 GCDs.** Reading another GCD's memory runs at
  177 GB/s for the card mate, 90 GB/s for the two other directly linked GCDs and 45 GB/s for the
  rest, against ~1,280 GB/s for a GCD's own memory. The work model on the real 120b routing gives,
  per layer: today (2 GCDs) 1,396 µs mean; 4 GCDs 1,159 mean, 1,260 for the slowest; 8 GCDs 1,033
  mean, 1,346 for the slowest. With 8, each GCD would send ~10 MB of rows and get ~20 MB of
  partial sums back per layer over 45–90 GB/s links: several hundred µs, more than the ~360 µs
  saved. With 4, ~23 MB → ~150 µs against ~240 µs saved: about +3 % for a large redesign, plus
  4-way waits (the 2-way waits already cost 5–6 %). The cheaper pair balance came first.

## 8. Questions to think about

1. Why does a fixed table help at all when the router's choices change every step? What part of
   the imbalance can a fixed table never remove?
2. Why does the wait kernel need an acquire fence *after* it sees the counter, and what could go
   wrong without it?
3. Calibrating on the evaluation prompts would balance the official run better (1.0 ms). Why is
   that still the wrong thing to ship?

## 9. Discussion notes

(empty: we write here during the walk-through)
