# Lesson 26: 120b — one input table, so the expert multiply fetches its rows itself

| | |
|---|---|
| Reference commit | `7a1833c` "Paired GCDs: one input table, so mlp1 gathers the partner's rows itself" |
| Starts from | `9eb944f` (lesson 25) |
| To type | 72 lines of program code in 2 files (70 removed); 6 lines of tools; 19 lines of tests and benchmarks and 98 of docs (optional) |
| Speed after | 120b 45,468 tok/s (+0.5 %); 20b 80,341 tok/s (unchanged path) |

**In plain words:** on 120b each GPU die received its partner's rows into a separate in-tray,
then photocopied them, together with its own rows, into a binder for the expert step. Now the
partner drops its rows straight into the binder, and the expert step reads them from there. One
copying job disappears; the answers stay exactly the same.

## 1. Goal

On a split GCD, keep the expert input of both GCDs' tokens in one table, so the 120b expert
multiply can gather its input rows itself (as 20b's already does since lesson 11) and the
separate gather kernel and its 11.8 MB block go away.

## 2. Why this, why now

- Lesson 25 made the norm push the partner's rows into this GCD's memory (`t16_peer`). But the
  receiving GCD still ran a gather kernel that copied about 2,000 rows (11.8 MB per layer) from its
  own table and from `t16_peer` into one block, which the expert multiply then read back: 18.7 /
  18.3 µs per layer on the two GCDs.
- 20b never needed that copy: its first expert multiply reads every input row itself from the
  table of token rows (lesson 11's fused gather + mlp1 + SwiGLU). It only did not work for 120b
  because the partner's rows were in a second buffer.
- The profile of the work after the experts on 120b (gather, partial sums, combine, signals,
  waits) was 202 µs of a 2.46 ms layer (8.3 %) before lesson 25; this is the next piece of it.

## 3. The technique

New words:
- **pair** — one token's choice of one expert (4 per token, numbered `token × 4 + choice`); the
  grouping sorts pairs by expert.
- **table** (`t16`) — the bf16 rows of normed activations, one row per token, that the first
  expert multiply reads.
- **link** — the connection between the two GCDs of one MI250 card: about 175 GB/s each way for
  long transfers, 100–145 GB/s for the few MB of one layer; a GCD's own memory gives about
  1,300 GB/s.
- **push** — the GCD that produces data writes it straight into the partner's memory.
- **gather** — copying the rows a step needs, in the order it needs them, into one block.

```
 before (each GCD of a pair, per layer):
   t16       [own 512 rows]                 ← own norm
   t16_peer  [partner's 512 rows]           ← partner's norm pushes here
   gather:   own + t16_peer rows → Xg (≈2,000 rows, 11.8 MB)   18.7 µs
   mlp1:     reads Xg

 after:
   t16       [own rows 0..511 | partner's rows 512..1023]      ← partner's norm pushes here
   mlp1:     row r of expert e = t16[row_pair[r] / 4]           (gathers while it multiplies)
```

The pair numbering makes "pair number / 4 = row in the table" true for both GCDs' tokens: an own
pair keeps its number p, a partner pair becomes `batch × 4 + p` (it was `-(p + 1)`). So
`row_pair / 4` is `p / 4` for an own token and `batch + p / 4` for a partner token.

Why the signals still protect the data: the partner reads rows 512.. after this GCD's "routed"
signal, in its expert multiply, before its own "experts" signal; this GCD writes those rows again
only in the next layer's norm, which comes after it has waited for that "experts" signal.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe_ep.hpp` | +33 −28 | the new pair numbering (`cap × k + p`) in `moe_group_ep` and the gather, given the batch capacity | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +39 −42 | `t16` sized `2 × batch` rows on a split GCD and zeroed; `t16_peer` removed; the norm writes into `p->t16 + p->batch × H`; `gpu_pair` checks both GCDs have the same batch; the fused gather + mlp1 + SwiGLU now also with a partner | yes |
| 3 | `tools/ep_split.py` | +6 −3 | recognise the merged grouping kernel (lesson 25) in profiler traces | yes |
| 4 | `getp-csrc/tests/small_kernels_ref.hpp`, `small_kernels_test.cpp` | +16 −14 | follow the new numbering | optional |
| 5 | `getp-csrc/bench/small_kernels_bench.cpp` | +3 −3 | same | optional |
| 6 | `docs/11-expert-parallelism.md` | +98 −6 | "One table for the expert input rows" | optional |

Read while typing (line numbers at `7a1833c`):
- `getp-csrc/kernels/moe_ep.hpp:9-13` — the pair numbering, in one comment.
- `getp-csrc/engine/gpu_engine.hpp:359-367` — `t16` with room for the partner's rows, zeroed (the
  gathering mlp1 reads its last 128-k step whole, so a row runs up to 64 values into the next).
- `getp-csrc/engine/gpu_engine.hpp:504` — the norm's second copy goes to rows `batch..` of the
  partner's `t16`; `:562-563` — the fused call, now with `B + B_peer` rows and no `!p` condition.
- `getp-csrc/engine/gpu_engine.hpp:445` — the check that both GCDs of a pair have the same batch
  (the table's second half starts at row `batch`).

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 7a1833c
git show 7a1833c -- getp-csrc/engine/gpu_engine.hpp      # the change to copy

git diff 7a1833c -- $(git show --name-only --format= 7a1833c)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (20b has no partner, so nothing changes for it):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (same as lesson 25)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 10e37a089467
```

The small kernels with the new numbering (one GCD, about 30 s):

```bash
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/tests/small_kernels_test.cpp -o small_kernels_test
HIP_VISIBLE_DEVICES=2 ./small_kernels_test    # 101 checks, last line: all identical
```

The exact path also uses the new numbering, and must stay bit-identical to the CPU (two GCDs of
one card, 120b, 4 prompts × 16 steps, about 75 s):

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
EP=2 HIP_VISIBLE_DEVICES=6,7 ./gpu_vs_cpu $M120 4 16
```

Expected (seen on the reference build): `PASS: GPU logits bit-identical to the CPU`.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-26 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-26 --score
```

Expected: 120b about 45,500 tok/s, fingerprint `973e97c40a85`, METEOR 0.552, BERTScore 0.981;
20b about 80,300 tok/s, fingerprint `58023f92ead7`, METEOR 0.533, BERTScore 0.978 (both
unchanged).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Paired GCDs: one input table, so mlp1 gathers the partner's rows itself"`

## 6. Result

Per layer on one pair, 1,024 requests, profiler (first / second GCD):

| 120b, per layer | before | after |
|---|---:|---:|
| gather of the expert inputs | 18.7 / 18.3 µs | — (gone) |
| first expert multiply (now also gathers its rows) | 929.1 / 901.5 µs | +2.8 / +2.9 µs |
| net per layer per GCD | | −15.5 µs |

Whole program, official input, 8 GCDs, old and new builds alternated twice:

| | before | after |
|---|---:|---:|
| 120b | 45,224 tok/s | **45,468 tok/s (+0.5 %)** |
| 20b | — | 80,341 tok/s (one run; path unchanged) |
| Output | | same fingerprints (`973e97c40a85`, `58023f92ead7`) |

(On the other machine of the same type the same pair of builds measured 45,792 / 45,762 →
46,081 / 46,030 tok/s, +0.6 %.)

## 7. Decisions and reasons

- **Decided:** one table per split GCD with the partner's rows in its second half, and the
  numbering `batch × 4 + p` for partner pairs. **Why:** with every row in one table, 120b can use
  the same fused gather + mlp1 + SwiGLU as 20b (same bits as gather + multiply), and the 11.8 MB
  copy disappears.
- **Decided:** the table stays in ordinary memory. **Why:** in fine-grained memory the partner's
  writes into it waited for the link inside the norm kernel (31 → 38 µs), and
  `tests/peer_write_test.cpp` (lesson 25) shows ordinary memory never served a stale row.
- **Not chosen** (each measured on top of this change; full runs alternated, 2 runs each):
  - pushing the expert choices as well: top-4 +1.5 µs, the grouping no faster; together with
    the fine-grained table +0.29 % instead of +0.6 %;
  - pushing the partial sums into the partner: the combine 61 → 30 µs but the partial-sum kernel
    18 → 49 µs (the same 5.9 MB still cross the link, now inside it); 45,858 against 45,931 tok/s;
  - raising the signals from the kernels that wrote the data (no separate signal kernel): +0.02 %,
    within the noise;
  - pushing the rows from a second stream while the router multiply runs: that multiply slowed
    from 33 to 45–50 µs.
- **Lesson of this round:** a push is not free. Writing into the partner runs at the link's speed
  and the writing kernel only ends when its writes have arrived. It pays only where it replaces
  slower reads (many small ones, or the same rows read twice).

## 8. Questions to think about

1. Why does the numbering `batch × 4 + p` let the same mlp1 kernel serve 20b (no partner) and
   120b (with a partner) without an `if`?
2. The partner writes into rows 512.. of this GCD's table while this GCD is running. What stops
   it from overwriting rows this GCD's expert multiply is still reading?
3. Pushing the partial sums made one kernel twice as fast and another nearly three times slower.
   Where did the time go, and what does that say about "moving work" across the link?

## 9. Discussion notes

(empty: we write here during the walk-through)
