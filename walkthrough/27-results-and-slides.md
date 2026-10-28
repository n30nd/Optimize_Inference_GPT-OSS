# Lesson 27: The results page and the slides

| | |
|---|---|
| Reference commit | `feddb1c` "Add the results to the README and the slides in docs/slides.md" |
| Starts from | `7a1833c` (lesson 26) |
| To type | no program code; 32 lines of `README.md` and 575 lines of `docs/slides.md` (docs: type or copy) |
| Shortcut | `git cherry-pick feddb1c` — copies the finished commit instead of typing (section 5) |
| Speed after | unchanged: 20b ~80,000 tok/s, 120b ~45,500 tok/s |

**In plain words:** the program is done for now. This last commit writes down what it achieves
on the project's front page and adds a slide deck that explains every idea to a newcomer. This
lesson also looks back over the whole road: what each step gave, where the time goes now, and
why the biggest part still reaches only about half of what the chip could do.

## 1. Goal

Anyone opening the repository sees the result first (speed and quality for both models, how to
run it), and can follow the ideas in the slides without reading the code.

## 2. Why this, why now

- After lesson 26 the code is in its final state; every number in the README can be measured
  again with one command.
- The course's README only described the starting program. A reader should not have to dig
  through 26 commit messages to learn that 20b now runs at ~80,000 tok/s.

## 3. The technique

There is no new technique here. New words used on this page:
- **Marp** — a tool that shows a Markdown file as slides; pages are separated by `---`. The
  Marp extension of VS Code shows `docs/slides.md` as a deck; any Markdown viewer shows the pages
  one after another.
- **math peak** — the most multiply-adds per second the matrix cores can do on paper: 181 TFLOP/s
  for bf16 on one GCD.
- **power cap** — the most electrical power a card may draw (560 W per MI250); at the cap the chip
  lowers its clock, so everything runs a little slower.

The whole road for 20b, as numbers (8 GCDs, official input; the CPU row: 3 prompts). The bars are on a log scale:
6 blocks = ×10, so equal steps of the bar are equal *factors* of speed.

```
 lesson  change                               tok/s
 01      CPU program                            3.3  ███
 03      first GPU version                    3,348  █████████████████████
 04      mixed precision                      5,217  ██████████████████████
 08      dense multiplies on matrix cores     7,858  ███████████████████████
 10      expert multiply on matrix cores     64,099  █████████████████████████████
 20      next token inside the LM head       75,811  █████████████████████████████
 26      final                               80,341  █████████████████████████████
```

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `README.md` | +32 | a new first section: the results table, how the work is split over the 8 GCDs, the vLLM yardstick, the larger-input result, links to the slides and the technique docs, how to run | optional (docs) |
| 2 | `docs/slides.md` | +575 | 32 pages: basics (2–5), task and results (6–10), one page per technique (11–26), what limits it now and surprises (27–29), backup words and numbers (30–32) | optional (docs) |

Nothing to read for code. Do read, in `docs/slides.md`, the pages "The journey (20b)", "Where the
time goes now" and "Why the experts reach only about half of the math peak": they summarise
sections 6 and 7 below.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat feddb1c
git cherry-pick feddb1c                   # or type README.md and docs/slides.md yourself

git diff feddb1c -- $(git show --name-only --format= feddb1c)   # must print nothing
git diff feddb1c --stat -- . ':!walkthrough'                    # the whole tree: must print nothing
make runomp
```

The last `git diff` is the real end of the walk-through: every file outside `walkthrough/` is now
exactly the finished program.

Quick checks (the program did not change since lesson 26):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 10e37a089467
```

Reproduce the README's table (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t final-20b --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t final-120b --score
```

Expected: 20b about 80,000 tok/s, fingerprint `58023f92ead7`, METEOR 0.533, BERTScore 0.978;
120b about 45,500 tok/s, fingerprint `973e97c40a85`, METEOR 0.552, BERTScore 0.981.

Larger input (optional): `python3 tools/make_input.py evaluation/input.txt 8192 > $W/input_8192.txt`,
then `GPTOSS_BATCH=1024 … bench.py -m $M20 -i $W/input_8192.txt …`. The README's 84,268 tok/s was
measured with the build of lesson 20; later builds were not measured on this input.

Save (only if you typed the files instead of cherry-picking):
`git add -A -- . ':!walkthrough' && git commit -m "Add the results to the README and the slides in docs/slides.md"`

## 6. Result

The whole journey, 8 GCDs, official input:

| Lesson | Commit | Change | 20b tok/s | 120b tok/s |
|---|---|---|---:|---:|
| 01 | `f0c4dbd` | the given program on the CPU, reading the download folder (3 prompts) | 3.3 | 0.40 |
| 03 | `e824840` | first GPU version, bit-identical to the CPU; 512 requests per GCD; data parallel | 3,348 | — |
| 04 | `3163e03` | mixed precision: bf16 into every multiply, fp32 sums | 5,217 | — |
| 05 | `d28c548` | 120b's experts split over a card's two GCDs (exact path) | — | 1,188 |
| 06 | `03e8e9d` | attention rewritten: grouped-query sharing, matrix cores, flash-decoding | 5,697 | — |
| 07 | `c61cc5d` | cache in 64-slot blocks (120b: fast path on the split model) | 5,741 | 2,835 |
| 08–09 | `30dd744`, `29f59e2` | dense multiplies on the matrix cores; parallel grouping | 7,858 | — |
| 10 | `9813b7a` | expert multiply on the matrix cores, straight from MXFP4 | **64,099** | **32,485** |
| 11 | `5d9e8e7` | SwiGLU fused into mlp1 | 66,710 | — |
| 13 | `85e8426` | top-4 with one wavefront per token (120b: with lesson 11) | 67,394 | 35,815 |
| 14 | `d68a11d` | 120b: one partial sum per token between the GCDs | — | 36,132 |
| 15 | `cdca2c5` | expert weights repacked in reading order | 72,258 | 40,164 |
| 16 | `c15ec95` | finished requests skip the experts | 72,453 | 40,633 |
| 17 | `265650c` | add and norm fused | 73,212 | 40,765 |
| 18 | `027549a` | V half of the cache in int8 (scores change) | 74,718 | 41,589 |
| 19 | `fa89592` | wave priority for small expert items | 74,746 | 41,737 |
| 20 | `9b2edc9` | next token picked inside the LM-head multiply | 75,811 | 42,113 |
| 22 | `1131a42` | stream-K | 76,149 | 42,176 |
| 23 | `a9b23f7` | 120b: pair balance and cheaper signals (120b scores change) | 76,461 | 42,981 |
| 24 | `25d09b2` | expert multiply: branch-free epilogue | 78,566 | 44,164 |
| 25 | `9eb944f` | small kernels, round two | 80,000 | 45,200 |
| 26 | `7a1833c` | 120b: one input table | 80,341 | 45,468 |

Final quality: 20b METEOR 0.533, BERTScore 0.978; 120b METEOR 0.552, BERTScore 0.981 (gates 0.30
and 0.90). Yardstick: vLLM runs 20b at 6,539 tok/s on one GCD; this program does about 10,000 per
GCD. With the official prompts repeated twice (1,024 per GCD), 20b reached 84,268 tok/s.

Where the time goes now (20b, one GCD, profile of the lesson-24 build, from the slides):

| Part | Share of the time | How close to its limit |
|---|---:|---|
| Experts | 54 % | 57 % of the math peak |
| Attention | 13 % | reads ~1.1 of the ~1.25 TB/s possible |
| Picking the next token (scoring 201,088 tokens) | 10 % | 129 of 181 TFLOP/s |
| Other multiplies | 14.5 % | ~100 TFLOP/s |
| Small kernels | ~6 % (was 8 % before lesson 25) | |

The GPU is busy about 100 % of the time: there are no gaps between kernels left to close. AMD's
library on the same dense multiplies reaches 83–97 TFLOP/s; ours reach 97–129.

Why the experts reach only about half of the math peak (measured before lesson 24):

| Loss | Factor |
|---|---:|
| the clock runs slower under the power cap (1.46 GHz instead of 1.7) | × 0.86 |
| the matrix cores wait for data 30 % of the time | × 0.70 |
| part of the work is padding: an expert with 5 tokens still fills a block of 16 | × 0.86 |
| together | ≈ 0.52 |

Lesson 24 raised it to 57 % (the first expert multiply of a layer 715 → 652 µs). 120b is harder:
about 2 of every 3 of its experts get fewer than 16 tokens per step (24 % none, 42 % 1–15), so
they wait on reading their weights, not on the math.

## 7. Decisions and reasons

- **Decided:** results first in the README, the course's own text kept below it. **Why:** the
  reader's first question is "how fast, and are the answers still good?"; the course's build and
  tokenizer instructions are still valid and needed.
- **Decided:** the headline numbers are the full official run on all 8 GCDs, with both quality
  scores. **Why:** that is what the course measures; single-GCD or short runs leave the GPU partly
  idle and are not at the power cap, so they flatter or understate.
- **Decided:** the vLLM yardstick is stated as measured: one GCD, 20b only. **Why:** its 8-GCD
  and 120b numbers were not measured, so no comparison is claimed for them.
- **Decided:** the slides explain each technique as problem → idea → whether it is used here and
  what it gave, including the well-known techniques we did *not* use (continuous batching,
  PagedAttention, tensor and pipeline parallelism, speculative decoding) and why. **Why:** a
  newcomer learns as much from "why not here" as from "what we did".

## 8. Questions to think about

1. One step (lesson 10) gave ×8.2 while most later steps gave 1–3 % each. Why did the order of
   the work matter so much, and what told us where to look each time?
2. The GPU is busy about 100 % of the time, yet the experts reach only 57 % of the math peak.
   Which of the three losses could still be attacked, and which are fixed by the hardware?
3. Two changes in this journey changed the answers (lesson 18 for both models, lesson 23 for
   120b). What made those acceptable, and what check would you insist on before accepting the
   next one?

## 9. Discussion notes

(empty: we write here during the walk-through)
