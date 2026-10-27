# Walk-through: rebuilding the fast gpt-oss program one commit at a time

This branch starts from the course's starting code and rebuilds, by hand, every change that made
gpt-oss about 24,000 times faster on 8 AMD GPU dies (from 3.3 to ~80,000 words per second for
gpt-oss-20b). One lesson = one commit of the finished branch `getp-gpu`. Each lesson page says what
the change is for, the idea behind it, what to type, how to check it and what it gave.

Words used everywhere below:
- **commit** — one saved step in the project's history, with a short name like `e824840`.
- **branch** — a line of commits. `getp-gpu` is the finished work; `walk-through-project` (this one)
  is where you retype it.
- **GCD** — one of the two GPU dies on an MI250 card; it works as one GPU. The machine has 4 cards,
  so 8 GCDs.
- **kernel** — a small program that runs on the GPU. The program here is about 30 kernels called
  one after another for every new word.
- **tok/s** — tokens (word pieces) produced per second, summed over all requests. The course
  measures this number.

## Where things are

| Folder | Branch | What it is for |
|---|---|---|
| `/home/namvu/gpt-oss-walkthrough` | `walk-through-project` | where you type; starts at commit `865fca5` (the course's last commit) |
| `/home/namvu/gpt-oss` | `getp-gpu` (also `main`) | the finished program; read it, run it, do not type here |

The two folders share one history (a git *worktree*: a second folder showing another branch of
the same repository), so `git show e824840` works in both.

**Why the start is `865fca5`, not `3163e03`:** `3163e03` is our 4th commit (the mixed-precision
fast path). The course's own code ends at `865fca5` "add kickoff slide"; our first commit is
`f0c4dbd`. Starting at `865fca5` lets you type the first GPU port (lesson 03) yourself.

## One-time setup

```bash
cd /home/namvu/gpt-oss-walkthrough
source /home/namvu/gpt-oss/.venv/bin/activate      # python with tiktoken, torch, bert-score
export M20=/remote/vast0/share-mv/models/openai/gpt-oss-20b
export M120=/remote/vast0/share-mv/models/openai/gpt-oss-120b
export MODELS_ROOT=/home/namvu/gptoss-models         # holds bs_model/roberta-large-mnli for scoring
export W=/home/namvu/walk-runs                       # run inputs and outputs, kept out of the repo
mkdir -p $W
(echo 64; sed -n 2,65p evaluation/input.txt) > $W/in64.txt   # first 64 prompts, for quick checks
```

Put the `source` and `export` lines in a small file and `source` it in every new terminal.
`make runomp` also writes `tokenizer.bin` (the word list) next to `./run`; git ignores it.

## How one lesson goes

Lesson 00 is the introduction and has its own layout. Lessons 01–27 all have the same
sections, in this order:

| Section | What is in it |
|---|---|
| In plain words | 2–3 lines without technical words |
| 1. Goal | the problem this commit solves |
| 2. Why this, why now | the measurement that pointed here |
| 3. The technique | the idea, with every new word explained and a text picture |
| 4. The change, file by file | the order to type the files in and what each part is for |
| 5. Type, compare, build, check | the commands, and the result you must see |
| 6. Result | numbers before → after |
| 7. Decisions and reasons | what was decided, why, and what was not chosen |
| 8. Questions to think about | three questions to discuss |
| 9. Discussion notes | empty at first; we write our discussion here |

The steps, the same for every lesson:

1. Read sections 1–3.
2. Look at the change: `git show --stat <commit>`, then `git show <commit> -- <file>` for each file
   (or the whole file as it was then: `git show <commit>:<file>`).
3. Type the files in the order of section 4. Program files first; tests, benchmarks and docs are
   marked "optional to type": you may copy them with `git checkout <commit> -- <file>`.
4. Compare with the original — this must print nothing:
   `git diff <commit> -- $(git show --name-only --format= <commit>)`
5. Build and run the checks of section 5.
6. Save it with the same name: `git add -A -- . ':!walkthrough' && git commit -m "<same subject>"`.
7. Discuss. The notes go into section 9 of the lesson page, saved in their own commit
   (`git commit -m "walkthrough: notes for lesson NN" walkthrough/`).

Quick check after a lesson (one GCD, 64 prompts, 128 positions, about a minute):

```bash
python3 /home/namvu/gpt-oss/tools/gpus.py              # pick a free GPU; use its HIP number
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12                    # the "fingerprint" of the answers
```

For 120b, use the two GCDs of one card (HIP 0,1 here), about 2–3 minutes:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12
```

Same fingerprint as the lesson page says = the same answers, word for word. The full measurement
(8 GCDs, all 4,096 prompts, with the quality scores) is in each lesson's section 5.

## The lessons

Speeds are for all 8 GCDs on the official 4,096 prompts (`evaluation/input.txt`). Runs on
different days vary by about 1 %. "—" = not measured for that model at that step.

| # | Commit | What it does, in plain words | Lines to type: program / tools / tests and benchmarks / docs | 20b tok/s after | 120b tok/s after |
|---|---|---|---|---:|---:|
| [00](00-the-project.md) | `865fca5` | The task, the given program, how to build, run, read the output and measure | — | — | — |
| [01](01-hf-weights.md) | `f0c4dbd` | Read the model straight from its download folder (CPU only) | 1,091 / – / 166 / – | 3.3 (CPU) | 0.40 (CPU) |
| [02](02-tools.md) | `e8c283e` | Tools: pick free GPUs, wait in line, measure, compare answers | – / 409 / – / 3 | — | — |
| [03](03-bit-exact-gpu-port.md) | `e824840` | First GPU version, giving exactly the CPU's numbers | 978 / – / 434 / 491 | 3,348 | — |
| [04](04-mixed-precision-and-first-profile.md) | `3163e03` | Smaller numbers for the big multiplications; first time profile | 467 / – / 2 / 81 | 5,217 | — |
| [05](05-120b-expert-split.md) | `d28c548` | 120b does not fit one GCD: split its experts over a card's two GCDs | 562 / – / 36 / 91 | — | 1,188 |
| [06](06-attention-rewrite.md) | `03e8e9d` | Attention rewritten: share each read, use the matrix units | 372 / – / 672 / 425 | 5,697 | — |
| [07](07-kv-cache-blocks.md) | `c61cc5d` | Store the attention memory in 64-slot blocks | 22 / – / – / – | 5,741 | 2,835 |
| [08](08-dense-gemm-matrix-cores.md) | `30dd744` | The ordinary multiplications on the matrix units | 511 / – / 558 / 267 | 7,858 (with 09) | — |
| [09](09-parallel-moe-grouping.md) | `29f59e2` | Sort tokens by expert with many workers instead of one | 88 / – / – / – | 7,858 (with 08) | — |
| [10](10-expert-gemm-matrix-cores.md) | `9813b7a` | The expert multiplications on the matrix units (the ×8 step) | 658 / – / 869 / 370 | 64,099 | 32,485 |
| [11](11-fuse-swiglu.md) | `5d9e8e7` | Do the step after the first expert multiply inside it | 24 / – / – / – | 66,710 | — |
| [12](12-make-input-tool.md) | `0214995` | A tool that makes bigger inputs by repeating prompts | – / 25 / – / – | 75,640 (8,192 prompts) | — |
| [13](13-topk-one-wave-per-token.md) | `85e8426` | Pick each token's 4 experts with 64 workers per token | 64 / – / – / – | 67,394 | 35,815 |
| [14](14-partial-sum-exchange.md) | `d68a11d` | 120b: send one sum per token between the two GCDs | 90 / – / – / – | — | 36,132 |
| [15](15-repack-expert-weights.md) | `cdca2c5` | Reorder the expert weights once, so reads are in order | 212 / – / 94 / 216 | 72,258 | 40,164 |
| [16](16-skip-finished-requests.md) | `c15ec95` | Finished requests no longer use the experts | 32 / – / – / – | 72,453 | 40,633 |
| [17](17-fuse-add-and-norm.md) | `265650c` | Merge "add" and "normalize" into one kernel | 131 / – / – / – | 73,212 | 40,765 |
| [18](18-int8-v-cache.md) | `027549a` (+ `6b315b5`) | Store half of the attention memory in 1 byte per number | 171 / – / 540 / 343 | 74,718 | 41,589 |
| [19](19-wave-priority.md) | `fa89592` | Let small expert jobs fetch first | 30 / – / 142 / 184 | 74,746 | 41,737 |
| [20](20-fused-argmax.md) | `9b2edc9` (+ `7faed85`) | Pick the next word inside the last multiplication | 150 / – / 122 / 31 | 75,811 | 42,113 |
| [21](21-expert-gemm-time-breakdown.md) | `64f5f16` | Where the expert multiply's time goes (measurement only) | – / – / 26 / 67 | — | — |
| [22](22-stream-k.md) | `1131a42` | Share the work evenly over all 104 compute units | 319 / – / 54 / 145 | 76,149 | 42,176 |
| [23](23-pair-balance-and-signals.md) | `a9b23f7` | 120b: give both GCDs of a card equal expert work | 251 / 610 / – / 142 | 76,461 | 42,981 |
| [24](24-branch-free-epilogue.md) | `25d09b2` | Expert multiply: store results without branches | 79 / – / 34 / 221 | 78,566 | 44,164 |
| [25](25-small-kernels-round-two.md) | `9eb944f` | Small kernels: start all reads first, merge two kernels | 648 / – / 1,365 / 160 | 80,000 | 45,200 |
| [26](26-one-input-table.md) | `7a1833c` | 120b: one input table, so the expert multiply fetches rows itself | 72 / 6 / 19 / 98 | 80,341 | 45,468 |
| [27](27-results-and-slides.md) | `feddb1c` | The results page and the slides | – / – / – / 607 | — | — |

Lessons 18 and 20 end by copying a docs-only commit (`git cherry-pick <commit>`): nothing to type.

## Tools in this folder

- `kernel_shares.py` — turns a `rocprofv3` kernel trace into a table: each kernel's share of
  the GPU time, calls and µs per call (first used in lesson 04).

## Where the numbers come from

- tok/s and quality scores: the full run of each commit, measured with `tools/bench.py`
  (lesson 02) on all 8 GCDs.
- Kernel times (µs): the commit messages (`git show <commit>`) and the technique docs in
  `docs/` (they arrive with the commits).
- Fingerprints of the quick check: each reference commit built and run once, 20b on one GCD,
  120b on a GCD pair. They agree with the fingerprints quoted in the commit messages.
