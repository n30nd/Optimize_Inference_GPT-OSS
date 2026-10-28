# Lesson 00: The project, the given program, and how to measure it

| | |
|---|---|
| Reference commit | `865fca5` "add kickoff slide" (the course's last commit; this branch starts here) |
| To type | nothing |
| Speed at this point | cannot run yet: the model files it expects are not on this machine (see 5) |

**In plain words:** we are given a working but very slow program that writes answers to 4,096
questions with a large language model. The job is to make it as fast as possible on 8 GPU dies,
without spoiling the answers. This lesson shows the starting program and the commands used in
every later lesson.

## 1. Goal

Know what is asked, what we may change, and how to build, run, read the output and measure.

## 2. The task (from `final-project-kickoff.pdf`)

- One command: `./run <model> -m getp -i <input file> -o <output file>`.
- **Input file:** the first line is the number of prompts, then one prompt per line. The official
  one is `evaluation/input.txt`: 4,096 prompts.
- **Output file:** one line per prompt with the generated **tokens** (a token is a word or a piece
  of a word; the model reads and writes numbers that stand for tokens).
- **Score to beat:** tok/s = generated tokens ÷ seconds. Only the `inference()` call is timed;
  loading the model in `warm_up()` is not (`getp-csrc/getp_eval.cpp:123-136`).
- **Answer length:** each prompt plus its answer is at most 1,024 tokens (`-n`, default 1024,
  `run.cpp:1154`).
- **Greedy choice:** the program always takes the single most likely next token, so the same
  input gives the same answer every time. The expected answers are in `evaluation/references/`.
- **Rules:**
  - `run.cpp`, `getp-csrc/getp_eval.cpp` and the `Makefile` must not change. Our code enters
    through `getp-csrc/getp_run.cpp` (`warm_up`, `inference`, `finish`).
  - Every GPU kernel is written by us. Ready-made GPU math libraries (rocBLAS, hipBLASLt, …)
    may only be used to compare speed, never inside the program.
  - One machine per run.
  - **Quality gates:** METEOR ≥ 0.30 and BERTScore ≥ 0.90 against the reference answers.
    - METEOR: a score of how many words and word orders match the reference answer.
    - BERTScore: a score of how close the meaning is, judged by another language model.
- **Models:** gpt-oss-20b (21 billion numbers, 13.7 GB as stored) and gpt-oss-120b (117 billion,
  ~65 GB as stored), read from `/remote/vast0/share-mv/models/openai/`.

## 3. The given program, in pieces

```
./run <model> -m getp -i in.txt -o out.txt
 └─ main()                                     run.cpp:1146
     └─ getp()                                 getp-csrc/getp_eval.cpp:105
         ├─ warm_up()          not timed       getp-csrc/getp_run.cpp:7     ← ours
         ├─ inference()        timed           getp-csrc/getp_run.cpp:109   ← ours
         │    └─ for each prompt, one after another (line 112):
         │         simple_getp_generate()      getp-csrc/getp_run.cpp:25
         │           └─ for each position: forward(transformer, token, pos)   line 68
         │                forward(): one token of one prompt, on the CPU      run.cpp:528
         ├─ write_outputfile()
         └─ finish()           not timed       getp-csrc/getp_run.cpp:16    ← ours
```

- `run.cpp` is a C++ port of "llama2.c": one file with the model's arithmetic. `forward()`
  computes the model for one token of one prompt; `matmul()` (`run.cpp:355`) is the plain
  multiply, spread over CPU cores with OpenMP (`#pragma omp parallel for`, line 360).
- The model in one breath (lesson 03 brings `docs/01-how-gpt-oss-makes-text.md` with the full
  story): each new token goes through 24 layers (20b) or 36 (120b). Each layer has an
  **attention** part (look back at the earlier tokens of the same prompt) and a **mixture of
  experts** part (the model picks 4 of its 32 or 128 small sub-networks, called experts, for
  each token). At the end, one big multiply scores all 201,088 possible next tokens and the
  best one is taken.
- The given `inference()` handles one prompt at a time, one token at a time. That is the
  slowest possible way; most of the later lessons are about doing many at once.

## 4. Build

```bash
cd /home/namvu/gpt-oss-walkthrough
source walkthrough/env.sh   # every new terminal: python with tiktoken, and $M20 $M120 $W
make runomp                 # hipcc, -O3, OpenMP; also writes tokenizer.bin
```

`make runomp` first runs `python3 export_tokenizer_bin.py` (`Makefile:42`), which needs the
`tiktoken` package. The system python does not have it; the environment turned on by `env.sh` does.

`make run` (no optimisation) and `make runfast` (-O3 without OpenMP) exist too; every lesson
uses `make runomp`, as the course measures it.

## 5. Run

**At this commit the program cannot run here.** It expects the course's exported `.bin` files
(`/nfs/gpu_trainee/final-project/modelbin/…`), which are not on this machine; the models here are
Hugging Face download folders. Exporting a `.bin` would make 84 GB (20b) or 467 GB (120b) of
fp32 numbers. Lesson 01 teaches the program to read the download folder directly.

To see where we are going, run the **finished** program in the other folder:

```bash
cd /home/namvu/gpt-oss                     # the finished program (branch getp-gpu)
make runomp
python3 tools/gpus.py                      # which GPUs are free; use the HIP column
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/final64.txt -n 128
```

What you see:
- `warm up elapsed time(s): …` — loading the model (20–40 s, mostly reading 13.7 GB).
- `elapsed time(s): …, achieved throughput TPS (tok/s): …` — the score.
- 64 prompts on one GCD are far too few to keep a GPU busy, so this number is small. The real
  score needs all prompts and all 8 GCDs (below).

## 6. Read the output

```bash
head -c 300 $W/final64.txt          # token numbers
make decode
./decode 0 -i $W/final64.txt        # answer 0 as text (-1 = all answers)
sha256sum $W/final64.txt | cut -c1-12   # expected: 446027378dd3
```

The last line is the **fingerprint** used in every lesson: the first 12 characters of a hash (a
short code computed from every byte of the file). Same fingerprint = the same answers, token for
token.

## 7. Measure

The full measurement uses `tools/bench.py` (it arrives in lesson 02; here we use the finished
folder's copy):

```bash
cd /home/namvu/gpt-oss
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t final-20b --score
```

It builds, waits until all 8 GPUs are free of our own jobs, runs all 4,096 prompts, then scores
the answers. Expect about **80,000 tok/s**, METEOR **0.533**, BERTScore **0.978**, fingerprint
`58023f92ead7`. For 120b (`-m $M120`): about **45,500 tok/s**, METEOR 0.552, BERTScore 0.981.

**When GPUs 4–7 are taken by someone else's job** (check with `python3 tools/gpus.py`), give
`bench.py` only the free ones: `-g 0,1,2,3` instead of `-g all`. `-g all` does not wait for other
people's jobs, only for ours; it shares their GPUs, and the slowest GPU sets the pace. On
2026-10-28, with GPUs 4–7 busy:

| `-g` | tok/s | METEOR | BERTScore | fingerprint |
|---|---:|---:|---:|---|
| `all` (4 of the 8 shared) | 44,085 | — | — | `58023f92ead7` |
| `0,1,2,3` | 40,478 | 0.533 | 0.978 | `58023f92ead7` |

With 4 GCDs, expect about half of every 8-GCD number in the lesson pages. Compare 4-GCD runs only
with 4-GCD runs. The answers do not change with the number of GCDs (same fingerprint).

`bench.py` prints nothing until it is done: about 2 minutes to run plus about 3 minutes to score
(BERTScore alone takes 2 minutes). To watch it, look at `run.log` and then `score.log` in the
newest folder under `$W/results/`.

## 8. Where we start and where we end

| | 20b tok/s | 120b tok/s |
|---|---:|---:|
| Given program (CPU, measured in lesson 01) | 3.3 | 0.40 |
| First GPU version (lesson 03) | 3,348 | — |
| Finished (lesson 26) | ~80,000 | ~45,500 |
| vLLM (a widely used open-source serving program), same prompts, 1 GCD | 6,539 | — |

## 9. Decisions and reasons

- **Decided:** work only on `getp` (the measured mode); `generate` and `chat` only have to keep
  working. **Why:** only `getp` is scored.
- **Decided:** two kinds of runs. A *quick check* (64 prompts, 128 positions, one GCD, about a
  minute) after every change, and a *full run* (4,096 prompts, 8 GCDs, a few minutes, with
  scores) for every speed number. **Why:** the quick check catches a wrong answer early through
  its fingerprint; only the full run gives numbers that can be compared with the course's.
- **Not chosen:** judging speed on the quick check. 64 prompts leave most of the GPU idle, so
  its tok/s says little about the full run.

## 10. Questions to think about

1. Why is `warm_up()` not timed, and what work would you move into it?
2. `inference()` handles one prompt at a time. Before lesson 03, guess: what part of the GPU's
   strength does that waste?
3. Two runs give the same fingerprint. What does that prove, and what does it not prove?

## 11. Discussion notes

**Why `source walkthrough/env.sh` and not `bash walkthrough/env.sh`.** `source` runs the file
inside the current terminal, so the settings it makes (the python environment, `$M20`, `$W`, …)
stay there. `bash env.sh` runs it in a separate shell that closes at once, and the settings are lost.

**`generate` mode and `chat` mode (`run.cpp`).** `-m` picks the function `main()` calls
(`run.cpp:1232-1238`).

| | `generate` (`run.cpp:923`) | `chat` (`run.cpp:1025`) |
|---|---|---|
| Where the text comes from | one string, `-i "…"` | the keyboard, one turn at a time |
| Turns | 1 | many, until `-n` positions are used up (`pos` is never reset, so the model still sees earlier turns) |
| Text put around your words | none | the Llama 2 template `[INST] … [/INST]` (`run.cpp:1068-1075`) |
| Stops on | token 199999 or 200002, gpt-oss's real end tokens (`run.cpp:978-982`) | token `2`, Llama 2's end token (`run.cpp:1093-1096`) |

- `chat` was copied from llama2.c and never adapted: in gpt-oss's word list token `2` is `#`, so
  the model hands the turn back whenever it writes `#`, and its real end (200002) is not noticed.
- Neither mode is scored. `simple_getp_generate()` (`getp-csrc/getp_run.cpp:25`) is `generate`'s
  loop without the printing, so understanding `generate()` means understanding the starting `getp`.

**Why a warm-up (question 1).**
- Some jobs are done once, before any question is answered. `warm_up()` is outside the clock
  (`getp-csrc/getp_eval.cpp:123-132`), and its own comment sets the one rule: "Do not inference
  here" — prepare, but do not answer.
- At `865fca5` it is empty because the CPU program has nothing left to prepare: `main()` already
  opened the model (`build_transformer`, `run.cpp:1218`) with `mmap` (the file looks as if it were
  in memory; each part is really read from disk the first time it is used, `run.cpp:288-289`).
  Catch: so the first question, inside the clock, pays for reading the model from disk.
- On the GPU there is much to prepare. In the finished program `gpu_warm_up()`
  (`getp-csrc/engine/gpu_engine.hpp:864`) copies each GCD's share of the weights into GPU memory,
  picks the batch size (how many questions a GCD handles at once), reserves the working memory and
  the KV cache (the model's memory of the earlier tokens of each question), and pairs the two GCDs
  of a card for 120b. Done in `inference()` instead, this would cost 20–40 s of score.
- The name: first runs are slow (disk reads, GPU code loading, first touch of memory). `generate()`
  does the same thing on a small scale: it starts its clock only after the first token
  (`run.cpp:990-993`).

**GPUs 4–7 are always busy: run on 0–3.** Someone else's job keeps GPUs 4–7 busy. `bench.py -g
all` still runs there because `tools/gpulock.sh` only waits for *our* lock files
(`/tmp/gptoss-gpu-locks/gpu<N>.lock`), not for other people. It gave 44,085 tok/s instead of
~80,000. That run finished at 13:25, but it left no `score.log`, so it was stopped during
scoring. Most likely it only *looked* stuck: `bench.py` prints nothing until the very end, and
scoring adds about 3 minutes after the run. Fix: `-g 0,1,2,3`. That gave 40,478 tok/s, METEOR
0.533, BERTScore 0.978 and the same fingerprint `58023f92ead7` (numbers in section 7). The finished
program needs no change: it uses however many GPUs it sees (`hipGetDeviceCount`,
`getp-csrc/engine/gpu_engine.hpp:866`). For 120b, GPUs 0,1 and 2,3 are the two dies of the same
card, so the pairs stay correct.

**A shortcut for every lesson: copy the commit instead of typing it.** Each lesson page (01–27)
now has a "Shortcut" row in its header and a "Shortcut — skip the typing" block at the top of
section 5: `git cherry-pick <commit>` copies the finished commit onto `walk-through-project` with
the same name, so there is nothing to type, compare or save; only the build and the checks are
left. The README shows how to skip several lessons at once (`git cherry-pick f0c4dbd^..9813b7a` =
lessons 01–10). Checked on 2026-10-28: all 29 commits, applied in order, go in without a conflict
and give exactly the files of `getp-gpu`. It is called "shortcut", not "fast path", because "fast
path" already means the program's mixed-precision route (lesson 04).
