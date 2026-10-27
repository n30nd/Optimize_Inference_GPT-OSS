# Lesson 12: A tool that makes bigger inputs by repeating prompts

| | |
|---|---|
| Reference commit | `0214995` "Add a tool that repeats an input file's prompts to a given count" |
| Starts from | `5d9e8e7` (lesson 11) |
| To type | 25 lines of tools in 1 file (`tools/make_input.py`); no program code |
| Speed after | official input unchanged (66,710 tok/s); with 8,192 prompts: **75,640** tok/s (20b) |

**In plain words:** the official input has 4,096 questions, which gives each of the 8 GPU dies
512 to work on at once. The GPU gets more done per second when it has more questions in flight,
because it fetches the model's numbers once per step no matter how many questions use them. This
small tool doubles the input by repeating the questions, so we can see how fast the program is
with 1,024 questions per die. It is an extra measurement, not the official score.

## 1. Goal

Make inputs of any size from the official prompts, keeping the official prompts first and in
order, so the first answers can still be scored against the reference answers.

## 2. Why this, why now

- After lesson 10 the expert multiply was fast, and it is faster per row with more rows: 79
  TFLOP/s at 512 requests per GCD, 99 at 1,024, 110 at 2,048 (20b mlp1, lesson 10's table).
- Each step reads all the model's weights once, whatever the number of requests. More requests
  per step = more work for each byte read.
- The library yardstick (lesson 02) says the same: a 20b step at library speed takes 77 ms at
  512 requests, 94 ms at 1,024 (twice the work for 1.22× the time), 149 ms at 2,048.
- With the official input, 512 requests per GCD is the most we can have: 4,096 ÷ 8.

## 3. The technique

New words:
- **batch** — the requests one GCD works on together, all moving one token forward per step.
  `GPTOSS_BATCH` sets its size (default 512, `GPU_DEFAULT_BATCH` in
  `getp-csrc/engine/gpu_engine.hpp`).
- **wave** — one batch from start to end. With 1,024 requests per GCD and a batch of 512, a GCD
  runs two waves one after the other; with a batch of 1,024, one.
- **KV cache** — the memory where each request keeps what attention needs from its earlier
  tokens; it grows with batch × positions. At 1,024 requests × 1,024 positions (bf16) it is 29 GB
  per GCD, next to 13.7 GB of 20b weights: it fits in 64 GB.

```
 make_input.py evaluation/input.txt 8192
   line 1:            8192
   lines 2 … 4097:    the 4,096 official prompts, in order
   lines 4098 … 8193: the same prompts again (prompt i % 4096)

 run with GPTOSS_BATCH=1024: each GCD gets 1,024 prompts and runs them as one wave

 one step on one GCD:   read all weights once ──► multiply 512 rows   (B=512)
                        read all weights once ──► multiply 1,024 rows (B=1024)
                        same bytes read, twice the useful work
```

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `tools/make_input.py` | 25 | reads an input file's prompts (skipping the count line and empty lines) and prints `count` prompts, repeating them in order | yes |

The whole tool is the loop at `tools/make_input.py:19–21`: `print(prompts[i % len(prompts)])`.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show 0214995
git diff 0214995 -- tools/make_input.py      # must print nothing
chmod +x tools/make_input.py

python3 tools/make_input.py evaluation/input.txt 8192 > $W/input_8192.txt
wc -l $W/input_8192.txt                      # 8193 (count line + 8,192 prompts)
sed -n 2p $W/input_8192.txt; sed -n 4098p $W/input_8192.txt   # the same prompt twice
```

There is no program change, so the quick check still gives `ccfc732a4a25` (lesson 11).

Full run on the larger input (20b; about 2 minutes of inference):

```bash
GPTOSS_BATCH=1024 GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i $W/input_8192.txt -g all -t lesson-12-8192 --score
```

`--score` works here because the scorer compares only as many lines as the reference has
(`evaluation/eval.py:101`, `n = min(len(refs), len(subm))`): the first 4,096 answers, which
belong to the official prompts.

Expected: about **75,640 tok/s**, METEOR 0.565, BERTScore 0.981, fingerprint `e82e6b3d00b4`.

The scores are not exactly those of the 4,096-prompt run (0.554 / 0.979): with 1,024 rows some
dense multiplies pick another tile shape or split their sums differently
(`gemm_pick_config`, `getp-csrc/kernels/fast_gemm.hpp:474`), which changes the rounding, and
greedy answers that differ in one token can drift apart afterwards.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Add a tool that repeats an input file's prompts to a given count"`

## 6. Result

20b, 8 GCDs, same build (lesson 11):

| input | requests per GCD | tok/s |
|---|---:|---:|
| official, 4,096 prompts | 512 | 66,710 |
| repeated, 8,192 prompts, `GPTOSS_BATCH=1024` | 1,024 | **75,640** (+13 %) |

The same larger-input measurement was repeated at two later points: 79,504 tok/s after lesson 15
and **84,268** after lesson 20 (the number on the README).

## 7. Decisions and reasons

- **Decided:** also measure 20b on the official prompts repeated twice (8,192 requests, 1,024
  per GCD), next to the official number.
  **Why:** the expert multiply is more efficient with more rows (79 → 99 TFLOP/s from 512 to
  1,024 requests); 1,024 per GCD fits (29 GB of KV cache + 13.7 GB of weights); repeating the
  input was allowed. It shows what the same program does when more work is available.
- **Decided:** the official prompts stay first and in order in the larger file.
  **Why:** so the first 4,096 answers can be scored against the reference answers with the
  same scorer.
- **Not chosen:** 2,048 requests per GCD. **Why:** its KV cache would need 58 GB, and the weights
  13.7 GB: more than the 64 GB of a GCD.
- **Not changed:** the official score is still measured on `evaluation/input.txt` as given.

## 8. Questions to think about

1. Why is "twice the requests per step" close to "twice the tok/s" for the weight reads, but not
   for attention? (Hint: what does attention read for each request?)
2. With 8,192 prompts and the default batch of 512, each GCD runs two waves of 512. Why is that
   slower than one wave of 1,024, even though the total work is the same?
3. The larger input repeats the same prompts. Could that make the measurement look better than a
   real input of 8,192 different prompts would? Think about what in the program depends on the
   prompt text.

## 9. Discussion notes

(empty: we write here during the walk-through)
