# Lesson 02: Tools to pick GPUs, wait in line, measure and compare

| | |
|---|---|
| Reference commit | `e8c283e` "Add tools to pick GPUs, queue jobs, benchmark and compare outputs" |
| Starts from | `f0c4dbd` (lesson 01) |
| To type | 409 lines of tools in 6 files, 3 lines of `.gitignore`; no program code |
| Shortcut | `git cherry-pick e8c283e` — copies the finished commit instead of typing (section 5) |
| Speed after | unchanged (no program change) |

**In plain words:** before changing the program we build the measuring tape. These small tools
find a free GPU, make our runs wait their turn instead of colliding, run the official
measurement the same way every time, and say how many answers match the expected ones.

## 1. Goal

Every later speed number is taken the same way, on GPUs nobody else is using, and is saved with
enough detail to compare later.

## 2. Why this, why now

- The machine is shared. A run on a GPU someone else is using gives a wrong, slow number.
- The GPU numbering is confusing: on these machines the number HIP uses
  (`HIP_VISIBLE_DEVICES`) is **not** the number `rocm-smi` (AMD's GPU status tool) prints for
  the same GPU. Picking "GPU 2" from `rocm-smi` and passing it to HIP lands on another GPU.
- Each speed claim later must be "before → after on the same machine, GPUs and input". That
  only works if one script does the run every time.

## 3. The technique

New words:
- **HIP index / rocm-smi index** — two numberings of the same 8 GPUs. Both list the **PCI bus
  id** (the slot address of the GPU in the machine), so the tools join the two on it.
- **lock file** — an empty file that one program holds "closed" while it uses something; a
  second program asking for the same file waits until it is released (`flock`).
- **fingerprint (hash)** — a short code computed from every byte of the output file. Same
  fingerprint = the same tokens.
- **yardstick** — a speed number from AMD's own libraries for the same multiplications. Never
  part of our program; it only tells us what "good" looks like.
- **HBM** — the GPU's own main memory (64 GB per GCD). **GB/s** = gigabytes read or written per
  second. **TFLOP/s** = trillions of multiply-or-add operations per second.

```
tools/bench.py -m <model> -i <input> -g all -t <name> --score
  1. make runomp
  2. tools/gpulock.sh all -- ./run …        waits until our other jobs leave those GPUs
  3. parse "warm up …" and "… TPS …" from the log
  4. fingerprint = sha256(output)[:12]
  5. --score: evaluation/eval.py on one free GPU → METEOR, BERTScore
  6. one JSON line appended to results/results.jsonl
```

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `tools/hip_devices.cpp` | 21 | prints "HIP index, PCI bus" for every GPU (a 20-line HIP program: a good first one to type) | yes |
| 2 | `tools/gpus.py` | 77 | table of GPUs by HIP index with rocm-smi index, bus, busy %, memory %; `--free N` prints N idle HIP indices | yes |
| 3 | `tools/gpulock.sh` | 30 | runs a command holding one lock file per GPU, then sets `HIP_VISIBLE_DEVICES` | yes |
| 4 | `tools/bench.py` | 134 | the measurement above | yes |
| 5 | `tools/ref_match.py` | 45 | share of output lines (and tokens) equal to the reference answers | yes |
| 6 | `tools/yardstick_gemm.py` | 102 | the library speed of our multiplications, and HBM copy speed (uses PyTorch) | optional |
| 7 | `.gitignore` | +3 | ignore `results/` | yes |

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick e8c283e
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout e8c283e -- $(git show --name-only --format= e8c283e)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show e8c283e -- tools/gpus.py
git diff e8c283e -- $(git show --name-only --format= e8c283e)   # must print nothing
chmod +x tools/*.py tools/*.sh                                 # typed files are not executable

python3 tools/gpus.py              # compiles tools/hip_devices on first use
python3 tools/gpus.py --free 2
tools/gpulock.sh 2 -- bash -c 'echo HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
python3 tools/ref_match.py $W/out3.txt evaluation/references/output_20b_token_ids.txt
```

Expected: the table shows two different numbers for most GPUs (HIP 0 is rocm-smi 2 on
mv-mi250-02); `gpulock.sh` prints `HIP_VISIBLE_DEVICES=2`; `ref_match.py` on lesson 01's output
says 3 of 3 lines match.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Add tools to pick GPUs, queue jobs, benchmark and compare outputs"`

## 6. Result

No speed change. What the yardstick said about one GCD (it guides every later decision):

| Measured with AMD's libraries | Number |
|---|---|
| HBM copy speed (read + write) | 1,249 GB/s (78 % of the 1.6 TB/s on paper) |
| bf16 multiply speed at ≥ 1,024 rows | 105–128 TFLOP/s (peak on paper: 181) |
| bf16 multiply speed at 64–128 rows | 40–70 TFLOP/s |
| A 20b step at 512 requests if every multiply ran at library speed | 77 ms → about 6,600 tok/s per GCD |

## 7. Decisions and reasons

- **Decided:** one script for every end-to-end number, recording the fingerprint.
  **Why:** a change that should not move any number must keep the fingerprint, and that check
  costs nothing once every run records it.
- **Decided:** our own lock files on top of looking at `rocm-smi`.
  **Why:** two of our own jobs must never share a GPU; the locks make the second one wait in
  line instead of failing or measuring nonsense. (They do not stop other people: always look
  at `tools/gpus.py` too.)
- **Decided:** library speed as the yardstick, measured on our exact shapes.
  **Why:** "fast" means nothing without a reference on the same hardware. **Not chosen:**
  other teams' numbers only — they used other inputs.

## 8. Questions to think about

1. Why would picking a GPU by its `rocm-smi` number give a wrong result here, and how does
   joining on the PCI bus fix it?
2. Two runs of the same build give tok/s 1 % apart. What could cause that, and how would you
   decide whether a change worth +0.8 % is real? (Lesson 16 meets exactly this.)
3. Why is a fingerprint a strong test for a speed change that "should not change the answers",
   but useless for one that changes the rounding?

## 9. Discussion notes

(empty: we write here during the walk-through)
