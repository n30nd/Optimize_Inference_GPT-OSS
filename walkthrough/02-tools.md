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
  5. --score: tools/score.py on every free GPU → METEOR, BERTScore
  6. one JSON line appended to results/results.jsonl
```

Step 5 on this branch comes from our own commit `ae385d4`, made after `e8c283e`; in `e8c283e`
itself it was `evaluation/eval.py` on one free GPU. `tools/score.py` calls `eval.py`'s own
functions and splits BERTScore over the GPUs in `eval.py`'s groups of 128 answers, so the
scores are the same: 20b, 4,096 answers, 148 s on 1 GPU → 57 s on 4, METEOR 0.532731 and
BERTScore F1 0.977937 both times. Once `ae385d4` is on the branch, the compare step of
section 5 (`git diff e8c283e -- …`) shows its changes to `bench.py` and `gpus.py`; that is
expected.

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

### 9.1 `tools/yardstick_gemm.py`, walked through and run (2026-10-28)

**In plain words:** the script asks AMD's own ready-made code to do each of our model's
multiplications, at 8 different numbers of requests at once, and writes down how long each took.
It also measures how fast one GPU can copy memory. Those are the "good" numbers our own code is
compared with later.

New words:
- **GEMM** (GEneral Matrix Multiply) — one big multiplication of two tables of numbers. Almost
  all of the model's work is GEMMs.
- **GCD** (Graphics Compute Die) — one of the two GPU chips inside one MI250 card. The machine
  has 4 cards = 8 GCDs; each one is "one GPU" to HIP.
- **rocBLAS / hipBLASLt** — AMD's ready-made, tuned multiply code. `torch.matmul` calls them.
- **M, N, K** — the sizes of a multiply. M = rows in = tokens (one per request in a decode step),
  K = numbers per row in, N = numbers per row out. The weight table is N × K.
- **memory-bound** — the time is set by how many bytes must be read, not by the arithmetic.
  **compute-bound** — the other way round.
- **launch overhead** — the fixed time every GPU call costs, even one with almost no work.

**The code, top to bottom**

| Lines | What it does |
|---|---|
| 15-16 | `torch.device("cuda")` — on ROCm "cuda" means the HIP GPU. `allow_bf16_reduced_precision_reduction = False` makes the library add up in fp32, the same as our kernels, so the race is fair. |
| 19-29 `timeit` | 5 warm-up calls (the library picks its kernel and gets scratch memory on the first call), then 20 timed calls between two **GPU events** (time stamps the GPU writes itself when it reaches them). Result = average µs per call. It includes any gap where the GPU waited for the CPU to send the next call. |
| 32-37 `bandwidth` | copies a 2 GiB bf16 block into another; bytes moved = read 2 GiB + write 2 GiB. Gives the real memory speed. |
| 40-49 `SHAPES`, `TOKENS` | the 7 weight tables of gpt-oss (N × K) and the 8 values of M tried. |
| 52-65 `gemm_table` | for every shape and M: `torch.matmul(x, w.t())` = x[M,K] · wᵀ → out[M,N]. `.t()` is only a view, nothing is copied. TFLOP/s = 2·M·N·K / time. GB/s = the smallest possible bytes (weight + input + output, each once) / time. |
| 68-83 `decode_step_estimate` | adds up one decode step if every multiply ran at library speed: per layer qkv + attn_out + router + (number of experts) × (expert_mlp1 + expert_mlp2), times the layers, plus lm_head once. Each token picks 4 experts, so an expert gets `batch·4/n_experts` rows if the picks are even. For an M it did not measure, it takes the next larger measured M (or scales the largest one up). |
| 86-98 `main` | prints device, bandwidth, the table, the estimates for 20b (24 layers, 32 experts) and 120b (36 layers, 128 experts) at 128…2048 requests, and saves JSON. |

What it does **not** count: attention itself (reading the saved keys and values), the norms, the
top-4 pick, and anything between layers. So the estimate is a floor for the multiplies only.

**Run** (HIP GPU 0, 1 min 14 s):

```bash
source walkthrough/env.sh
python3 tools/gpus.py --free 1                       # → 0
tools/gpulock.sh 0 -- python tools/yardstick_gemm.py $W/yardstick_gemm.json | tee $W/yardstick_gemm.log
```

```
device: AMD Instinct MI250X / MI250, 68.7 GB, torch 2.8.0+rocm6.4
HBM copy bandwidth (read+write): 1246 GB/s
         qkv M=    1 N=  5120 K= 2880       41.5 us     0.7 TFLOP/s     712 GB/s
         qkv M=  128 N=  5120 K= 2880       54.0 us    69.9 TFLOP/s     584 GB/s
         qkv M=  256 N=  5120 K= 2880       82.7 us    91.3 TFLOP/s     406 GB/s
         qkv M= 2048 N=  5120 K= 2880      512.7 us   117.8 TFLOP/s     121 GB/s
  router_20b M=    1 N=    32 K= 2880       27.6 us     0.0 TFLOP/s       7 GB/s
  router_20b M= 2048 N=    32 K= 2880       29.2 us    12.9 TFLOP/s     414 GB/s
 expert_mlp1 M=   64 N=  5760 K= 2880       48.7 us    43.6 TFLOP/s     704 GB/s
 expert_mlp2 M=   64 N=  2880 K= 2880       33.3 us    31.9 TFLOP/s     521 GB/s
     lm_head M=    1 N=201088 K= 2880     1176.5 us     1.0 TFLOP/s     985 GB/s
     lm_head M=  256 N=201088 K= 2880     2909.3 us   101.9 TFLOP/s     434 GB/s
     lm_head M=  512 N=201088 K= 2880     7198.1 us    82.4 TFLOP/s     190 GB/s
     lm_head M= 1024 N=201088 K= 2880     9515.7 us   124.6 TFLOP/s     166 GB/s
{'model': '20b', 'batch': 512, 'rows_per_expert': 64, 'step_ms': 77.77, 'tok_per_s_per_gcd': 6584}
{'model': '120b', 'batch': 512, 'rows_per_expert': 16, 'step_ms': 371.53, 'tok_per_s_per_gcd': 1378}
```
(shortened; the full output is in `$W/yardstick_gemm.log`, the numbers in `$W/yardstick_gemm.json`)

**What the numbers say**

1. **Few rows → memory-bound; many rows → compute-bound.** A multiply does about M operations
   per byte of weight. The GCD can do ~115 TFLOP/s and read ~1.25 TB/s, so the turn-over point is
   ~90 rows. Below it, the time hardly moves with M (qkv: 41 µs at 1 row, 47 µs at 64 rows);
   above it, the time doubles when M doubles (qkv 1024 → 2048: 290 → 513 µs).
2. **Big tables reach the memory speed, small ones do not.** lm_head (1.16 GB of weight) reads at
   985 GB/s = 79 % of the copy speed. qkv (29 MB) only gets 712 GB/s: the call ends before the
   memory system is fully busy.
3. **Every call costs at least ~27 µs.** The router multiply at 1 row and at 2,048 rows both take
   27-29 µs. Check run (not in the script): a tiny add costs 5.4 µs, a tiny 1×32×32 matmul 26.9 µs,
   and the router M=64 drops from 25.6 to 16.1 µs when 20 calls are recorded once and replayed as
   a **HIP graph** (the GPU replays a recorded list of calls, so the CPU does no work per call).
   So ~10 µs is CPU and library bookkeeping per call, ~16 µs is the library's kernel itself.
4. **The library has bad spots.** lm_head at 512 rows takes 7.2 ms, more than twice the 2.9 ms
   at 256 rows (82 TFLOP/s instead of ~100). It picked a poor kernel for that size.
5. **Experts are almost the whole step.** 20b at 512 requests, per layer: qkv 162 + attn_out 126
   + router 28 + 32 × (48.7 + 33.3) = 2,624 µs of experts → 2,940 µs, i.e. **89 % experts**.
   × 24 layers + lm_head 7.2 ms = 77.8 ms → 512 / 0.0778 s = 6,584 tok/s per GCD.
   32 experts' bf16 weights are 1.59 GB per layer; at 1,246 GB/s that is 1.28 ms, but 32
   separate library calls take 2.62 ms — the per-call cost and the small-table effect (points 2
   and 3) double it.
6. **120b is worse:** 16 rows per expert and 128 experts per layer → 128 × (44 + 32) µs ≈ 9.8 ms
   per layer, nearly all of it fixed per-call cost. One call per expert is hopeless here.

**Limits of the yardstick**

- It times the experts in bf16. Our program keeps expert weights in **mxfp4** (4 bits per
  number, with one shared scale per 32 numbers), ~¼ of the bytes, so a good expert kernel can beat
  the bf16 library line when memory-bound.
- It is one GCD, and 120b does not fit on one (lesson 05 splits it over a pair); for 120b read the
  estimate as "work per GCD", not as a runnable speed.
- The same multiply is repeated 20 times in a row. Only the router table (184 KB) fits in the
  GCD's 8 MB L2 cache and is read from cache after the first call; the other tables are re-read
  from memory every time, as in a real step.

**What this decides later**

- All experts in one call, not one call each (`gemm_grouped_mxfp4` from lesson 04; made fast on
  the matrix units in lesson 10): removes the per-call cost of points 3 and 6, and reads 4-bit
  weights.
- The ordinary multiplies on the matrix units (lesson 08), aiming at the 100+ TFLOP/s the
  library shows above ~256 rows.
- Lesson 12 quotes the 77 ms / ~6,600 tok/s line as the "library speed" bar.
