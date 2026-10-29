# Lesson 05: 120b does not fit one GCD — split its experts over a card's two GCDs

| | |
|---|---|
| Reference commit | `d28c548` "Split gpt-oss-120b's experts between the two GCDs of each card" |
| Starts from | `3163e03` (lesson 04) |
| To type | 562 lines of program code in 2 files (+421 −179 in the engine, 141 new); 36 lines of tests (optional); 91 lines of docs (optional) |
| Shortcut | `git cherry-pick d28c548` — copies the finished commit instead of typing (section 5) |
| Speed after | 120b 1,188 tok/s — the first GPU run of 120b, measured on the **exact** path; 20b unchanged (5,217) |

**In plain words:** the big model is slightly larger than one GPU die's memory. Most of it is the
"experts", and each word only uses 4 of the 128. So the two dies of one card each keep half of the
experts and everything else in full. Each die serves its own questions; only in the expert step
do the two dies hand work to each other.

## 1. Goal

Run gpt-oss-120b on the GPUs at all, with the same exact numbers as the CPU, using all 8 GCDs.

## 2. Why this, why now

- 120b stored compactly (experts in MXFP4) is **65 GB**; one GCD has **64 GB**. It does not fit,
  even before any room for the requests' memory.
- 20b already ran on the GPUs (lessons 03–04); 120b was still the CPU at 0.40 tok/s.
- We split as little as possible: 95 % of the bytes are experts, and the rest (attention,
  router, embeddings: 4.2 GB) is small enough to copy to both GCDs.

## 3. The technique

New words:
- **expert parallelism (EP)** — splitting the experts of each layer over several GPUs. Here
  EP = 2: GCD A holds experts 0–63, GCD B experts 64–127, of every layer.
- **data parallelism** — copies working side by side on different requests, never talking. The
  8 GCDs form 4 *groups* (pairs); groups are data parallel, the two GCDs of a pair are not.
- **package (card)** — one MI250 card: two GCDs joined by their fastest link. HIP devices
  (0,1), (2,3), (4,5), (6,7) are the pairs on our machines.
- **peer access** — one GPU reading another GPU's memory directly over the link
  (`hipDeviceEnablePeerAccess`).
- **stream** — a queue of GPU work; the GPU runs it in order. The engine uses one per GCD.
- **event** — a marker put into a stream (`hipEventRecord`). Another stream can be told "wait
  until that marker is passed" (`hipStreamWaitEvent`); the GPU itself waits, the host does not.
- **KV cache** — the stored keys and values of all earlier tokens of each request (lesson 06
  explains them); it grows with every token.

One layer on a pair, for one step:

```
        GCD A (experts 0–63)                         GCD B (experts 64–127)
        its own 512 requests                         its own 512 requests
 attention, router for its rows               attention, router for its rows
        │ topk_A, t_A ready ── event ──┐  ┌── event ── topk_B, t_B ready │
        ▼                              ▼  ▼                              ▼
 group the rows that chose 0–63           group the rows that chose 64–127
 from BOTH A's and B's tokens              from BOTH A's and B's tokens
 (reads topk_B, t_B in B's memory)         (reads topk_A, t_A in A's memory)
 run experts 0–63 → Z_A                    run experts 64–127 → Z_B
        │ Z_A ready ── event ──┐        ┌── event ── Z_B ready               │
        ▼                      ▼        ▼                                    ▼
 for each own token: add its 4 expert     same on B
 outputs, reading rows from Z_A or Z_B
```

Two waits per layer, both on the GPU. One host thread drives both GCDs of a pair, so it always
queues the marker before the wait that refers to it. One copy of each shared buffer is enough:
a GCD overwrites `t`, `topk` or `Z` only in the next layer, after waiting for a marker its
partner sets once it has finished reading them.

Each GCD **reads** (pulls) its partner's buffers instead of the partner writing (pushing) into
it: another team saw a receiving GPU's cache keep stale contents after pushed copies.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe_ep.hpp` | +141 (new) | `moe_group_ep_kernel` (group own and partner rows by local expert), `moe_gather_ep_kernel<T>` (copy each row's input from own or partner memory), `moe_combine_ep_kernel<kExact>` (add each own token's 4 outputs, some read from the partner) | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +421 −179 | `GpuWorker` gains `e_begin`/`e_count`, `peer`, two events; new `GpuGroup`; `gpu_pair()`; the step cut into `step_embed`, `layer_attention`, `layer_route`, `layer_experts`, `layer_combine`, `step_logits`; `gpu_step_group()`; waves served per group (`WaveSlot`, `gpu_run_wave`); `gpu_group_size()`; groups built in `gpu_warm_up()` | yes |
| 3 | `getp-csrc/tests/gpu_vs_cpu.cpp` | +36 −51 | `EP=2` splits the experts over GPUs 0 and 1 in the bit-exact test | optional |
| 4 | `docs/11-expert-parallelism.md` | +91 | this lesson's technique for newcomers | optional |

Read while typing (line numbers at `d28c548`):
- `getp-csrc/kernels/moe_ep.hpp:7-8` — a row refers to an own pair as `p` and a partner pair as
  `-(p + 1)`; the sign tells the kernels whose memory to read.
- `getp-csrc/engine/gpu_engine.hpp:302` `gpu_pair()` — peer access switched on in both directions.
- `getp-csrc/engine/gpu_engine.hpp:448` `gpu_step_group()` — the three loops per layer; the
  markers at lines 462 and 471, the waits at 468 and 477.
- `getp-csrc/engine/gpu_engine.hpp:602` `gpu_group_size()` — 2 when the weights would fill more
  than 60 % of a GCD; `GPTOSS_EP` overrides.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick d28c548
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout d28c548 -- $(git show --name-only --format= d28c548)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat d28c548
git show d28c548 -- getp-csrc/kernels/moe_ep.hpp
git diff d28c548 -- $(git show --name-only --format= d28c548)   # must print nothing
make runomp
```

Quick check, 20b (one GCD; 20b is not split, so nothing may change):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # b4c73f55a422, the same as lesson 04
```

Quick check, 120b (one pair = two GCDs of one card; this runs the default fast path):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 78d2c8fd9f66
```

The startup line says `fast path, 2 GPU(s) in 1 group(s) of 2 …`.

Bit-exact test with the experts split (build and run commands from the test's first lines):

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/gpu_vs_cpu.cpp tokenizer.cpp -o gpu_vs_cpu
EP=2 tools/gpulock.sh 2,3 -- ./gpu_vs_cpu $M20 4 16      # 20b, 4 prompts x 16 steps
EP=2 tools/gpulock.sh 2,3 -- ./gpu_vs_cpu $M120 2 12     # 120b, 2 prompts x 12 steps
```

Expected: one line per step with `logits differing 0`, then the last line (about 35 s for 20b,
about 1 minute for 120b, most of it the CPU side):

```
pos  15: logits differing 0 / 804352, max |diff| 0, argmax differing 0
PASS: GPU logits bit-identical to the CPU
```

Full run, 120b. **The first 120b number was taken on the exact path** (fp32 cache, so only 256
requests per GCD fit: two waves of 256), hence `GPTOSS_EXACT=1` in front. At 1,188 tok/s it takes
about an hour:

```bash
GPTOSS_EXACT=1 GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-05 --score
```

Expect about **1,188 tok/s**, METEOR **0.988**, BERTScore **0.99987**, fingerprint `33c21d6e726b`.
The 20b full run is unchanged (lesson 04: 5,217 tok/s).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Split gpt-oss-120b's experts between the two GCDs of each card"`

## 6. Result

| 120b | before | after |
|---|---|---|
| Where it runs | CPU | 8 GCDs = 4 pairs |
| tok/s (official 4,096 prompts) | 0.40 | **1,188** (×3,000) |
| METEOR / BERTScore | — | 0.988 / 0.99987 |
| Answers identical to the reference | — | 3,992 of 4,096 lines |
| Logits vs the CPU, experts split over 2 GCDs | — | bit-identical (20b 4 × 16, 120b 2 × 12) |

Memory of one 120b GCD:

| | per GCD |
|---|---:|
| all dense weights (bf16) | 4.2 GB |
| half of the experts (MXFP4) | 30.4 GB |
| KV cache, 512 requests, bf16 (fast path) | 21.8 GB |
| read from the partner per layer, on average: inputs (bf16) / expert outputs (fp32) | ≈ 5.9 MB / ≈ 11.8 MB |

**This first 120b number is the exact path.** The exact path keeps its KV cache in fp32 (twice
the bytes), so only 256 requests per GCD fit: the 1,188 run served each GCD's 512 requests in two
waves of 256. The fast path also runs at this commit (the quick check above), but its first
full-run number comes in lesson 07: 2,835 tok/s.

## 7. Decisions and reasons

- **Decided:** split each layer's experts between the two GCDs of one card (EP = 2), four such
  pairs side by side; each GCD keeps all dense weights and serves its own requests.
  **Why:** half the experts plus all dense weights is 34.6 GB per GCD, leaving room for 512
  requests' bf16 KV cache (21.8 GB). The two GCDs of a card have the fastest link. Each GCD reads
  only ~6 MB + ~12 MB of its partner's data per layer.
  **Not chosen:**
  - spreading the experts over all 8 GCDs (EP = 8): every token's work crosses slower links to
    many GPUs, and every layer waits for 8 of them; another team measured 63 % of the time spent
    waiting on that exchange;
  - **tensor parallelism** (cutting every matrix of every layer in two): it needs an exchange
    in attention too, every layer.
- **Decided:** the first 120b run uses the exact path. **Why:** the same rule as lesson 03 —
  correct first. With bit-identical logits, any later difference is a real bug, not rounding.
- **Decided:** one host thread per pair, GPU-side waits (events). **Why:** with one thread per
  GCD, one thread could queue "wait for the partner's marker" before the partner's thread queued
  the marker, and the wait would pass at once.
- Measured later: splitting 20b the same way (so each expert sees twice the rows) was **4 %
  slower** — the bigger multiplies did not pay for reading the partner's data. 20b stays one
  model copy per GCD.

## 8. Questions to think about

1. Why split the experts and not the attention? Count what each GCD would have to send to its
   partner in each case.
2. A GCD reads its partner's `t` buffer during the expert step. What stops the partner from
   overwriting `t` with the next layer's values while it is still being read?
3. Each GCD runs its 64 experts on rows from 1,024 tokens (its own and its partner's). Why is that
   better for the expert multiply than 512 tokens on 128 experts?

## 9. Discussion notes

### 9.1 The forward pass: 20b on one GCD, 120b on a pair

Question: walk through the forward pass for 20b and 120b; for 120b, the flow of cooperation, then each operation and kernel. Line numbers are at `d28c548`.

In short: every step, each GCD runs the same chain of about 15 kernels per layer on its own batch of 512 tokens. For 20b, one GCD runs the whole chain alone. For 120b, the two GCDs of one card each run the whole chain for their own 512 tokens, except the expert step. There, each GCD runs only its 64 experts, on the tokens of both GCDs that picked them. Then, in the combine step, each GCD adds the expert outputs into its own tokens' `x`, reading some rows from its own `Zg` and some from the partner's. The two GCDs wait for each other twice per layer (two events).

#### Words used below

- **kernel** — a small program that runs on the GPU. A launch `kernel<<<grid, threads>>>` starts `grid` *blocks*. A **block** is a team of threads that runs on one compute unit, shares a small fast scratch memory (**LDS**, local data share) and can wait for each other (`__syncthreads()`).
- **step** — one turn of the main loop: every request gets one new token.
- **fp32 / bf16** — numbers stored in 32 bits / in 16 bits (bf16 keeps the same range but fewer digits).
- **MXFP4** — 4 bits per number, plus one shared scale (a power of two) for each group of 32 numbers. The experts are stored this way.
- **`x`, the running row** — the 2,880 numbers that stand for one token. Every layer adds its result to it (`x += …`).
- **RMSNorm** — scale a row so that the average of its squares is 1, then multiply each entry by a learned weight.
- **GEMM** (general matrix multiply) — `out = in × Wᵀ + bias`, here for all requests at once.
- **RoPE** (rotary position embedding) — turn pairs of numbers by an angle that depends on the token's position, so attention can tell how far apart two tokens are.
- **attention** — the token's *query* is compared with the *keys* of earlier tokens. The scores become weights, and the output is the weighted sum of their *values*.
- **softmax** — turns scores into positive weights that add up to 1: `exp(s) / Σ exp`.
- **sink** — one extra learned score per head. It joins the softmax but has no value, so a head can give part of its weight to "nothing".
- **router** — a small GEMM that gives each token one score per expert. **top-k** means keeping the 4 best.
- **SwiGLU** — the step between the two expert multiplies. Half of the numbers act as a "gate" that scales the other half.
- **logits** — one score for every word of the 201,088-word vocabulary. **argmax** is the position of the biggest score.
- **prefix sum** — running total: counts `[2, 3]` become start positions `[0, 2, 5]`.
- **stream / event / peer access** — see lesson section 3: a GPU work queue / a marker in it / one GPU reading another's memory.

#### 1. The sizes

| | 20b | 120b |
|---|---:|---:|
| layers | 24 | 36 |
| experts per layer | 32 | 128 |
| experts held by one GCD | 32 | 64 |
| GCDs per group | 1 | 2 (one card) |
| requests per GCD (`B`) | 512 | 512 fast path, 256 exact path |

The same in both: `H` = 2,880 (row width), 64 query heads × 64 = 4,096, 8 key/value heads × 64 = 512, so one `qkv` row is 4,096 + 512 + 512 = 5,120. The expert inner width is `I` = 2,880, `k` = 4 experts per token, vocabulary `V` = 201,088. Even layers only look at the last 128 positions (*sliding window*); odd layers look at all of them.

#### 2. 20b: one step on one GCD

**The host side (the CPU).**
- `gpu_group_size`: 20b is 13 GB, less than 60 % of 64 GB, so `ep = 1`. Every GCD holds all 32 experts (`e_begin = 0`, `e_count = 32`) and is a group of one: 8 groups.
- `gpu_inference`: one CPU thread per group. 4,096 / 8 = 512 requests per group, which is one *wave*.
- `gpu_run_wave`, every step: copy the `B` current tokens to the GPU, queue the step (`gpu_step_group` with one member), copy the `B` guesses back, and wait (`hipStreamSynchronize`). Then the CPU picks each request's next input. If the request is still inside its prompt, it takes the next prompt token and ignores the guess. Otherwise it takes the guess and writes it to the output. Token 199999 or 200002 ends a request.

**The GPU side: the kernels of one step, in order** (`B` = 512 rows; grids at `B` = 512).

Before the layers (`step_embed`):

| # | kernel | grid × threads | in → out | what each block does |
|---|---|---|---|---|
| 1 | `embed_kernel` | 512 × 256 | `tokens` → `x` (B, 2880) fp32 | copies row `tokens[b]` of the bf16 embedding table, as fp32 |

Each layer, attention part (`layer_attention`, fast path lines 345–353):

| # | kernel | grid × threads | in → out | what |
|---|---|---|---|---|
| 2 | `rmsnorm_bf16_kernel` | 512 × 256 | `x` → `t16` (B, 2880) bf16 | each thread sums the squares of ~12 entries; a tree sum in LDS gives the total; each entry is written as `w·x/rms` in bf16 |
| 3 | `gemm_bf16` (qkv) | (80, 8) × 256 | `t16 · W_qkv(5120, 2880)ᵀ + b` → `qkv` (B, 5120) fp32 | a block computes one 64 × 64 tile of the output, each thread 4 × 4 of it; the 2,880 inputs go through LDS 32 at a time |
| 4 | `rope_kv_bf16_kernel` | 512 × 256 | turns the 64 query heads and 8 key heads in place; writes key and value (bf16) into slot `kv_slot` of this layer's cache | `kv_slot` = `pos`, or `pos % 128` on sliding layers (a ring of 128) |
| 5 | `attention_bf16_kernel` | (64 heads, 512) × 64 | `qkv` + cache → `att16` (B, 4096) bf16 | one block per (head, request). Thread `tid` scores positions `tid, tid+64, …`, adds the sink, takes the softmax, then sums the values for dimension `tid`. Query heads 0–7 share key/value head 0, 8–15 head 1, … (`h / 8`) |
| 6 | `gemm_bf16` (out) | (45, 8) × 256 | `att16 · W_o(2880, 4096)ᵀ + b` → `proj` | |
| 7 | `add_inplace_kernel` | 5760 × 256 | `x += proj` | one thread per number |

Each layer, routing part (`layer_route`):

| # | kernel | grid × threads | in → out | what |
|---|---|---|---|---|
| 8 | `rmsnorm_bf16_kernel` | 512 × 256 | `x` → `t16` | the same as #2, with the second norm weight |
| 9 | `gemm_bf16` (router) | (1, 8) × 256 | `t16 · W_router(32, 2880)ᵀ + b` → `router` (B, 32) fp32 | |
| 10 | `router_topk_exact` | 8 × 64 | `router` → `topk_e`, `topk_w` (B·4 each) | one thread per token: picks the 4 largest scores (on a tie, the lower expert), then a softmax over those 4 |

From here on, a **pair** is one (token, choice). Its number is `p = b·4 + j`, so token `b`'s four pairs are `4b … 4b+3`.

Each layer, expert part (`layer_experts`, fast path lines 396–403). For 20b there is no partner: `B_peer = 0`, `p = nullptr`.

| # | kernel | grid × threads | in → out | what |
|---|---|---|---|---|
| 11 | `moe_group_ep_kernel` | 1 × 64 | `topk_e` → `offsets` (33), `row_pair`, `own_row` | sorts the 2,048 pairs by expert: rows `offsets[e] … offsets[e+1]-1` belong to expert `e` (details in 3.4) |
| 12 | `moe_gather_ep_kernel<bf16>` | 2048 × 256 | `t16` → `Xg16` (rows in expert order) | row `r` gets a copy of the `t16` row of token `row_pair[r] / 4` |
| 13 | `gemm_grouped_mxfp4` (mlp1) | (90, 8, 32) × 256 | `Xg16 · W1_e(5760, 2880)ᵀ + b` → `Yg` (R, 5760) fp32 | grid z = expert, y = which 64-row tile of that expert, x = which 64 output columns. Blocks past the expert's last row stop at once. Each weight is decoded from 4 bits while it is loaded: table value × 2^(scale − 127) (`Mxfp4Rows`, `gemm.hpp:33-43`) |
| 14 | `swiglu_bf16_kernel` | R·2880/256 × 256 | `Yg` → `Gg16` (R, 2880) bf16 | even columns are the gate `g`, odd ones are `u`: `out = min(g,7)·sigmoid(1.702·min(g,7)) · (clamp(u, −7, 7) + 1)` |
| 15 | `gemm_grouped_mxfp4` (mlp2) | (45, 8, 32) × 256 | `Gg16 · W2_e(2880, 2880)ᵀ + b` → `Zg` (R, 2880) fp32 | |

Each layer, combine part (`layer_combine`):

| # | kernel | grid × threads | in → out | what |
|---|---|---|---|---|
| 16 | `moe_combine_ep_kernel<false>` | 512 × 256 | `x[b] += Σ_j topk_w[b,j] · Zg[own_row[4b+j]]` | one block per token, each thread owns the columns `tid, tid+256, …` |

After the last layer (`step_logits`):

| # | kernel | grid × threads | in → out |
|---|---|---|---|
| 17 | `rmsnorm_bf16_kernel` | 512 × 256 | `x` → `t16`, with the final norm weight |
| 18 | `gemm_bf16` (LM head, the last multiply) | (3142, 8) × 256 | `t16 · W_unemb(201088, 2880)ᵀ` → `logits` (B, 201088) fp32, 412 MB |
| 19 | `argmax_kernel` | 512 × 256 | `logits` → `next[b]` |

That is 15 kernels per layer: 1 + 24 × 15 + 3 = **364 launches per step** for 20b.

#### 3. 120b: how the two GCDs of a card cooperate

##### 3.1 Setup, once (`gpu_warm_up`)

- `gpu_group_size`: 65 GB is more than 60 % of 64 GB, so `ep = 2` (`GPTOSS_EP` can force it).
- One CPU thread per GCD builds its worker at the same time (`lines 635–641`): `rank = d % 2`, so the even HIP devices hold experts 0–63 and the odd ones 64–127.
- `gpu_upload_weights` copies the dense weights in full. For the expert tensors it copies only a slice. Experts are stored one after another, so experts `e0 … e0+63` are one unbroken byte range starting at `e0 × (bytes of one expert)`. `g1 = 2·I·(H/32)` = 518,400 groups of 32 numbers for mlp1, each 16 bytes of codes + 1 scale byte = 8.8 MB per expert. mlp2 is 4.4 MB. 13.2 MB × 64 experts × 36 layers = **30.4 GB**, the number in section 6.
- The expert buffers (`Xg`, `Yg`, `Gg`, `Zg`, `row_pair`) get room for `R = B·4·2` = 4,096 rows, twice 20b's, because they may also hold the partner's rows (`gpu_expert_rows`).
- The same batch on every GCD, the largest that the fullest one allows (`lines 645–647`).
- Groups: `g_groups[d / 2]` = {GCD 2g, GCD 2g+1}. `gpu_pair` turns on peer access both ways (A may read B's memory and B may read A's).

##### 3.2 Serving

- `gpu_inference`: one CPU thread per group (4 threads), 1,024 requests per group. A wave is `batch × 2` = 1,024 requests, split evenly: the first 512 go to member 0, the next 512 to member 1 (`lines 692–696`). Each member has its own `WaveSlot`: its own requests, prompts and outputs. Its KV cache only ever holds its own requests, so **attention never crosses to the partner**.
- `gpu_run_wave` works as for 20b, for both members: copy the tokens to each, queue the step, copy the guesses back, wait for both streams, update both. The two GCDs always move to the next position together, because they meet in every layer.

##### 3.3 One step: the order the CPU queues the work (`gpu_step_group`)

A = member 0, B = member 1. One CPU thread queues everything:

```
CPU queues, in this order               A's stream                    B's stream
loop 1  i=0  A: attention + route       attn(l) route(l) ●rA
        i=1  B: attention + route                                     attn(l) route(l) ●rB
loop 2  i=0  A: wait rB, experts        ⏸rB experts(l) ●xA
        i=1  B: wait rA, experts                                      ⏸rA experts(l) ●xB
loop 3  i=0  A: wait xB, combine        ⏸xB combine(l)
        i=1  B: wait xA, combine                                      ⏸xA combine(l)

● = hipEventRecord (puts a marker)    ⏸ = hipStreamWaitEvent (the GPU waits; the CPU goes on)
rA/rB = ev_routed of A/B              xA/xB = ev_experts of A/B
```

The CPU queues all 36 layers without stopping, then waits in `hipStreamSynchronize`. On the GPUs, over time:

```
time →
A: [attn+route]●rA ⏸…until rB [group gather mlp1 swiglu mlp2]●xA ⏸…until xB [combine] [attn+route l+1] …
B: [attn+route]●rB ⏸…until rA [group gather mlp1 swiglu mlp2]●xB ⏸…until xA [combine] [attn+route l+1] …
                                  ↑ reads partner's topk_e, t16            ↑ reads partner's Zg, peer_row
```

What A reads in B's memory each layer (B does the same the other way):

| A's kernel | reads in B's memory | why it is ready |
|---|---|---|
| `moe_group_ep_kernel` | `B.topk_e` (512·4 ints) | A waited for rB, which B records after its route |
| `moe_gather_ep_kernel` | the `B.t16` rows of B's tokens that chose A's experts: ≈ 1,024 rows × 5.8 KB ≈ **5.9 MB** | the same |
| `moe_combine_ep_kernel` | `B.Zg` rows of A's tokens that chose B's experts: ≈ 1,024 rows × 11.5 KB ≈ **11.8 MB**, plus `B.peer_row` | A waited for xB, which B records after its experts |

Nothing is ever written into the partner's memory. Each GCD only reads (pulls).

**Why one copy of each buffer is enough** (question 2 of section 8). A next overwrites its `t16` and `topk_e` in its own layer l+1 attention/route. In A's queue, that comes after "⏸xB" of layer l. B records xB after experts(l), the very kernels that read A's `t16` and `topk_e`. So once A gets past ⏸xB, B has finished reading them. The same holds for A's `Zg` and `peer_row`. A next writes them in experts(l+1), after "⏸rB" of layer l+1. B records rB(l+1) after its route(l+1), which comes after combine(l), the kernel that read them.

**Why one CPU thread drives both GCDs.** `hipStreamWaitEvent` waits for the *latest* `hipEventRecord` of that event *queued so far*. Suppose two threads were used and A's thread queued "⏸rB" before B's thread queued layer l's "●rB". A would then wait for layer l−1's marker, which has long passed, and read B's buffers while they are still being written. With one thread, loop 1 queues both markers before loop 2 queues any wait.

##### 3.4 The three new kernels, on a tiny example

Toy sizes: 4 experts (A holds 0–1, B holds 2–3), `k` = 2, 2 tokens per GCD. Pair `p = token·2 + choice`.

```
topk_e on A: [1, 3 | 0, 1]    A token 0 chose experts 1, 3; A token 1 chose 0, 1
topk_e on B: [2, 0 | 1, 2]    B token 0 chose 2, 0;         B token 1 chose 1, 2
```

**`moe_group_ep_kernel`** (`moe_ep.hpp:22-72`). One block; thread `e` looks after local expert `e`.
- Lines 32–39: thread `e` counts the own pairs and the partner's pairs that chose expert `e_begin + e`. At the same time (lines 40–49), all threads write −1 into `own_row` / `peer_row` for pairs whose expert is not on this GCD.
- Lines 51–57: thread 0 turns the counts into start positions (prefix sum) in LDS and copies them to `offsets`.
- Lines 59–71: thread `e` walks the pairs again and fills its rows from `start[e]`, own pairs first, then the partner's. `row_pair[row]` is `p` for an own pair and `-(p+1)` for a partner pair. The `+1` keeps own pair 0 (`0`) apart from partner pair 0 (`-1`). `own_row` / `peer_row` are the reverse lookup: pair → row.

On A (experts 0, 1): expert 0 has A's pair 2 and B's pair 1 (2 rows); expert 1 has A's pairs 0, 3 and B's pair 2 (3 rows). `offsets = [0, 2, 5]`.

```
row       0      1      2      3      4
expert    0      0      1      1      1
row_pair  2     -2      0      3     -3
meaning   A t1   B t0   A t0   A t1   B t1
own_row_A  = [2, -1, 0, 3]     (A's pair 1 chose expert 3: not here)
peer_row_A = [-1, 1, 4, -1]    (B's pairs that chose 0 or 1, and their rows on A)
```

On B (experts 2, 3): `offsets = [0, 2, 3]`, `row_pair = [0, 3, -2]`, `own_row_B = [0, -1, -1, 1]`, `peer_row_B = [-1, 2, -1, -1]`. (I checked both tables with a small Python copy of the kernel.)

Cost: each thread walks all pairs one by one (2 × 2,048 for 120b), in a single block, so it uses 1 of the 104 compute units. Lesson 09 replaces this.

**`moe_gather_ep_kernel`** (`moe_ep.hpp:77-92`). One block per row. The CPU launches `R` = 4,096 blocks because it does not know the real row count (that is only on the GPU). Blocks past `offsets[e_count]` return at once. If `p ≥ 0`, it copies own token `p / k` from `x_own` (A's `t16`). If `p < 0`, it copies partner token `(-p-1) / k` from `x_peer` (B's `t16`, read over the link between the two GCDs). In the toy: row 1 on A has `p = -2` → B's pair 1 → B's token 0. ✓

Next come the expert GEMMs and SwiGLU: **the same kernels as for 20b**, now with 64 experts (grid z = 64). An expert can get up to `B + B_peer` = 1,024 rows, so grid y = 16.

**`moe_combine_ep_kernel`** (`moe_ep.hpp:100-141`). One block per **own** token `b`. For each of its `k` choices it finds the row: if the expert is here, `Z_own[own_row[p]]`; if not, `Z_peer[peer_rows_of_mine[p]]`. Here `Z_peer` is the partner's `Zg`, and `peer_rows_of_mine` is the partner's `peer_row`: the row the partner gave to *my* pair `p` when it grouped. Then `x[b] += Σ w·Z`, each thread over columns `tid, tid+256, …`.
Toy, A's token 0 (pairs 0, 1): pair 0 chose expert 1 (here) → `Z_A` row `own_row_A[0]` = 2. Pair 1 chose expert 3 (on B) → `Z_B` row `peer_row_B[1]` = 2. Check: B's row 2 has `row_pair = -2`, which is A's pair 1. ✓
`kExact = true` (exact path) first sorts the 4 choices by expert number and adds with the same rounding as the CPU, so the result stays bit-identical. The fast path adds in choice order.

##### 3.5 20b and 120b side by side

| | 20b | 120b |
|---|---|---|
| group | 1 GCD | the 2 GCDs of one card |
| expert rows per GCD per layer | 2,048 (all its own) | 2,048 on average (≈ half own, half the partner's); room for 4,096 |
| average rows per expert | 2,048 / 32 = 64 | 4,096 / 128 = 32 |
| GPU-side waits per layer | 0 | 2 |
| read from the partner per layer | 0 | ≈ 5.9 MB + ≈ 11.8 MB |
| kernel launches per step per GCD | 364 | 1 + 36 × 15 + 3 = 544 |

### 9.2 `moe_group_ep_kernel` in full: inputs, outputs, shapes

Question: the one-line table row for this kernel in 9.1 was not enough — what are the inputs and outputs, their shapes, and what do they look like?

In short: the input is `topk_e` of both GCDs, the 4 experts picked by every token. The output is four int arrays: `offsets` (where each local expert's rows start), `row_pair` (which token and pick each row belongs to, own or partner), and `own_row` / `peer_row` (the reverse lookup: pick → row). The gather, the expert GEMMs and the combine all use these to know which row goes where.

Numbers below are for 120b on **GCD B** (the odd HIP device, holding experts 64–127). "Own" means B's 512 tokens; "peer" means A's 512 tokens. 20b differences are at the end of part 1.

#### 1. Operation

New words:
- **pair** — one (token, choice). Every token picks `k` = 4 experts, so 512 tokens make 2,048 pairs. Pair number `p = b·4 + j` (b = token 0–511, j = choice 0–3).
- **int32** — a whole number stored in 4 bytes.
- **local expert** `e` — numbering of the experts this GCD holds: on B, local 0 = expert 64, local 63 = expert 127. `e_begin` = 64 changes one into the other.

**Inputs** (launch at `gpu_engine.hpp:384-386`):

| name | argument | shape, type | where it lives | what it holds |
|---|---|---|---|---|
| `T_own` | `topk_own` = `w->topk_e` | [2048] int32, read as [512 tokens, 4 choices] | this GCD | the experts (0–127) each own token picked, best first |
| `T_peer` | `topk_peer` = `p->topk_e` | [2048] int32, read as [512, 4] | **the partner's memory** | the same for the partner's tokens |
| `N_own`, `N_peer` | `n_own`, `n_peer` | numbers | — | `B·4` = 2,048 each |
| `e_begin`, `E_loc` | `e_begin`, `e_count` | numbers | — | 64 and 64 on B (0 and 64 on A) |

Both come from `router_topk_exact` in the route step. What they look like (first 3 tokens; random choices made in Python, only to show the layout. Real choices are not uniform):

```
T_own  (B's tokens)   token 0: [ 98,  52,  75,  24]   pairs 0..3
                      token 1: [111,   1,  89,  23]   pairs 4..7
                      token 2: [104,  39,  28,  45]   pairs 8..11   ... 512 tokens
T_peer (A's tokens)   token 0: [ 34,  16,  65,  30]
                      token 1: [126, 115, 120,  97]
                      token 2: [ 53,  24, 124,   7]   ... 512 tokens
```

On B, the picks of 64–127 are "here" (98, 75, 111, 89, 104 in T_own; 65, 126, 115, 120, 97, 124 in T_peer). The rest belong to A.

**Outputs** (all new values, written over the previous layer's; none of them is an input):

| name | argument | shape, type | what it holds |
|---|---|---|---|
| `O` | `offsets` | [E_loc + 1] = [65] int32 | `O[e]` = first row of local expert `e`; `O[64]` = number of rows in use |
| `RP` | `row_pair` | [4096] int32 (room for all own + all peer pairs); only rows `0 … O[64]-1` are written | what each row is: own pair `p` → `p`; peer pair `q` → `-(q+1)` |
| `OR` | `own_row` | [2048] int32, one per own pair | the row that own pair got, or −1 if its expert is on the partner |
| `PR` | `peer_row` | [2048] int32, one per **partner** pair | the row that partner pair got here, or −1 |

**Math.** For local expert `e = 0 … 63` (global expert `g = 64 + e`):

```
count[e] = #{ p : T_own[p] = g } + #{ q : T_peer[q] = g }          p, q = 0 … 2047
O[e]     = count[0] + … + count[e-1]          O[0] = 0,  e = 0 … 64
Rows O[e] … O[e+1]-1 belong to expert g, in this order:
   first the own pairs p with T_own[p] = g, in increasing p,
   then the peer pairs q with T_peer[q] = g, in increasing q.
RP[r] = p        if row r is own pair p
RP[r] = -(q+1)   if row r is peer pair q
OR[p] = r if row r is own pair p,   −1 if T_own[p]  is not in 64 … 127
PR[q] = r if row r is peer pair q,  −1 if T_peer[q] is not in 64 … 127
```

In words: line up every pick (both GCDs') of my experts, sorted by expert, and write down the row each one got, both ways.

What the outputs look like for the random choices above (computed in Python with the same rules):

```
count[0..5] = [40, 36, 36, 26, 47, 31] ...   (between 19 and 47; 32 on average)
O           = [0, 40, 76, 112, 138, 185, 216, ..., 2058]   65 numbers
                └ expert 64 = rows 0..39 ┘└ expert 65 = rows 40..75 ┘ ...

RP rows 0..39 (expert 64):
  368  641  809  851  866  898  958 1042 1084 1298 1467 1722 1803 1843 1990    ← 15 own pairs
 -137 -168 -265 -285 -353 -357 -411 -461 -690 -713 -776 -788 -849 -901        ← 25 peer pairs
 -1020 -1064 -1130 -1248 -1256 -1473 -1625 -1671 -1675 -1718 -1830
  e.g. 368 = own pair 368 = token 92, choice 0;  -137 = peer pair 136 = A's token 34, choice 0

OR[0..11] = [1097, -1, 369, -1, 1547, -1, 823, -1, 1313, -1, -1, -1]   (1,015 of 2,048 are ≥ 0)
PR[0..11] = [-1, -1, 59, -1, 2015, 1690, 1837, 1079, -1, -1, 1945, -1] (1,043 of 2,048 are ≥ 0)
  OR[0] = 1097: own pair 0 (token 0 → expert 98) is row 1097.
  PR[2] = 59:   A's pair 2 (A's token 0 → expert 65) is row 59 here; expert 65 starts at O[1] = 40.
```

Rows used: 2,058 of the 4,096 reserved. On average half the picks of both GCDs land here, 2,048 rows.

**Who uses each output next:**

| output | read by | to do what |
|---|---|---|
| `O` | `moe_gather_ep_kernel` (only `O[64]`), both `gemm_grouped_mxfp4` | how many rows to fill; which rows go with which expert's weights |
| `RP` | `moe_gather_ep_kernel` | where to copy each row's input from: `t16` of own token `p/4`, or the partner's `t16` of token `q/4` |
| `OR` | this GCD's `moe_combine_ep_kernel` | where my token's result is in my `Zg` |
| `PR` | **the partner's** `moe_combine_ep_kernel` (as `peer_rows_of_mine`) | where A's token's result is in B's `Zg` |

The −1 values are never read later. The combine checks the expert number itself (`moe_ep.hpp:125-127`), so the −1 only keeps the arrays tidy.

**20b:** no partner, so `T_peer = nullptr` and `N_peer = 0`, `e_begin = 0`, `E_loc = 32`. `O` is [33], every row is an own pair (`RP` ≥ 0, 2,048 rows exactly), `OR` has no −1, and `PR` is not touched.

#### 2. Thread distribution

Launch: `<<<1, max(64, ⌈E_loc/64⌉·64), 0, s>>>` = **1 block of 64 threads** for both models (`gpu_engine.hpp:384`). The block has one shared array `start[1025]` (LDS), room for up to 1,024 experts.

| level | count | owns |
|---|---|---|
| grid | 1 block | the whole job |
| block | 64 threads | phase 1 and 4: thread `e` = local expert `e`. Phase 2: pairs `tid, tid+64, …`. Phase 3: thread 0 only |

The four phases do **not** use the same split:

| phase | lines | who | loop | work per thread (120b) |
|---|---|---|---|---|
| 1 count | `32-39` | thread `e` < 64 | all 2,048 own + 2,048 peer picks | 4,096 reads |
| 2 mark −1 | `40-49` | all 64 | own pairs `tid, tid+64, …` then peer pairs | 32 + 32 pairs |
| — barrier | `50` | | | |
| 3 running total | `51-57` | thread 0 | 64 adds, 65 writes | |
| — barrier | `58` | | | |
| 4 fill rows | `59-71` | thread `e` < 64 | all 4,096 picks again | ~32 rows written |

```
phase 1/4: thread e scans every pick, keeps only expert 64+e
   T_own  [ 98  52  75  24 | 111 1 89 23 | ... ]   ← all 64 threads read every cell
   T_peer [ 34  16  65  30 | 126 ...      ]
phase 2:   thread t marks pairs t, t+64, t+128, ...
   T_own   p:  0  1  2 ... 63 | 64 65 ... 127 | ...
   thread:     0  1  2 ... 63 |  0  1 ...  63 | ...
```

Why the barriers: phase 3 (thread 0) needs every thread's `count` from phase 1, and phase 4 (all threads) needs thread 0's running totals. These go through `start[]` in LDS, and `__syncthreads()` makes everyone wait until they are written.

Cost: one block runs on 1 of the 104 compute units, and every thread reads all 4,096 picks twice. Lesson 09 makes this parallel.

#### 3. Implementation details: one thread, with numbers

The real sizes are too big to write out, so this uses the toy from 9.1. 4 experts: A holds 0–1, B holds 2–3. `k` = 2, 2 tokens per GCD, so 4 pairs per GCD. We follow **GCD A, thread 1** (local expert 1 = global expert 1; `e_begin = 0`, `E_loc = 2`). Thread 0 is the one that also does phase 3, so we pick thread 1.

```
T_own  (A) = [1, 3, 0, 1]     pairs 0,1 = A token 0;  pairs 2,3 = A token 1
T_peer (B) = [2, 0, 1, 2]     pairs 0,1 = B token 0;  pairs 2,3 = B token 1
```

**Phase 1** (lines 30–39), `e = 1`, `ge = 0 + 1 = 1`:

```
count = (T_own[0]==1) + (T_own[1]==1) + (T_own[2]==1) + (T_own[3]==1)    = 1+0+0+1 = 2
      + (T_peer[0]==1)+ (T_peer[1]==1)+ (T_peer[2]==1)+ (T_peer[3]==1)   = 0+0+1+0 = 1
start[2] = 3
```
(Thread 0 does the same for expert 0 and writes `start[1] = 2`.)

**Phase 2** (lines 40–49). The loop is `p = 1; p < 4; p += 64`, so thread 1 only handles pair 1 of each list:

```
own  pair 1: x = T_own[1] - 0 = 3   → not in 0..1 → own_row[1]  = -1
peer pair 1: x = T_peer[1] - 0 = 0  → in 0..1     → nothing written
```
(Thread 0 handles pair 0: `own_row[0]` stays, `peer_row[0] = -1` because B's pair 0 chose 2. Threads 2, 3 handle pairs 2, 3: `peer_row[3] = -1`.)

**Phase 3** (lines 51–57), thread 0 only: `start = [0, 2, 3]` → `start = [0, 2, 5]`, copied to `offsets = [0, 2, 5]`.

**Phase 4** (lines 59–71), thread 1. `row = start[1] = 2`. It reads `start[1]`, which thread 0 wrote in phase 3; that is why the barrier on line 58 is there.

| code | numbers | result |
|---|---|---|
| own p = 0: `T_own[0] == 1` | yes | `row_pair[2] = 0`, `own_row[0] = 2`, row → 3 |
| own p = 1, 2: | 3, 0: no | — |
| own p = 3: `T_own[3] == 1` | yes | `row_pair[3] = 3`, `own_row[3] = 3`, row → 4 |
| peer p = 2: `T_peer[2] == 1` | yes | `row_pair[4] = -(2+1) = -3`, `peer_row[2] = 4`, row → 5 |

It stops at row 5 = `offsets[2]`, exactly where its block ends.

Phase 2 and phase 4 never write the same cell. Phase 2 writes −1 only for picks of experts that are *not* here, and phase 4 writes only picks that *are* here. So the different splits cannot collide.

Result on A (thread 0 filled rows 0–1, thread 1 rows 2–4):

```
offsets  = [0, 2, 5]
row      0    1    2    3    4
row_pair 2   -2    0    3   -3
own_row  = [2, -1, 0, 3]
peer_row = [-1, 1, 4, -1]      ← B's combine reads this: B's pair 1 (expert 0) is row 1 on A
```

Checked by running the same rules in Python (both the toy and the random 512-token case above).

The other 63 threads run the same code for their own expert. Together they fill every row `0 … offsets[64]-1` exactly once, and every pick of experts 64–127 (B) or 0–63 (A) gets exactly one row.

### 9.3 The whole expert step (`layer_experts`): inputs and outputs, then each kernel

Question: explain the input and output of the expert phase first, then go into each of its kernels.

In short: the expert step takes the `t16` row of every token (own and partner's) that picked one of this GCD's 64 experts, and runs each row through that expert: GEMM, SwiGLU, GEMM. The output is `Zg`: one row of 2,880 numbers per (token, pick), sorted by expert. Nothing is added into `x` yet; `layer_combine` does that.

Numbers: 120b, **GCD B** (experts 64–127), fast path. `B` = 512 own tokens, `B_peer` = 512 partner tokens, `K` = 4, `H` = 2,880, `I` = 2,880, `E_loc` = `e_count` = 64, so `R = (512 + 512)·4` = **4,096** rows of room. 20b: `B_peer` = 0, `E_loc` = 32, `R` = 2,048. The exact path differences are at the end.

#### A. The whole phase (`layer_experts`, `gpu_engine.hpp:375-405`)

**Inputs**

| name | buffer | shape, type | made by | what it is |
|---|---|---|---|---|
| `t_own` | `w->t16` | [512, 2880] bf16 | `layer_route`: RMSNorm of `x` | each own token's row, ready for the experts |
| `t_peer` | `p->t16` | [512, 2880] bf16, **partner's memory** | the partner's `layer_route` | the same for the partner's tokens |
| `T_own`, `T_peer` | `w->topk_e`, `p->topk_e` | [512, 4] int32 each | `router_topk_exact` | the 4 experts each token picked |
| `W1` | `g.mlp1_blocks`, `g.mlp1_scales` | [64 experts, 5760, 90, 16] uint8 and [64, 5760, 90] uint8 | upload (this GCD's slice) | first expert matrix, 5,760 × 2,880 per expert, in MXFP4 |
| `b1` | `g.b_mlp1` | [64, 5760] fp32 | upload | its bias |
| `W2` | `g.mlp2_blocks`, `g.mlp2_scales` | [64, 2880, 90, 16] and [64, 2880, 90] uint8 | upload | second expert matrix, 2,880 × 2,880 per expert |
| `b2` | `g.b_mlp2` | [64, 2880] fp32 | upload | its bias |

`90 = 2880 / 32` groups per weight row; 16 bytes hold the 32 four-bit codes of one group; 1 scale byte per group. The weights of one layer on one GCD are 564 MB (`W1`) + 282 MB (`W2`).

**Outputs**

| name | buffer | shape, type | what it is |
|---|---|---|---|
| `Z` | `w->Zg` | [4096, 2880] fp32; rows `0 … O[64]-1` used (~2,048) | row `r` = the expert's answer for the (token, pick) `RP[r]`, sorted by expert |
| `O`, `RP`, `OR`, `PR` | `offsets`, `row_pair`, `own_row`, `peer_row` | [65], [4096], [2048], [2048] int32 | the row bookkeeping (9.2). The combine needs `OR` here and the partner needs `PR` |

**Scratch buffers in between** (only used inside the phase): `Xg16` [4096, 2880] bf16 (23.6 MB), `Yg` [4096, 5760] fp32 (94.4 MB), `Gg16` [4096, 2880] bf16 (23.6 MB).

**Not touched:** `x` (the running row) and `topk_w` (the 4 weights per token). Both are used only by `layer_combine`.

**Math of the whole phase.** For a used row `r`, let `RP[r]` say it is pick `j` of token `b` on GCD `d` (own or partner), with expert `g = T_d[b, j]`:

```
Z[r, :] = W2_g · SwiGLU( W1_g · t_d[b, :] + b1_g ) + b2_g
             [2880×2880]  [5760×2880] [2880]  [5760]   [2880]
```

In words: each row is the token's row, sent through the expert it picked.

**The five kernels and the buffers between them:**

```
T_own, T_peer ─► ① group ─► O, RP, OR, PR
t_own, t_peer, RP, O ─► ② gather ─► Xg16 [R,2880] bf16
Xg16, O, W1, b1 ─► ③ mlp1 multiply ─► Yg [R,5760] fp32
Yg ─► ④ SwiGLU ─► Gg16 [R,2880] bf16
Gg16, O, W2, b2 ─► ⑤ mlp2 multiply ─► Z = Zg [R,2880] fp32
```

#### B. Each kernel

##### ① `moe_group_ep_kernel` — sort the picks by expert

Inputs `T_own`, `T_peer` [2048] int32 each, `e_begin` = 64, `E_loc` = 64. Outputs `O` [65], `RP` [4096], `OR` [2048], `PR` [2048]. Grid 1 block × 64 threads. Full details in 9.2. What matters for the next kernels:
- rows `O[e] … O[e+1]-1` belong to expert `64+e`;
- `RP[r]` ≥ 0 means own pair `RP[r]`, so own token `RP[r]/4`. `RP[r]` < 0 means partner pair `q = -RP[r]-1`, so partner token `q/4`.

In the random example of 9.2: `O = [0, 40, 76, …, 2058]`, `RP[0] = 368` (own token 92), `RP[15] = -137` (partner pair 136, so partner token 34).

##### ② `moe_gather_ep_kernel<bf16_t>` — copy each row's input into expert order (`moe_ep.hpp:77-92`)

| | name | shape, type |
|---|---|---|
| in | `t_own` = `x_own` | [512, 2880] bf16 |
| in | `t_peer` = `x_peer` | [512, 2880] bf16, partner's memory |
| in | `RP` = `row_pair`, `O[64]` = `offsets[e_count]` | [4096] int32, one number |
| out | `Xg16` = `Xg` | [4096, 2880] bf16, rows `0 … O[64]-1` written |

```
Xg[r, i] = t_own [ RP[r] / 4,        i ]   if RP[r] ≥ 0
Xg[r, i] = t_peer[ (-RP[r]-1) / 4,   i ]   if RP[r] < 0          r = 0 … O[64]-1,  i = 0 … 2879
```

In words: a pure copy. Row `r` gets the row of the token it stands for. A token that picked 2 of my experts appears twice.

Split: grid **4,096 blocks** × 256 threads, one block per row. The CPU does not know `O[64]` (it is only on the GPU), so it launches the maximum, and blocks with `r ≥ O[64]` return at line 85. Thread `tid` copies columns `tid, tid+256, …`: 2,880 / 256 = 11.25, so threads 0–63 copy 12 numbers and threads 64–255 copy 11. No cooperation between threads.

Example (random case): block 0 → `RP[0] = 368 ≥ 0` → copies `t_own[92, :]`. Block 15 → `RP[15] = -137` → `(137-1)/4 = 34` → copies `t_peer[34, :]` over the link from the partner (5.8 KB). Block 3000 → `3000 ≥ 2058` → returns.

```
t_own  (512 rows)        Xg16 (expert order)
 row 92 ──────────────►  row 0   (expert 64)
 ...                     ...
t_peer (512 rows)
 row 34 ──────────────►  row 15  (expert 64)
```

##### ③ `gemm_grouped_mxfp4` for mlp1 — the first expert multiply (`fast_moe_gemm.hpp:55-81`)

| | name | shape, type |
|---|---|---|
| in | `Xg16` = `A` | [4096, 2880] bf16 (used rows only) |
| in | `O` = `offsets` | [65] int32 |
| in | `W1` = `blocks`, `scales` | per expert e: [5760, 90, 16] and [5760, 90] uint8 |
| in | `b1` = `bias` | [64, 5760] fp32 |
| out | `Yg` = `C` | [4096, 5760] fp32, rows `0 … O[64]-1` written |

```
Y[r, n] = Σ_{k=0..2879} X[r, k] · W1_e[n, k]  +  b1_e[n]
          r in O[e] … O[e+1]-1  (so e = the expert row r belongs to),  n = 0 … 5759

W1_e[n, k] = LUT[code] · 2^(S_e[n, k/32] − 127)
  code = low 4 bits (k even) or high 4 bits (k odd) of byte  blocks_e[n, k/32, (k%32)/2]
  LUT  = [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]
```

Decoding example: byte `0x3A`, scale 126. For an even `k`, code `A` = 10 → −1.0 × 2⁻¹ = **−0.5**. For the next (odd) `k`, code 3 → 1.5 × 2⁻¹ = **0.75**. (Checked in Python.) The decoding is done by `Mxfp4Rows` (`gemm.hpp:33-43`) while the weight is loaded into LDS.

What `Y` looks like: 5,760 numbers per row, **two kinds interleaved**. Column `2j` is the "gate" `j` and column `2j+1` is the "up" value `j`, for `j = 0 … 2879`. That is how the checkpoint orders `W1`'s rows, and SwiGLU reads them that way.

Split: grid **(90, 16, 64)** × 256 threads.

| grid axis | count | means |
|---|---|---|
| z | 64 | expert `e` (local) |
| y | 16 = (`B + B_peer` = 1,024) / 64 | which 64-row tile of that expert. 1,024 bounds any expert, because a token picks an expert at most once |
| x | 90 = 5,760 / 64 | which 64 output columns |
| thread | 256 | `tx = tid % 16` → 4 columns, `ty = tid / 16` → 4 rows: a 4 × 4 piece of the 64 × 64 tile |

An expert with 40 rows uses only tile `y = 0` (rows 0–39; rows 40–63 of the tile are filled with zeros and not written). The other 15 `y` tiles return at once (`m0 >= rows`, line 63). So of the 92,160 blocks, only ~64 × 90 = 5,760 do work. Each working block walks the 2,880 inputs 32 at a time (`GEMM_BK`): it loads 64 rows of `X` and 64 decoded rows of `W1` into LDS, waits (`__syncthreads`), and does 32 × 16 multiply-adds per thread.

The cost is mostly **reading the weights**: every expert's whole `W1` (8.8 MB) is read for its ~32 rows. More rows per expert would reuse the same weight read. That is question 3 of section 8.

##### ④ `swiglu_bf16_kernel` — the squashing step between the multiplies (`fast_misc.hpp:66-78`)

| | name | shape, type |
|---|---|---|
| in | `Yg` = `Y` | [4096, 5760] fp32 |
| out | `Gg16` = `G` | [4096, 2880] bf16 |

```
a = min(Y[r, 2j], 7)                 gate, cut at 7 from above
u = min(max(Y[r, 2j+1], -7), 7)      up, cut to -7 … 7
G[r, j] = a · sigmoid(1.702 · a) · (u + 1)          r = 0 … 4095,  j = 0 … 2879
```
`sigmoid(v) = 1 / (1 + e^(−v))`, a smooth step from 0 to 1. `a · sigmoid(1.702·a)` is close to `a` for large positive `a` and close to 0 for negative `a`, so the gate lets the up value through or blocks it.

Examples (checked in Python): gate 1.0, up 0.5 → 1 · 0.8458 · 1.5 = **1.2687**. Gate 10, up −9 → cut to 7 and −7 → 7 · 1.0000 · (−6) = **−42.0**. Gate −2, up 3 → −2 · 0.0322 · 4 = **−0.2574**.

Split: one thread per output number, grid `4096 · 2880 / 256` = **46,080 blocks** × 256. It runs over **all 4,096 rows**, including the ~2,048 unused ones past `O[64]`. Those hold leftovers and nobody reads them, so this is harmless but wasted work. The 20b version has no unused rows.

##### ⑤ `gemm_grouped_mxfp4` for mlp2 — the second expert multiply

The same kernel as ③, with other sizes:

| | name | shape, type |
|---|---|---|
| in | `Gg16` = `A` | [4096, 2880] bf16 |
| in | `O` | [65] |
| in | `W2` | per expert: [2880, 90, 16] and [2880, 90] uint8 (2,880 × 2,880) |
| in | `b2` | [64, 2880] fp32 |
| out | `Zg` = `C` | [4096, 2880] fp32 |

```
Z[r, n] = Σ_{k=0..2879} G[r, k] · W2_e[n, k] + b2_e[n]       n = 0 … 2879
```

Grid **(45, 16, 64)**: 45 = 2,880 / 64 column tiles. `Z` is the output of the whole phase.

#### C. One row from start to end (random case of 9.2, GCD B)

```
RP[0] = 368  → own token 92, pick 0, expert 64 (rows 0..39 = expert 64)
② Xg16[0, :]  = t16_B[92, :]                       2,880 bf16
③ Yg[0, :]    = W1_64 · Xg16[0] + b1_64            5,760 fp32 (gate, up, gate, up, …)
④ Gg16[0, :]  = SwiGLU(Yg[0])                      2,880 bf16
⑤ Zg[0, :]    = W2_64 · Gg16[0] + b2_64            2,880 fp32
combine on B, token 92: OR[368] = 0 → reads Zg[0], multiplies by topk_w[92, 0], adds into x[92]
```

For a partner row, such as row 15 (`RP = -137`, A's token 34): the same chain runs on B. Then **A's** combine reads `B.Zg[PR_B[136]] = B.Zg[15]` for its token 34.

#### D. The exact path (`GPTOSS_EXACT=1`)

Same five steps and same shapes. `Xg` and `Gg` are fp32 instead of bf16 (`moe_gather_ep_kernel<float>`, `swiglu_exact`), and `gemm_grouped_mxfp4_exact` adds each output's 2,880 products in fp64 in increasing `k` order, as the CPU does. This keeps it bit-identical to the CPU.

### 9.4 What the expert step does and what it outputs, on a toy

Question: the output table of 9.3 A (`Z`, `w->Zg`, `rows 0 … O[64]-1 used`, `(token, pick) RP[r]`) is hard to follow. What does this phase do, and what is its output? Every name defined before use, with its toy content.

In short: the router has already written, for every token, the numbers of the 4 experts that token chose (buffer `topk_e`). The expert step runs only the experts stored on this GCD, for every token on both GCDs that chose one of them. The results go into the buffer `Zg`: one row of 2,880 numbers for each (token, chosen expert) combination. A second buffer, `row_pair`, records which token and which of its 4 choices each `Zg` row belongs to. The step does not change `x`.

Everything below uses a toy that is small enough to write out completely. Every name is defined before it is used, with its toy content.

#### 1. The toy sizes

| | real 120b | toy |
|---|---|---|
| `H` = numbers in one token's row | 2,880 | **3** |
| experts per layer | 128 | **4** (experts 0, 1, 2, 3) |
| experts stored on GCD A / GCD B | 0–63 / 64–127 | **0–1 / 2–3** |
| `B` = tokens on one GCD (its batch) | 512 | **2** (token 0, token 1) |
| `K` = experts each token chooses | 4 | **2** |

A real expert computes GEMM → SwiGLU → GEMM (section 9.3). In the toy, **expert `e` multiplies every number by (e+1)·10**, so expert 0 = ×10, 1 = ×20, 2 = ×30, 3 = ×40. That lets you see from the result which expert made it.

#### 2. What already exists when the expert step starts

The earlier kernels of this layer (`layer_attention`, `layer_route`) have filled four buffers on each GCD. A *buffer* is an array in that GCD's memory.

**`x`**: the running value of each token, `B` rows of `H` numbers. The expert step reads nothing from it and does not write it. Only `layer_combine` adds to it.

**`t16`**: `x` after RMSNorm, `B` rows of `H` numbers. This is the input the experts multiply.

```
t16 on GCD A:  token 0: [1, 2, 3]      t16 on GCD B:  token 0: [7, 8, 9]
               token 1: [4, 5, 6]                     token 1: [2, 0, 1]
```

**`topk_e`**: for each token, the `K` expert numbers it chose, best first. Its shape is `B` rows × `K` columns, stored flat as `B·K` ints. Column `j` is **choice `j`** of that token.

```
topk_e on GCD A:              choice 0   choice 1
               token 0:           1          3
               token 1:           0          1
topk_e on GCD B:
               token 0:           2          0
               token 1:           1          2
```

So A's token 0 chose experts 1 and 3. B's token 1 chose experts 1 and 2.

**`topk_w`**: the same shape as `topk_e`. It holds the weight of each choice (from the router's softmax; the `K` weights of a token add up to 1). The expert step does not use it; `layer_combine` does. Toy values:

```
topk_w on GCD A:  token 0: [0.7, 0.3]   token 1: [0.6, 0.4]
topk_w on GCD B:  token 0: [0.5, 0.5]   token 1: [0.8, 0.2]
```

**Entry number `p`.** `topk_e` is stored flat, so each (token, choice) has one index into it: `p = token·K + choice`. With `K` = 2:

```
p        0              1              2              3
means    token 0 ch.0   token 0 ch.1   token 1 ch.0   token 1 ch.1
A: topk_e[p]   1          3              0              1
B: topk_e[p]   2          0              1              2
```

In the code, `p` is called a "pair" (`p = b * k + j`, `moe_ep.hpp:7`). Every `p` stands for one job: run expert `topk_e[p]` on the `t16` row of token `p / K`.

#### 3. The job of GCD A in this step

A stores experts 0 and 1 only. So A must do every job, on **both** GCDs, whose `topk_e[p]` is 0 or 1:

```
from A's topk_e:  p=0 (token 0, expert 1),  p=2 (token 1, expert 0),  p=3 (token 1, expert 1)
from B's topk_e:  p=1 (token 0, expert 0),  p=2 (token 1, expert 1)
```

That makes 5 jobs. A's p=1 (expert 3) and B's p=0, p=3 (expert 2) belong to GCD B, so A skips them.

#### 4. Kernel ① `moe_group_ep_kernel`: put the 5 jobs in order

The expert GEMM wants all rows of one expert next to each other, so the jobs are sorted by expert. Within one expert, A's own jobs come first, then B's, each in increasing `p`. The position of a job in this order is its **row number `r`**. The same `r` is used in `Xg16`, `Yg`, `Gg16` and `Zg`.

```
r     expert   job
0       0      A's p=2  (A token 1, choice 0)
1       0      B's p=1  (B token 0, choice 1)
2       1      A's p=0  (A token 0, choice 0)
3       1      A's p=3  (A token 1, choice 1)
4       1      B's p=2  (B token 1, choice 0)
```

The kernel writes this order into four buffers:

**`offsets`** (`E_loc + 1` ints; `E_loc` = experts on this GCD = 2): the first row of each expert, and at the end the total row count.
`offsets = [0, 2, 5]` means expert 0 is rows 0–1, expert 1 is rows 2–4, and 5 rows are used.

**`row_pair`** (one int per row): the "job" column of the table, as a number. A's own job `p` is written as `p`. B's job `p` is written as `-(p+1)`. The minus sign means "this `p` is an index into B's `topk_e`, not A's". The `+1` is needed because B's p=0 written as `-0` would look the same as A's p=0.

```
r          0     1     2     3     4
row_pair   2    -2     0     3    -3
```
To read it back: `row_pair[1] = -2` → negative, so B's job; `p = -(-2) - 1 = 1` → B token 1/2 = 0, choice 1%2 = 1. ✓

**`own_row`** (one int per A's `p`, so 4): the row where A's job `p` sits on A, or −1 if its expert is on B.
`own_row = [2, -1, 0, 3]`. A's p=0 is row 2. A's p=1 (expert 3) is −1, not here. A's p=2 is row 0. A's p=3 is row 3.

**`peer_row`** (one int per B's `p`, so 4): the row where B's job `p` sits **on A**, or −1.
`peer_row = [-1, 1, 4, -1]`. B's p=1 is row 1 on A; B's p=2 is row 4 on A.

`own_row` and `peer_row` are the reverse of `row_pair`: `row_pair` goes row → job, and these two go job → row. `layer_combine` needs that direction (part 7).

#### 5. Kernel ② `moe_gather_ep_kernel`: copy each row's input

**`Xg16`** (one row of `H` numbers per row `r`): the `t16` row of the job's token. A's jobs copy from A's `t16`; B's jobs copy from B's `t16`, read directly from B's memory (peer access).

```
r   row_pair   source            Xg16[r]
0      2       A t16, token 1    [4, 5, 6]
1     -2       B t16, token 0    [7, 8, 9]
2      0       A t16, token 0    [1, 2, 3]
3      3       A t16, token 1    [4, 5, 6]
4     -3       B t16, token 1    [2, 0, 1]
```

A's token 1 appears twice (rows 0 and 3) because it chose two of A's experts.

#### 6. Kernels ③ ④ ⑤: run the expert on each row → `Zg`

For rows `offsets[e] … offsets[e+1]-1`, expert `e` is applied to `Xg16[r]`. In the real model that is GEMM `mlp1` → SwiGLU → GEMM `mlp2` (9.3). In the toy it is ×(e+1)·10.

**`Zg`** (one row of `H` numbers per row `r`): the output of the expert for that job.

```
r   expert   Xg16[r]      Zg[r] on A
0     0      [4, 5, 6]    × 10 = [ 40,  50,  60]
1     0      [7, 8, 9]    × 10 = [ 70,  80,  90]
2     1      [1, 2, 3]    × 20 = [ 20,  40,  60]
3     1      [4, 5, 6]    × 20 = [ 80, 100, 120]
4     1      [2, 0, 1]    × 20 = [ 40,   0,  20]
```

**This is the output of the expert step on A:** `Zg` with 5 rows, together with `offsets`, `row_pair`, `own_row`, `peer_row`.

At the same time GCD B does the same for experts 2 and 3:

```
r   expert   job                          row_pair   Zg[r] on B
0     2      B's p=0 (B token 0, choice 0)     0      [7,8,9] × 30 = [210, 240, 270]
1     2      B's p=3 (B token 1, choice 1)     3      [2,0,1] × 30 = [ 60,   0,  30]
2     3      A's p=1 (A token 0, choice 1)    -2      [1,2,3] × 40 = [ 40,  80, 120]

B: offsets = [0, 2, 3]   own_row = [0, -1, -1, 1]   peer_row = [-1, 2, -1, -1]
```

Check: A did 5 jobs and B did 3, so all 8 `p` values (4 on each GCD) are done exactly once.

#### 7. What `layer_combine` does with this output (so you see why it has this shape)

For each of its own tokens, a GCD adds `Σ_j topk_w[token, j] · (Zg row of choice j)` into `x[token]`. To find the `Zg` row of A's job `p`:
- if `topk_e[p]` is on A → A's `Zg[own_row[p]]`;
- if not → B's `Zg[peer_row_of_B[p]]`, read from B's memory. (B's `peer_row` holds the rows of A's jobs on B.)

A's token 0 (p = 0 and p = 1):
```
p=0: expert 1, on A → own_row[0] = 2        → A's Zg[2] = [20, 40, 60],   weight 0.7
p=1: expert 3, on B → B's peer_row[1] = 2   → B's Zg[2] = [40, 80, 120],  weight 0.3
x[A token 0] += 0.7·[20,40,60] + 0.3·[40,80,120] = [26, 52, 78]
```

A's token 1 (p = 2 and p = 3), both on A:
```
p=2: own_row[2] = 0 → A's Zg[0] = [40, 50, 60],   weight 0.6
p=3: own_row[3] = 3 → A's Zg[3] = [80, 100, 120], weight 0.4
x[A token 1] += 0.6·[40,50,60] + 0.4·[80,100,120] = [56, 70, 84]
```

#### 8. The same at real size (GCD B of 120b)

| buffer | toy (GCD A) | real (GCD B) |
|---|---|---|
| `t16` | 2 rows × 3 | 512 rows × 2,880 bf16 |
| `topk_e`, `topk_w` | 2 × 2 | 512 × 4 (`p` = 0 … 2,047) |
| jobs looked at | 4 own + 4 partner | 2,048 own + 2,048 partner |
| jobs kept (= rows used) | 5 | ~2,048 (those with expert 64–127) |
| `offsets` | 3 ints | 65 ints (64 experts + total) |
| `row_pair` | 5 used | room for 4,096, ~2,048 used |
| `own_row`, `peer_row` | 4 each | 2,048 each |
| `Xg16` | 5 × 3 | room 4,096 × 2,880 bf16 |
| `Zg` | 5 × 3 | room 4,096 × 2,880 fp32 |

Room for 4,096 rows is reserved because, in the worst case, all 1,024 tokens × 4 choices go to this GCD. The rows past `offsets[64]` are not written in this layer and hold old values that nobody reads.

Now the table row you asked about, in these words: `Zg` (`w->Zg` = the `Zg` buffer of this GCD; `w` is the variable for this GCD's `GpuWorker`) is `[4096, 2880] fp32`. Rows `0 … offsets[64]-1` are used. Row `r` holds the expert output for the job `row_pair[r]`, and the rows are sorted by expert.

### 9.5 What `Xg16` is, and what the expert really computes (no stand-in)

Question: in 9.4 part 5, what is `Xg16`, why does "A t16, token 1" give `[4, 5, 6]`, and why is `Zg` ×10, ×20?

In short: `[4, 5, 6]` is not computed. It is the toy value I chose in part 2 for A's token 1 in `t16` (9.4 part 2), and row 0 of `Xg16` is simply a copy of it. `Xg16` is a copy of `t16` rows rearranged into job order, which is the order the expert GEMM needs. "×10, ×20" was a stand-in I made up for the expert's real computation, and 9.4 did not say that clearly enough. Below is the real computation on a tiny expert, so nothing is made up.

#### 1. Where `[4, 5, 6]` comes from

9.4 part 2 picked toy numbers for `t16` (in the real model each row has 2,880 numbers; the toy uses 3):

```
t16 on GCD A:  row 0 = token 0: [1, 2, 3]
               row 1 = token 1: [4, 5, 6]
t16 on GCD B:  row 0 = token 0: [7, 8, 9]
               row 1 = token 1: [2, 0, 1]
```

Row `r = 0` of the job table is "A's p = 2", and `p = 2` means token `2 / K = 2 / 2 = 1`, choice `2 % 2 = 0`. Its job is "run expert 0 on A's token 1". The input for that job is A's `t16` row 1 = `[4, 5, 6]`. So `Xg16[0] = [4, 5, 6]` is a plain copy.

Each row of the table follows the same rule:

```
r   row_pair   decode row_pair                         copy from          Xg16[r]
0      2       ≥ 0 → A's p=2 → token 2/2 = 1           A's t16 row 1      [4, 5, 6]
1     -2       < 0 → B's p=-(-2)-1 = 1 → token 1/2 = 0 B's t16 row 0      [7, 8, 9]
2      0       ≥ 0 → A's p=0 → token 0/2 = 0           A's t16 row 0      [1, 2, 3]
3      3       ≥ 0 → A's p=3 → token 3/2 = 1           A's t16 row 1      [4, 5, 6]
4     -3       < 0 → B's p=-(-3)-1 = 2 → token 2/2 = 1 B's t16 row 1      [2, 0, 1]
```

This is exactly what the kernel does (`moe_ep.hpp:87-91`):

```cpp
const int p = row_pair[row];
const T *src = p >= 0 ? x_own + (size_t)(p / k) * H            // own:     t16 row p/k on this GCD
                      : x_peer + (size_t)((-p - 1) / k) * H;   // partner: t16 row (-p-1)/k on the partner
for (int i = threadIdx.x; i < H; i += blockDim.x)
  Xg[(size_t)row * H + i] = src[i];                            // copy the H numbers
```

`x_own` is this GCD's `t16`, and `x_peer` is the partner's `t16` (`gpu_engine.hpp:396`).

#### 2. What `Xg16` is and why the copy is needed

`Xg16` is the **input buffer of the expert GEMM**. The name means X (input) + g (grouped by expert) + 16 (bf16, 16-bit numbers).

```
copied from                      →  Xg16 on A
A's t16 row 1 (token 1) [4,5,6]  →  row 0   ┐ expert 0
B's t16 row 0 (token 0) [7,8,9]  →  row 1   ┘
A's t16 row 0 (token 0) [1,2,3]  →  row 2   ┐
A's t16 row 1 (token 1) [4,5,6]  →  row 3   │ expert 1
B's t16 row 1 (token 1) [2,0,1]  →  row 4   ┘
```

`t16` cannot be used directly by the GEMM, for three reasons:
1. **Order.** The grouped GEMM takes expert `e`'s input as one unbroken block of rows, `offsets[e] … offsets[e+1]-1`. In `t16` the rows are in token order, and the tokens of one expert are scattered.
2. **Duplicates.** A's token 1 chose two of A's experts (0 and 1), so its row is needed twice (rows 0 and 3).
3. **The partner's tokens.** Rows 1 and 4 are B's tokens. Their `t16` rows are in B's memory, so they are copied over.

#### 3. What the expert really computes on one row

The real expert `e` is three kernels on each of its rows (`gpu_engine.hpp:398-403`):

```
Yg[r]  = W1_e · Xg16[r] + b1_e        GEMM: W1_e has 5,760 rows × 2,880 columns → 5,760 numbers
Gg16[r] = SwiGLU(Yg[r])               5,760 numbers → 2,880 (each gate/up pair becomes one number)
Zg[r]  = W2_e · Gg16[r] + b2_e        GEMM: W2_e has 2,880 × 2,880 → 2,880 numbers
```

SwiGLU of one (gate, up) pair: `a = min(gate, 7)`, `u = clamp(up, −7, 7)`, `result = a · sigmoid(1.702·a) · (u + 1)`, where `sigmoid(v) = 1/(1+e^(−v))`.

**Tiny real expert.** Use the same row width 3, and make the inner width 1 (one gate/up pair instead of 2,880). Then W1 has 2 rows × 3 columns and W2 has 3 rows × 1 column. Weights I chose for expert 0 (biases 0):

```
W1_0 = [[0.1, 0,   0],      ← row 0 makes the gate
        [0,   0.1, 0]]      ← row 1 makes the up value
W2_0 = [[1], [2], [3]]
```

Row r = 0, input `Xg16[0] = [4, 5, 6]`:

```
Yg[0]   = W1_0 · [4,5,6] = [0.1·4 + 0·5 + 0·6,  0·4 + 0.1·5 + 0·6] = [0.4, 0.5]     (gate, up)
Gg16[0] = 0.4 · sigmoid(1.702·0.4) · (0.5 + 1) = 0.4 · 0.6639 · 1.5 = 0.3984
Zg[0]   = W2_0 · [0.3984] = [1·0.3984, 2·0.3984, 3·0.3984] = [0.398, 0.797, 1.195]
```

(Checked in Python.) Note: `Zg[0]` again has row width 3, the same as `x`. That is why `layer_combine` can add it into `x`.

Every other row works the same way, with the weights of its own expert: rows 0–1 use `W1_0`, `W2_0`, and rows 2–4 use `W1_1`, `W2_1`.

#### 4. Why 9.4 wrote ×10, ×20

The weights of one real expert are about 25 million numbers (W1: 5,760 × 2,880, W2: 2,880 × 2,880), and the result cannot be checked by eye. To show **where the data goes** (which row, which GCD, which token), 9.4 replaced the whole three-kernel chain with "expert `e` multiplies by (e+1)·10". That is not what gpt-oss does. It was only a label, so that in the combine step you could see that `[20, 40, 60]` came from expert 1 and `[40, 80, 120]` from expert 3.

A cleaner way to write part 6 of 9.4, without the stand-in:

```
r   expert   Xg16[r]     Zg[r] on A
0     0      [4,5,6]     expert_0([4,5,6])  = W2_0 · SwiGLU(W1_0 · [4,5,6] + b1_0) + b2_0
1     0      [7,8,9]     expert_0([7,8,9])
2     1      [1,2,3]     expert_1([1,2,3])
3     1      [4,5,6]     expert_1([4,5,6])
4     1      [2,0,1]     expert_1([2,0,1])
```

#### 5. Row 0 from start to end

```
A's topk_e[2] = 0              A's token 1, choice 0, chose expert 0
group:   row_pair[0] = 2       job p=2 is placed at row 0 (first row of expert 0)
gather:  Xg16[0] = A's t16[1]  = [4, 5, 6]                     (copy)
mlp1:    Yg[0]   = W1_0 · Xg16[0] + b1_0                       (GEMM)
swiglu:  Gg16[0] = SwiGLU(Yg[0])
mlp2:    Zg[0]   = W2_0 · Gg16[0] + b2_0                       (GEMM)
combine: own_row[2] = 0 → x[A token 1] += topk_w[A token 1, choice 0] · Zg[0]
```

### 9.6 `layer_combine`: collecting a token's 4 expert rows from both GCDs

Question: in `layer_combine`, a GCD must get the 4 `Zg` rows of each of its tokens, some from itself and some from the other GCD. How does that work?

In short: yes, the picture in the question is right. For each of its own tokens, a GCD needs 4 `Zg` rows, one per chosen expert. Each row is either in its own `Zg` or in the partner's `Zg`. The GCD takes the rows from its own memory or from the partner's memory, multiplies each by its weight from `topk_w`, and adds the sum into `x[token]`. One correction: there is no "call" to the other GPU. The kernel holds a pointer into the partner's memory and reads from it like ordinary memory. The partner does nothing during the read; it only had to finish writing `Zg` first, and an event guarantees that.

#### 1. Input and output of `layer_combine`

`layer_combine` (`gpu_engine.hpp:408-423`) launches one kernel, `moe_combine_ep_kernel`. Shapes are for 120b on GCD A (experts 0–63), `B` = 512 own tokens, `K` = 4, `H` = 2,880.

| kernel argument | buffer | shape, type | where | content |
|---|---|---|---|---|
| `x` (in and out) | `w->x` | [512, 2880] fp32 | own | each own token's running value; the kernel adds into it |
| `Z_own` | `w->Zg` | [4096, 2880] fp32 (~2,048 rows used) | own | outputs of A's experts, from the expert step |
| `own_row` | `w->own_row` | [2048] int32 | own | for A's entry `p`: its row in A's `Zg`, or −1 |
| `Z_peer` | `p->Zg` | [4096, 2880] fp32 | **partner's memory** | outputs of B's experts |
| `peer_rows_of_mine` | `p->peer_row` | [2048] int32 | **partner's memory** | for A's entry `p`: its row in **B's** `Zg`, or −1 (B wrote it in its group kernel) |
| `topk_w` | `w->topk_w` | [512, 4] fp32 | own | weight of each choice |
| `topk_e` | `w->topk_e` | [512, 4] int32 | own | expert number of each choice |
| `e_begin`, `e_count` | | numbers | | 0 and 64 on A: which experts are "here" |

Output: `x` is updated in place. Nothing else is written.

```
x[b, :] += Σ_{j=0..3} topk_w[b, j] · R_j          b = 0 … 511
  where p = b·4 + j  and
  R_j = A.Zg[ own_row[p] ]           if e_begin ≤ topk_e[p] < e_begin + e_count   (expert here)
  R_j = B.Zg[ B.peer_row[p] ]        otherwise                                     (expert on partner)
```

For 20b there is no partner: `Z_peer` and `peer_rows_of_mine` are `nullptr`, and every expert is "here", so the second case never happens.

#### 2. A real-size example: the 4 rows of one token

Using the random choices from 9.2 and both GCDs' grouping run in Python: **GCD B's token 0** chose experts `[98, 52, 75, 24]`, so its entries are `p = 0, 1, 2, 3`. B holds 64–127:

```
choice j  p  expert  where      lookup                     row      check: that row's row_pair
   0      0    98    on B       B.own_row[0]   = 1097      B.Zg[1097]   B.row_pair[1097] = 0     (own p=0)   ✓
   1      1    52    on A       A.peer_row[1]  = 1655      A.Zg[1655]   A.row_pair[1655] = -2    (partner p=1) ✓
   2      2    75    on B       B.own_row[2]   = 369       B.Zg[369]    B.row_pair[369]  = 2     (own p=2)   ✓
   3      3    24    on A       A.peer_row[3]  = 775       A.Zg[775]    A.row_pair[775]  = -4    (partner p=3) ✓

x[B token 0] += topk_w[0,0]·B.Zg[1097] + topk_w[0,1]·A.Zg[1655] + topk_w[0,2]·B.Zg[369] + topk_w[0,3]·A.Zg[775]
```

Two rows come from B's own memory and two from A's memory. On average it is 2 + 2, because each GCD holds half of the experts. A single token can also have all 4 here, or all 4 on the partner.

`A.peer_row` is in A's memory, and B reads it too, because only A knows where it put B's entries.

#### 3. The kernel, line by line (`moe_ep.hpp:100-141`)

Launch: `<<<B, 256>>>` = 512 blocks × 256 threads (`gpu_engine.hpp:420-422`). **Block `b` = own token `b`.** Following block `b = 0` on GCD B (the token above).

**Lines 109–121: the order of the 4 choices.**
```cpp
const int b = blockIdx.x;                 // b = 0
int order[MOE_MAX_TOPK];
for (int j = 0; j < k; j++) order[j] = j; // order = [0, 1, 2, 3]
if (kExact) ... sort order by topk_e      // exact path only: [3, 1, 2, 0] (experts 24, 52, 75, 98)
```
On the fast path (`kExact = false`), `order` stays `[0, 1, 2, 3]`: choice order. The exact path sorts by expert number because the CPU adds the experts in that order, and a different order would round differently.

**Lines 122–128: find the 4 rows.** Every thread of the block computes the same 4 pointers.
```cpp
for (int j = 0; j < k; j++) {
  const int p = b * k + order[j];         // p = 0, 1, 2, 3
  const int e = topk_e[p] - e_begin;      // 98-64=34, 52-64=-12, 75-64=11, 24-64=-40
  rows[j] = (e >= 0 && e < e_count)       // here?  yes, no, yes, no
      ? Z_own  + own_row[p] * H                     // j=0: B.Zg + 1097·2880
      : Z_peer + peer_rows_of_mine[p] * H;          // j=1: A.Zg + 1655·2880
}                                                   // j=2: B.Zg + 369·2880,  j=3: A.Zg + 775·2880
```
`rows[j]` is a pointer to the first number of a 2,880-number row. Two of the pointers point into A's memory.

**Lines 129–140: the weighted sum.** The 2,880 columns are shared out: thread `tid` does columns `tid, tid+256, tid+512, …` (2,880 / 256 = 11.25, so threads 0–63 do 12 columns and threads 64–255 do 11).
```cpp
for (int i = threadIdx.x; i < H; i += blockDim.x) {      // thread 5: i = 5, 261, 517, …, 2821 (12 columns)
  float acc = 0.0f;
  for (int j = 0; j < k; j++)
    acc = fmaf(rows[j][i], topk_w[b * k + j], acc);       // acc += Zrow_j[i] · w_j
  x[b * H + i] += acc;
}
```
For thread 5, column 5:
```
acc = B.Zg[1097][5]·w0 + A.Zg[1655][5]·w1 + B.Zg[369][5]·w2 + A.Zg[775][5]·w3
x[0][5] += acc
```
`A.Zg[1655][5]` and `A.Zg[775][5]` are plain loads through a pointer. Their addresses are in A's memory, so the hardware sends the load over the link between the two GCDs of the card, and the value comes back. No code runs on A for this.

Threads never share data here: each thread owns its columns and writes `x[b][i]` alone. So there is no LDS and no barrier.

#### 4. Why reading the partner's memory works and is safe

**It works** because `gpu_pair` (`gpu_engine.hpp:302-312`) called `hipDeviceEnablePeerAccess` both ways at start-up. After that, a kernel on B may load from an address that `hipMalloc` gave out on A. `Z_peer = p->Zg` (`gpu_engine.hpp:413`) is exactly such an address.

**It is safe** because of the wait right before the launch (`gpu_engine.hpp:473-479`):
```cpp
hipStreamWaitEvent(w->stream, w->peer->ev_experts, 0);   // B's stream waits for A's ev_experts
layer_combine(w, l, B[i]);
```
A records `ev_experts` right after its last expert kernel (mlp2, which writes `A.Zg`). So B's combine starts only when `A.Zg` and `A.peer_row` are complete for this layer.

#### 5. The toy from 9.4, all four tokens

Same toy as 9.4 (K = 2, A holds experts 0–1, B holds 2–3). `A.Zg[r]` means row `r` of A's `Zg` (its content is `expert_e(Xg16[r])`, 9.5).

```
GCD A, token 0  (p=0: expert 1, p=1: expert 3)
  p=0  here      → A.own_row[0] = 2    → A.Zg[2]
  p=1  partner   → B.peer_row[1] = 2   → B.Zg[2]       (read from B)
  x[A0] += 0.7·A.Zg[2] + 0.3·B.Zg[2]

GCD A, token 1  (p=2: expert 0, p=3: expert 1)
  p=2  here      → A.own_row[2] = 0    → A.Zg[0]
  p=3  here      → A.own_row[3] = 3    → A.Zg[3]
  x[A1] += 0.6·A.Zg[0] + 0.4·A.Zg[3]                    (nothing read from B)

GCD B, token 0  (p=0: expert 2, p=1: expert 0)
  p=0  here      → B.own_row[0] = 0    → B.Zg[0]
  p=1  partner   → A.peer_row[1] = 1   → A.Zg[1]       (read from A)
  x[B0] += 0.5·B.Zg[0] + 0.5·A.Zg[1]

GCD B, token 1  (p=2: expert 1, p=3: expert 2)
  p=2  partner   → A.peer_row[2] = 4   → A.Zg[4]       (read from A)
  p=3  here      → B.own_row[3] = 1    → B.Zg[1]
  x[B1] += 0.8·A.Zg[4] + 0.2·B.Zg[1]
```

Every `Zg` row of both GCDs (A: 5 rows, B: 3 rows) is used exactly once. Each row is read by the GCD that owns the token, which is the only GCD that has that token's `x`.

#### 6. How much is read from the partner

Per layer, per GCD: about half of its 2,048 entries are on the partner, so about 1,024 rows × 2,880 × 4 bytes ≈ **11.8 MB** of `Zg`, plus 4 bytes of `peer_row` per entry. This is the "≈ 11.8 MB" in section 6 of the lesson. (The other ≈ 5.9 MB is the gather reading the partner's `t16` before the expert GEMM.)
