---
marp: true
paginate: true
title: Making gpt-oss fast on AMD MI250
---

# Making a language model answer fast on AMD GPUs

## gpt-oss on one machine with 8 AMD MI250 GPU dies, every GPU program written by hand

| | Before | After |
|---|---:|---:|
| gpt-oss-20b (smaller model) | 3.3 tokens/s | **80,000 tokens/s** |
| gpt-oss-120b (bigger model) | 0.40 tokens/s | **45,500 tokens/s** |

- **Token:** a piece of a word, about ¾ of an English word. Speed = tokens written per second.
- The answers still pass both quality checks.
- **The talk:** basics (pages 2–5) → task and results (6–10) → the techniques, one page each (11–26) → what limits it now, and surprises (27–29).

<!-- One sentence: I took a one-file C++ program that runs the model on the CPU and turned it into
a batch server on 8 GPU dies, writing every GPU program myself and checking correctness at every step.
20b is about 24,000 times faster. -->

---

# Basics (1/4): how a language model writes text

**In plain words:** the model reads the question, then writes the answer one token at a time. Each new token is added to the text, and the model runs again to get the next one.

```
"Explain rain to a 5-year-old"              → [ model ] → "Rain"
"Explain rain to a 5-year-old Rain"         → [ model ] → "is"
"Explain rain to a 5-year-old Rain is"      → [ model ] → "water"   … up to 1,024 tokens
```

- **Weights:** the numbers the model learned. 20b has about 21 billion, 120b about 117 billion.
- **Layers:** the model is a stack of 24 (20b) or 36 (120b) layers. Each layer changes the token's list of 2,880 numbers a little.
- At the end, the model scores all **201,088** possible tokens and takes the top one. Always taking the top one is called **greedy** picking.
- **Prefill / decode:** reading the question / writing the answer one token at a time.

---

# Basics (2/4): inside one layer

```
the token's 2,880 numbers
   │
   ├─ attention   compare this token with all earlier tokens of the same request,
   │              and mix in what they left behind (the saved history, page 13)
   │
   ├─ router      score the 32 experts, keep the best 4
   │
   └─ experts     each of the 4 chosen experts reworks the numbers; the 4 results are mixed
   ▼
next layer
```

- **Attention** is how the model uses context. For example, "it" looks back to find what "it" refers to. 64 **heads** do this side by side, each looking for something different.
- **Experts (mixture of experts, MoE):** 32 (20b) or 128 (120b) small sub-networks. Each token uses only 4, so the model can be big while each token stays cheap.
- **Matrix multiply:** almost all of this work is "rows of token numbers × a big table of weights". That is what the GPU must do fast.

---

# Basics (3/4): the GPU in plain words

```
one GCD (GPU die)
┌──────────────────────────────────────────────────────────────────┐
│ 104 compute units, each with: 4 groups of calculators,           │
│                               matrix cores, 64 KB fast scratchpad │
│ 8 MB shared cache                                                │
└──────────── 64 GB main memory, read at ~1.25 TB/s ───────────────┘
```

- **GCD (graphics compute die):** one GPU die. An MI250 card has two. The machine has 4 cards, so 8 GCDs.
- **Kernel:** a small program that runs on the GPU. One step of the model is a few hundred kernel runs.
- **Wavefront:** the GPU runs threads in groups of 64 that do the same instruction on different numbers.
- **Matrix cores:** circuits that multiply a 16 × 16 block of numbers in one instruction. They do **181 trillion** operations per second on 16-bit numbers; the ordinary calculators do 22.6 trillion on 32-bit numbers, 8× less.
- **Links:** the two GCDs of a card talk at 177 GB/s; other pairs at 90 or 45 GB/s.

---

# Basics (4/4): reading is slow, math is fast

**In plain words:** the GPU spends most of its time fetching the model's numbers, not doing math. So one fetch should serve many requests.

```
one request per step                    512 requests per step
fetch weights  ████████████████         fetch weights  ████████████████
math           ▏                        math           ████████████████████
→ the calculators wait almost all       → the calculators have enough work
  the time
```

- One GCD can do about **145 math operations** in the time it reads **1 byte**.
- Writing one token for one request needs only **1–4 operations per byte** read.
- **Memory-bound / compute-bound:** a kernel whose time is set by reading / by math. Every technique in this talk either reads fewer bytes, or keeps the calculators busier.

<!-- 181 TFLOP/s bf16 peak / 1.25 TB/s measured = 145 FLOP per byte (the roofline "ridge point").
One request: 2 FLOP per weight; bf16 = 2 bytes -> 1 FLOP/B; MXFP4 = 0.53 bytes -> 3.8 FLOP/B. -->

---

# The task and the rules

- **Given:** a simple one-file C++ program that runs gpt-oss on the CPU (the computer's main processor).
- **Build:** a batch mode that reads **4,096 questions** and writes all the answers. Score = tokens per second. Only the answering part is timed.
- **Rules**
  - Every GPU program is written by hand. Ready-made GPU math libraries may be used only to compare against, never inside the program.
  - Only our own source file, and the files it includes, may change.
  - One machine: 8 GCDs.
  - The answers must stay good. Two automatic scores compare them with reference answers:
    **METEOR** (how many words match) ≥ 0.30 and **BERTScore** (how close the meaning is) ≥ 0.90.

<!-- Not allowed: rocBLAS, hipBLASLt, RCCL, CK. Fixed files: run.cpp, getp-csrc/getp_eval.cpp, Makefile.
Our entry point: getp-csrc/getp_run.cpp. Weights: the Hugging Face checkpoints. -->

---

# What the questions and the machine tell us

| Fact | What it means for the design |
|---|---|
| Questions are short (~20 tokens). **95 %** of answers run to the full 1,024-token limit | almost all the time is decode. Everyone finishes together, so a fixed batch is fine (page 12) |
| All 4,096 questions fit in memory at once: 512 per GCD | nobody waits in a queue |
| The expert weights come compressed: **MXFP4**, 4 bits per number plus one shared scale for every 32 numbers | memory carries almost 4× less than with 16-bit numbers, but we must unpack them ourselves (page 16) |
| 120b is **65 GB**, one GCD has **64 GB** | 120b must be split over 2 GCDs (page 23) |
| The two GCDs of a card share a **560 W** power limit | under full load the clock drops to 1.4–1.5 GHz instead of 1.7, so saving energy saves time |

---

# Results

|  | gpt-oss-20b | gpt-oss-120b |
|---|---:|---:|
| Given CPU program | 3.3 tok/s | 0.40 tok/s |
| First GPU version (same math as the CPU, bit for bit) | 3,348 | 1,188 |
| **Now** (official questions, 8 GCDs) | **80,000** | **45,500** |
| Word match / meaning score (must be ≥ 0.30 / ≥ 0.90) | 0.533 / 0.978 | 0.552 / 0.981 |

- **Yardstick:** vLLM (a widely used open-source program that serves a language model to many users at once, built on AMD's tuned libraries) writes 20b at **6,539 tok/s on one GCD**. Ours does ≈ 10,000 per GCD, about **1.5×** faster.
- **More questions** (the official set twice, so 1,024 per GCD): 20b reaches **84,268 tok/s**.

<!-- vLLM was measured on 1 GCD only; its 8-GCD and 120b numbers were not measured. -->

---

# The journey (20b): which technique gave what

| Step | Technique (page) | tok/s |
|---|---|---:|
| 0 | the CPU program | 3.3 |
| 1 | move to the GPU, same math bit for bit; **batching** 512 requests per GCD (12); **grouping by expert** (19); **data parallel** on 8 GCDs (23) | 3,348 |
| 2 | **mixed precision:** 16-bit inputs, 32-bit sums (16) | 5,217 |
| 3 | attention: **grouped-query sharing, online softmax, flash-decoding** (14–15); cache in 64-position blocks (22) | 5,741 |
| 4 | plain multiplies in **tiles on the matrix cores** (18) | 7,858 |
| 5 | **the expert multiply on the matrix cores, straight from the 4-bit weights** (20) | **64,099 (×8.2)** |
| 6 | **kernel fusion** (21), **weights in reading order** (22), faster router | 72,258 |
| 7 | skip finished requests (12), 8-bit values in the cache (16), next-token pick fused (21), stream-K (18) | 76,149 |
| 8 | faster write-out of results, faster small kernels (20, 21) | 80,000 |

- The log has 36 steps for both models. Each has before → after, the reason and a commit. Failed ideas are logged too.

---

# How I worked

1. **Copy exactly first.** The first GPU version gives bit-for-bit the CPU's answers, so any difference is a bug, not rounding.
   - This found a GPU square-root function that was wrong for 7.5 % of its inputs.
2. **Then make it fast** with smaller numbers, judged by the two quality scores.
3. **Prefer changes that keep the answers identical.** A fingerprint (hash) of all 4,096 answers proves "same output".
4. **Measure before choosing.** A profile (a record of how long every kernel takes) names the biggest cost. The decision is written down with those numbers.
5. **Compare fairly.** The old and new versions run in turns, twice each, on all 8 GCDs, because single runs differ by about 1 %.
6. **Write everything down.** One accepted step = one log row (before → after, reason) + one commit. The repo also has 12 beginner-level docs, one per technique.

<!-- Gates: G1 kernel unit tests, G2 output hash of the full run, G3 METEOR/BERTScore. The square-root bug:
HIP __fsqrt_rn, found by a 1-layer bisection against the CPU. -->

---

# Map: each problem and the technique that answers it

| Problem | Technique | Here? | Page |
|---|---|---|---:|
| one fetch of the weights serves one request | batching | ✓ | 12 |
| finished requests leave empty seats | continuous batching | ✗ (cheap part ✓) | 12 |
| redoing the past at every step | KV cache | ✓ | 13 |
| cache memory reserved but not used | PagedAttention | ✗ | 13 |
| 8 heads read the same history | grouped-query sharing | ✓ | 14 |
| scores must all be known before mixing | online softmax (FlashAttention) | ✓ | 14 |
| one long history for one worker | flash-decoding | ✓ | 15 |
| big numbers are slow to read and to multiply | mixed precision, compressed weights and cache | ✓ | 16–17 |
| multiplying one number at a time | tiles on the matrix cores, stream-K | ✓ | 18 |
| each token's experts are scattered | grouping by expert, grouped multiply | ✓ | 19–20 |
| many small steps, each a memory round trip | kernel fusion | ✓ | 21 |
| reads that jump around in memory | data in reading order | ✓ | 22 |
| one GPU is too small, or not enough | data / expert parallel (tensor / pipeline ✗) | ✓ | 23–24 |
| reading the prompt one token per step | batched prefill | ✗ | 25 |
| the CPU between steps; guessing ahead | GPU-side loop; speculative decoding | ✗ | 26 |

---

# Batching: one trip, many passengers

**Problem:** with one request per pass, each fetched weight is used once, and the calculators wait almost all the time (page 5).

**Idea:** a bus instead of taxis. Each pass makes the next token for many requests, so each weight is fetched once and used by all of them.

```
static batching (ours)                       continuous batching (vLLM)
seat 1  ■■■■■■■■■■■■                         seat 1  ■■■■■■■■■■■■
seat 2  ■■■■■■□□□□□□  finished, seat empty   seat 2  ■■■■■■▲▲▲▲▲▲  a waiting request takes the seat
seat 3  ■■■■■■■■■■■■                         seat 3  ■■■■■■■■■■■■
```

- **Static batching (used):** 512 requests per GCD start together and move one token per step (lockstep). Measured: 8× more requests (64 → 512) cost only 2.5× more time per step.
- **Continuous batching (not used):** when a request finishes, a waiting one takes its seat at once. It is the right choice for a live chat service, where answer lengths vary and people keep arriving.
- **Why not here:** all 4,096 questions are in the first batch, so nobody is waiting for a seat, and 95 % of answers run to the full length anyway. Empty seats are only **3.1 %** of the work.
- **The cheap part we took:** empty seats skip the experts: **+0.8 %** (20b), **+1.2 %** (120b).

---

# KV cache: keep notes instead of re-reading

**Problem:** attention compares each new token with all earlier tokens. Recomputing what every earlier token offers, at every step, repeats work that grows with the answer: about 500 times on average over a 1,024-token answer.

**Idea:** like taking notes on each page as you read, instead of re-reading the book for every new sentence. Each token's **key** ("what I contain") and **value** ("what I hand over") are computed once and saved: the **KV cache**.

- **Used from the start.** The cost is memory: for 512 requests per GCD, 20b needs ~11 GB, 120b ~16 GB. Even-numbered layers keep only the last 128 tokens (a sliding window).
- **PagedAttention (vLLM, not used):** hands out cache memory in small pages as an answer grows, like a hotel giving rooms on demand instead of booking a whole floor per guest. It saves memory when lengths vary.
- **Why not here:** almost every answer reaches 1,024 tokens, so reserving the full length wastes nothing, and everything fits.
- **What we borrowed:** fixed-size blocks (64 positions of one head in one 8 KB piece), because they read faster: **+0.8 %** (page 22).

---

# Attention (1/2): read the history once

**Problem:** at the last position, one layer must read **1.07 GB** of saved history for 512 requests: 0.86 ms at full memory speed, the floor. The first version read every saved item **8 times** and took 9.5 ms.

**Grouped-query sharing (GQA):** gpt-oss has 64 query heads but only 8 key/value heads, so 8 heads look at the same history. Read each saved item once and use it for all 8.
- Like one helper reading a book aloud to 8 students, instead of 8 students each fetching the book.

**Online softmax (the trick behind FlashAttention):** attention turns scores into weights that add up to 1, and the usual way needs all the scores first: two passes and a long list. Online: keep a running "best score so far" and a running total, and correct the total when a better score turns up. One pass, no list.

```
for each block of 16 earlier tokens:
    scores → new best = max(old best, block's best)
    shrink the old total to the new best, add the block's part
end: answer = total ÷ sum of weights
```

- The two small multiplies inside attention also moved to the matrix cores.

---

# Attention (2/2): flash-decoding

**Problem:** at position 1,023 one request has 1,024 history entries per head. If one worker reads them all, too few reads are in flight at once, and the slowest workers finish long after the rest.

**Idea:** cut the history into pieces, give each piece to a different worker, then merge. The running best and running total from page 14 make the merge exact.

```
history of one request:  [0–31] [32–63] [64–95] [96–127] [128–159] … [992–1023]
worker:                     0       1       2        3        0     …      3
each worker keeps its own best score, total and mix  →  merge: rescale all to the overall best, add up
```

- **Here:** 4 workers per head, merged inside the same kernel. (Usually the merge is a second kernel, but 512 requests × 8 heads = 4,096 jobs already fill the GPU.)
- **Measured:** no split is 12 % slower on full layers and 35 % slower on window layers; 16 splits are slower again.
- **Attention, all together:** 9.5 → 1.25 ms per full layer at the last position (**7.6×**), then 0.95 ms with the 64-position blocks. Whole program **+9.2 %**. PyTorch's attention on this GPU: 9.9 ms.

<!-- Workgroup = 2 neighbouring KV heads x 4 splits (8 waves); 32 positions per wave step; lazy rescale (only
when the max grows by more than 2^8); merge through LDS. Code: getp-csrc/kernels/fast_attention.hpp. -->

---

# Smaller numbers: less to read, faster math

**Problem:** the given program uses 32-bit numbers. Every byte read costs memory time, and the matrix cores want 16-bit inputs.

```
fp32   ████████████████████████████████   32 bits, about 7 digits
bf16   ████████████████                   16 bits, about 3 digits, same range as fp32
int8   ████████                           8 bits: a whole number −127…127 × one shared scale
MXFP4  ████                               4 bits: one of 16 codes × one shared power of 2 per 32 numbers
```

- **Mixed precision (used):** multiply 16-bit (bf16) numbers, add up the results in 32 bits. **+56 %**, and it opens the matrix cores.
- **Compressed weights (used):** the expert weights stay 4-bit (MXFP4) in memory and are unpacked inside the compute unit, right before the matrix cores use them (page 20).
- **Compressed cache (used for half):** the values in the KV cache are stored as 8-bit (int8), one scale per 64 numbers: **+2 %**. The keys stay 16-bit.
- What each step cost in answer quality: next page.

---

# Precision: speed against quality

**In plain words:** smaller numbers are faster but can change the answers, so every such change was scored.

| Change | Speed | Word match (METEOR) | Meaning (BERTScore) |
|---|---:|---:|---:|
| Same math as the CPU | baseline | 0.997 | 0.99998 |
| 16-bit numbers, 32-bit sums | +56 % | 0.552 | 0.979 |
| 8-bit values in the cache | +2 % | −0.021 | −0.001 |
| 8-bit keys too (measured by another team) | | 0.342 | not taken |

- **Why the word score drops but the meaning score does not:** the model always takes its top choice. When two tokens are nearly tied, a tiny rounding difference flips the pick, and from there the answer goes a different but equally good way. Different words, same meaning.
- **Every step after that keeps the answers identical,** so there is no further quality risk.

---

# Matrix multiply: work in tiles on the matrix cores

**Problem:** done one number at a time on the ordinary calculators, the multiplies were slow, and the same inputs were fetched from memory again and again.

**Idea:** cut the output into **tiles** (blocks). One group of threads loads the inputs of its tile once into the compute unit's fast scratchpad, and the matrix cores multiply 16 × 16 blocks from there. The next pieces are loaded while the current ones are multiplied.

```
weights (columns) ──────▶
tokens  ┌────┬────┬────┬────┐
(rows)  │ T1 │ T2 │ T3 │ …  │   each tile: load its inputs once, reuse them many times
  │     ├────┼────┼────┼────┤
  ▼     │ …  │    │    │    │
```

- **Measured:** the plain multiplies of one step: 191 → 12.1 ms, whole program **+37 %**. AMD's tuned library on the same shapes: 83–97 trillion operations/s; ours: 97–129.
- **Stream-K:** 160 tiles on 104 compute units means 56 units do two rounds while 48 wait. Stream-K cuts the leftover work into slices so all units finish together. Kernel 155 → 141 µs, but whole program only **+0.3 %**.

---

# Grouping by expert: fetch each expert once

**Problem:** each token picks 4 of the 32 experts. With 512 tokens that makes **2,048 token–expert jobs per layer**. One expert's weights are 13 MB.

```
the simple way: one fetch per job          grouped: one fetch per expert
token A → expert 3   fetch 13 MB           expert 3: fetch 13 MB once → tokens A, C, F, …
token B → expert 7   fetch 13 MB           expert 7: fetch 13 MB once → tokens B, D, …
token C → expert 3   fetch 13 MB again     …
2,048 fetches = 27 GB per layer            32 fetches = 0.42 GB per layer
```

- **Like a post office:** sort the letters by city first, then send one truck per city, instead of one trip per letter.
- **64× less to read.** Without it, reading alone would take 22 ms per layer, about 0.5 s per step.
- **Used from step 1.** Sorting costs ~15 µs per layer. Each expert then gets about 64 tokens at once, but unevenly: the busiest got 473 of 512, and 10 % got none.

---

# The ×8 step: the expert multiply on the matrix cores, straight from 4-bit weights

**Problem:** after grouping, the expert multiply was still **68 %** of a step (96 % once the other multiplies were fast). It had three faults:

| Fault | Fix |
|---|---|
| one number at a time, on the ordinary calculators | feed the **matrix cores** (8× faster at this job) |
| the matrix cores cannot read the 4-bit format | unpack inside the compute unit, 4 numbers per instruction, right before use; memory carries only 4-bit numbers |
| experts get very different numbers of tokens | **one shared to-do list** of work pieces (64 tokens × 256 outputs), biggest experts first, taken by one kernel that stays running, so no compute unit sits idle |

- **Measured:** the expert multiply became **15× faster**, and 1.5–3.3× faster than AMD's library run once per expert (with 4× bigger 16-bit weights). Whole program 7,858 → **64,099 tok/s**.
- **Later rounds on the same kernel:** weights in reading order (**+7.2 %**, page 22), three steps in one pass (**+4.1 %**, page 21), a faster way to write the results (16 → 4 µs per piece, **+3 %**).

<!-- Technical names: grouped GEMM; persistent kernel with an atomic ticket queue sorted by expert size;
MFMA 16x16x16 bf16; MXFP4 dequantized in registers; wave priorities for items of <= 32 rows;
epilogue through buffer descriptors + DPP. Code: getp-csrc/kernels/fast_moe_gemm.hpp. Doc 08. -->

---

# Kernel fusion: do several steps in one pass

**Problem:** every kernel reads its input from memory and writes its output back. Small steps (add, normalize, pick the top token) do little math, but each pays a full trip to memory.

**Idea:** merge neighbouring steps into one kernel, so the in-between results stay inside the chip. Like cooking the whole dish at the stove, instead of putting each half-done step back in the fridge.

```
before:  [add] → memory → [normalize] → memory → [multiply]
after:   [add + normalize] ─────────→ memory → [multiply]
```

| Fused | What it saves | Gain (20b) |
|---|---|---:|
| collect the tokens + expert multiply + activation | 47 MB written and read back per layer | +4.1 % |
| add + normalize | one pass over all rows | +0.9 % |
| top-token pick inside the last multiply | 412 MB of scores per step never written | +1.3 % |
| small kernels: start all reads first, merge two counting kernels | waiting | +1.4 % (120b +2.3 %) |

- **Not every merge wins:** the router multiply + top-4 pick merged was slower in the real run (its code was not in the GPU's fast code store, page 29).

---

# Data in reading order

**Problem:** memory is fastest when a kernel reads long continuous pieces. Reads that jump around get much less than the full speed.

**Idea:** rearrange the data once, then read it fast every time. Like putting a book's pages on the shelf in reading order.

- **Expert weights, rearranged when the model loads:** each block the expert kernel needs is one continuous 16 KB piece. Reading speed **855 → 1,296 GB/s**. Whole program **+7.2 %** (20b), **+11.2 %** (120b, whose experts get fewer tokens, so reading matters more).
- **KV cache in 64-position blocks:** the 64 positions of one head sit together in one 8 KB piece. Full layer at the last position 1.25 → 0.95 ms, whole program **+0.8 %**.
- Both keep the answers bit-identical: only where the numbers live changes, not the math.

---

# Using many GPUs: four ways to split the work

```
data parallel      each GCD: the whole model + its own requests       no talking
tensor parallel    each layer's math cut into slices across GCDs     talks in every layer
pipeline parallel  GCD 0: layers 1–18, GCD 1: layers 19–36           hands over between halves
expert parallel    each GCD: half of the experts                     swaps tokens every layer
```

- **20b → data parallel (used):** the model (13.7 GB) fits on each GCD, so the 8 GCDs never talk.
- **120b must be split:** 65 GB > 64 GB. **Expert parallel (used)** over the two GCDs of one card, which have the fastest link (page 24).
- **Tensor parallel (not used):** attention would need a swap too, in every layer, over links 7× slower than memory.
- **Pipeline parallel (not used; rough estimate, not built):** to keep both GCDs busy, the batch must be cut in two halves, so each expert is fetched twice per step with half the tokens. Our cost model gives ~1.4× the expert time; expert parallel's swaps cost ~8 %.

<!-- Pipeline estimate with the tools/ep_split.py cost model at average rows: EP=2 32 rows per expert ->
20.2 us x 64 experts x 36 layers = 46.6 ms per step; PP 16 rows -> 14.7 us x 128 x 18 layers x 2 halves =
67.5 ms. Expert parallel exchange: 202 us of 2.46 ms per layer (8.3 %). Links: 177 GB/s vs 1.25 TB/s. -->

---

# 120b: two GCDs working as one

**In plain words:** the two GCDs of a card split the experts and swap work at every layer.

```
GCD 0: its own 512 requests,                GCD 1: its own 512 requests,
       half of the experts                         the other half
1. each token picks its 4 experts            1. same
2. tokens whose expert is on the other GCD are sent there   ⇄
3. each GCD runs its experts on the tokens of both GCDs
4. the results go back and are added up
```

- **Balance the load:** some experts are much more popular. A table made in advance from our own test prompts (never the scored questions) splits them so both GCDs finish together. Gap 6.7 → 2.6 ms per step, **+1.7 %**.
- **Cheaper waiting:** a shared counter replaces the standard GPU signal: 18 → 13 µs per swap.
- **Push, don't pull:** each GCD writes straight into its partner's memory. A test of 5,000 rounds never saw an old value.
- **Why not 4 or 8 GCDs per copy:** the links between cards are 2–4× slower. A cost model says the transfers eat the saving.

<!-- Per-layer split table: tools/ep_split.py -> getp-csrc/engine/ep_split_120b.hpp, calibrated on 4 kinds of
self-written prompts. Counter in fine-grained memory with system-scope release/acquire. Fresh 3-6 MB reads
over the link run at only 95-145 GB/s. -->

---

# Prefill and decode: reading the question vs writing the answer

- **Prefill:** all the question's tokens are known in advance, so they can go through the model together, as one big efficient multiply.
- **Decode:** each answer token depends on the one before, so it is one token per request per step.

```
question (~20 tokens)       answer (~1,000 tokens)
■■■■■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■…
```

- **Here:** like the given program, we feed the question one token per step. Those steps are only ~2 % of all steps, and cheap ones (short history).
- **Batched prefill (not done):** about **1 %** to gain. It needs each request to keep its own position, because questions have different lengths.
- **Chunked prefill** (long prompts cut into pieces mixed into decode steps) matters for long documents. It does not matter here.

---

# Other well-known techniques

| Technique | What it is | Here |
|---|---|---|
| GPU-side step loop (like CUDA/HIP graphs) | the GPU runs the steps on its own, without the CPU starting each one | **tried: +0.2 %**, removed. The GPU was already busy 99.6 % of the time |
| Speculative decoding | a fast guesser proposes several tokens; the model checks them all in one pass | **not tried.** It pays when the calculators are idle (few users). With 512 requests per GCD they are busy, and checking 4 guesses costs ~4× the math (reasoning, not measured) |
| Sampling vs greedy | pick a random token by its score, or always the top one | the task uses greedy, so the top-token pick is fused into the last multiply (+1.3 %, page 21) |
| Vendor libraries | ready-made multiply and communication code | not allowed inside; used only as yardsticks |

---

# Where the time goes now (20b, one GCD)

| Part | Share of time | How close to the limit |
|---|---:|---|
| Experts | **54 %** | 57 % of the math peak |
| Attention | 13 % | reads ~1.1 of the ~1.25 TB/s possible |
| Picking the next token (scoring 201,088 tokens) | 10 % | 129 of 181 trillion operations/s |
| Other multiplies | 14.5 % | ~100 trillion operations/s |
| Small kernels | ~6 % (was 8 %) | after the last round |

- The GPU is busy about 100 % of the time. There are no gaps between kernels.
- **Yardstick:** AMD's tuned library (hipBLASLt) on the same multiplies reaches 83–97 trillion operations/s. Ours reach 97–129.

<!-- Profile: rocprofv3 kernel trace, build 25d09b2, 512 requests x 1,024 steps, before the small-kernel round.
Other multiplies: QKV 8 %, attention-out 6.5 %. -->

---

# Why the experts reach only about half of the math peak

**In plain words:** three separate losses, and they multiply.

| Loss (measured before the last round) | Factor |
|---|---:|
| The clock runs slower under the power limit (1.46 GHz instead of 1.7) | × 0.86 |
| The matrix cores wait for data 30 % of the time | × 0.70 |
| Some work is padding: an expert with 5 tokens still fills a block of 16 | × 0.86 |
| **Together** | **≈ 0.52** |

- The last round raised it to **57 %** (715 → 652 µs for the first expert multiply of a layer).
- **120b is harder:** about 2 of every 3 of its experts get **fewer than 16 tokens** per step. They wait on reading their weights, not on math.

<!-- 0.86 x 0.70 x 0.86 = 0.52. The padding factor also includes column and k tails (rows padded to 16: -10.5 %).
120b routing at B=512: 24 % of experts empty, 42 % with 1-15 rows. Expert cost model (tools/ep_split.py):
3.03*(n>0) + 6.05*ceil(n/64) + 3.28*ceil(n/16) + 0.143*n us per expert. -->

---

# Surprises worth knowing

- **The power limit decides the speed.** In the real run every card sits at 560 W. A kernel timed alone is not the truth; only the full 8-GCD run is.
- **A hidden wait.** The compiler used a general-purpose write instruction, and the GPU counts those writes in the same queue as its fast scratchpad. So every swap of data between neighbouring threads waited for all earlier writes. Fixed: **16 → 4 µs** per work piece, **+1.5 %**.
- **Cold code.** A kernel took 24 µs in a test loop but **45 µs** in the real run, because its instructions had dropped out of the GPU's small fast store for code.
- **Uneven last round.** Stream-K fixed the 56-of-104 problem in the kernel, but the full run gained only **+0.3 %**.
- **Links are slower for fresh data:** 95–145 GB/s instead of the 177 GB/s of a large copy.
- **A profile can mislead.** "7.5 % idle" was start-up time, not the timed part. The fix for it gained 0.2 % and was removed.

<!-- Hidden wait: flat_store counts toward lgkmcnt, so each ds_bpermute wait waited for the stores. Fix: buffer
descriptors + DPP. Uneven last round = wave quantization. Cold code = instruction cache. -->

---

# Backup: words (1/2)

| Word | Plain meaning |
|---|---|
| token | a piece of a word, about ¾ of an English word |
| weights | the numbers the model learned |
| layer | one stage of the model; 20b has 24, 120b has 36 |
| attention, head | looking back at earlier tokens / one of 64 side-by-side looks |
| KV cache (keys, values) | the saved history attention looks at: what each token contains / hands over |
| expert, router (MoE) | small sub-networks / the part that picks 4 of them per token (mixture of experts) |
| prefill / decode | reading the question at once / writing the answer one token at a time |
| greedy | always take the top-scoring token |
| GCD | one GPU die; an MI250 card has two |
| compute unit | one group of calculators inside a GCD; a GCD has 104 |
| wavefront | 64 threads that run each instruction together |
| kernel | a small program that runs on the GPU |
| matrix core (MFMA) | a circuit that multiplies a 16 × 16 block of numbers in one instruction |
| memory-bound / compute-bound | time set by reading / by math |

---

# Backup: words (2/2)

| Word | Plain meaning |
|---|---|
| GEMM | a matrix multiply: rows of numbers times a table of weights |
| tile | one block of the output, worked on by one group of threads |
| stream-K | cutting the leftover tiles into slices so every compute unit finishes together |
| bf16 / int8 / MXFP4 | 16-bit numbers / 8-bit whole numbers × a scale / 4-bit codes × a shared scale per 32 |
| mixed precision | multiply small numbers, add up the results in bigger ones |
| static / continuous batching | all requests start and end together / a finished seat is refilled at once |
| PagedAttention | cache memory handed out in small pages as an answer grows |
| online softmax (FlashAttention) | turning scores into weights in one pass, with a running best and total |
| flash-decoding | splitting one long history across several workers, then merging |
| grouped-query attention (GQA) | 8 query heads sharing one set of keys and values |
| kernel fusion | several steps in one kernel, so in-between results stay in the chip |
| data / tensor / pipeline / expert parallel | copy per GPU / slices of each layer / halves of the layers / shares of the experts |
| speculative decoding | guess several tokens fast, check them in one pass |
| METEOR / BERTScore | word-match score / meaning-closeness score |
| profile / A/B test / hash | a record of kernel times / old and new run in turns / a short fingerprint of the output |

---

# Backup: numbers to remember

- **Work:** 4,096 questions × ~1,000 tokens ≈ **4.1 million** tokens. 20b takes ~51 s, 120b ~90 s.
- **20b step:** ~50 ms for 512 tokens on each GCD. One layer ≈ 1.9 ms, of which the experts take ~1.0 ms.
- **120b step:** ~89 ms. One layer ≈ 2.46 ms per GCD.
- **One expert:** 13.2 MB compressed. 20b reads ~420 MB of experts per layer (32 experts); 120b ~850 MB per layer per GCD (64 experts).
- **Next-token scoring:** 512 × 201,088 × 2,880 × 2 = 593 billion operations per step, ~4.6 ms.
- **Saved history per GCD at 512 requests:** 20b ~11 GB, 120b ~16 GB.
- **Attention floor:** 1.07 GB read per full layer at the last position = 0.86 ms at 1.25 TB/s.
