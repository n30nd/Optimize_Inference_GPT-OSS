# Lesson 06: Attention rewritten — read each stored number once, use the matrix units

| | |
|---|---|
| Reference commit | `03e8e9d` "Rewrite decode attention around grouped-query sharing and matrix cores" |
| Starts from | `d28c548` (lesson 05) |
| To type | 372 lines of program code in 1 file (+372 −9); 672 lines of tests and benchmarks (optional); 425 lines of docs (optional) |
| Speed after | 20b 5,697 tok/s (was 5,217) |

**In plain words:** at every step each request looks back at everything it has written so far.
The first GPU version fetched the same stored numbers 8 times and did its arithmetic one number
at a time. The new kernel fetches each stored number once, lets 8 "readers" share it, does the
arithmetic on the matrix units, and splits a long history over 4 helpers who combine their
notes at the end.

## 1. Goal

Make decode attention close to the time it takes just to read the stored history once.

## 2. Why this, why now

- The time profile of lesson 04 put attention under 1 % of a step, but that profile looked only
  at the first ~24 positions of each answer. Attention's work grows with every token, and 95 %
  of the answers run to 1,024 tokens.
- Near the end of an answer the old kernel read about **8.6 GB per full layer** (every stored key
  and value 8 times) — roughly 100 ms per step at 512 requests.
- The floor: one full layer at position 1,023 for 512 requests must read **1.07 GB** of stored
  history; at the 1.25 TB/s a GCD really delivers that is **0.86 ms**. The old kernel took 9.5 ms.
- Lessons 06–10 rebuild the three big parts (attention, the dense multiplies, the expert
  multiplies). They are in the order the changes were finished, not in order of size.

## 3. The technique

New words:
- **query, key, value** — three lists of 64 numbers computed for every token. The newest token's
  query asks "what am I looking for?", each earlier token's key says "what do I contain?", and its
  value is what it hands over.
- **head** — one independent copy of this look-back; gpt-oss has 64 query heads per layer.
- **grouped-query attention (GQA)** — 8 query heads share one key/value head, so a layer stores
  only 8 key/value heads.
- **KV cache** — the stored keys and values of all earlier tokens, one *slot* per token, in bf16.
- **softmax** — turns scores into positive weights that add up to 1:
  `weight = exp(score − max) / sum of exp(score − max)`.
- **attention sink** — one learned extra score per head that joins the softmax but brings no
  value: a head can give part of its weight to "nothing".
- **sliding window** — every other layer only looks at the last 128 tokens.
- **wavefront (wave), lane** — 64 GPU threads that run the same instruction together; each one is
  a lane. A **workgroup** is a group of waves on one compute unit that share **LDS** (64 KB of
  fast scratch memory).
- **MFMA (matrix fused multiply-add)** — one instruction that multiplies two small matrices on the
  matrix cores (here 16 × 16 by 16 × 16 bf16, sums in fp32).
- **online softmax** — a softmax in one pass, keeping only a running max and a running sum.
- **flash-decoding** — splitting one request's history over several workers and merging their
  partial results.
- **memory-bound** — the time is set by the bytes fetched, not by the arithmetic.

```
before: one 64-thread workgroup per (query head h, request b)   → 64 × 512 = 32,768 workgroups
        the 8 heads of one group each fetch the SAME keys and values      → 8× the bytes

after:  waves of one (key/value head g, request b) compute all 8 query heads of g together
        every key and value is fetched once and used by 8 heads

        4 waves split the history, 32 positions per step, round-robin:
          step:  0  1  2  3  4  5  6  7 ... 31
          wave:  0  1  2  3  0  1  2  3 ...  3
        each wave keeps (m = running max, l = running sum, acc = running output);
        at the end:  M = max(sink, m_0..m_3)
                     L = exp(sink − M) + Σ l_w · exp(m_w − M)
                     out = Σ acc_w · exp(m_w − M) / L
```

Two more ideas make it fast:
- **Both products on the matrix cores.** Per 16-position tile, "scores = keys · queries" and
  "output += weights · values" are small matrix products: 4 MFMAs each. An MFMA has 16 columns
  but a group has only 8 heads, so column c holds head c rounded to bf16 (the *hi* part) and
  column c + 8 holds the rounding error (the *lo* part). Adding the two columns afterwards gives
  about 17 bits of precision instead of bf16's 8, for free.
- **Lazy rescale.** The running output only needs rescaling when the max grows a lot; the kernel
  checks once per step and almost never does it.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_attention.hpp` | +372 −9 | `bf16_piece` (hi/lo split), `row_swap8`, `make_rsrc` (buffer loads that read zeros past the end), `KVLayout`, the new `attention_gqa8_kernel`, `attention_bf16()` now picks it (2 key/value heads × 4 splits per workgroup); the old kernel stays as `attention_bf16_simple_kernel`; `kv_index_chunk64` and `attention_bf16_chunk64` added for lesson 07, not yet used | yes |
| 2 | `getp-csrc/tests/attn_test.cpp` | +258 | kernel against an fp64 CPU reference; unused cache slots filled with NaN to catch stray reads | optional |
| 3 | `getp-csrc/bench/attn_bench.cpp` | +353 | times before vs after on gpt-oss shapes; `variants` times every idea tried | optional |
| 4 | `getp-csrc/bench/attn_torch_yardstick.py` | +61 | PyTorch's attention on the same shapes | optional |
| 5 | `docs/09-flash-decoding-attention.md` | +425 | the full story, with every measured idea | optional |

The engine does not change: it already called `attention_bf16()`.

Read while typing (line numbers at `03e8e9d`, file `getp-csrc/kernels/fast_attention.hpp`):
- `:72` `bf16_piece()` and `:94` `row_swap8()` — the hi/lo trick; `:207` adds the two columns of
  the scores.
- `:122` `attention_gqa8_kernel` — the kernel; `:166` the buffer loads whose range ends at the
  last real slot.
- `:231` `if (__any(mx > m + 8.0f))` — the lazy rescale, one decision for the whole wave.
- `:393` `attention_bf16()` — the entry point the engine calls.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 03e8e9d
git show 03e8e9d -- getp-csrc/kernels/fast_attention.hpp
git diff 03e8e9d -- $(git show --name-only --format= 03e8e9d)   # must print nothing
make runomp
```

Quick check, 20b (one GCD):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # 0635c70cdb17
```

The fingerprint changes from lesson 05 (`b4c73f55a422`): the sums are now added in another order,
so a few answers differ in late tokens. Only the full run's scores can say whether that is fine.

120b uses the same attention kernel, so its quick check changes too (lesson 05: `78d2c8fd9f66`):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 59aa33af7244
```

Kernel test and benchmark (commands from the files' first lines):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/attn_test.cpp -o attn_test
tools/gpulock.sh 2 -- ./attn_test         # about 1 minute
hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/bench/attn_bench.cpp -o attn_bench
tools/gpulock.sh 2 -- ./attn_bench        # before (old kernel) vs after
```

`attn_test` ends with:

```
largest fp32-output error: 4.19e-05
ALL PASS
```

`attn_bench` prints one line per shape: kernel, layer, requests, position, positions read, µs per
call, GB/s. The lines at 512 requests, position 1023, from one run (numbers move ~1 % between runs):

```
before (simple)                full     512  1023  1024     9409.4      114
after (attention_bf16)         full     512  1023  1024     1238.8      867
plain read (yardstick)         full     512  1023  1024      817.4     1314
```

Full run (20b, 8 GCDs, about 15 minutes):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-06 --score
```

Expect about **5,697 tok/s**, METEOR **0.548**, BERTScore **0.979**, fingerprint `ddfa580a980c`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Rewrite decode attention around grouped-query sharing and matrix cores"`

## 6. Result

One GCD, 512 requests, µs per call (`bench/attn_bench.cpp`; GB/s counts only the cache bytes a
call must read):

| Layer | position | before | after | faster |
|---|---:|---:|---:|---:|
| window (128 positions) | 1023 | 1,406 µs (95 GB/s) | 151 µs (887 GB/s) | 9.3× |
| full | 256 | 2,452 µs | 340 µs | 7.2× |
| full | 1023 | 9,468 µs (113 GB/s) | **1,246 µs** (861 GB/s) | **7.6×** |

- Over a whole answer: attention costs about **74 ms per step before and 10 ms after** on average
  (130 → 17 ms at the last step).
- Yardstick: PyTorch's attention has no grouped-query decode kernel for this GPU: 9,908 µs on the
  full layer at position 1023.
- Accuracy: largest fp32 error 4.2e-5 against fp64 (the bf16 output's own rounding is ~100×
  larger).

End to end (20b, 8 GCDs, 4,096 prompts):

| | before | after |
|---|---:|---:|
| tok/s | 5,217 | **5,697** (+9.2 %) |
| METEOR / BERTScore | 0.552 / 0.979 | 0.548 / 0.979 |

The end-to-end gain is smaller than 7.6× because at this point a step is mostly the expert
multiply (lessons 09–10).

## 7. Decisions and reasons

- **Decided:** rebuild attention now although the short-answer profile gave it < 1 %. **Why:**
  its cost grows with position; the old kernel read each key and value 8 times, ≈ 8.6 GB per full
  layer at position 1023, about 100 ms per step near the end of the answers.
- **Decided:** 2 key/value heads × 4 history splits per workgroup, split *inside* the workgroup.
  **Why:** fastest of every combination measured (1,245 µs full / 151 µs window). With 512
  requests × 8 heads = 4,096 (request, head) pairs there is plenty of parallel work, so the merge
  can stay in LDS instead of a second kernel.
  **Not chosen** (µs per call, full layer / window layer):
  - grouped-query sharing with plain vector math: 3,825 / 284 — too much arithmetic, one number
    per lane at a time;
  - no split: 1,391 / 203 (12 % / 35 % slower): too little waiting on memory at once;
  - 16 splits: 1,298 / 244: most waves get one step, the merge costs more;
  - 8 key/value heads per workgroup: 1,282 / 206: workgroups too large, poor tail;
  - head-major cache layout: 3× slower at short histories;
  - hi/lo parts truncated instead of rounded: 3× less accurate.
- **Decided:** keep the cache layout for now, and add the better one ("chunk64") as a separate
  step. **Why:** with all the arithmetic removed, the same loads on the old layout reach only
  ~870 GB/s against 1,312 GB/s for a plain read, so the layout is the next limit — and a separate
  step lets us measure it alone (lesson 07).

## 8. Questions to think about

1. Online softmax never stores the list of all scores. Why is subtracting the running max still
   correct when a bigger score shows up later?
2. Why does sharing each key among 8 heads move the limit from "memory" to "arithmetic" — and why
   does moving the arithmetic to the matrix cores move it back?
3. Flash-decoding is usually two kernels (split, then merge). Why could we do it in one here, and
   when would that stop working?

## 9. Discussion notes

(empty: we write here during the walk-through)
