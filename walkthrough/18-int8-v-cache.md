# Lesson 18: Store half of the attention memory in 1 byte per number

| | |
|---|---|
| Reference commit | `027549a` "Store the V half of the fast-path KV cache as int8", then the docs-only commit `6b315b5` |
| Starts from | `265650c` (lesson 17) |
| To type | 171 lines of program code in 3 files; 540 lines of tests and benchmarks (optional); 232 + 111 lines of docs (optional / cherry-picked) |
| Shortcut | `git cherry-pick 027549a 6b315b5` — copies the finished commits instead of typing (section 5) |
| Speed after | 20b 74,718 tok/s, 120b 41,589 tok/s (8 GCDs, official input) |

**In plain words:** for every word already written, the program keeps two lists of numbers per
head that attention reads again at every later step. Reading them is what attention spends its time
on. We now store one of the two lists with 1 byte per number instead of 2, plus one "unit size" per
list. The numbers are slightly rounded, so the answers change a little. That is a deliberate trade
of a little quality for speed, so it was accepted only after measuring the quality scores.

## 1. Goal

Cut the bytes attention reads per position and head from 256 to 196 by storing the values (V) of
the cache as int8 codes with one scale each, keep the keys (K) in bf16, and keep both quality
scores far above the gates.

## 2. Why this, why now

- A full-length profile (the lesson-15 build, 20b, 512 requests, one GCD) put attention at
  **14.4 %** of the time, and the attention kernel is limited by memory: it reached 861–1,100 GB/s
  against a plain-read ceiling of about 1,312 GB/s. Its arithmetic is not the problem; its bytes are.
- V is half of those bytes.
- Another team had measured both halves on the same course: int8 V moved their METEOR from 0.534
  to 0.519, while int8 K dropped it to 0.342. So only V is worth trying.

## 3. The technique

New words:
- **KV cache** — for every earlier position of a request, each layer stores a **key** (K) and a
  **value** (V) vector per key/value head (64 numbers each). Attention compares the new token's
  query with every key, turns the comparisons into weights (softmax), and sums the values with
  those weights.
- **int8** — a whole number stored in 1 byte; we use −127..127.
- **quantize** — replace a number by the nearest step on a coarse grid. Here the grid is
  `scale × (−127 … 127)`. The stored whole number is the **code**; the step size is the **scale**,
  one fp32 number per 64 values.
- **outlier** — one value much larger than the others in its group of 64. It sets the scale, so
  the other 63 get coarser steps.
- **memory-bound** — a kernel whose time is set by how many bytes it reads, not by its arithmetic.
- **matrix cores (MFMA)** — the GPU units that multiply small tiles of bf16 numbers very fast
  (lessons 06, 08). bf16 holds every whole number from −256 to 256 exactly.
- **METEOR / BERTScore** — the course's two answer-quality scores: word overlap with the reference
  answers, and closeness of meaning. Gates 0.30 / 0.90.

```
 one slot, one key/value head: 64 values (fp32, just computed)
   v      =  0.31  -1.20  0.05 ...  2.54 ... -0.48        largest |v| = 2.54
   scale  =  2.54 / 127 = 0.0200                          1 fp32  =  4 bytes
   codes  =  round(v / scale)
          =   16    -60     2  ...  127  ...  -24          64 int8 = 64 bytes
   back   =  scale × code = 0.320 -1.200 0.040 ... 2.540   error ≤ scale / 2 = 0.01

 bytes per slot and head:  K 128 + V 128 = 256   →   K 128 + codes 64 + scale 4 = 196  (−23 %)

 in attention:  out = Σ p_t·v_t / L  ≈  Σ (p_t · s_t) · c_t / L
                the scale goes into each position's weight, the matrix cores multiply the codes
                themselves (exact in bf16); L, the sum of the weights, does NOT get the scale
```

So the only new error is the rounding to int8, made once when the token is stored.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_misc.hpp` | +20 −6 | `rope_kv_bf16_kernel`: K stays bf16; each wave finds its head's largest \|v\| with 6 `__shfl_xor` steps, writes 64 codes and one scale | yes |
| 2 | `getp-csrc/kernels/fast_attention.hpp` | +137 −12 | `sbyte_to_f32` / `sbytes_to_bf16x4` (4 codes → one matrix operand); `attention_gqa8_kernel` gets `Q8`: loads codes and scales, multiplies each weight by its scale; `vq_index_chunk64`, `vs_index_chunk64`, `attention_q8v_chunk64` | yes |
| 3 | `getp-csrc/engine/gpu_engine.hpp` | +14 −8 | `GpuLayer` gets `vq8` (codes) and `vs` (scales) instead of `vc16`; memory estimate per slot; allocation; the new attention call | yes |
| 4 | `getp-csrc/tests/attn_test.cpp` | +294 −2 | second part: the int8 V kernel against an fp64 reference, and the cache writer against the CPU rule | optional |
| 5 | `getp-csrc/tests/attn_vstats.cpp` | +141 | statistics of the real value vectors (how peaky they are) on real prompts | optional |
| 6 | `getp-csrc/bench/attn_bench.cpp` | +105 −11 | times bf16 V against int8 V | optional |
| 7 | `docs/09-flash-decoding-attention.md` | +232 −20 | section "A smaller cache: the values stored as int8" | optional |
| 8 | (`git cherry-pick 6b315b5`) `docs/12-small-kernels-and-fusion.md` new, `docs/11-expert-parallelism.md` +16 | +111 | docs only: kernel fusion, top-4 per wavefront, parallel grouping, skipping finished requests (lessons 09, 13, 16, 17); the partial-sum exchange (lesson 14) | nothing to type |

Read while typing (line numbers at `027549a`):
- `getp-csrc/kernels/fast_misc.hpp:66-78` — the quantizer: `__shfl_xor` max over the 64 lanes
  (`:71`), `rintf` and clamp (`:73`), the scale written by lane 0 of the head (`:77`).
- `getp-csrc/kernels/fast_attention.hpp:317` — `p * scale` goes into the weight; `:326` the MFMA
  takes `sbytes_to_bf16x4(vq[t][nn])`, the codes themselves.
- `getp-csrc/kernels/fast_attention.hpp:123` — 4 codes become a bf16 operand: each byte to fp32 in
  one instruction (SDWA: one instruction reading a single byte of a register), then the upper
  halves of two fp32 values packed with one `v_perm` (an instruction that builds a word from any 4
  bytes of two registers).
- `getp-csrc/kernels/fast_attention.hpp:544` — `vq_index_chunk64`: inside each 16-slot tile the
  codes sit in the order the lanes load them, so each lane gets its 16 codes with one 16-byte load.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit, and the docs-only commit `6b315b5` after it, as they are:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick 027549a 6b315b5
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout 027549a -- $(git show --name-only --format= 027549a)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 027549a
git show 027549a -- getp-csrc/kernels/fast_misc.hpp       # the change to copy, file by file

git diff 027549a -- $(git show --name-only --format= 027549a)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: 446027378dd3 (NEW: was ccfc732a4a25)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: e205be8f59c8
```

The fingerprint **changes** here, as it must: V is rounded differently. From now on the new value
is the reference for the following lessons. Since the fingerprint can no longer prove correctness,
the unit test and the quality scores do.

Unit test (one GCD, under a minute):

```bash
hipcc --std=c++17 -O3 -fopenmp --offload-arch=gfx90a getp-csrc/tests/attn_test.cpp -o attn_test
HIP_VISIBLE_DEVICES=2 ./attn_test
```

Expected last lines (seen on the reference build):

```
largest bf16-output error vs the original values (no outliers): int8 V 1.16e-02, bf16 V 7.75e-03
writer (rope_kv_bf16_kernel): 56832 codes and their scales vs the CPU rule: identical
ALL PASS
```

Kernel timing (optional, one GCD, about two minutes): `hipcc --std=c++17 -O3 --offload-arch=gfx90a
getp-csrc/bench/attn_bench.cpp -o attn_bench && HIP_VISIBLE_DEVICES=2 ./attn_bench`. On the
reference build, full layer, 512 requests, position 1023: bf16 V 948.4 µs, int8 V 694.9 µs.

Value statistics (optional, real 20b model, about 20 s after the build):

```bash
hipcc --std=c++17 -O3 -fopenmp -march=native --offload-arch=gfx90a \
  getp-csrc/tests/attn_vstats.cpp tokenizer.cpp -o attn_vstats
HIP_VISIBLE_DEVICES=2 ./attn_vstats $M20
```

One line per layer (largest |v| over the root mean square: 2.61–2.92), then, on the reference build:
`all layers: noise 0.61%, bias 0.60%, smooth 0.58%, both 0.57%` — the int8 rounding noise, and what
two rewrites would leave of it (section 7).

Full run with scores (8 GCDs; here the scores matter, not only the speed):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-18 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-18 --score
```

Expected: 20b about 74,700 tok/s, METEOR 0.533, BERTScore 0.978, fingerprint `58023f92ead7`; 120b
about 41,600 tok/s, METEOR 0.562, BERTScore 0.981, fingerprint `141e0b9dbc17`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Store the V half of the fast-path KV cache as int8"`

Then copy the docs-only commit (nothing to type): `git cherry-pick 6b315b5`

## 6. Result

| One GCD, 512 requests | bf16 V | int8 V | change |
|---|---:|---:|---:|
| Attention, full layer, position 1023 | 935 µs | **696 µs** | −26 % |
| Attention, window layer | 123 µs | **107 µs** | −13 % |
| Cache for 512 requests × 1,024 positions, 20b / 120b | 14.5 / 21.7 GB | **11.1 / 16.6 GB** | −23 % |

| 8 GCDs, 4,096 requests | before | after | change |
|---|---|---|---:|
| 20b tok/s | 73,212 | **74,718** | +2.1 % |
| 20b METEOR / BERTScore | 0.554 / 0.979 | **0.533 / 0.978** | −0.021 / −0.001 |
| 120b tok/s | 40,765 | **41,589** | +2.0 % |
| 120b METEOR / BERTScore | 0.576 / 0.982 | **0.562 / 0.981** | −0.014 / −0.001 |

The end-to-end gain is much smaller than the kernel gain because attention is only about 14 % of a
step. (`docs/09` shows a second measurement against the older lesson-15 build: 72,258 → 73,475 and
40,164 → 40,949.)

## 7. Decisions and reasons

- **Decided:** store V as int8 codes with one fp32 scale per 64-value head vector.
  **Why:** attention was 14.4 % of the time and limited by the bytes it reads; V is half of them.
- **Decided:** accept it although METEOR drops by 0.021 (20b) and 0.014 (120b).
  **Why:**
  - +2 % throughput on both models;
  - BERTScore (meaning) moves by only 0.001;
  - both scores stay far above the gates (0.53 ≥ 0.30, 0.978 ≥ 0.90);
  - it is the same kind of trade as the bf16 step of lesson 04 (METEOR 0.997 → 0.552 for +56 %),
    and the same size another team saw (0.534 → 0.519);
  - it is one commit: reverting it brings bf16 V back.
  The drop is real, not noise: bf16 builds differ among themselves by about ±0.005 (0.548–0.557
  for 20b).
- **Decided:** keys stay bf16. **Why:** an error in a key changes a score, and scores go through
  `exp()`, so a small error can move weight from one position to another; an error in a value only
  shifts the output a little, in proportion to its weight. Another team's int8 K run: METEOR 0.342.
- **Decided:** multiply the scale into each position's weight, not into the values.
  **Why:** one multiply per weight instead of per value, and no extra bf16 rounding of `s × c`.
- **Not chosen:**
  - keep V in bf16 — no quality cost, but 2 % slower;
  - int8 K — quality (above);
  - subtracting the value bias or evening out the dims before rounding — the real value vectors
    are not peaky (largest |v| only 2.6–2.9× their root mean square, `attn_vstats`), so the
    rounding noise would only fall from 0.61 % to 0.57–0.60 %.

## 8. Questions to think about

1. Why must the scale go into the weighted sum but not into L, the sum of the weights? What would
   the output look like if it went into both?
2. Why is an error in a key more dangerous than the same error in a value?
3. The scale is chosen per 64 values from the largest one. When would that choice be bad, and how
   did we check that it is not bad for this model?

## 9. Discussion notes

(empty: we write here during the walk-through)
