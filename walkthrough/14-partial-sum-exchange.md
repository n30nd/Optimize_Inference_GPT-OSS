# Lesson 14: 120b — send one sum per token between the two GCDs

| | |
|---|---|
| Reference commit | `d68a11d` "Exchange one partial sum per token between expert-split GCDs" |
| Starts from | `85e8426` (lesson 13) |
| To type | 90 lines of program code in 2 files (78 in `getp-csrc/kernels/moe_ep.hpp`, 12 in the engine); no tests or docs |
| Speed after | 120b 35,815 → **36,132** tok/s (+0.9 %); 20b unchanged (it has no partner) |

**In plain words:** for gpt-oss-120b, the two halves of a card each hold half of the experts
(lesson 05). After the experts run, each half had to fetch, for every one of its tokens, the
separate results of the experts that ran on the other half — about two long rows per token.
Now the other half first adds those results up, weighted, and sends one finished row per token.
Half the traffic between the halves, and one tidy read per token.

## 1. Goal

On a split GCD pair, replace "read each of the partner's expert outputs for my tokens" by "the
partner adds them up for my tokens; I read one row per token". Fast path only.

## 2. Why this, why now

A 120b profile on one GCD pair (after lesson 13):
- the expert-sum kernel took **106 µs per layer**, against 45 µs on a GCD without a partner —
  because it read about 2 fp32 rows of 2,880 numbers per token from the partner's memory;
- the gather of the partner's input rows: 42 µs per layer, against 30 µs without a partner;
- **5.4 %** of the time was spent waiting for the partner.

Reading the partner's memory goes over the link between the two GCDs of a card (a kernel reading the
partner's memory measured about 177 GB/s, against 1,270–1,290 GB/s from its own memory), so every
byte saved there counts.

## 3. The technique

New words:
- **expert split (expert parallelism)** — each GCD of a pair holds 64 of the 128 experts of
  every layer; each GCD runs *its* experts on every token that chose them, its own tokens and
  its partner's (lesson 05).
- **partner memory** — a GCD can read the other GCD's memory directly (peer access), but over
  the slower link.
- **partial sum** — part of a total: here, the weighted sum of only the outputs one GCD's experts
  produced for a token. The other part comes from the other GCD's experts.
- **order of additions** — fp32 addition rounds after every step, so `(a + b) + c` and
  `a + (b + c)` can differ in the last bit. Changing the order changes the rounding.

Each token picks 4 experts; on average about 2 are on the partner.

```
 before (GCD A combining its token t; experts e1, e2 on A, e3, e4 on B):
   A reads its own rows   Z_A[e1], Z_A[e2]
   A reads over the link  Z_B[e3], Z_B[e4]          2 rows × 2,880 fp32 = 23 KB per token
   x[t] += w1·Z1 + w2·Z2 + w3·Z3 + w4·Z4            (in choice order)

 after:
   B, right after its experts:  S_B[t] = w3·Z_B[e3] + w4·Z_B[e4]   moe_partial_ep_kernel
   A reads over the link        S_B[t]               1 row × 2,880 fp32 = 11.5 KB per token
   x[t] += (w1·Z_A[e1] + w2·Z_A[e2]) + S_B[t]        moe_combine_partial_kernel
```

For 512 tokens per GCD and layer: about 11.8 MB over the link before, 5.9 MB after.

Because the additions now happen in another order, the 120b answers change slightly (last-bit
rounding differences that can grow into different words later); so this lesson is checked with
the quality scores, not with the fingerprint.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe_ep.hpp` | +78 | `moe_partial_ep_kernel` (the partner's sums for my tokens, into `S`) and `moe_combine_partial_kernel` (my experts + the partner's `S` into `x`) | yes |
| 2 | `getp-csrc/engine/gpu_engine.hpp` | +12 | the buffer `S` (B × 2,880 fp32) in `GpuWorker`, its size and allocation; the call to `moe_partial_ep_kernel` after mlp2 in `layer_experts()`; `layer_combine()` uses `moe_combine_partial_kernel` on the fast path with a partner | yes |

Read while typing (line numbers at `d68a11d`):
- `getp-csrc/kernels/moe_ep.hpp:139–180` — the old `moe_combine_ep_kernel`, which builds a list of
  4 row pointers, some into the partner's memory (`Z_peer`).
- `getp-csrc/kernels/moe_ep.hpp:186` and `:220` — the two new kernels: each block is one token,
  `float4` loads (4 numbers at a time), and only the choices that fall in this GCD's experts
  (`e_begin` … `e_begin + e_count`) are added.
- `getp-csrc/engine/gpu_engine.hpp:437–439` — `S` is computed for the partner's `B_peer` tokens,
  from this GCD's `Zg`, with the partner's choices (`p->topk_w`, `p->topk_e`).
- `getp-csrc/engine/gpu_engine.hpp:455–458` — the combine reads the partner's `p->S`.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show d68a11d
git diff d68a11d -- $(git show --name-only --format= d68a11d)   # must print nothing
make runomp
```

Quick check, 20b (one GCD, 64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12      # ccfc732a4a25 — 20b has no partner, nothing changes
```

Quick check, 120b (a GCD pair):

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12  # 4ed6e262b73a — differs from lesson 13 (new order of additions)
```

No new test file. The exact path is untouched (it still adds the four outputs in expert order,
as the CPU does), so `GPTOSS_EXACT=1` runs keep reproducing the CPU.

Full run, 120b with scores:

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-14-120b --score
```

Expected: about **36,132 tok/s**, METEOR **0.576**, BERTScore **0.982**, fingerprint `e723ce53480f`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Exchange one partial sum per token between expert-split GCDs"`

## 6. Result

| 120b, 8 GCDs, 4,096 prompts | before | after |
|---|---:|---:|
| tok/s | 35,815 | **36,132** (+0.9 %) |
| METEOR / BERTScore | 0.568 / 0.981 | 0.576 / 0.982 |
| bytes read over the link per layer and GCD (expert outputs) | ~11.8 MB | ~5.9 MB |

Both before and after are measured back to back on the same machine and input. The scores
moved up a little; that is chance from the new rounding, not a quality gain.

## 7. Decisions and reasons

- **Decided:** each GCD sums its experts' outputs for the partner's tokens; the partner reads
  one row per token.
  **Why:** the expert-sum kernel took 106 µs per layer with a partner vs 45 µs without, because
  of ~2 fp32 rows per token read over the link; 5.4 % of the time was spent waiting.
- **Not chosen:** sending the expert rows in bf16 instead (half the bytes too).
  **Why:** it rounds every expert output to bf16 before the sum, a real loss of precision; the
  partial sum halves the bytes and stays in fp32.
- **Not chosen:** splitting 20b's experts over pairs too (`GPTOSS_EP=2`), so that each expert
  sees 1,024 tokens instead of 512 and its multiply runs faster (79 → 99 TFLOP/s).
  **Why:** measured on 8 GCDs, 4,096 prompts: 67,394 (no split) → 64,687 tok/s (split), −4 %,
  same answers. The faster multiplies do not pay for reading ~18 MB of the partner's rows and
  outputs per layer, nor for the waiting. A re-test after lessons 14, 15 and 20 still lost
  5.5 %. 20b keeps a full copy of the model on each GCD.

## 8. Questions to think about

1. Why does summing *before* sending halve the traffic? When would it not help (think of a token
   whose 4 experts are all on the partner, or all at home)?
2. This change alters the 120b answers although the arithmetic is "the same sum". Why, and why
   is that acceptable here but would not be on the exact path?
3. For 20b, splitting the experts made each multiply faster yet the whole program slower. What
   costs did the split add, and how would you estimate them before trying?

## 9. Discussion notes

(empty: we write here during the walk-through)
