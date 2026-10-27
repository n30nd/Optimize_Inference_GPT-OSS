# Lesson 17: Merge "add" and "normalize" into one kernel

| | |
|---|---|
| Reference commit | `265650c` "Fuse the residual adds with the RMSNorms that follow them" |
| Starts from | `c15ec95` (lesson 16) |
| To type | 131 lines of program code in 3 files |
| Speed after | 20b 73,212 tok/s, 120b 40,765 tok/s (8 GCDs, official input) |

**In plain words:** twice per layer, the program added a result into each request's running list
of numbers, wrote the list back to memory, and then a second small program read the whole list
again just to scale it. Now one program does both while the numbers are still in hand. Same
arithmetic in the same order, so the answers are identical, and two trips through memory per layer
are gone.

## 1. Goal

Remove two passes over the residual stream per layer by fusing each residual add with the RMSNorm
that comes right after it, with bit-identical results.

## 2. Why this, why now

- After the big multiplies got fast (lessons 08–15), the small kernels started to show. A
  full-length profile (20b, 512 requests, one GCD, 52.0 s) showed: expert sum 2.1 %, RMSNorm
  1.8 %, residual add 0.7 % of the time.
- Each of them is one separate pass over the whole residual stream: 512 rows × 2,880 numbers ×
  4 bytes = 5.9 MB read (and often written) per pass. They do almost no arithmetic: their time is
  memory traffic, so fewer passes is the only way to make them faster.
- The data dependence makes the merge natural: the norm needs exactly the row the add has just
  produced.

## 3. The technique

New words:
- **residual stream** — the 2,880 fp32 numbers per request (`x`) that every layer adds its result
  into. Attention adds its output; the experts add theirs.
- **RMSNorm** — "root mean square" normalisation: divide the row by the square root of the mean of
  its squares, then multiply by a learned weight per position. The result (`t16`, in bf16) is the
  input of the next multiply.
- **kernel fusion** — merging two kernels into one, so the data is read from memory once.
- **pass** — one full read (and maybe write) of a table from GPU memory.
- **reduction** — combining many values into one (here: a sum of squares). In a block it is done
  in shared memory (LDS, the fast on-chip memory a block shares), halving the number of active
  threads each round.
- **bit for bit** — exactly the same 32 bits in every result, not just "close".

```
 before (per layer, fast path)                       after
 attention-out ─► add_inplace:   x += proj  (pass 1)  attention-out ─► add_rmsnorm:   x += proj
              ─► rmsnorm(ffn):   t16 = N(x) (pass 2)                                 t16 = N(x)   (1 pass)
 experts ─► combine:   x += Σ w·Z           (pass 3)  experts ─► combine_rmsnorm:    x += Σ w·Z
 next layer ─► rmsnorm(attn): t16 = N(x)    (pass 4)                                 t16 = N'(x)  (1 pass)
                                                      N' = next layer's attention norm,
                                                           or the final norm before the LM head
```

Why the bits stay the same: in the fused kernel each thread keeps the new values of its own
elements (j = t, t + 256, t + 512, …: at most 16 of them, `FUSED_ROW_MAX`) in registers, and
`rmsnorm_row_bf16` then runs the same sum of squares, in the same element order, with the same
`fmaf`, the same reduction tree and the same formula as the old `rmsnorm_bf16_kernel`.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/fast_misc.hpp` | +42 | `rmsnorm_row_bf16()` (the norm's arithmetic for values a thread already holds), `add_rmsnorm_bf16_kernel` (add, then norm) | yes |
| 2 | `getp-csrc/kernels/moe_ep.hpp` | +69 | `moe_combine_rmsnorm_kernel` (20b: expert sum + norm), `moe_combine_partial_rmsnorm_kernel` (120b pair: own experts + partner's partial sum + norm) | yes |
| 3 | `getp-csrc/engine/gpu_engine.hpp` | +20 −13 | `layer_attention` runs the first norm only in layer 0 and calls `add_rmsnorm_bf16_kernel`; `layer_route` and `step_logits` no longer norm; `layer_combine` picks the next layer's norm weight | yes |

Read while typing (line numbers at `265650c`):
- `getp-csrc/kernels/fast_misc.hpp:12` `rmsnorm_bf16_kernel` next to `:109` `rmsnorm_row_bf16`:
  the same loops, except that the second one reads `xs[c]` from registers instead of `x` from
  memory.
- `getp-csrc/kernels/fast_misc.hpp:141` — `__fadd_rn(xr[j], yr[j])`: an add the compiler may not
  merge into a multiply-add, so it rounds exactly like the old `x += y`.
- `getp-csrc/engine/gpu_engine.hpp:493` — `norm_w`: the next layer's `rms_attn`, or `rms_out`
  after the last layer.
- `getp-csrc/kernels/moe_ep.hpp:269` — a finished request (lesson 16, expert −1) keeps its `x`
  but still gets its norm, because the next layer's attention reads `t16` for every row.

## 5. Type, compare, build, check

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat 265650c
git show 265650c -- getp-csrc/kernels/fast_misc.hpp       # the change to copy, file by file

git diff 265650c -- $(git show --name-only --format= 265650c)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: ccfc732a4a25 (same as lessons 10–16)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 4ed6e262b73a
```

A fused kernel that added in another order would show up here as a new fingerprint.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-17 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-17 --score
```

Expected: 20b about 73,200 tok/s, fingerprint `496d7ce70b4e`; 120b about 40,770 tok/s, fingerprint
`e723ce53480f` (both unchanged from lesson 16, so the scores are unchanged too).

Save: `git add -A -- . ':!walkthrough' && git commit -m "Fuse the residual adds with the RMSNorms that follow them"`

## 6. Result

| 8 GCDs, 4,096 requests | before | after | change |
|---|---:|---:|---:|
| 20b (means of 2 alternating runs each) | 72,540 | **73,212** | +0.9 % |
| 120b | 40,633 | **40,765** | +0.3 % (noise level) |
| Output | | same fingerprints | |
| Memory passes over the 512 × 2,880 residual stream | 4 per layer | 2 per layer | −2 |

The two 20b pairs were 72,613 / 72,467 → 73,146 / 73,279. The 120b gain is within the noise: a
120b step takes about twice as long as a 20b step (~93 vs ~48 ms), so the same kind of saving is a
smaller share of it.

## 7. Decisions and reasons

- **Decided:** fuse residual add + RMSNorm, and expert sum + the next layer's RMSNorm.
  **Why:** the profile put expert sum, RMSNorm and add at 2.1 + 1.8 + 0.7 % of the time, each a
  separate pass over the same 5.9 MB; they are bound by memory, so removing passes removes time.
- **Decided:** keep every thread's element order and every operation identical to the separate
  kernels. **Why:** then the fingerprint proves the fusion correct at once; any change in the last
  bit of a sum would change later tokens and make the check useless.
- **Decided:** the exact path keeps its separate kernels. **Why:** it must stay bit-identical to
  the CPU, and its speed does not matter.

## 8. Questions to think about

1. Why can the fused kernel keep each row's values in registers, and what limits how big a row
   can be (`FUSED_ROW_MAX`)?
2. The combine kernel now needs the *next* layer's norm weight. What happens after the last layer,
   and why does layer 0 still need its own norm kernel?
3. The gain is +0.9 % for 20b although the three kernels were 4.6 % of the time. Where does the
   rest of their time go after fusion?

## 9. Discussion notes

(empty: we write here during the walk-through)
