# Lesson 16: Finished requests no longer use the experts

| | |
|---|---|
| Reference commit | `c15ec95` "Route finished requests to no expert" |
| Starts from | `cdca2c5` (lesson 15) |
| To type | 32 lines of program code in 3 files |
| Shortcut | `git cherry-pick c15ec95` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 72,453 tok/s, 120b 40,633 tok/s (8 GCDs, official input) |

**In plain words:** all requests on a GPU move forward together, one word per step. Some answers
end early, but their seats stay in the group until everyone is done, and the experts kept working
for them. Now a finished request gets "no expert", so the most expensive part of each step skips
it. The answers of the requests still running do not change at all.

## 1. Goal

Stop spending expert work on rows whose answer has already ended, without changing a single bit of
the other rows' results.

## 2. Why this, why now

- **Static batching** (lesson 03: up to 512 requests per GCD advance one position per step, all
  together) keeps a finished request in its row until the whole group is done. That row still went
  through every kernel.
- Counting on the reference answers: **3.1 %** of the row-steps of a 4,096-request run belong to
  requests that have already ended. (A **row-step** is one request's row in one step.)
- The expert multiplies were about **53 %** of a step's time in a profile of that time, and they
  computed those rows like any other.
- The cheapest way to recover that waste: make the router say "no expert" for them. Grouping
  (lesson 09) and the expert sums (`moe_combine_ep_kernel`, and `moe_combine_partial_kernel`
  of lesson 14) then simply skip them.

## 3. The technique

New words:
- **request / row** — one prompt and its answer; on the GPU it is one row of every activation table
  (row b of `x`, `t16`, …).
- **router** — the small multiply plus top-4 kernel that picks each token's 4 experts (lesson 13).
  It writes `topk_e` (the expert numbers) and `topk_w` (their weights).
- **expert −1** — a number no expert has. The grouping kernels count rows per expert number, so a
  pair with expert −1 is never counted and never placed: the expert multiplies never see it.
- **done flag** — one `int` per row on the GPU, 1 when that request has ended. The **host** (the
  CPU side of the program, which reads each new token and decides when a request ends) uploads the
  flags only when one of them changed.
- **continuous batching** — the well-known alternative: as soon as a request ends, a new one takes
  its seat, so no seat is ever empty. This program does not build it; section 7 says why.
- **noise** — the up-and-down of tok/s between two runs of the very same build (about 1 %).

```
 step 900, one GCD, 512 rows
 row:      0     1     2   …  37   …  511
 ended?    no    no    yes     no      yes
           │     │     │       │       │
 router:  4 ex  4 ex  -1 -1   4 ex    -1 -1 -1 -1     done[b] = 1 → expert −1 for all 4 choices
           │     │     ╳       │       ╳
 grouping: counts/places only pairs with a real expert number
 experts:  multiply fewer rows (3.1 % fewer over a whole run)
 combine:  row of a finished request: x left as it is
 attention, dense multiplies: unchanged (they still compute every row; see section 7)
```

The rows still running are computed exactly as before, so the answers stay identical; the
finished rows' answers were already written, so nothing they compute matters any more.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/kernels/moe.hpp` | +10 −1 | `router_topk_wave_kernel` gets a `done` array (may be null); a done row writes weight 0 and expert −1 for its k choices and returns | yes |
| 2 | `getp-csrc/kernels/moe_ep.hpp` | +7 −2 | `moe_combine_ep_kernel`: a row whose first expert is −1 returns before touching `x`; `moe_combine_partial_kernel` the same | yes |
| 3 | `getp-csrc/engine/gpu_engine.hpp` | +15 −1 | `row_done` buffer per GCD; cleared at the start of a wave; uploaded before a step when `done_changed`; the router gets it on the fast path only | yes |

Read while typing (line numbers at `c15ec95`):
- `getp-csrc/kernels/moe.hpp:67-73` — the early exit for a done row: lanes 0..k−1 write −1.
- `getp-csrc/kernels/moe_ep.hpp:31-36` (`moe_count_ep_kernel`) — counting is `topk == ge` for a real
  expert number `ge`, so −1 is never counted. No change was needed there.
- `getp-csrc/engine/gpu_engine.hpp:616` (clear the flags), `:626-629` (upload only if changed),
  `:660` (the host marks a row done when it reads an end token).
- `getp-csrc/engine/gpu_engine.hpp:416` — `w->exact ? nullptr : w->row_done`: the exact path keeps
  computing every row, so it stays bit-identical to the CPU.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick c15ec95
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout c15ec95 -- $(git show --name-only --format= c15ec95)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat c15ec95
git show c15ec95 -- getp-csrc/kernels/moe.hpp            # the change to copy, file by file

git diff c15ec95 -- $(git show --name-only --format= c15ec95)   # must print nothing
make runomp
```

Quick check, 20b on one GCD (64 prompts, 128 positions):

```bash
HIP_VISIBLE_DEVICES=2 ./run $M20 -m getp -i $W/in64.txt -o $W/out64.txt -n 128
sha256sum $W/out64.txt | cut -c1-12          # expected: ccfc732a4a25 (same as lesson 15)
```

Quick check, 120b on a GCD pair:

```bash
HIP_VISIBLE_DEVICES=0,1 ./run $M120 -m getp -i $W/in64.txt -o $W/out64_120.txt -n 128
sha256sum $W/out64_120.txt | cut -c1-12      # expected: 4ed6e262b73a
```

The fingerprint must not move: this change only skips work whose results nobody reads.

Full run (8 GCDs, a few minutes each):

```bash
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t lesson-16 --score
GPTOSS_RESULTS=$W/results python tools/bench.py -m $M120 -i evaluation/input.txt -g all -t lesson-16 --score
```

Expected: 20b about 72,450 tok/s, fingerprint `496d7ce70b4e`; 120b about 40,630 tok/s, fingerprint
`e723ce53480f` — both the same fingerprints as lesson 15, so the scores are lesson 15's (20b METEOR
0.554, BERTScore 0.979; 120b 0.576 / 0.982).

Optional — measuring a gain this small. +0.8 % is close to the noise, so one run of each build
proves nothing. We ran the old and the new build **alternately**, twice each, and compared the
means; any slow drift of the machine (temperature, other users) then hits both builds alike:

```bash
mkdir -p $W/prev && git archive cdca2c5 | tar -x -C $W/prev      # lesson 15's code
for i in 1 2; do
  (cd $W/prev && GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t prev)
  GPTOSS_RESULTS=$W/results python tools/bench.py -m $M20 -i evaluation/input.txt -g all -t new
done
grep -h '"tag"' $W/results/results.jsonl | tail -4    # compare the "tps" values
```

Save: `git add -A -- . ':!walkthrough' && git commit -m "Route finished requests to no expert"`

## 6. Result

| 8 GCDs, 4,096 requests | before | after | change |
|---|---:|---:|---:|
| 20b (means of 2 alternating runs each) | 71,853 | **72,453** | +0.8 % |
| 120b | 40,164 | **40,633** | +1.2 % |
| Output | | same fingerprints (20b `496d7ce70b4e`, 120b `e723ce53480f`) | |

The two 20b pairs were 71,889 / 71,816 → 72,506 / 72,400: each new run beat each old one, which is
what makes a +0.8 % claim believable.

## 7. Decisions and reasons

- **Decided:** give finished requests "no expert" in the router.
  **Why:** 3.1 % of all row-steps were finished requests, and the expert multiplies (53 % of the
  time) computed them anyway. The change is a few lines and cannot change any active row.
  **Not chosen:** compacting the batch (moving the live rows together so that attention and the
  dense multiplies skip finished rows too): much more code for the remaining ~1 %.
- **Decided (earlier, confirmed here):** no continuous batching and no PagedAttention.
  **Why:**
  - 95 % of the reference answers run to the full 1,024 tokens, so seats seldom free up early.
  - All 4,096 requests fit at once (512 per GCD; the bf16 cache of 512 requests is 14.5 GB of a
    64 GB GCD), so there is no queue of waiting requests to fill a freed seat with.
  - The whole waste continuous batching would remove is the 3.1 % above, and this lesson already
    takes the expert part of it.
  - PagedAttention (handing out cache memory in small pages on demand) solves memory waste; here
    every request's cache is reserved once and nothing is short of memory.
- **Decided:** judge gains near the noise with alternating runs, not single runs.
  **Why:** two runs of the same build differ by about 1 %, the size of the gain.

## 8. Questions to think about

1. A finished request still goes through attention and the dense multiplies every step. Why does
   that not change the answers of the other requests?
2. Why does 120b gain more (+1.2 %) than 20b (+0.8 %) from the same 3.1 % of rows?
3. If the input had 40,000 prompts instead of 4,096, which of the reasons against continuous
   batching would stop holding?

## 9. Discussion notes

(empty: we write here during the walk-through)
