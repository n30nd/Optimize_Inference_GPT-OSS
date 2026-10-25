#!/usr/bin/env python3
"""Chooses which experts each GCD of a pair holds (gpt-oss-120b, two GCDs
share every layer's 128 experts), so that both GCDs get about the same expert
work in every step, and writes the table the engine reads.

The engine runs a layer's expert step on both GCDs at once, and each GCD then
waits for the other's results. The GCD holding the more popular experts
finishes later and the other one waits. This tool picks, per layer, the 64
experts of the first GCD (the rest go to the second) that make the predicted
work of the two as equal as possible over many recorded steps.

1. Record how many rows each expert gets, per step and layer, on prompts that
   are not the evaluation set. Which experts are popular depends on the kind
   of text, so record each kind of tools/make_calib_prompts.py on its own:

       for k in prose essay code math; do
         tools/make_calib_prompts.py 1024 --kind $k > calib_$k.txt
         GPTOSS_EXPERT_LOAD=load_$k ./run <120b> -m getp -i calib_$k.txt -o out_$k.txt
       done

   (one GCD pair, through tools/gpulock.sh). Each run writes load_<kind>.0
   (int16, steps x 36 layers x 128 experts).

2. Make the table from all of them (it then balances every kind):

       tools/ep_split.py make load_*.0 --note "how it was recorded"

3. See how well a table balances another recording (first and second half of
   the experts with --halves):

       tools/ep_split.py check load.0 --table getp-csrc/engine/ep_split_120b.hpp

Work model: an expert that got n rows in a step costs
    expert * (n > 0) + tile * ceil(n / 64) + sub * ceil(n / 16) + row * n
microseconds (the numbers in COST): the expert's weights are read once, every 64-row work item of
the grouped matrix multiply decodes them again, the matrix-core work grows
with the 16-row tiles in use, and gathering and adding up grow with the rows.
`fit` estimates these numbers from a rocprofv3 kernel trace of a run recorded
with GPTOSS_EXPERT_LOAD:

    tools/ep_split.py fit kernel_trace.csv load.0 --halves
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

LAYERS, EXPERTS, HIDDEN = 36, 128, 2880
# From `fit` on a 1,024-prompt run of tools/make_calib_prompts.py on one MI250
# pair (R^2 0.88; the predicted per-step difference between the two GCDs
# correlates 0.99 with the measured one).
COST = dict(expert=3.03, tile=6.05, sub=3.28, row=0.143)


def load(paths, layers, experts):
    parts = []
    for p in paths:
        a = np.fromfile(p, dtype=np.int16)
        if a.size % (layers * experts):
            sys.exit(f"{p}: size is not a multiple of {layers} x {experts}")
        parts.append(a.reshape(-1, layers, experts).astype(np.int64))
    x = np.concatenate(parts)
    return x[x.sum(axis=(1, 2)) > 0]  # steps with work


def features(n):
    """n: rows per expert (any shape). Returns (..., 4): the four work terms."""
    return np.stack([(n > 0), -(-n // 64), -(-n // 16), n], axis=-1).astype(np.float64)


def cost(n):
    f = features(n)
    return f @ np.array([COST["expert"], COST["tile"], COST["sub"], COST["row"]])


def read_table(path, layers, experts):
    text = Path(path).read_text()
    body = text[text.index("= {") + 3:]
    nums = [int(v) for v in re.findall(r"\d+", body)]
    t = np.array(nums[: layers * experts]).reshape(layers, experts)
    for l in range(layers):
        if sorted(t[l]) != list(range(experts)):
            sys.exit(f"{path}: layer {l} is not a permutation")
    return t


def first_gcd_sets(args, layers, experts):
    """Boolean (layers, experts): True for experts the first GCD holds."""
    if args.halves or not args.table:
        m = np.zeros((layers, experts), bool)
        m[:, : experts // 2] = True
        return m
    t = read_table(args.table, layers, experts)
    m = np.zeros((layers, experts), bool)
    for l in range(layers):
        m[l, t[l, : experts // 2]] = True
    return m


def imbalance(c, first):
    """c: (steps, experts) costs, first: (experts,) bool. Returns per-step A - B."""
    return c[:, first].sum(1) - c[:, ~first].sum(1)


def balance_layer(c):
    """Picks half of the experts: greedy by mean cost, then best swaps while
    they lower the sum over steps of |A - B|."""
    S, E = c.shape
    half = E // 2
    mean = c.mean(0)
    first = np.zeros(E, bool)
    n_first = n_second = 0
    tot = [0.0, 0.0]
    for e in np.argsort(-mean, kind="stable"):
        if n_first < half and (tot[0] <= tot[1] or n_second >= E - half):
            first[e] = True
            n_first += 1
            tot[0] += mean[e]
        else:
            n_second += 1
            tot[1] += mean[e]
    d = imbalance(c, first)
    best = np.abs(d).sum()
    while True:
        a_idx, b_idx = np.where(first)[0], np.where(~first)[0]
        # new d for swapping a (first -> second) with b (second -> first)
        nd = d[:, None, None] - 2 * c[:, a_idx][:, :, None] + 2 * c[:, b_idx][:, None, :]
        obj = np.abs(nd).sum(0)
        i, j = np.unravel_index(np.argmin(obj), obj.shape)
        if obj[i, j] >= best - 1e-9:
            break
        first[a_idx[i]], first[b_idx[j]] = False, True
        d = nd[:, i, j]
        best = obj[i, j]
    return first


def report(x, first, label):
    L = x.shape[1]
    per = np.zeros(L)
    mean_d = np.zeros(L)
    tot = 0.0
    for l in range(L):
        c = cost(x[:, l, :])
        d = imbalance(c, first[l])
        per[l] = np.abs(d).mean()
        mean_d[l] = d.mean()
        tot += c.sum(1).mean()
    print(f"{label}: predicted mean |A - B| summed over layers {per.sum() / 1e3:.2f} ms per step, "
          f"|mean A - mean B| {np.abs(mean_d).sum() / 1e3:.2f} ms, "
          f"A + B {tot / 1e3:.1f} ms; worst layers (us): " +
          ", ".join(f"{l}:{per[l]:.0f}" for l in np.argsort(-per)[:5]))
    return per


def cmd_make(args):
    x = load(args.load, args.layers, args.experts)
    E = args.experts
    halves = np.zeros((args.layers, E), bool)
    halves[:, : E // 2] = True
    report(x, halves, "halves")
    first = np.zeros((args.layers, E), bool)
    for l in range(args.layers):
        first[l] = balance_layer(cost(x[:, l, :]))
    report(x, first, "table ")
    rows = []
    for l in range(args.layers):
        order = list(np.where(first[l])[0]) + list(np.where(~first[l])[0])
        rows.append("    {" + ", ".join(str(int(e)) for e in order) + "},")
    src = args.note or ", ".join(Path(p).name for p in args.load)
    text = f"""// Generated by tools/ep_split.py; do not edit.
// Recorded loads: {x.shape[0]} steps of {src}.
// Work model (us per expert and step): {COST['expert']} * (rows > 0) + {COST['tile']} * ceil(rows / 64)
//   + {COST['sub']} * ceil(rows / 16) + {COST['row']} * rows.
//
// ep_split_120b[l][j]: the checkpoint expert in slot j of layer l. The first
// GCD of a pair holds slots 0 .. {E // 2 - 1}, the second {E // 2} .. {E - 1}.
#pragma once

#include <stdint.h>

#define EP_SPLIT_LAYERS {args.layers}
#define EP_SPLIT_EXPERTS {E}
#define EP_SPLIT_HIDDEN {HIDDEN}

static const uint8_t ep_split_120b[EP_SPLIT_LAYERS][EP_SPLIT_EXPERTS] = {{
""" + "\n".join(rows) + "\n};\n"
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")


def cmd_check(args):
    x = load(args.load, args.layers, args.experts)
    first = first_gcd_sets(args, args.layers, args.experts)
    per = report(x, first, "halves" if args.halves or not args.table else Path(args.table).name)
    if args.per_layer:
        for l, v in enumerate(per):
            print(f"  layer {l:2d}: {v:6.0f} us")


def trace_phases(path, layers):
    """Per agent: the expert phase (start of the grouping kernel,
    moe_group_ep_kernel or the older moe_count_ep_kernel, to the end of the
    moe_partial_ep_kernel after it), in us, shape (steps, layers)."""
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for d in r:
            name = d["Kernel_Name"]
            if name.startswith("void "):  # template kernels, e.g. moe_group_ep_kernel<16>
                name = name[5:]
            if name.startswith(("moe_group_ep_kernel", "moe_count_ep_kernel")):
                k = 0
            elif name.startswith("moe_partial_ep_kernel"):
                k = 1
            else:
                continue
            rows.setdefault(d["Agent_Id"], []).append((int(d["Start_Timestamp"]), int(d["End_Timestamp"]), k))
    out = {}
    for agent, ev in rows.items():
        ev.sort()
        starts = [s for s, _, k in ev if k == 0]
        ends = [e for _, e, k in ev if k == 1]
        n = min(len(starts), len(ends)) // layers * layers
        out[agent] = ((np.array(ends[:n]) - np.array(starts[:n])) / 1e3).reshape(-1, layers)
    return dict(sorted(out.items()))


def nnls(A, y):
    keep = list(range(A.shape[1]))
    while True:
        coef, *_ = np.linalg.lstsq(A[:, keep], y, rcond=None)
        if (coef >= 0).all() or len(keep) == 1:
            full = np.zeros(A.shape[1])
            full[keep] = np.maximum(coef, 0)
            return full
        keep.pop(int(np.argmin(coef)))


def cmd_fit(args):
    a = np.fromfile(args.load[0], dtype=np.int16).astype(np.int64).reshape(-1, args.layers, args.experts)
    phases = trace_phases(args.trace, args.layers)
    if len(phases) != 2:
        sys.exit(f"expected two GCDs with expert steps in the trace, found {len(phases)}")
    first = first_gcd_sets(args, args.layers, args.experts)
    best = None
    for agents in (list(phases), list(phases)[::-1]):
        X, y = [], []
        for rank, agent in enumerate(agents):
            t = phases[agent]
            n = min(len(t), len(a))
            for l in range(args.layers):
                held = first[l] if rank == 0 else ~first[l]
                f = features(a[:n, l][:, held]).sum(1)  # (steps, 4)
                X.append(np.concatenate([f, np.ones((n, 1))], 1))
                y.append(t[:n, l])
        X, y = np.concatenate(X), np.concatenate(y)
        coef = nnls(X, y)
        res = y - X @ coef
        r2 = 1 - res.var() / y.var()
        if best is None or r2 > best[0]:
            best = (r2, coef, agents, np.abs(res).mean())
    r2, coef, agents, mae = best
    print(f"GCD order {agents}; R^2 {r2:.3f}, mean |error| {mae:.1f} us per layer and step")
    print(f"COST.update(expert={coef[0]:.2f}, tile={coef[1]:.2f}, sub={coef[2]:.2f}, row={coef[3]:.3f})"
          f"  # plus {coef[4]:.1f} us per layer that both GCDs pay")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--experts", type=int, default=EXPERTS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="compute the table from recorded loads")
    m.add_argument("load", nargs="+")
    m.add_argument("--note", help="where the recording came from, for the header")
    m.add_argument("-o", "--out", default=str(Path(__file__).resolve().parents[1] /
                                              "getp-csrc/engine/ep_split_120b.hpp"))
    for name in ("check", "fit"):
        c = sub.add_parser(name)
        if name == "fit":
            c.add_argument("trace")
        c.add_argument("load", nargs="+")
        c.add_argument("--table", help="table header to evaluate (default: halves)")
        c.add_argument("--halves", action="store_true", help="first and second half of the experts")
        if name == "check":
            c.add_argument("--per-layer", action="store_true")
    args = ap.parse_args()
    {"make": cmd_make, "check": cmd_check, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
