#!/usr/bin/env python3
"""Compares getp output token ids with reference token ids, line by line.

    tools/ref_match.py <output.txt> <reference.txt>

Line i of the output must answer the same prompt as line i of the reference
(true for any input made of the first lines of evaluation/input.txt). Lines
are compared over the shorter of the two lengths: a run with a smaller -n is
compared with the reference's first tokens, and a line that runs one token
past the reference (the reference stops at 1,024 positions in total, getp at
1,025) still counts as matching. Prints the share of matching lines and the
share of tokens before the first difference.
"""

import sys


def main():
    out = [l.split() for l in open(sys.argv[1])]
    ref = [l.split() for _, l in zip(range(len(out)), open(sys.argv[2]))]
    full = 0
    same_prefix = 0
    total = 0
    first_diff = []
    for o, r in zip(out, ref):
        n = 0
        while n < min(len(o), len(r)) and o[n] == r[n]:
            n += 1
        if n == min(len(o), len(r)):
            full += 1
        else:
            first_diff.append(n)
        same_prefix += n
        total += len(o)
    lines = len(out)
    print(f"lines identical to the reference: {full} / {lines} ({100 * full / max(lines, 1):.2f} %)")
    print(f"tokens before the first difference: {same_prefix} / {total} ({100 * same_prefix / max(total, 1):.2f} %)")
    if first_diff:
        first_diff.sort()
        print(f"differing lines: first difference at token {first_diff[0]} (min), "
              f"{first_diff[len(first_diff) // 2]} (median)")


if __name__ == "__main__":
    main()
