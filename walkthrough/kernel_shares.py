#!/usr/bin/env python3
"""Sums a rocprofv3 kernel trace per kernel: share of GPU busy time, calls, µs per call.

    rocprofv3 --kernel-trace --output-format csv -d prof -o p -- ./run ...
    python3 walkthrough/kernel_shares.py prof/p_kernel_trace.csv [top=25]

"busy" is the sum of kernel durations on that GPU. One step = one call of
embed_kernel (every step starts with it), so "ms per step" = busy / steps.
"""

import csv
import re
import sys
from collections import defaultdict


def short(name):
    name = re.sub(r"\(.*", "", name)
    return name.replace("void ", "")


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    per_gpu = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in csv.DictReader(open(path)):
        t = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
        k = per_gpu[row["Agent_Id"]][short(row["Kernel_Name"])]
        k[0] += t
        k[1] += 1
    for gpu, kernels in sorted(per_gpu.items()):
        busy = sum(t for t, _ in kernels.values())
        steps = kernels.get("embed_kernel", [0, 0])[1]
        print(f"{gpu}: busy {busy / 1e9:.2f} s, {steps} steps"
              + (f", {busy / 1e6 / steps:.1f} ms per step" if steps else ""))
        print(f"  {'share':>6} {'total s':>8} {'calls':>7} {'us/call':>9}  kernel")
        for name, (t, n) in sorted(kernels.items(), key=lambda x: -x[1][0])[:top]:
            print(f"  {100 * t / busy:5.1f}% {t / 1e9:8.3f} {n:7d} {t / n / 1e3:9.1f}  {name[:80]}")


if __name__ == "__main__":
    main()
