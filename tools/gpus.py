#!/usr/bin/env python3
"""Shows every GPU by HIP index (the number HIP_VISIBLE_DEVICES uses) with its
rocm-smi index, PCI bus, busy % and memory %, and which ones are free.

    tools/gpus.py              # table
    tools/gpus.py --free 2     # print 2 free HIP indices, e.g. "4,5" (exit 1 if not enough)
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
HELPER = HERE / "hip_devices"
BUSY_MAX = 10  # percent
VRAM_MAX = 5  # percent


def hip_buses():
    if not HELPER.exists() or HELPER.stat().st_mtime < (HERE / "hip_devices.cpp").stat().st_mtime:
        subprocess.run(["hipcc", "-O2", "-o", str(HELPER), str(HERE / "hip_devices.cpp")], check=True)
    env = dict(os.environ)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    out = subprocess.run([str(HELPER)], capture_output=True, text=True, check=True, env=env).stdout
    buses = {}
    for line in out.strip().splitlines():
        idx, bus = line.split()
        buses[int(idx)] = bus.lower()
    return buses


def smi_cards():
    out = subprocess.run(["rocm-smi", "--showbus", "--showuse", "--showmemuse", "--json"],
                         capture_output=True, text=True, check=True).stdout
    cards = {}
    for name, d in json.loads(out).items():
        if not name.startswith("card"):
            continue
        cards[d["PCI Bus"].lower()] = {
            "smi": int(name[4:]),
            "busy": int(d.get("GPU use (%)", 0)),
            "vram": int(d.get("GPU Memory Allocated (VRAM%)", 0)),
        }
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", type=int, metavar="N", help="print N free HIP indices, comma-separated")
    args = ap.parse_args()

    buses, cards = hip_buses(), smi_cards()
    rows = []
    for idx in sorted(buses):
        c = cards.get(buses[idx], {"smi": -1, "busy": 100, "vram": 100})
        free = c["busy"] <= BUSY_MAX and c["vram"] <= VRAM_MAX
        rows.append((idx, c["smi"], buses[idx], c["busy"], c["vram"], free))

    if args.free is not None:
        ids = [str(r[0]) for r in rows if r[5]]
        if len(ids) < args.free:
            print(f"only {len(ids)} free GPU(s): {','.join(ids)}", file=sys.stderr)
            sys.exit(1)
        print(",".join(ids[: args.free]))
        return

    print(f"{'HIP':>3} {'rocm-smi':>8} {'PCI bus':>14} {'busy%':>6} {'VRAM%':>6}  state")
    for idx, smi, bus, busy, vram, free in rows:
        print(f"{idx:>3} {smi:>8} {bus:>14} {busy:>6} {vram:>6}  {'free' if free else 'IN USE'}")


if __name__ == "__main__":
    main()
