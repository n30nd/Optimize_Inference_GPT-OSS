#!/usr/bin/env python3
"""Yardstick for the expert matrix multiply: the vendor library (hipBLASLt via
PyTorch, bf16 weights) running one matrix multiply per expert on the same real
routing as getp-csrc/bench/moe_gemm_bench.cpp. Not part of ./run.

    python3 getp-csrc/bench/moe_torch_lib.py <dump B=512> <dump B=1024> <dump B=2048> [iters=20]

A dump file holds lines "B pos layer offsets[0] .. offsets[32]". For each
configuration it prints the mean time of one grouped call (all experts, each
`torch.addmm(bias, A_e, W_e^T)`, captured in a graph), over `iters` calls that
cycle through the dumped layers. The library gets bf16 weights (4x the bytes of MXFP4) and
writes bf16 output (half the bytes of our fp32 output).
"""

import sys

import numpy as np
import torch


def read_dump(path):
    sets = []
    for line in open(path):
        v = list(map(int, line.split()))
        sets.append(v[3:])
    return sets


def synth(E, B, seed, shape=0.45, E_all=None):
    """B tokens pick 4 of E_all experts; offsets of the first E experts."""
    E_all = E_all or E
    rng = np.random.default_rng(seed)
    p = rng.gamma(shape, 1.0, E_all)
    c = np.zeros(E_all, dtype=int)
    for _ in range(B):
        q = p.copy()
        for _ in range(4):
            e = rng.choice(E_all, p=q / q.sum())
            c[e] += 1
            q[e] = 0
    return [0] + list(np.cumsum(c[:E]))


def main():
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    dev = torch.device("cuda")
    K = 2880
    cfgs = []
    for i, B in enumerate((512, 1024, 2048)):
        cfgs.append(("20b", 32, B, read_dump(sys.argv[1 + i])))
    for i, B in enumerate((512, 1024, 2048)):
        # one GCD of gpt-oss-120b: experts 0..63 of 128, rows of both GCDs' B tokens
        cfgs.append(("120b", 64, B, [synth(64, 2 * B, 1000 * i + s, E_all=128) for s in range(8)]))
    print(f"{'model':5} {'shape':5} {'B':>5} | {'library us':>10}")
    for model, E, B, sets in cfgs:
        for name, N in (("mlp1", 5760), ("mlp2", 2880)):
            R = max(o[E] for o in sets)
            W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16)
            bias = torch.randn(E, N, device=dev, dtype=torch.bfloat16)
            A = torch.randn(R, K, device=dev, dtype=torch.bfloat16)
            C = torch.empty(R, N, device=dev, dtype=torch.bfloat16)

            def call(off):
                for e in range(E):
                    a, b = off[e], off[e + 1]
                    if b > a:
                        torch.addmm(bias[e], A[a:b], W[e].t(), out=C[a:b])

            # One CUDA/HIP graph per routing set, so Python launch overhead
            # does not count against the library.
            n = min(len(sets), iters)
            graphs = []
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(n):
                    call(sets[i])  # warm-up outside the capture
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            for i in range(n):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    call(sets[i])
                graphs.append(g)
            for g in graphs[:3]:
                g.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(iters):
                graphs[i % n].replay()
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000 / iters
            print(f"{model:5} {name:5} {B:5d} | {us:10.1f}", flush=True)
            del W, bias, A, C, graphs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
