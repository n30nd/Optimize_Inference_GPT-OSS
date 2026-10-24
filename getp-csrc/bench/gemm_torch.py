"""Library yardstick for bench/gemm_bench.cpp: the same shapes through
torch (hipBLASLt / rocBLAS), on the same GCD, timed the same way (warm-up,
then the mean of --iters calls between two events). Never part of ./run.

  bf16 out : torch.matmul(x, w.t()), bf16 result (what the yardstick JSON has)
  fp32 out : torch.ops.aten.mm.dtype(x, w.t(), torch.float32), fp32 result,
             i.e. the same output format as our gemm_bf16()

Run on one GCD (repo venv):
  tools/gpulock.sh $(tools/gpus.py --free 1) -- python getp-csrc/bench/gemm_torch.py
"""

import argparse

import torch

SHAPES = [("qkv", 5120, 2880), ("attn_out", 2880, 4096), ("router_20b", 32, 2880),
          ("router_120b", 128, 2880), ("lm_head", 201088, 2880)]


def timeit(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ms", default="64,128,256,512,1024,2048")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ms = [int(m) for m in args.ms.split(",")]
    print(f"{'shape':<12} {'M':>5} {'N':>7} {'K':>5}  {'bf16 out us':>11} {'fp32 out us':>11}")
    for name, N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.05
        for M in ms:
            x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            t16 = timeit(lambda: torch.matmul(x, w.t()), args.iters)
            try:
                t32 = timeit(lambda: torch.ops.aten.mm.dtype(x, w.t(), torch.float32), args.iters)
            except RuntimeError:
                t32 = float("nan")
            print(f"{name:<12} {M:>5} {N:>7} {K:>5}  {t16:>11.1f} {t32:>11.1f}", flush=True)
        del w


if __name__ == "__main__":
    main()
