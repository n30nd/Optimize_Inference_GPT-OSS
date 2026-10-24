#!/usr/bin/env python3
"""Yardstick for decode attention: PyTorch's scaled_dot_product_attention on
the shapes of gpt-oss (64 query heads, 8 key/value heads, head_dim 64, one new
token per sequence). Never part of ./run; it only says how fast a library is.

    tools/gpulock.sh <gpu> -- .venv/bin/python getp-csrc/bench/attn_torch_yardstick.py

PyTorch has no attention sink, and it wants the cache as (B, heads, n, 64)
(head-major), so this is the same amount of reading with a friendlier layout.
GB/s counts the K and V bytes only, like attn_bench.cpp.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def bench(fn, calls=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(calls):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / calls  # us per call


def main():
    dev = "cuda"
    print(torch.__version__, torch.cuda.get_device_name())
    print(f"{'backend':<22} {'layer':<6} {'B':>5} {'pos':>5} {'n':>5} {'us/call':>10} {'GB/s':>8}")
    for B in (512, 1024):
        kf = torch.randn(B, 8, 1024, 64, device=dev, dtype=torch.bfloat16)
        vf = torch.randn(B, 8, 1024, 64, device=dev, dtype=torch.bfloat16)
        q = torch.randn(B, 64, 1, 64, device=dev, dtype=torch.bfloat16)
        for window in (128, 0):
            for pos in (64, 256, 512, 1023):
                n = min(pos + 1, window) if window else pos + 1
                k, v = kf[:, :, :n], vf[:, :, :n]
                nbytes = B * n * 8 * 64 * 2 * 2
                for name, be in (("sdpa math", SDPBackend.MATH),
                                 ("sdpa efficient", SDPBackend.EFFICIENT_ATTENTION),
                                 ("sdpa flash", SDPBackend.FLASH_ATTENTION)):
                    try:
                        with sdpa_kernel(be):
                            us = bench(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
                    except Exception as e:  # backend not available for this shape / GPU
                        print(f"{name:<22} {'window' if window else 'full':<6} {B:>5} {pos:>5} {n:>5}"
                              f" {'n/a':>10}   ({type(e).__name__})")
                        continue
                    print(f"{name:<22} {'window' if window else 'full':<6} {B:>5} {pos:>5} {n:>5}"
                          f" {us:>10.1f} {nbytes / (us * 1e-6) / 1e9:>8.0f}")
        del kf, vf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
