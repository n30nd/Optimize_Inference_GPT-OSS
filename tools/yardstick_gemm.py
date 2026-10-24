"""Yardstick: how fast AMD's libraries (rocBLAS / hipBLASLt via torch.matmul)
run the matrix multiplies of gpt-oss on one MI250 GCD, plus achievable HBM
bandwidth. Our own kernels are compared against these numbers.

Run (repo venv, one free GPU):
  tools/gpulock.sh <hip id> -- python tools/yardstick_gemm.py [out.json]
"""

import json
import sys

import torch

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


def bandwidth():
    n = 2 * 1024**3 // 2  # 2 GiB of bf16
    a = torch.empty(n, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    us = timeit(lambda: b.copy_(a), iters=10)
    return 2 * a.numel() * 2 / (us * 1e-6) / 1e9  # read + write, GB/s


# (name, N = output features, K = input features)
SHAPES = [
    ("qkv", 5120, 2880),
    ("attn_out", 2880, 4096),
    ("router_20b", 32, 2880),
    ("router_120b", 128, 2880),
    ("expert_mlp1", 5760, 2880),
    ("expert_mlp2", 2880, 2880),
    ("lm_head", 201088, 2880),
]
TOKENS = [1, 16, 64, 128, 256, 512, 1024, 2048]


def gemm_table():
    rows = []
    for name, N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
        for M in TOKENS:
            x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            us = timeit(lambda: torch.matmul(x, w.t()))
            tflops = 2 * M * N * K / (us * 1e-6) / 1e12
            gbs = (N * K + M * K + M * N) * 2 / (us * 1e-6) / 1e9
            rows.append({"op": name, "M": M, "N": N, "K": K, "us": round(us, 1),
                         "tflops": round(tflops, 1), "GB/s": round(gbs)})
            print(f"{name:>12} M={M:>5} N={N:>6} K={K:>5}  {us:9.1f} us  {tflops:6.1f} TFLOP/s  {gbs:6.0f} GB/s", flush=True)
    return rows


def decode_step_estimate(rows, model, n_layers, n_experts, batch):
    """Time of one decode step's matrix multiplies at `batch` sequences on one
    GCD if every GEMM ran at library speed (experts get batch*4/n_experts rows
    each, assuming even routing)."""
    def t(op, M):
        # nearest measured M at or above the request, scaled linearly when larger
        cands = [r for r in rows if r["op"] == op]
        best = min((r for r in cands if r["M"] >= M), key=lambda r: r["M"], default=cands[-1])
        return best["us"] * max(1.0, M / best["M"])
    per_expert = max(1, batch * 4 // n_experts)
    router = "router_20b" if n_experts == 32 else "router_120b"
    layer = (t("qkv", batch) + t("attn_out", batch) + t(router, batch)
             + n_experts * (t("expert_mlp1", per_expert) + t("expert_mlp2", per_expert)))
    step = n_layers * layer + t("lm_head", batch)
    return {"model": model, "batch": batch, "rows_per_expert": per_expert,
            "step_ms": round(step / 1e3, 2), "tok_per_s_per_gcd": round(batch / (step * 1e-6))}


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}, {props.total_memory / 1e9:.1f} GB, torch {torch.__version__}, hip {torch.version.hip}")
    bw = bandwidth()
    print(f"HBM copy bandwidth (read+write): {bw:.0f} GB/s")
    rows = gemm_table()
    est = []
    for batch in (128, 256, 512, 1024, 2048):
        est.append(decode_step_estimate(rows, "20b", 24, 32, batch))
        est.append(decode_step_estimate(rows, "120b", 36, 128, batch))
    for e in est:
        print(e)
    json.dump({"bandwidth_GBs": bw, "gemm": rows, "decode_estimate": est},
              open(sys.argv[1] if len(sys.argv) > 1 else "yardstick_gemm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
