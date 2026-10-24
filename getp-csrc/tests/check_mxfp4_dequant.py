"""Compares our C++ MXFP4 decoding with the course exporter's dequantize_fp4().

Usage (repo root, venv active):
    ./cpu_forward_equiv <model folder> --dump-expert L E /tmp/expert.f32
    python getp-csrc/tests/check_mxfp4_dequant.py <model folder> L E /tmp/expert.f32

Also reports the range of scale bytes over every expert tensor, since
scales of 255 (2^128) would overflow fp32.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open
import torch


def load_exporter(repo: Path):
    path = repo / "export_model_bin" / "gpt-oss-20b" / "export_model_bin.py"
    spec = importlib.util.spec_from_file_location("exporter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weights_dir(model: Path) -> Path:
    return model / "original" if (model / "original" / "config.json").exists() else model


def find_file(wdir: Path, key: str) -> Path:
    for f in sorted(wdir.glob("*.safetensors")):
        with safe_open(f, framework="pt") as st:
            if key in st.keys():
                return f
    raise KeyError(key)


def main():
    model, layer, expert, dump = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    repo = Path(__file__).resolve().parents[2]
    exporter = load_exporter(repo)
    wdir = weights_dir(model)

    key = f"block.{layer}.mlp.mlp1_weight"
    f = find_file(wdir, key + ".blocks")
    with safe_open(f, framework="pt") as st:
        blocks = st.get_slice(key + ".blocks")[expert]
        scales = st.get_slice(key + ".scales")[expert]
    ref = exporter.dequantize_fp4(blocks, scales, dtype=torch.bfloat16).to(torch.float32).numpy()
    ours = np.fromfile(dump, dtype=np.float32).reshape(ref.shape)
    same = np.array_equal(ref.view(np.uint32), ours.view(np.uint32))
    print(f"{key} expert {expert}: shape {ref.shape}, bit-identical: {same}")
    if not same:
        bad = np.count_nonzero(ref.view(np.uint32) != ours.view(np.uint32))
        print(f"  differing values: {bad}")

    lo, hi = 255, 0
    for f in sorted(wdir.glob("*.safetensors")):
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                if k.endswith(".scales"):
                    t = st.get_tensor(k)
                    lo, hi = min(lo, int(t.min())), max(hi, int(t.max()))
    print(f"scale bytes over all expert tensors: min {lo} (2^{lo - 127}), max {hi} (2^{hi - 127})")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
