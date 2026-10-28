#!/usr/bin/env python3
"""Scores an output file like evaluation/eval.py (METEOR, BERTScore F1, the same
thresholds), but runs BERTScore on every visible GPU at once.

    tools/gpulock.sh 0,1,2,3 -- tools/score.py -s output.txt \
        -r evaluation/references/output_20b_token_ids.txt

- BERTScore is cut into eval.py's chunks of 128 lines; each GPU takes the next
  chunk until none are left, and the chunk sums are added in eval.py's order,
  so the scores are the ones eval.py prints.
- METEOR runs on the CPU while the GPUs work.
- Prints "meteor\t..." and "bertscore_f1\t..." like eval.py and exits 1 if a
  score is below evaluation/threshold.json. Needs MODELS_ROOT to hold
  bs_model/roberta-large-mnli.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("course_eval", REPO / "evaluation/eval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["course_eval"] = ev  # its METEOR workers find their function by module name
_spec.loader.exec_module(ev)

CHUNK = 128  # eval.py's chunk_size
BATCH = 64  # eval.py's bs_batch

_scorer = None


def _init(devices, model):
    global _scorer
    import torch
    dev = devices.get()
    torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    _scorer = ev.BERTScorer(lang="en", model_type=model, num_layers=ev._infer_num_layers(model),
                            device=f"cuda:{dev}")


def _chunk_f1_sum(job):
    i, cands, refs = job
    _, _, f1 = _scorer.score(cands, refs, batch_size=BATCH)
    return i, float(f1.sum().item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--submission", type=Path, required=True)
    ap.add_argument("-r", "--references", type=Path, required=True)
    ap.add_argument("-e", "--encoding", default="o200k_harmony")
    args = ap.parse_args()
    t0 = time.time()

    # Starting the GPU runtime takes ~13 s in every process, so the main process
    # counts the GPUs without it, and the workers start before the files are read.
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible:
        n_gpu = len(visible.split(","))
    else:
        import torch
        n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        sys.exit("score.py: no GPU visible")
    model = str(ev.ensure_bs_model(Path(os.environ["MODELS_ROOT"]) / "bs_model" / "roberta-large-mnli"))

    ctx = mp.get_context("spawn")
    devices = ctx.Queue()
    for d in range(n_gpu):
        devices.put(d)
    with ctx.Pool(n_gpu, _init, (devices, model)) as pool:
        enc = ev.tiktoken.get_encoding(args.encoding)
        subm = ev.read_completions(args.submission, enc)
        refs = ev.read_completions(args.references, enc)
        n = min(len(refs), len(subm))
        jobs = [(i, subm[i:i + CHUNK], refs[i:i + CHUNK]) for i in range(0, n, CHUNK)]
        pending = pool.map_async(_chunk_f1_sum, jobs, chunksize=1)
        _, meteor = ev.compute_meteor(refs, subm)  # CPU, while the GPUs work
        sums = dict(pending.get())
    bsf1 = sum(sums[i] for i, _, _ in jobs) / max(1, n)

    print(f"items\t{n}")
    print(f"meteor\t{round(meteor, 6)}")
    print(f"bertscore_f1\t{round(bsf1, 6)}")
    print(f"gpus\t{n_gpu}")
    print(f"seconds\t{time.time() - t0:.1f}")
    thresholds = json.loads((REPO / "evaluation/threshold.json").read_text())
    passed = meteor >= thresholds["meteor"] and bsf1 >= thresholds["bertscore_f1"]
    if not passed:
        print(f"score.py: below threshold {thresholds}", file=sys.stderr)
    # Importing torch starts the GPU runtime in the background, and a normal exit
    # waits ~14 s for it. Release the pool's locks, then leave at once.
    mp.util._exit_function()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
