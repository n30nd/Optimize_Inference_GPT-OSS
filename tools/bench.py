#!/usr/bin/env python3
"""Runs ./run in getp mode once and records the result.

    tools/bench.py -m /remote/vast0/share-mv/models/openai/gpt-oss-20b \
        -i evaluation/input.txt -g all -t baseline --score

- Builds ./run with `make runomp` first (skip with --no-build).
- Waits for the GPUs through tools/gpulock.sh.
- Writes <results>/<time>_<tag>/{run.log,output.txt}, appends one JSON line to
  <results>/results.jsonl and prints a summary. <results> is $GPTOSS_RESULTS,
  or ./results.
- hash = first 12 hex digits of sha256(output file). Same hash means the same
  generated tokens.
- --score compares the output with the reference answers using
  evaluation/eval.py (METEOR, BERTScore F1) on one free GPU. Only meaningful
  when the input is evaluation/input.txt. Needs MODELS_ROOT to hold
  bs_model/roberta-large-mnli.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
REFERENCES = {
    "20b": REPO / "evaluation/references/output_20b_token_ids.txt",
    "120b": REPO / "evaluation/references/output_120b_token_ids.txt",
}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def model_size(model: str) -> str:
    return "120b" if "120b" in model else "20b"


def build():
    env = dict(os.environ)
    venv = REPO / ".venv" / "bin"
    if venv.exists():
        env["PATH"] = f"{venv}:{env['PATH']}"
    subprocess.run(["make", "runomp"], cwd=REPO, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_log(text: str) -> dict:
    out = {}
    m = re.search(r"warm up elapsed time\(s\): ([0-9.]+)", text)
    if m:
        out["warmup_s"] = float(m.group(1))
    m = re.search(r"\nelapsed time\(s\): ([0-9.]+), achieved throughput TPS \(tok/s\): ([0-9.]+)", text)
    if m:
        out["inference_s"] = float(m.group(1))
        out["tps"] = float(m.group(2))
    return out


def score(output: Path, reference: Path, run_dir: Path) -> dict:
    gpu = sh([str(TOOLS / "gpus.py"), "--free", "1"])
    cmd = [str(TOOLS / "gpulock.sh"), gpu, "--", sys.executable,
           str(REPO / "evaluation/eval.py"), "-s", str(output), "-r", str(reference)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (run_dir / "score.log").write_text(proc.stdout + proc.stderr)
    res = {"score_passed": proc.returncode == 0}
    for key in ("meteor", "bertscore_f1"):
        found = re.findall(rf"^{key}\t([0-9.]+)", proc.stdout, re.M)
        if found:
            res[key] = float(found[-1])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="model folder or .bin")
    ap.add_argument("-i", "--input", required=True, help="getp input file")
    ap.add_argument("-g", "--gpus", default="all", help="HIP indices for gpulock.sh, or 'all'")
    ap.add_argument("-n", "--steps", type=int, help="pass -n to ./run")
    ap.add_argument("-t", "--tag", default="run")
    ap.add_argument("--score", action="store_true", help="score against the reference answers")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    results = Path(os.environ.get("GPTOSS_RESULTS", "results")).resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results / f"{stamp}_{args.tag}"
    run_dir.mkdir(parents=True)

    if not args.no_build:
        build()
    commit = sh(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO) != ""

    output = run_dir / "output.txt"
    cmd = [str(TOOLS / "gpulock.sh"), args.gpus, "--", str(REPO / "run"), args.model,
           "-m", "getp", "-i", str(Path(args.input).resolve()), "-o", str(output)]
    if args.steps:
        cmd += ["-n", str(args.steps)]
    with open(run_dir / "run.log", "w") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    log_text = (run_dir / "run.log").read_text(errors="replace")

    rec = {
        "time": stamp, "host": socket.gethostname(), "tag": args.tag,
        "commit": commit + ("+dirty" if dirty else ""), "model": args.model,
        "input": args.input, "requests": int(open(args.input).readline()),
        "steps": args.steps, "gpus": args.gpus, "exit_code": proc.returncode,
    }
    rec.update(parse_log(log_text))
    if output.exists():
        rec["hash"] = hashlib.sha256(output.read_bytes()).hexdigest()[:12]
        rec["generated_tokens"] = sum(len(line.split()) for line in open(output))
    if args.score and output.exists() and proc.returncode == 0:
        rec.update(score(output, REFERENCES[model_size(args.model)], run_dir))

    with open(results / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    (run_dir / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    for k, v in rec.items():
        print(f"{k:>17}: {v}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
