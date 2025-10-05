#!/bin/bash
set -e

./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/INPUT5120.txt -o data/OUTPUT5120.txt
cp data/OUTPUT5120.txt evaluation/submission/output_20b_token_ids.txt
cp evaluation/references/output_20b_token_ids_5120.txt evaluation/references/output_20b_token_ids.txt
cp data/INPUT5120.txt evaluation/input.txt

cd evaluation
python eval.py -m 20b
#How to run: source .venv/bin/activate -> srun --gres=gpu:8 ./run_and_eval_20b_5120.sh