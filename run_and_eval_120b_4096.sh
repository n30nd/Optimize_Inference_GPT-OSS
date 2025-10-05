#!/bin/bash
set -e

./run "${MODELBIN_ROOT}/gpt-oss-120b.bin" -m getp -i data/INPUT4096.txt -o data/OUTPUT4096.txt
cp data/OUTPUT4096.txt evaluation/submission/output_120b_token_ids.txt
cp evaluation/references/output_120b_token_ids_4096.txt evaluation/references/output_120b_token_ids.txt
cp evaluation/input4096.txt evaluation/input.txt

cd evaluation
python eval.py -m 120b
#How to run: source .venv/bin/activate -> srun --gres=gpu:8 ./run_and_eval_120b_4096.sh