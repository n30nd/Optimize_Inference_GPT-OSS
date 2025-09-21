#!/bin/bash
set -e

GETP_MOE_BLOCK=16 GETP_MAX_BATCH=512 ./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/INPUT4096.txt -o data/OUTPUT4096.txt
cp data/OUTPUT4096.txt evaluation/submission/output_20b_token_ids.txt
cd evaluation
python eval.py -m 20b
#How to run: source .venv/bin/activate -> srun --gres=gpu:8 ./run_and_eval.sh