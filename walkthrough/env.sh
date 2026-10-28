# Run once in every new terminal:  source walkthrough/env.sh
source /home/namvu/gpt-oss/.venv/bin/activate      # python with tiktoken, torch, bert-score
export M20=/remote/vast0/share-mv/models/openai/gpt-oss-20b
export M120=/remote/vast0/share-mv/models/openai/gpt-oss-120b
export MODELS_ROOT=/home/namvu/gptoss-models         # holds bs_model/roberta-large-mnli for scoring
export W=/home/namvu/walk-runs                       # run inputs and outputs, kept out of the repo
mkdir -p $W
[ -f $W/in64.txt ] || (echo 64; sed -n 2,65p /home/namvu/gpt-oss-walkthrough/evaluation/input.txt) > $W/in64.txt
