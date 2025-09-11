export export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
uv pip install --upgrade evaluate bert-score nltk pandas tiktoken jsonschema

RUN COMMAND: 
cd evaluation
srun --gres=gpu:8 python eval.py -m 20b
