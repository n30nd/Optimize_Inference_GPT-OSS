# gpt-oss on 8× AMD MI250: results

Batch generation (`-m getp`) for gpt-oss-20b and gpt-oss-120b on one machine with 4 AMD MI250
cards (8 GCDs). Every GPU kernel is written from scratch in HIP; the program uses no GPU libraries.

| | gpt-oss-20b | gpt-oss-120b |
|---|---:|---:|
| Given CPU program | 3.3 tok/s | 0.40 tok/s |
| First GPU version (bit-identical to the CPU) | 3,348 tok/s | 1,188 tok/s |
| **Now** | **~80,000 tok/s** | **~45,500 tok/s** |
| METEOR / BERTScore (must be ≥ 0.30 / ≥ 0.90) | 0.533 / 0.978 | 0.552 / 0.981 |

- Input: `evaluation/input.txt` (4,096 prompts, answers up to 1,024 tokens), all 8 GCDs.
- 20b: a full model copy on each GCD, 512 requests each. 120b: each layer's experts are split
  between the two GCDs of one card, 4 such pairs.
- Yardstick: vLLM runs 20b at 6,539 tok/s on one GCD; this program does about 10,000 per GCD.
- With the input repeated twice (8,192 prompts, 1,024 per GCD), 20b reaches 84,268 tok/s.

**Slides:** [docs/slides.md](docs/slides.md) — the problem, each technique (what it fixes, whether it
is used here, what it gave) and the results, written for newcomers. The Marp extension shows them
as a deck; any Markdown viewer shows the pages one after another.

**Technique docs:** [docs/](docs/) — one beginner-level document per technique, 01 to 12.

Run with the Hugging Face model folder as the model path:

```bash
make runomp
./run /path/to/gpt-oss-20b -m getp -i evaluation/input.txt -o output.txt
python3 tools/bench.py -m /path/to/gpt-oss-20b -i evaluation/input.txt -g all -t mytag --score
```

# Environment Setup

```bash
# gpt-oss
export GPT_OSS_REPO_ROOT="/nfs/gpu_trainee/user/gpt-oss" # TODO(user): change to your account name
cd $GPT_OSS_REPO_ROOT

# Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Model paths
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
```

# Main program

The main function at `run.cpp`
Do not modify `run.cpp` `getp_csrc/getp_eval.cpp` `Makefile`

## Build

```bash
make run  # Default compilation, very slow
make runfast  # Compiled with -O3 optimization
make runomp # Compiled with -O3 and -fopenmp
```

## Run

Example:

```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/input.txt -o data/output.txt
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m chat
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m generate -i "1+1="
```

## Visualize `getp` mode output

For `getp` mode the output file contains list of output tokens index of each requests. To convert those indexes into text, you should build and run `decode.cpp`

```bash
make decode
./decode -1 -i data/output.txt
```

# Tokenizer

## Export Tokenizer

```bash
make tokenizer-bin
```

## Build & Run Tokenizer Test (C++)

```bash
make tokenizer-test
./test_tokenizer -t tokenizer.bin -i "Hello world"
# Expected output: 13225 2375
```

## Verify Compatibility with Tiktoken

```bash
python3 test_tokenizer.py \
  --bin ./test_tokenizer \
  --tok ./tokenizer.bin \
  --verbose \
  --prompt data/input.txt
```

### Example Results

```
PROMPT: 'ฉันรักทะเล'
  C  encoded: [97797, 6560, 151737, 37899, 17758]
  PY encoded: [97797, 6560, 151737, 37899, 17758]
  C  decoded: 'ฉันรักทะเล'
  PY decoded: 'ฉันรักทะเล'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: 'naïve façade — déjà vu'
  C  encoded: [1503, 9954, 737, 114665, 2733, 21229, 12005]
  PY encoded: [1503, 9954, 737, 114665, 2733, 21229, 12005]
  C  decoded: 'naïve façade — déjà vu'
  PY decoded: 'naïve façade — déjà vu'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: '🍣 sushi and 🍜 ramen'
  C  encoded: [102415, 96, 85535, 326, 197348, 250, 90938]
  PY encoded: [102415, 96, 85535, 326, 197348, 250, 90938]
  C  decoded: '🍣 sushi and 🍜 ramen'
  PY decoded: '🍣 sushi and 🍜 ramen'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: 'email: test@example.com'
  C  encoded: [4261, 25, 1746, 81309, 1136]
  PY encoded: [4261, 25, 1746, 81309, 1136]
  C  decoded: 'email: test@example.com'
  PY decoded: 'email: test@example.com'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: 'newlines:'
  C  encoded: [1389, 10105, 25]
  PY encoded: [1389, 10105, 25]
  C  decoded: 'newlines:'
  PY decoded: 'newlines:'
  [ENCODE MATCH] [DECODE MATCH]
```
