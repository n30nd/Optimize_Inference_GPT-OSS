# Lesson 01: Read the model straight from its download folder

| | |
|---|---|
| Reference commit | `f0c4dbd` "Load gpt-oss weights straight from the Hugging Face folder" |
| Starts from | `865fca5` (lesson 00) |
| To type | 1,091 lines of program code in 6 files (432 of them are file-reading plumbing); 166 lines of tests (optional) |
| Shortcut | `git cherry-pick f0c4dbd` — copies the finished commit instead of typing (section 5) |
| Speed after | 20b 3.3 tok/s, 120b 0.40 tok/s — still the CPU, one prompt at a time |

**In plain words:** the program could only read the model from one special file that does not
exist on this machine. We teach it to read the model the way it is downloaded, without changing
the files we are not allowed to touch. After this lesson the slow starting program runs, and we
can measure it.

## 1. Goal

`./run <model folder> …` works in every mode (`getp`, `generate`, `chat`) with the Hugging Face
folder, and gives exactly the numbers the course's `.bin` route would give.

## 2. Why this, why now

- The `.bin` files of the course are not on this machine (lesson 00, section 5).
- Making one ourselves means expanding every weight to fp32: 84 GB for 20b, 467 GB for 120b.
  The home disk had 99 GB free.
- Nothing else can start before the program runs: we need the starting speed and a correct
  CPU result to compare the GPU against.

## 3. The technique

New words:
- **Hugging Face folder** — the model as downloaded: a `config.json` with the sizes, and the
  weights in `.safetensors` files. Here we read the `original/` subfolder OpenAI ships.
- **safetensors** — a simple file format: 8 bytes giving the header length, a JSON header
  listing each tensor's name, number type, shape and byte range, then the raw bytes.
- **tensor** — a block of numbers with a shape, like a table (2-D) or a stack of tables.
- **memory-mapping (mmap)** — asking the operating system to make a file look like memory. The
  program reads the file as if it were an array; only the parts touched are read from disk.
- **fp32 / bf16** — two ways to store a number with a decimal point. fp32 uses 32 bits; bf16
  uses 16 bits: the top half of an fp32, so converting bf16 → fp32 is "append 16 zero bits".
- **MXFP4** — a 4-bit way to store the expert weights. Each value is one of 16 numbers
  (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6); two values share one byte (low half first), and each
  group of 32 values shares one scale byte `s` meaning "multiply by 2^(s − 127)".
- **logits** — the score the model gives every possible next token; the highest wins.

The trick that keeps `run.cpp` untouched: `run.cpp` includes `getp_run.cpp` *before* `main()`.
So a `#define` at the end of `getp_run.cpp` renames the functions `main()` calls:

```
run.cpp                                 getp-csrc/getp_run.cpp (included above main)
───────                                 ────────────────────────────────────────────
main() {                                #define build_transformer gptoss_build_transformer
  build_transformer(&t, path);   ──►    #define generate          gptoss_generate
  …                                     …
  generate(…);                   ──►    gptoss_*() in engine/cli_modes.hpp:
}                                         folder? → read Hugging Face files
                                          file?   → the old .bin code, unchanged
```

How the weights end up in memory:

```
original/*.safetensors (memory-mapped, nothing copied yet)
 ├─ small tensors (bf16): attention, router, norms, embedding
 │     └─ converted once to fp32, into run.cpp's own TransformerWeights layout
 └─ expert tensors (MXFP4, ~95 % of the bytes)
       └─ left in the file; decoded value by value inside the expert multiply
```

Decoding MXFP4 to fp32 gives exactly the value the course's exporter writes, and the multiply
adds the products in the same order as `run.cpp`'s `matmul()`. So the logits are the same bits.

## 4. The change, file by file (typing order)

| # | File | Lines | What it is for | Type it? |
|---|---|---:|---|---|
| 1 | `getp-csrc/engine/json.hpp` | 260 | a small JSON reader for `config.json` and the safetensors header | optional (plumbing) |
| 2 | `getp-csrc/engine/safetensors.hpp` | 172 | opens and memory-maps every `.safetensors` file; `st_require()` finds a tensor by name and checks its type and shape | optional (plumbing) |
| 3 | `getp-csrc/engine/hf_model.hpp` | 281 | reads `config.json`; `bf16_to_f32`; the MXFP4 table `kFp4Values` and `mxfp4_to_f32`; `hf_build_transformer()` fills `run.cpp`'s structures | yes |
| 4 | `getp-csrc/engine/cpu_forward.hpp` | 208 | `matmul_mxfp4()` (the expert multiply reading 4-bit weights) and `hf_cpu_forward()` (`forward()` with that multiply) | yes |
| 5 | `getp-csrc/engine/cli_modes.hpp` | 159 | `model_forward()` and the `gptoss_*` functions `main()` now reaches | yes |
| 6 | `getp-csrc/getp_run.cpp` | +11 −1 | include `cli_modes.hpp`, call `model_forward()`, the four `#define`s | yes |
| 7 | `getp-csrc/tests/cpu_forward_equiv.cpp` | 95 | checks our CPU forward against `run.cpp`'s on fully expanded fp32 weights | optional |
| 8 | `getp-csrc/tests/check_mxfp4_dequant.py` | 71 | checks our MXFP4 decoding against the course exporter's | optional |

Read while typing: `hf_model.hpp` lines 23–27 (the 16 MXFP4 values) and `mxfp4_to_f32()`;
`cpu_forward.hpp` `matmul_mxfp4()` next to `run.cpp:355` `matmul()` — the loops are the same,
only the weight read differs.

## 5. Type, compare, build, check

**Shortcut — skip the typing.** Take the finished commit as it is:

```bash
cd /home/namvu/gpt-oss-walkthrough
git status --short -- . ':!walkthrough'   # must print nothing (no half-typed files)
git cherry-pick f0c4dbd
```

`git cherry-pick` copies a commit from the finished branch onto yours, with the same name, so
there is nothing to type, compare or save. Go straight to `make runomp` and the checks below.
If you already typed part of it and want to stop there, take the finished version of every file
of the commit over yours, then save as usual:
`git checkout f0c4dbd -- $(git show --name-only --format= f0c4dbd)`

**Or type it:**

```bash
cd /home/namvu/gpt-oss-walkthrough
git show --stat f0c4dbd
git show f0c4dbd -- getp-csrc/engine/hf_model.hpp          # the change to copy, file by file
git checkout f0c4dbd -- getp-csrc/engine/json.hpp          # only for files you choose not to type

git diff f0c4dbd -- $(git show --name-only --format= f0c4dbd)   # must print nothing
make runomp
```

Check 1 — it talks (CPU, 20b, about 10 s; it should end with "Paris"):

```bash
./run $M20 -m generate -i "The capital of France is" -n 24
```

Check 2 — `getp` gives the reference answers (3 prompts, 48 positions):

```bash
(echo 3; sed -n 2,4p evaluation/input.txt) > $W/in3.txt
./run $M20 -m getp -i $W/in3.txt -o $W/out3.txt -n 48
python3 -c "
ref=[l.split() for l in open('evaluation/references/output_20b_token_ids.txt')][:3]
for i,(o,r) in enumerate(zip([l.split() for l in open('$W/out3.txt')],ref)):
    n=0
    while n<len(o) and o[n]==r[n]: n+=1
    print(f'prompt {i}: {len(o)} tokens, first {n} equal to the reference')"
```

Expected: every token written is equal to the reference (38, 37 and 34 tokens), and
`achieved throughput TPS` between 3.3 and 3.8 (it uses all CPU cores, so other work on the
machine moves it). The run takes about 30 s.

Check 3 (optional) — bit-identical to `run.cpp`: build and run `tests/cpu_forward_equiv.cpp`
with the commands in its first lines (`./cpu_forward_equiv $M20 8`, about 20 s). Expected last
line: `PASS: bit-identical logits`.

Save: `git add -A -- . ':!walkthrough' && git commit -m "Load gpt-oss weights straight from the Hugging Face folder"`

## 6. Result

| | before | after |
|---|---|---|
| Can it run here? | no | yes, every mode |
| 20b, CPU, 3 prompts, `-n 48` | — | 3.3 tok/s; 38 / 37 / 34 tokens = reference |
| 120b, CPU, 3 prompts, `-n 40` | — | 0.40 tok/s; 30 / 29 / 26 tokens = reference |
| Logits vs `run.cpp` `forward()` (20b, 8 steps) | — | bit-identical |

## 7. Decisions and reasons

- **Decided:** read the Hugging Face `original/` folder directly, in every mode.
  **Why:** no `.bin` here, no disk room for one, and `generate`/`chat` must keep working.
  **Not chosen:**
  - a full fp32 export: 84 / 467 GB;
  - a small "pointer" `.bin` that only our `getp` understands: `generate` and `chat` would break.
- **Decided:** keep the expert weights in MXFP4 even on the CPU, decoding each value when used.
  **Why:** the experts are ~95 % of the bytes; expanding them is exactly the 467 GB problem.
- **Decided:** the CPU forward must give the same bits as `run.cpp`.
  **Why:** it becomes the reference for the GPU in lesson 03. If it were only "close", every
  later difference would be ambiguous.

## 8. Questions to think about

1. Why does appending 16 zero bits turn a bf16 into the same number in fp32?
2. MXFP4 stores 32 values in 16 bytes plus 1 scale byte. How many bits per value is that, and
   how much smaller than bf16?
3. Why can a `#define` in `getp_run.cpp` change what `main()` calls, but only because of where
   `run.cpp` includes the file?

## 9. Discussion notes

(empty: we write here during the walk-through)
