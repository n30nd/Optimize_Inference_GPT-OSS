# 01b — gpt-oss-20b operation by operation, with real numbers

[01](01-how-gpt-oss-makes-text.md) gives the overall picture. This page takes one step of
gpt-oss-20b on a batch of **B = 512** requests and walks through every operation in order. For
each one it gives the inputs, the weights, the formula, the output, the shapes and the real values
that come out.

## Short answer

One step reads one word piece from each of 512 conversations and guesses the next piece for each of
them. Each conversation's piece becomes a list of 2,880 numbers. That list goes through 24 identical
stages. Each stage first looks back at the earlier words, then hands the list to 4 of 32 small
specialist networks. At the end, the list becomes a score for each of the 201,088 known word pieces,
and the highest score wins. The 512 conversations never mix with each other. They only share the
model's numbers, which are read once per step and used for all 512.

## Where the numbers come from

- Model: `gpt-oss-20b/original/model.safetensors`. The settings come from its `config.json`.
- Input: the first 512 prompts of `evaluation/input.txt`, encoded with our `tokenizer.bin`.
- The step shown: **position P = 4**, the 5th token of every prompt. The shortest prompt has 5
  tokens, so at P = 4 every row is still reading its prompt. Positions 0–3 are already in the
  key/value memory (explained below).
- The arithmetic was rerun in numpy, in fp32 (normal 32-bit decimal numbers), with the same formulas
  as `run.cpp` `forward()`. The script is `tools/trace_20b.py`. Our exact CPU path adds up in
  64-bit numbers and our fast GPU path uses bf16 (16-bit numbers), so their results differ from
  these from about the 4th significant digit.
- Sanity check: for row 0 (`"What is the biggest misconception"`), the model's top guess is
  `" about"`, which is the word that really comes next in the prompt.

How the real values are printed below: the shape first, then mean / std / min / max over the whole
tensor (std, the standard deviation, is how far values typically sit from the mean), then the first 5 and the last 2 numbers of rows 0, 1 and 2.

## The symbols (all from `config.json`)

| Symbol | Value | Meaning |
|---|---:|---|
| B | 512 | requests (rows) in the batch, chosen by us |
| P | 4 (here) | the position being processed, the same for every row (see below) |
| H | 2,880 | `hidden_size`: numbers per token in the hidden state |
| D | 64 | `head_dim`: numbers per attention head |
| Hq | 64 | `num_attention_heads`: query heads |
| Hkv | 8 | `num_key_value_heads`: key/value heads |
| G | 8 | Hq / Hkv: query heads that share one key/value head |
| QKV | 5,120 | (Hq + 2·Hkv)·D = 4,096 + 512 + 512 |
| E | 32 | `num_experts` per layer |
| K | 4 | `experts_per_token` |
| I | 2,880 | `intermediate_size`: width inside one expert |
| L | 24 | `num_hidden_layers` |
| V | 201,088 | `vocab_size`: number of known tokens |
| W | 128 | `sliding_window`, used on layers 0, 2, 4, … |
| θ, s, ctx₀ | 150,000, 32, 4,096 | `rope_theta`, `rope_scaling_factor`, `initial_context_length` |
| α, β | 1, 32 | `rope_ntk_alpha`, `rope_ntk_beta` |
| clip | 7.0 | `swiglu_limit` |

**Why one P for all 512 rows.** Our engine uses *static batching*: the 512 requests start
together and all move forward one position per step (`getp-csrc/engine/gpu_engine.hpp:9-10`). A row
that is still reading its prompt is given its next prompt token. A row that is already answering is
given the token it produced in the previous step. The math is the same in both cases.

**Rows never meet.** In every operation below, row b's output depends only on row b's input and on
row b's own key/value memory. The batch exists only so that each weight is read from memory once
and then used for 512 rows.

## The whole step in one picture

```
tokens [512] (int)                                 e.g. row 0 = 174654 " misconception"
  │ embedding lookup              table [201088, 2880]
  ▼
x [512, 2880]  fp32  ──────────────────────────────── the "residual stream", kept for the whole step
  │
  │  ┌──────────── layer l = 0 … 23 ───────────────────────────────────────────────────────────┐
  │  │ t   = RMSNorm(x)                          [512, 2880]                                    │
  │  │ qkv = t·Wqkvᵀ + b                         [512, 5120]  → q [512,64,64] k,v [512,8,64]    │
  │  │ q,k = RoPE(q,k, P)                        same shapes                                    │
  │  │ write k,v into KV cache at P              cache [512, slots, 8, 64] ×2                   │
  │  │ o   = attention(q, cache, sink)           [512, 64, 64] → [512, 4096]                    │
  │  │ x  += o·Woᵀ + b                           [512, 2880]                                    │
  │  │ t   = RMSNorm(x)                          [512, 2880]                                    │
  │  │ r   = t·Wrᵀ + b                           [512, 32]                                      │
  │  │ (idx, w) = softmax(top4(r))               [512, 4] ints, [512, 4] weights                │
  │  │ for each (row, chosen expert): 2048 pairs                                                │
  │  │     h = t·W1ᵀ + b1     [5760] → gate, up [2880] each → a = SwiGLU [2880]                 │
  │  │     y = a·W2ᵀ + b2     [2880]                                                            │
  │  │ x  += Σₖ wₖ·yₖ                            [512, 2880]                                    │
  │  └──────────────────────────────────────────────────────────────────────────────────────────┘
  ▼
x̂ = RMSNorm(x) [512, 2880]  →  logits = x̂·Uᵀ [512, 201088]  →  argmax → next tokens [512]
```

---

## 1. Embedding: token id to 2,880 numbers

*An embedding is a fixed table with one row of 2,880 numbers for every known token. A token id
simply picks a row.*

| | shape | type |
|---|---|---|
| input `tokens` | [512] | int32 |
| weight `embedding.weight` | [201,088, 2,880] | bf16 on disk, 1.16 GB |
| output `x` | [512, 2,880] | fp32 |

**Math:** `x[b, :] = Emb[tokens[b], :]`. It is a copy, with no arithmetic.

At P = 4 the tokens are row 0 `" misconception"` (174654), row 1 `" meaning"` (10915), row 2
`" guide"` (9165), row 3 `"5"` (20), …

```
--- embedding x   shape=[512, 2880]  mean=-0.0002 std=2.7587 min=-62.75 max=56.75
   row 0: [ 0.2695  0.5     0.4121  0.1104 -0.2949] ... [-0.0041 -0.2168]
   row 1: [-0.3398  0.9102  0.2559  0.0427  3.5469] ... [ 0.0051  0.2266]
   row 2: [ 0.1855 -0.7383  0.7656 -0.0015  0.7812] ... [-0.0043 -0.5977]
```

The values look like short bf16 numbers (0.2695, 0.4121) because that is how they are stored.
Most values are small, but a few columns hold large values (±60). Those large columns matter later.

---

## 2. RMSNorm (before attention)

*RMSNorm (root-mean-square normalization) rescales a row so that its typical size is 1, then
multiplies each column by a learned per-column factor.*

| | shape |
|---|---|
| input `x` | [512, 2,880] |
| weight `block.l.attn.norm.scale` (γ) | [2,880], values 0.47 … 7.75 in layer 0 |
| output `t` | [512, 2,880] |

**Math** (for each row b):

```
rms_b   = sqrt( (1/2880) · Σⱼ x[b,j]²  + 1e-5 )
t[b,j]  = γ[j] · x[b,j] / rms_b
```

Row 0 worked by hand: rms₀ = 2.7496, x[0,0] = 0.2695, γ[0] = 1.3906, so
t[0,0] = 1.3906 · 0.2695 / 2.7496 = **0.1363**.

```
--- t = rmsnorm(x)   shape=[512, 2880]  mean=-0.0054 std=0.7856 min=-33.73 max=15.03
   row 0: [ 0.1363  0.2287  0.1616  0.1492 -0.124 ] ... [-0.0071 -0.1577]
   row 1: [-0.1722  0.4172  0.1005  0.0579  1.4947] ... [ 0.0089  0.1651]
```

`x` is not changed. `t` is a normalized copy that feeds the next matrix multiply.

---

## 3. QKV projection

*A matrix multiply (GEMM, general matrix multiply) turns every row of 2,880 numbers into 5,120 new
numbers. Each output is a weighted sum of all 2,880 inputs.*

| | shape |
|---|---|
| input `t` | [512, 2,880] |
| weight `attn.qkv.weight` | [5,120, 2,880] bf16, 29.5 MB |
| bias `attn.qkv.bias` | [5,120] |
| output `qkv` | [512, 5,120] |

**Math:** `qkv[b, o] = Σⱼ t[b,j] · Wqkv[o,j] + bias[o]`. This is 512 × 2,880 × 5,120 =
**7.55 billion multiply-adds**.

The 5,120 output columns are three blocks laid side by side, then viewed per head:

```
columns    0 … 4095  →  q  [512, 64 heads, 64]   "query": what this token is looking for
columns 4096 … 4607  →  k  [512,  8 heads, 64]   "key":   what this token offers to be found by
columns 4608 … 5119  →  v  [512,  8 heads, 64]   "value": what this token hands over if found
```

```
--- qkv   shape=[512, 5120]  mean=0.0262 std=2.0150 min=-63.62 max=67.59
   row 0: [ 0.2226  0.014  -0.1914  0.2761  0.1807] ... [-0.4628  0.0405]
```

---

## 4. RoPE with YaRN: put the position into q and k

*RoPE (rotary position embedding) turns pairs of numbers by an angle that grows with the token's
position. When a query and a key are later multiplied, only the difference of their angles is
left, so attention can tell how far apart two tokens are. YaRN is a fixed rule for choosing the
turning speeds so that the model works on texts longer than the ones it was trained on.*

| | shape |
|---|---|
| input `q` | [512, 64, 64] |
| input `k` | [512, 8, 64] |
| tables `cos`, `sin` | [32] for position P (the same for every row, because every row is at P) |
| output | same shapes, rotated in place |

**The turning speeds** (one per pair i = 0 … 31, computed once at start-up):

```
freq[i]   = θ^(2i/64)                                  θ = 150,000
low       = 32·ln(4096 / (β·2π)) / ln θ  = 8.093
high      = 32·ln(4096 / (α·2π)) / ln θ  = 17.398
ramp[i]   = clamp((i − low)/(high − low), 0, 1)
inv[i]    = ramp·(1/(32·freq[i])) + (1 − ramp)·(1/freq[i])
conc      = 0.1·ln 32 + 1 = 1.346574
cos[i]    = conc · cos(P · inv[i]),   sin[i] = conc · sin(P · inv[i])
```

Pairs 0–8 turn at the original speed 1/freq. Pairs above 17 turn 32× slower, which stretches them
to cover long texts. Pairs in between are a blend. Real values: inv[0, 1, 8, 9, 16] =
1, 0.689, 0.0508, 0.0317, 0.0005. At P = 4: cos[0..3] = −0.8802, −1.2478, −0.4342, 0.3491 and
sin[0..3] = −1.0191, 0.5062, 1.2746, 1.3005. The numbers can go above 1 because of the 1.3466
factor (conc).

**The rotation.** Element i is paired with element i + 32 of the same head (first half with
second half):

```
x1 = x[i],  x2 = x[i+32]
x[i]    = x1·cos[i] − x2·sin[i]
x[i+32] = x2·cos[i] + x1·sin[i]
```

Row 0, query head 0, i = 0: x1 = 0.2226, x2 = −0.0272

```
new x[0]  = 0.2226·(−0.8802) − (−0.0272)·(−1.0191) = −0.2237
new x[32] = (−0.0272)·(−0.8802) + 0.2226·(−1.0191) = −0.2029
```

```
q head0 before RoPE [0:4] = [ 0.2226  0.014  -0.1914  0.2761]   [32:36] = [-0.0272 -0.0596  0.3827  0.2041]
q head0 after  RoPE [0:4] = [-0.2237  0.0127 -0.4046 -0.169 ]   [32:36] = [-0.2029  0.0814 -0.4101  0.4303]
```

v is not rotated.

---

## 5. Write k and v into the KV cache

*The KV cache (key/value memory) keeps each earlier token's k and v, so they are computed only
once. Each step adds one position.*

| | shape (bf16 in the fast path) |
|---|---|
| input `k`, `v` | [512, 8, 64] each |
| cache on a full layer (1, 3, 5, …) | K and V each [512, 1,024 slots, 8, 64] = 537 MB each |
| cache on a sliding layer (0, 2, 4, …) | K and V each [512, 128 slots, 8, 64] = 67 MB each, reused in a ring (slot = P mod 128) |
| whole model, B = 512, 1,024 positions | 12 × 1.07 GB + 12 × 0.134 GB = **14.5 GB** |

**Math:** `Kc[l][b, slot(P), g, :] = k[b, g, :]`, and the same for v. It is a copy, with no
arithmetic.

Row 0, key/value head 0, layer 0: the five positions now in the cache (after RoPE):

```
k  shape=[5, 64]           v  shape=[5, 64]
t=0 [-2.41 -5.40  2.26 ...]   t=0 [-0.96  1.08 -0.64 ...]   "What"
t=1 [ 1.03 -5.20  3.74 ...]   t=1 [-0.89 -1.33  0.20 ...]   " is"
t=2 [ 8.07 -0.01  4.60 ...]   t=2 [-0.37  2.09  0.56 ...]   " the"
...                                                          t=3 " biggest", t=4 " misconception"
```

---

## 6. Attention (with GQA, sliding window and sink)

*Attention lets the current token read from earlier tokens. For each earlier token it computes a
score that says how relevant that token is, turns the scores into shares that add up to 1, and
takes the matching mix of the earlier tokens' values.*

*GQA (grouped-query attention): 8 query heads share one key/value head. Query head h reads
key/value head h / 8 (integer division), so heads 0–7 use kv head 0, heads 8–15 use kv head 1, and
so on.*

*Sink: each head has one extra learned score with no value attached. It joins the softmax like an
extra token, and whatever share it takes is simply thrown away. This lets a head look at nothing.*

| | shape |
|---|---|
| input `q` | [512, 64, 64] |
| cache K, V | positions 0 … P of the same row |
| weight `attn.sinks` | [64], one per head (layer 0: 2.52, 0.56, 1.72, 0.91, …) |
| output `o` | [512, 64, 64], viewed as [512, 4,096] |

**Math**, for row b, head h, kv head g = ⌊h/8⌋:

```
s[t]    = ( Σᵢ q[b,h,i] · K[b,t,g,i] ) / √64          for each earlier position t
          only t with P − t < 128 on even layers (sliding window), all t ≤ P on odd layers
s[P+1]  = sink[h]                                       the extra "nothing" score
p       = softmax(s[0 … P+1])   i.e. p[t] = e^(s[t]−max) / Σ e^(s[u]−max)
o[b,h,:] = Σ_{t≤P} p[t] · V[b,t,g,:]                    the sink's share is dropped
```

Real values, row 0, layer 0, P = 4 (6 entries: 5 tokens plus the sink):

```
                What     is      the     biggest  misconc.  SINK
head 0 score  [ 0.2873 -1.2808 -4.9433 -2.6544  -1.0918   2.5156]
head 0 p      [ 0.0926  0.0193  0.0005  0.0049   0.0233   0.8594]   sums to 1
head 1 p      [ 0.0238  0.0505  0.0259  0.3001   0.0752   0.5245]
```

Head 0 gives 86% of its attention to the sink, so it passes on very little. Head 1 mostly reads
`" biggest"`. Check of the sink share: e^0 / (e^−2.228 + e^−3.796 + e^−7.459 + e^−5.170 + e^−3.607
+ e^0) = 1 / 1.1636 = 0.859.

Cost per step: 2 × 512 × 64 × 64 × (P+1) multiply-adds (scores plus the mix of values). That is
tiny at P = 4. At P = 1,023 it is 4.29 billion on a full layer and 0.54 billion on a sliding layer.

```
--- attn out   shape=[512, 4096]  mean=-0.0015 std=0.5521 min=-6.31 max=6.68
   row 0: [-0.1025  0.0781 -0.0481 -0.0216  0.011 ] ... [-0.0093 -0.4329]
```

---

## 7. Output projection and the first residual add

| | shape |
|---|---|
| input `o` | [512, 4,096] |
| weight `attn.out.weight` | [2,880, 4,096] bf16, 23.6 MB, and a bias [2,880] |
| output `proj` | [512, 2,880] |

**Math:** `proj = o · Woᵀ + b` (512 × 4,096 × 2,880 = **6.04 billion multiply-adds**), then
**`x = x + proj`**.

*Residual: the layer adds its result onto x instead of replacing x, so the original information
stays and each layer only contributes a correction.*

```
--- proj       row 0: [-0.2358 -0.1013 -0.1636  0.0078 -0.4083] ...   std=0.644
--- x (after)  row 0: [ 0.0337  0.3987  0.2485  0.1182 -0.7032] ...   std=2.780
                      = 0.2695 + (−0.2358) = 0.0337 ✓
```

---

## 8. RMSNorm (before the experts)

This is the same formula as step 2, with its own weights `mlp.norm.scale` [2,880] (layer 0: 0.05 …
9.63).

```
--- t   shape=[512, 2880]  std=0.3699 min=-7.92 max=3.29
   row 0: [ 0.0208  0.1872  0.1071  0.2219 -0.2622] ... [ 0.143  -0.3757]
```

---

## 9. Router: pick 4 of 32 experts per row

*The router is a small matrix multiply that gives each expert a score for this token. The 4
highest-scoring experts are used, and their scores are turned into weights that add up to 1.*

| | shape |
|---|---|
| input `t` | [512, 2,880] |
| weight `mlp.gate.weight` | [32, 2,880], and a bias [32] |
| `r` (router scores) | [512, 32] |
| outputs `idx`, `w` | [512, 4] int, [512, 4] fp32 |

**Math:**

```
r[b,e]          = Σⱼ t[b,j] · Wr[e,j] + bias[e]           512 × 2880 × 32 = 47 M multiply-adds
idx[b,0..3]     = the 4 e with the largest r[b,e], largest first
w[b,k]          = softmax over those 4 only = e^(r[b,idx_k]) / Σ_{k'} e^(r[b,idx_k'])
```

Real values, layer 0:

```
row 0  r[0:8] = [-0.747 -0.909  1.418 -0.257  1.063 -0.829  0.547  0.164] ...
row 0  top-4 experts [ 2 16  9  4]  r = [1.4179 1.4153 1.2360 1.0626] → w = [0.2831 0.2824 0.2360 0.1984]
row 1  top-4 experts [29 16  7  2]  r = [1.6807 1.2590 0.7785 0.5320] → w = [0.4204 0.2758 0.1705 0.1333]
row 2  top-4 experts [19 15 11 21]  r = [1.4708 1.3069 0.9512 0.6560] → w = [0.3465 0.2941 0.2061 0.1534]
row 3  top-4 experts [19 15 31 11]  r = [2.4827 1.2622 1.2122 0.9231] → w = [0.5599 0.1652 0.1572 0.1177]
```

**Load per expert.** 512 rows × 4 picks = 2,048 (row, expert) pairs. That would be 64 per expert if
they were spread evenly. The real counts at layer 0:

```
expert:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
picks : 44  34  73  55 121  66  43  59  62  99  83 157  38  51  32  49
expert: 16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
picks :190  88  85  69  42  42 102  25  86 137  22  26  41  12   0  15
```

Expert 16 gets 190 rows and expert 30 gets none. Later layers are even more uneven: the busiest
expert gets 225–509 rows, and up to 13 of the 32 experts get no rows at all. One reason is that
these 512 prompts are about similar topics and all sit at the same position. Our real workload
behaves the same way.

---

## 10. One expert: MLP1 → SwiGLU → MLP2

Each expert is a small two-matrix network. Every (row, expert) pair from step 9 goes through it:
2,048 pairs in total. Row 0 goes through experts 2, 16, 9 and 4.

### 10a. How the expert weights are stored: MXFP4

*MXFP4 (a 4-bit number format with a shared scale per group): each weight is a 4-bit code for one
of 16 values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, and each group of 32 weights shares one
power-of-two scale.*

| tensor | shape | meaning |
|---|---|---|
| `mlp1_weight.blocks` | [32, 5,760, 90, 16] u8 | expert, output row, group of 32, 16 bytes = 32 codes |
| `mlp1_weight.scales` | [32, 5,760, 90] u8 | one scale per group, value 2^(byte − 127) |
| `mlp2_weight.blocks` | [32, 2,880, 90, 16] u8 | |
| `mlp2_weight.scales` | [32, 2,880, 90] u8 | |
| `mlp1_bias`, `mlp2_bias` | [32, 5,760], [32, 2,880] | bf16 |

Per layer this is 423 MB, or 13.2 MB per expert, against about 50 MB if stored as bf16.

**Decoding one group** (real bytes: layer 0, expert 0, row 0, group 0). The scale byte is 122, so
the scale is 2^(122−127) = 1/32. Each byte holds two codes, the **low 4 bits first**:

```
byte 0x00 → codes 0x0, 0x0 → 0,      0         → ×1/32 → 0,       0
byte 0xc0 → codes 0x0, 0xc → 0,     −2         → ×1/32 → 0,      −0.0625
byte 0x80 → codes 0x0, 0x8 → 0,     −0         → ×1/32 → 0,      −0
byte 0xa9 → codes 0x9, 0xa → −0.5,  −1         → ×1/32 → −0.0156, −0.0312
```

The decoded matrices are W1 [5,760, 2,880] and W2 [2,880, 2,880] for each expert.

### 10b. MLP1

**Math:** `h = t[b] · W1[e]ᵀ + b1[e]`, giving [5,760] per pair. Over the batch that is 2,048 × 2,880 ×
5,760 = **33.97 billion multiply-adds**, the largest piece of each layer.

The 5,760 outputs are **interleaved**: `gate[j] = h[2j]` and `up[j] = h[2j+1]`, for j = 0 … 2,879.

```
row 0 / expert 2:  h[0:6] = [-1.5649 -0.5417 -0.3342 -2.1964 -1.0359 -0.5888]
                   gate[0:3] = [-1.5649 -0.3342 -1.0359]   up[0:3] = [-0.5417 -2.1964 -0.5888]
```

### 10c. SwiGLU (with gpt-oss's clamp and +1)

*SwiGLU is an elementwise rule: one half (gate) decides how much of the other half (up) passes
through. gpt-oss also caps very large values at 7 and adds 1 to `up`.*

```
g    = min(gate, 7)                 (only capped from above)
u    = clamp(up, −7, 7)
a[j] = g · σ(1.702·g) · (u + 1)     σ(z) = 1 / (1 + e^(−z))
```

Row 0 / expert 2, j = 0: g = −1.5649, σ(1.702 · −1.5649) = σ(−2.6635) = 0.0652, and u + 1 = 0.4583,
so a = −1.5649 · 0.0652 · 0.4583 = **−0.0467**.

```
a[0:4] = [-0.0467  0.1446 -0.0624  0.0038]     shape [2880]; the largest |a| in this pair is 4.40
```

### 10d. MLP2

**Math:** `y = a · W2[e]ᵀ + b2[e]`, giving [2,880] per pair. Over the batch that is 2,048 × 2,880 ×
2,880 = **16.99 billion multiply-adds**.

```
row 0 / expert 2  (w = 0.2831): y[0:5] = [-0.0556 -0.4253  0.3977 -0.4005  1.0613]
row 0 / expert 16 (w = 0.2824): y[0:5] = [-0.2613 -0.7328 -0.3321  0.0010 -1.0101]
row 0 / expert 9  (w = 0.2360): y[0:5] = [ 0.1214 -0.3339  0.3258  0.1767 -0.9492]
row 0 / expert 4  (w = 0.1984): y[0:5] = [-0.2915 -1.4122 -1.0622 -0.0311  0.6599]
```

---

## 11. Combine the 4 experts and the second residual add

**Math:** `moe[b] = Σₖ w[b,k] · y[b,k]`, then **`x = x + moe`**.

Row 0, column 0: 0.2831·(−0.0556) + 0.2824·(−0.2613) + 0.2360·0.1214 + 0.1984·(−0.2915) = −0.1187.

```
--- moe out          row 0: [-0.1187 -0.6864 -0.1151 -0.0776 -0.0778] ...
--- x (layer 0 out)  row 0: [-0.085  -0.2877  0.1335  0.0406 -0.781 ] ...
                     shape=[512, 2880]  std=3.2457  min=-97.2  max=113.3
```

This `x` is the input of layer 1. Steps 2–11 then repeat with layer 1's weights, 24 times in all.

---

## 12. How x grows through the 24 layers

Each layer adds onto x, so x gets larger and larger. Real values for row 0 at P = 4:

| after layer | rms of x, row 0 | largest abs(x) over all 512 rows | row 0's experts |
|---:|---:|---:|---|
| (embedding) | 2.75 | 62.8 | — |
| 0 | 3.54 | 113 | 2 16 9 4 |
| 5 | 10.1 | 255 | 6 24 17 21 |
| 11 | 28.5 | 721 | 30 11 24 21 |
| 17 | 109 | 2,537 | 15 7 28 17 |
| 23 | 464 | 15,109 | 31 0 18 23 |

This is why our fast GPU path keeps x in fp32 even though it feeds bf16 into the matrix multiplies.
RMSNorm brings every layer's input back to size about 1, but x itself, the running total, needs
the extra precision.

---

## 13. Final RMSNorm, logits and the next token

| | shape |
|---|---|
| input `x` | [512, 2,880] |
| weight `norm.scale` | [2,880] (values around 10, up to 25.5) |
| weight `unembedding.weight` (U) | [201,088, 2,880] bf16, 1.16 GB |
| output `logits` | [512, 201,088] fp32, 412 MB |
| output `next` | [512] int |

**Math:**

```
x̂             = RMSNorm(x)                        same formula as step 2
logits[b, v]  = Σⱼ x̂[b,j] · U[v,j]                512 × 2880 × 201088 = 296.5 billion multiply-adds
next[b]       = argmax_v logits[b, v]              the largest score wins (greedy decoding)
```

*Logits are the raw scores, one per known token. Applying softmax to them would give
probabilities, but taking the largest one does not need that step.*

Real results:

| row | prompt so far | top 5 guesses (score) | real next word in the prompt |
|---:|---|---|---|
| 0 | What is the biggest misconception | ` about` 15.67, ` people` 14.44, ` that` 14.35, ` you` 13.46, ` in` 12.66 | ` about` ✓ |
| 1 | Explain Life purpose & meaning | ` in` 11.83, ` of` 11.69, `:` 11.68, `,` 11.44, `\n\n` 11.39 | ` to` |
| 2 | Outline a beginner's guide | ` to` 16.49, ` for` 15.20, ` on` 14.30, ` or` 13.75, ` that` 13.49 | ` to` ✓ |
| 3 | List the top 5 | ` most` 10.70, ` countries` 9.70, ` movies` 9.27, ` companies` 9.16, ` books` 8.98 | ` challenges` |
| 5 | Should institutions prioritize advances in | ` technology` 11.75, ` the` 10.79, ` medical` 10.43, ` AI` 10.35, ` imaging` 10.31 | ` Life` |

Across the batch, the top guess equals the prompt's real next word for 91 of the 510 rows that have
a 6th token. A person wrote these prompts, so many different continuations are reasonable. While a
row is still in its prompt, its guess is thrown away and the real prompt token is fed instead.

---

## 14. The cost of one step at B = 512

"MACs" are multiply-adds. Weight sizes are as stored on disk.

| operation | per layer, MACs | × 24 layers | weights read per layer |
|---|---:|---:|---:|
| QKV projection | 7.55 G | 181 G | 29.5 MB |
| attention (P = 1,023; full / sliding) | 4.29 G / 0.54 G | 58 G | — (reads the KV cache instead) |
| output projection | 6.04 G | 145 G | 23.6 MB |
| router | 0.05 G | 1.1 G | 0.18 MB |
| expert MLP1 (2,048 pairs) | 33.97 G | 815 G | 282 MB (32 experts) |
| expert MLP2 (2,048 pairs) | 16.99 G | 408 G | 141 MB |
| logits (once per step) | — | 296.5 G | 1.16 GB |
| **total** | | **≈ 1.90 T** | 13.8 GB for the whole model |

Not counting attention, this is 1.85 T MACs, or 3.6 billion per row. That matches the "≈ 3.6 billion
multiply-adds per generated token" in [01](01-how-gpt-oss-makes-text.md). Experts are 64% of the
work and logits are 16%. Every operation except attention reads its weights once per step and
uses them for all 512 rows, which is the reason for batching.

---

## Where it is in our code

| step | CPU reference (`run.cpp`) | GPU (`getp-csrc/engine/gpu_engine.hpp`) |
|---|---|---|
| 1 embedding | `forward()` `memcpy` of the table row | `step_embed` |
| 2, 8, 13 RMSNorm | `rmsnorm()` | `rmsnorm_exact` / `rmsnorm_bf16_kernel` |
| 3 QKV | `matmul()` + bias loop | `layer_attention` |
| 4 RoPE (YaRN) | `compute_concentration_and_inv_freq`, `compute_cos_sin`, `apply_rotary_emb` | `rope_exact` / `rope_kv_bf16_kernel` |
| 5 KV write | `s->k`, `s->v` point into the cache | `kv_write_kernel` (exact path), fused into RoPE (fast path) |
| 6 attention | the `for h` loop, sink at `att[pos + 1]` | `attention_exact` / `attention_bf16` |
| 9 router | `matmul` + `topk()` + `softmax(…, 4)` | `layer_route`, `router_topk_exact` |
| 10 experts | the `for e` loop, SwiGLU with clamp | `layer_experts` (grouped by expert) |
| 11 combine | `e_agg` | `layer_combine` |
| 13 logits | final `rmsnorm` + `matmul(…, vocab_size)` | `step_logits`, `argmax_kernel` |

To regenerate every number on this page: `python3 tools/trace_20b.py <model folder> <ids file> 4`.
The script's header explains how to make the ids file.
