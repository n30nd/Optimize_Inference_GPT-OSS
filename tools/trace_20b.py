"""Run gpt-oss-20b in numpy (fp32) on real prompts, positions 0..P, and print
the tensors of the step at position P: layer 0 in full, one line per later
layer, then the logits. Same formulas as run.cpp forward(). Produces the
numbers in docs/01b-architecture-math-20b.md.

Usage (repo root):
    gcc -DTESTING -O3 test_tokenizer.cpp tokenizer.cpp -o test_tokenizer
    sed -n 2,513p evaluation/input.txt | while IFS= read -r p; do
        ./test_tokenizer -t tokenizer.bin -i "$p"; done > /tmp/ids512.txt
    python3 tools/trace_20b.py <gpt-oss-20b folder> /tmp/ids512.txt 4

Every prompt needs at least P+1 tokens. Needs only numpy; about 7 minutes
and 24 GB of memory for 512 prompts on 96 cores.
"""
import json, math, os, struct, sys, time
import numpy as np

M = sys.argv[1]
if os.path.isfile(f"{M}/original/config.json"):
    M = f"{M}/original"
IDS = sys.argv[2]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 4
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
np.set_printoptions(precision=4, suppress=True, linewidth=140)

cfg = json.load(open(f"{M}/config.json"))
H, D, NQ, NKV = cfg["hidden_size"], cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
E, K, I, L, V = cfg["num_experts"], cfg["experts_per_token"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]

f = open(f"{M}/model.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n))
raw = np.memmap(f"{M}/model.safetensors", dtype=np.uint8, mode="r")
base = 8 + n

def T(name):
    h = hdr[name]; a, b = h["data_offsets"]
    buf = raw[base + a: base + b]
    if h["dtype"] == "BF16":
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(h["shape"])
    return np.frombuffer(buf, dtype=np.uint8).reshape(h["shape"])

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def deq(blocks, scales):  # blocks [rows, 90, 16] u8, scales [rows, 90] u8 -> [rows, 2880]
    lo = FP4[blocks & 0x0F]; hi = FP4[blocks >> 4]
    out = np.empty(blocks.shape[:-1] + (32,), np.float32)
    out[..., 0::2] = lo; out[..., 1::2] = hi
    out = np.ldexp(out, (scales.astype(np.int32) - 127)[..., None])
    return out.reshape(blocks.shape[0], -1)

# tokenizer pieces for printing
tb = open(f"{REPO}/tokenizer.bin", "rb").read()
off = 4; pieces = []
while off < len(tb):
    sc, ln = struct.unpack_from("<fi", tb, off); off += 8
    pieces.append(tb[off:off + ln].decode("utf-8", "replace")); off += ln
def tok(i): return repr(pieces[i]) if i < len(pieces) else f"<{i}>"

ids = [list(map(int, l.split())) for l in open(IDS)]
B = len(ids); NP = P + 1
tokens = np.array([r[:NP] for r in ids], np.int64)  # [B, NP]

def show(name, a, rows=3, cols=5):
    a = np.asarray(a)
    print(f"--- {name}  shape={list(a.shape)}  mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}")
    if a.ndim == 1: print("   ", a[:cols], "...", a[-2:])
    elif a.ndim == 2:
        for r in range(min(rows, a.shape[0])): print(f"   row {r}:", a[r, :cols], "...", a[r, -2:])

# YaRN RoPE tables
half = D // 2
theta, sf, ictx = cfg["rope_theta"], cfg["rope_scaling_factor"], cfg["initial_context_length"]
beta, alpha = cfg["rope_ntk_beta"], cfg["rope_ntk_alpha"]
freq = np.array([theta ** (2 * i / D) for i in range(half)], np.float32)
conc = 0.1 * math.log(sf) + 1.0
low = half * math.log(ictx / (beta * 2 * math.pi)) / math.log(theta)
high = half * math.log(ictx / (alpha * 2 * math.pi)) / math.log(theta)
ramp = np.clip((np.arange(half) - low) / (high - low), 0, 1); mask = 1 - ramp
inv_freq = (1 / (sf * freq)) * (1 - mask) + (1 / freq) * mask
print(f"YaRN: concentration={conc:.6f} low={low:.4f} high={high:.4f}")
print("inv_freq[0,1,8,9,16,17,31] =", inv_freq[[0, 1, 8, 9, 16, 17, 31]])
pos = np.arange(NP)[:, None] * inv_freq[None]
COS, SIN = np.cos(pos) * conc, np.sin(pos) * conc  # [NP, 32]
print(f"cos[P={P}][:4] =", COS[P, :4], " sin[P][:4] =", SIN[P, :4])

def rope(x):  # x [B, NP, heads, 64]
    x1, x2 = x[..., :half], x[..., half:]
    c, s = COS[None, :, None, :], SIN[None, :, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

def rms(x, w):
    return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5))

emb = T("embedding.weight")
x = emb[tokens].astype(np.float32)  # [B, NP, H]
print(f"B={B} step position P={P}; row 0 tokens:", [tok(t) for t in tokens[0]])
show("embedding x (step P)", x[:, P])

for l in range(L):
    t0 = time.time()
    full = (l == 0)
    pre = f"[layer {l}] "
    t = rms(x, T(f"block.{l}.attn.norm.scale"))
    if full: show(pre + "t = rmsnorm(x)", t[:, P]); print("    rms of x row0 =", float(np.sqrt((x[0, P] ** 2).mean())))
    Wqkv, bqkv = T(f"block.{l}.attn.qkv.weight"), T(f"block.{l}.attn.qkv.bias")
    qkv = t @ Wqkv.T + bqkv  # [B, NP, 5120]
    if full: show(pre + "qkv = t @ Wqkv^T + b", qkv[:, P])
    q = qkv[..., :NQ * D].reshape(B, NP, NQ, D)
    k = qkv[..., NQ * D:NQ * D + NKV * D].reshape(B, NP, NKV, D)
    v = qkv[..., NQ * D + NKV * D:].reshape(B, NP, NKV, D)
    if full: print(f"    row0 q head0 before RoPE [:4],[32:36]:", q[0, P, 0, :4], q[0, P, 0, 32:36])
    q, k = rope(q), rope(k)
    if full:
        print(f"    row0 q head0 after  RoPE [:4],[32:36]:", q[0, P, 0, :4], q[0, P, 0, 32:36])
        show(pre + "k (kv head 0, row0, all positions)", k[0, :, 0])
        show(pre + "v (kv head 0, row0, all positions)", v[0, :, 0])
    # attention for step P (also earlier positions, needed for later layers)
    kq = np.repeat(k, NQ // NKV, axis=2); vq = np.repeat(v, NQ // NKV, axis=2)  # [B,NP,64,64]
    sc = np.einsum("bthd,bshd->bhts", q, kq) / math.sqrt(D)  # [B,64,NP,NP]
    causal = np.triu(np.ones((NP, NP), bool), 1)
    if l % 2 == 0:
        win = np.tril(np.ones((NP, NP), bool), -cfg["sliding_window"])
        causal |= win
    sc = np.where(causal[None, None], -np.inf, sc)
    sink = T(f"block.{l}.attn.sinks")
    scs = np.concatenate([sc, np.broadcast_to(sink[None, :, None, None], (B, NQ, NP, 1))], -1)
    pr = np.exp(scs - scs.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
    if full:
        print(f"    sinks[:8] =", sink[:8])
        print(f"    row0 head0 scores at P (t=0..P, then sink):", scs[0, 0, P])
        print(f"    row0 head0 weights at P              :", pr[0, 0, P], " sum=", pr[0, 0, P].sum())
        print(f"    row0 head1 weights at P              :", pr[0, 1, P])
    o = np.einsum("bhts,bshd->bthd", pr[..., :NP], vq).reshape(B, NP, NQ * D)
    if full: show(pre + "attn out (64 heads x 64 concatenated)", o[:, P])
    proj = o @ T(f"block.{l}.attn.out.weight").T + T(f"block.{l}.attn.out.bias")
    if full: show(pre + "proj = o @ Wo^T + bo", proj[:, P])
    x = x + proj
    if full: show(pre + "x after attention residual", x[:, P])
    t = rms(x, T(f"block.{l}.mlp.norm.scale"))
    if full: show(pre + "t = rmsnorm(x) (mlp)", t[:, P])
    router = t @ T(f"block.{l}.mlp.gate.weight").T + T(f"block.{l}.mlp.gate.bias")  # [B,NP,32]
    top = np.argsort(-router, -1, kind="stable")[..., :K]
    tv = np.take_along_axis(router, top, -1)
    tw = np.exp(tv - tv.max(-1, keepdims=True)); tw /= tw.sum(-1, keepdims=True)
    hist = np.bincount(top[:, P].ravel(), minlength=E)
    if full:
        show(pre + "router logits", router[:, P], cols=8)
        for r in range(4): print(f"    row{r} top4 experts {top[r, P]} scores {tv[r, P]} -> weights {tw[r, P]}")
        print("    picks per expert at step P (2048 = 512 x 4 total):", hist.tolist())
    b1, b2 = T(f"block.{l}.mlp.mlp1_bias"), T(f"block.{l}.mlp.mlp2_bias")
    blk1, sc1 = T(f"block.{l}.mlp.mlp1_weight.blocks"), T(f"block.{l}.mlp.mlp1_weight.scales")
    blk2, sc2 = T(f"block.{l}.mlp.mlp2_weight.blocks"), T(f"block.{l}.mlp.mlp2_weight.scales")
    if full:
        print("    MXFP4 example: mlp1 expert0 row0 group0: scale byte", int(sc1[0, 0, 0]),
              "bytes", [hex(b) for b in blk1[0, 0, 0, :4]], "-> values", deq(blk1[0, 0:1, 0:1], sc1[0, 0:1, 0:1])[0, :8])
    tf = t.reshape(-1, H); topf = top.reshape(-1, K); twf = tw.reshape(-1, K)
    agg = np.zeros_like(tf)
    row0P = 0 * NP + P
    for e in range(E):
        rows, slot = np.nonzero(topf == e)
        if len(rows) == 0: continue
        W1 = deq(blk1[e], sc1[e]); W2 = deq(blk2[e], sc2[e])  # [5760,2880], [2880,2880]
        h1 = tf[rows] @ W1.T + b1[e]
        g, u = h1[:, 0::2], h1[:, 1::2]
        g = np.minimum(g, cfg["swiglu_limit"]); u = np.clip(u, -cfg["swiglu_limit"], cfg["swiglu_limit"])
        act = g * (1 / (1 + np.exp(-1.702 * g))) * (u + 1)
        y = act @ W2.T + b2[e]
        agg[rows] += y * twf[rows, slot][:, None]
        if full and row0P in rows:
            j = list(rows).index(row0P)
            print(f"    -- row0 through expert {e} (weight {twf[row0P, slot[j]]:.4f}):")
            print(f"       mlp1 out [:6] (gate,up,gate,up..):", h1[j, :6])
            print(f"       gate[:4] {g[j, :4]}  up[:4] {u[j, :4]}")
            print(f"       swiglu[:4] {act[j, :4]}  (max |.| {np.abs(act[j]).max():.3f})")
            print(f"       mlp2 out y[:5] {y[j, :5]}")
    agg = agg.reshape(B, NP, H)
    if full: show(pre + "moe out = sum_k w_k * y_k", agg[:, P])
    x = x + agg
    if full: show(pre + "x after moe residual (layer output)", x[:, P])
    print(f"layer {l:2d}: row0 top4 {top[0, P]}  rms(x row0)={np.sqrt((x[0, P] ** 2).mean()):.3f}  "
          f"max|x| all rows={np.abs(x[:, P]).max():.1f}  experts used by >0 rows {int((hist > 0).sum())}/32, "
          f"min/max picks {hist.min()}/{hist.max()}  ({time.time() - t0:.1f}s)", flush=True)

xf = rms(x[:, P], T("norm.scale"))
show("final rmsnorm", xf)
U = T("unembedding.weight")
logits = xf @ U.T
show("logits", logits)
nxt = logits.argmax(-1)
for r in range(6):
    o = np.argsort(-logits[r])[:5]
    print(f"row{r}: prompt so far {''.join(pieces[t] for t in tokens[r])!r} -> top5",
          [(tok(i), round(float(logits[r, i]), 2)) for i in o], " prompt's real next:", tok(ids[r][NP]) if len(ids[r]) > NP else "-")
print("argmax == prompt's real next token for", int(sum(nxt[r] == ids[r][NP] for r in range(B) if len(ids[r]) > NP)), "of",
      sum(len(r) > NP for r in ids), "rows")
