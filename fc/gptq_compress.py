"""GPTQ version of compress.py: same plan and artifact format, but expert and attention weights are quantized with
GPTQ (second-order error compensation) using calibration statistics from the pruned bf16 model itself.

python gptq_compress.py --base BASE --plan plan.json --prunedbf PRUNED_BF16_CKPT --calib items.jsonl --answers ans.jsonl --out ART
"""
import argparse, json, os, shutil, time, torch, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from qpack import quantize, pack

ap = argparse.ArgumentParser()
for k in ["base", "plan", "prunedbf", "calib", "answers", "out"]: ap.add_argument("--" + k, required=True)
ap.add_argument("--ntok", type=int, default=300000); ap.add_argument("--damp", type=float, default=0.01)
a = ap.parse_args()
plan = json.load(open(a.plan)); G = plan.get("group", 128); B = plan["bits"]; ebits = plan.get("expert_bits", {})
keep = plan["keep"]; K = max(len(k) for k in keep)

# ---------------- 1. calibration statistics on the pruned bf16 model
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feedback import FB  # loads with router offsets (padding slots off)
fb = FB(a.prunedbf); m = fb.m; tok = fb.tok
L = len(fb.layers); H = m.config.hidden_size; I = m.config.moe_intermediate_size
Hin = [torch.zeros(len(keep[l]), H, H) for l in range(L)]; Hdn = [torch.zeros(len(keep[l]), I, I) for l in range(L)]
Hq = [torch.zeros(H, H, device="cuda") for _ in range(L)]; Ho = [torch.zeros(4096, 4096, device="cuda") for _ in range(L)]
NE = [torch.zeros(len(keep[l])) for l in range(L)]; state = {}
def moe_hook(l):
    def h(mod, args):
        x = args[0].reshape(-1, H)[state["valid"]].float()
        _, tv, ti = mod.gate(x.to(torch.bfloat16)); ex = mod.experts
        for e in range(len(keep[l])):
            t = torch.where((ti == e).any(-1))[0]
            if len(t) == 0: continue
            xe = x[t]; Hin[l][e] += (xe.T @ xe).cpu(); NE[l][e] += len(t)
            g, u = F.linear(xe.to(torch.bfloat16), ex.gate_up_proj[e]).chunk(2, -1)
            hh = (ex.act_fn(g) * u).float(); Hdn[l][e] += (hh.T @ hh).cpu()
    return h
def lin_hook(store, l, d):
    def h(mod, args):
        x = args[0].reshape(-1, d)[state["valid"]].float(); store[l] += x.T @ x
    return h
hs = []
for l, layer in enumerate(fb.layers):
    hs.append(layer.mlp.register_forward_pre_hook(moe_hook(l)))
    hs.append(layer.self_attn.q_proj.register_forward_pre_hook(lin_hook(Hq, l, H)))
    hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lin_hook(Ho, l, 4096)))
items = {json.loads(x)["id"]: json.loads(x) for x in open(a.calib)}
ans = [json.loads(x) for x in open(a.answers)]
import random; random.Random(0).shuffle(ans)
seqs = []
for r in ans:
    it = items[r["id"]]
    p = tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False)
    seqs.append(tok(p, add_special_tokens=False).input_ids + tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids)
t0 = time.time(); ntok = 0; i = 0
with torch.no_grad():
    while i < len(seqs) and ntok < a.ntok:
        batch = []; T = 0
        while i < len(seqs) and (len(batch) + 1) * max(T, len(seqs[i])) <= 24000:
            batch.append(seqs[i]); T = max(T, len(seqs[i])); i += 1
        if not batch: batch = [seqs[i]]; T = len(seqs[i]); i += 1
        ids = torch.zeros(len(batch), T, dtype=torch.long, device="cuda"); att = torch.zeros_like(ids)
        for b, s in enumerate(batch): ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        state["valid"] = att.reshape(-1).bool()
        m(input_ids=ids, attention_mask=att); ntok += int(att.sum())
print(f"calibration: {ntok} tokens in {time.time()-t0:.0f}s", flush=True)
for h in hs: h.remove()
del fb, m; torch.cuda.empty_cache()

# ---------------- 2. GPTQ (batched over the matrices that share one Hessian)
def gptq(W, Hs, bits, damp=a.damp):
    """W (E, out, in) float cuda, Hs (E, in, in). Returns codes uint8, scale, mn like quantize()."""
    W = W.clone(); E, R, N = W.shape; Hs = Hs.clone()
    d = torch.diagonal(Hs, dim1=1, dim2=2); dead = d == 0
    Hs[dead.unsqueeze(1).expand_as(Hs) & dead.unsqueeze(2).expand_as(Hs)] = 0
    W.masked_fill_(dead.unsqueeze(1), 0)
    d = torch.diagonal(Hs, dim1=1, dim2=2); d += dead.float()
    d += damp * d.mean(-1, keepdim=True)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hs)), upper=True)
    Q = torch.zeros(E, R, N, dtype=torch.uint8, device=W.device)
    S = torch.zeros(E, R, N // G, dtype=torch.float16, device=W.device); M = torch.zeros_like(S)
    qmax = 2 ** bits - 1
    for i1 in range(0, N, G):
        i2 = i1 + G; W1 = W[:, :, i1:i2].clone(); Err = torch.zeros_like(W1); Hi = Hinv[:, i1:i2, i1:i2]
        _, s, mn = quantize(W1, bits, G); S[:, :, i1 // G] = s[..., 0]; M[:, :, i1 // G] = mn[..., 0]
        s = s.float(); mn = mn.float()
        for j in range(G):
            w = W1[:, :, j]
            q = ((w - mn[..., 0]) / s[..., 0]).round().clamp(0, qmax)
            Q[:, :, i1 + j] = q.to(torch.uint8)
            err = (w - (q * s[..., 0] + mn[..., 0])) / Hi[:, j, j].unsqueeze(1)
            W1[:, :, j:] -= err.unsqueeze(2) * Hi[:, j, j:].unsqueeze(1)
            Err[:, :, j] = err
        W[:, :, i2:] -= Err @ Hinv[:, i1:i2, i2:]
    return Q, S, M

os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"]
hd = {f: safe_open(os.path.join(a.base, f), "pt", device="cuda") for f in set(idx.values())}
get = lambda n: hd[idx[n]].get_tensor(n)
out, meta = {}, {}
def put(name, codes, s, mn, bits, shape):
    out[name + ".qcodes"] = pack(codes, bits).cpu(); out[name + ".qscale"] = s.cpu(); out[name + ".qmin"] = mn.cpu()
    meta[name] = {"bits": bits, "shape": list(shape)}
def rtn(name, w, bits):
    c, s, mn = quantize(w, bits, G); put(name, c, s, mn, bits, w.shape)
rtn("model.embed_tokens.weight", get("model.embed_tokens.weight"), B["embed"])
rtn("lm_head.weight", get("lm_head.weight"), B["head"])
out["model.norm.weight"] = get("model.norm.weight").cpu()
t0 = time.time()
for l in range(L):
    p = f"model.layers.{l}."
    for n in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
        out[p + n] = get(p + n).cpu()
    qkv = [get(p + f"self_attn.{n}_proj.weight").float() for n in "qkv"]
    Wc = torch.cat(qkv, 0)[None]; Qc, Sc, Mc = gptq(Wc, Hq[l][None], B["attn"])
    r0 = 0
    for n, w in zip("qkv", qkv):
        r1 = r0 + w.shape[0]; put(p + f"self_attn.{n}_proj.weight", Qc[0, r0:r1], Sc[0, r0:r1], Mc[0, r0:r1], B["attn"], w.shape); r0 = r1
    wo = get(p + "self_attn.o_proj.weight").float()[None]; Qo, So, Mo = gptq(wo, Ho[l][None], B["attn"])
    put(p + "self_attn.o_proj.weight", Qo[0], So[0], Mo[0], B["attn"], wo.shape[1:])
    out[p + "mlp.gate.weight"] = get(p + "mlp.gate.weight")[keep[l]].contiguous().cpu()
    bits_l = [ebits.get(f"{l}.{e}", B["expert"]) for e in keep[l]]
    for bits in sorted(set(bits_l)):
        js = [j for j, b in enumerate(bits_l) if b == bits]
        for c0 in range(0, len(js), 16):
            jj = js[c0:c0 + 16]; ee = [keep[l][j] for j in jj]
            gu = torch.stack([torch.cat([get(p + f"mlp.experts.{e}.gate_proj.weight"), get(p + f"mlp.experts.{e}.up_proj.weight")], 0) for e in ee]).float()
            Hg = torch.stack([Hin[l][j] for j in jj]).cuda()
            Q1, S1, M1 = gptq(gu, Hg, bits)
            dn = torch.stack([get(p + f"mlp.experts.{e}.down_proj.weight") for e in ee]).float()
            Hd = torch.stack([Hdn[l][j] for j in jj]).cuda()
            Q2, S2, M2 = gptq(dn, Hd, bits)
            for k, j in enumerate(jj):
                put(p + f"mlp.experts.{j}.gate_proj.weight", Q1[k, :I], S1[k, :I], M1[k, :I], bits, (I, H))
                put(p + f"mlp.experts.{j}.up_proj.weight", Q1[k, I:], S1[k, I:], M1[k, I:], bits, (I, H))
                put(p + f"mlp.experts.{j}.down_proj.weight", Q2[k], S2[k], M2[k], bits, (H, I))
    if l % 8 == 0: print("layer", l, f"{time.time()-t0:.0f}s", flush=True)
save_file(out, os.path.join(a.out, "weights.safetensors"))
cfg = json.load(open(os.path.join(a.base, "config.json"))); cfg["num_experts"] = K
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
json.dump({"group": G, "tensors": meta, "keep": keep, "K": K}, open(os.path.join(a.out, "qmeta.json"), "w"))
for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]: shutil.copy(os.path.join(a.base, f), a.out)
tot = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
print(f"artifact bytes {tot} ({tot/1e9:.3f} GB) in {time.time()-t0:.0f}s")
