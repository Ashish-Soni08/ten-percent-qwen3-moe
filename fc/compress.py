"""Build the shipped artifact from the base checkpoint: keep a subset of experts per layer, quantize, pack.

python compress.py --base BASE --plan plan.json --out ARTIFACT_DIR
plan.json: {"keep": [[expert ids for layer 0], ...], "bits": {"expert": 4, "attn": 4, "embed": 4, "head": 6},
            "group": 128, "expert_bits": optional {"l.e": bits} overrides, "imp": optional path to per-expert input E[x^2] stats}
The artifact holds packed codes + fp16 scales/mins (+ small bf16 tensors), the tokenizer and config.
"""
import argparse, json, os, shutil, time, torch
from safetensors import safe_open
from safetensors.torch import save_file
from qpack import quantize, pack

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
plan = json.load(open(a.plan)); G = plan.get("group", 128); B = plan["bits"]; ebits = plan.get("expert_bits", {})
keep = plan["keep"]; K = max(len(k) for k in keep)
imp = torch.load(plan["imp"]) if plan.get("imp") else None
os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"]
files = sorted(set(idx.values())); handles = {f: safe_open(os.path.join(a.base, f), "pt", device="cuda") for f in files}
def get(n): return handles[idx[n]].get_tensor(n)

out, meta = {}, {}
def q(name, w, bits, im=None):
    if bits >= 16:
        out[name] = w.to(torch.bfloat16).contiguous().cpu(); return
    c, s, m = quantize(w, bits, G, imp=im)
    out[name + ".qcodes"] = pack(c, bits).cpu(); out[name + ".qscale"] = s.cpu(); out[name + ".qmin"] = m.cpu()
    meta[name] = {"bits": bits, "shape": list(w.shape)}

t0 = time.time()
cfg = json.load(open(os.path.join(a.base, "config.json")))
L = cfg["num_hidden_layers"]
q("model.embed_tokens.weight", get("model.embed_tokens.weight"), B["embed"])
q("lm_head.weight", get("lm_head.weight"), B["head"])
out["model.norm.weight"] = get("model.norm.weight").cpu()
for l in range(L):
    p = f"model.layers.{l}."
    for n in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
        out[p + n] = get(p + n).cpu()
    for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        q(p + f"self_attn.{n}.weight", get(p + f"self_attn.{n}.weight"), B["attn"])
    out[p + "mlp.gate.weight"] = get(p + "mlp.gate.weight")[keep[l]].contiguous().cpu()  # (len(keep[l]), H); padded at unpack
    for j, e in enumerate(keep[l]):
        bits = ebits.get(f"{l}.{e}", B["expert"])
        for n in ["gate_proj", "up_proj", "down_proj"]:
            w = get(p + f"mlp.experts.{e}.{n}.weight")
            im = None
            if imp is not None:
                im = imp["down" if n == "down_proj" else "in"][l][e].cuda()
            q(p + f"mlp.experts.{j}.{n}.weight", w, bits, im)
    if l % 8 == 0: print("layer", l, f"{time.time()-t0:.0f}s", flush=True)
save_file(out, os.path.join(a.out, "weights.safetensors"))
cfg["num_experts"] = K
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
json.dump({"group": G, "tensors": meta, "keep": keep, "K": K}, open(os.path.join(a.out, "qmeta.json"), "w"))
for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
    shutil.copy(os.path.join(a.base, f), a.out)
tot = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
print(f"artifact bytes {tot} ({tot/1e9:.3f} GB) in {time.time()-t0:.0f}s")
