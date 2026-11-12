"""Turn a packed artifact into a plain bf16 HF checkpoint (runtime cache, not shipped) that SGLang/transformers load."""
import argparse, json, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
from qpack import unpack, dequantize

def unpack_artifact(art, dst, edits=None, router_off=None):
    """edits: {tensor name: dense delta}; router_off: list of (layer, slot) switched off by feedback."""
    os.makedirs(dst, exist_ok=True)
    meta = json.load(open(os.path.join(art, "qmeta.json"))); G = meta["group"]
    f = safe_open(os.path.join(art, "weights.safetensors"), "pt", device="cuda")
    names = list(f.keys()); shard, n_shard, size = {}, 0, 0; wmap = {}
    cfg = json.load(open(os.path.join(art, "config.json"))); H = cfg["hidden_size"]; inter = cfg["moe_intermediate_size"]
    def flush():
        nonlocal shard, n_shard, size
        if not shard: return
        fn = f"model-{n_shard:05d}.safetensors"; save_file(shard, os.path.join(dst, fn))
        for k in shard: wmap[k] = fn
        shard, size = {}, 0; n_shard += 1
    for n in names:
        if n.endswith((".qscale", ".qmin")): continue
        if n.endswith(".qcodes"):
            base = n[: -len(".qcodes")]; m = meta["tensors"][base]
            codes = unpack(f.get_tensor(n), m["bits"], m["shape"])
            w = dequantize(codes, f.get_tensor(base + ".qscale"), f.get_tensor(base + ".qmin"), G)
        else:
            base = n; w = f.get_tensor(n)
        if base.endswith("mlp.gate.weight"):
            l = int(base.split(".")[2]); n_real = w.shape[0]; K = meta["K"]
            if n_real < K: w = torch.cat([w, torch.zeros(K - n_real, w.shape[1], dtype=w.dtype, device=w.device)])
            b = torch.zeros(K, dtype=torch.bfloat16); b[n_real:] = -1e4
            for (ll, e) in (router_off or []):
                if ll == l: b[e] = -1e4
            shard[base.replace(".weight", ".bias")] = b
            for j in range(n_real, K):  # zero padding experts (never routed)
                for nm, shp in (("gate_proj", (inter, H)), ("up_proj", (inter, H)), ("down_proj", (H, inter))):
                    shard[f"model.layers.{l}.mlp.experts.{j}.{nm}.weight"] = torch.zeros(shp, dtype=torch.bfloat16)
        if edits and base in edits: w = (w.float() + edits[base].to(w.device).float()).to(w.dtype)
        shard[base] = w.cpu().contiguous(); size += w.numel() * 2
        if size > 4e9: flush()
    flush()
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        shutil.copy(os.path.join(art, fn), dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("art"); ap.add_argument("dst"); a = ap.parse_args()
    unpack_artifact(a.art, a.dst)
