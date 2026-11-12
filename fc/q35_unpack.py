"""Turn a packed Qwen3.5 artifact into a plain bf16 checkpoint (a runtime cache, not shipped) that SGLang and
transformers load. router_off: (layer, slot) pairs switched off by feedback; they get router offset -1e4.
Every other router offset is 0, so with an empty list the cache computes exactly the shipped model."""
import argparse, json, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
from qpack import unpack, dequantize

def unpack_artifact(art, dst, router_off=None, with_bias=True):
    os.makedirs(dst, exist_ok=True)
    meta = json.load(open(os.path.join(art, "qmeta.json"))); G = meta["group"]; K = meta["K"]
    f = safe_open(os.path.join(art, "weights.safetensors"), "pt", device="cuda")
    shard, n_shard, size, wmap = {}, 0, 0, {}
    def flush():
        nonlocal shard, n_shard, size
        if not shard: return
        fn = f"model-{n_shard:05d}.safetensors"; save_file(shard, os.path.join(dst, fn), metadata={"format": "pt"})
        for k in shard: wmap[k] = fn
        shard, size = {}, 0; n_shard += 1
    for n in f.keys():
        if n.endswith((".qscale", ".qmin")): continue
        if n.endswith(".qcodes"):
            base = n[: -len(".qcodes")]; m = meta["tensors"][base]
            w = dequantize(unpack(f.get_tensor(n), m["bits"], m["shape"]), f.get_tensor(base + ".qscale"), f.get_tensor(base + ".qmin"), G)
        else:
            base = n; w = f.get_tensor(n)
        if base.endswith("mlp.gate.weight") and with_bias:
            l = int(base.split(".layers.")[1].split(".")[0])
            b = torch.zeros(K, dtype=torch.bfloat16)
            for (ll, e) in (router_off or []):
                if ll == l: b[e] = -1e4
            shard[base[: -len(".weight")].replace("model.language_model.", "model.") + ".bias"] = b
        shard[base.replace("model.language_model.", "model.")] = w.cpu().contiguous(); size += w.numel() * w.element_size()
        if size > 4e9: flush()
    flush()
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    for fn in os.listdir(art):
        if fn.endswith(".json") and fn not in ("qmeta.json", "keep.json", "base_config.json") or fn.endswith((".jinja", ".txt")):
            shutil.copy(os.path.join(art, fn), dst)
    return dst

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("art"); ap.add_argument("dst"); a = ap.parse_args()
    unpack_artifact(a.art, a.dst)
