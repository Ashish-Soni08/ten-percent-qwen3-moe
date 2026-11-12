"""Build a packed Qwen3.5-MoE artifact (text model only) without any training.

  1. Expert selection: keep the top-K routed experts per layer by routing mass measured on the task's calibration
     prompts with the base model's own answers (fc/q35_route_stats.py). Router rows are sliced to match.
  2. Quantization (round-to-nearest with MSE-optimal clipping, groups of 128, fp16 scale and min):
     routed experts 3-bit, attention / linear-attention / shared expert / embeddings 4-bit, LM head 6-bit.
     Routers, norms, convolutions and other small tensors stay bf16. The vision tower and MTP layer are dropped.
Usage: q35_build.py --model M --stats s.pt --k K --out ART
"""
import argparse, json, os, shutil, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
from safetensors.torch import save_file
from qpack import quantize, pack

FP16_BYTES = 71_903_655_008  # Qwen/Qwen3.5-35B-A3B total_size (bf16) from model.safetensors.index.json
G = 128

def bits_for(name, t):
    if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"): return EXPERT_BITS
    if name == "lm_head.weight": return 6
    if t.dim() != 2 or t.numel() < 1_000_000 or t.shape[-1] % G: return None
    if name.endswith("mlp.gate.weight") or name.endswith("shared_expert_gate.weight"): return None
    return 4

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for x in ["model", "stats", "out"]: ap.add_argument("--" + x, required=True)
    ap.add_argument("--k", type=int, required=True); ap.add_argument("--expert-bits", type=int, default=3)
    a = ap.parse_args(); EXPERT_BITS = a.expert_bits; os.makedirs(a.out, exist_ok=True)
    mass = torch.load(a.stats)["mass"]; L = mass.shape[0]
    keep = [sorted(mass[l].topk(a.k).indices.tolist()) for l in range(L)]
    print("min routing-mass coverage", f"{min((mass[l, keep[l]].sum() / mass[l].sum()).item() for l in range(L)):.4f}", flush=True)
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    out, meta = {}, {"group": G, "K": a.k, "tensors": {}}
    for fn in sorted(set(idx.values())):
        with safe_open(os.path.join(a.model, fn), "pt", device="cuda") as f:
            for k in f.keys():
                if k.startswith("mtp.") or k.startswith("model.visual."): continue
                t = f.get_tensor(k)
                if ".layers." in k and ".mlp." in k:
                    l = int(k.split(".layers.")[1].split(".")[0]); ix = torch.tensor(keep[l], device=t.device)
                    if k.endswith("mlp.gate.weight") or k.endswith("experts.gate_up_proj") or k.endswith("experts.down_proj"):
                        t = t[ix].contiguous()
                b = bits_for(k, t)
                if b is None:
                    out[k] = t.cpu().contiguous(); continue
                codes, sc, mn = quantize(t, b, G)
                out[k + ".qcodes"] = pack(codes, b).cpu(); out[k + ".qscale"] = sc.cpu(); out[k + ".qmin"] = mn.cpu()
                meta["tensors"][k] = {"bits": b, "shape": list(t.shape), "dtype": str(t.dtype)}
                del codes, sc, mn
        print("read", fn, flush=True)
    save_file(out, os.path.join(a.out, "weights.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    tc = dict(cfg["text_config"]); tc["num_experts"] = a.k; tc["mtp_num_hidden_layers"] = 0
    tc["architectures"] = ["Qwen3_5MoeForCausalLM"]; tc["tie_word_embeddings"] = cfg.get("tie_word_embeddings", False)
    json.dump(tc, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(a.out, "base_config.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(a.out, "qmeta.json"), "w"))
    json.dump(keep, open(os.path.join(a.out, "keep.json"), "w"))
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "vocab.json", "merges.txt"]:
        if os.path.exists(os.path.join(a.model, f)): shutil.copy(os.path.join(a.model, f), a.out)
    tot = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"ARTIFACT {a.out} bytes {tot} = {100 * tot / FP16_BYTES:.3f}% of fp16 ({FP16_BYTES})", flush=True)
