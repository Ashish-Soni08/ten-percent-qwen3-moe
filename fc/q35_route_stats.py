"""Routing mass per (layer, expert) for an MoE model via HF forward hooks on router linears (out_features == n_experts).
Usage: q35_route_stats.py --model M --items calib.jsonl --answers base_answers.jsonl --out stats.pt [--n 1500]"""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
for k in ["items", "answers", "out"]: ap.add_argument("--" + k, action="append", required=True)
ap.add_argument("--n", type=int, default=600); ap.add_argument("--k", type=int, default=8); ap.add_argument("--E", type=int, default=256)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
try:
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
except Exception as e:
    print("CausalLM load failed:", e)
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
m.eval(); print(type(m).__name__, flush=True)
gates = [(n, mod) for n, mod in m.named_modules() if n.endswith("mlp.gate") and "mtp" not in n]
if not gates:  # some MoE blocks keep the router as a bare Parameter
    gates = [(n, mod) for n, mod in m.named_modules() if n.endswith(".gate") and hasattr(mod, "weight") and mod.weight.shape[0] == a.E]
print("router modules", len(gates), gates[0][0] if gates else None, flush=True)
L = len(gates); mass = torch.zeros(L, a.E, dtype=torch.float64); cnt = torch.zeros(L, a.E, dtype=torch.float64)
mask = {"m": None}
def hook(i):
    def h(mod, args, out):
        x = args[0].reshape(-1, args[0].shape[-1]); lg = (x.float() @ mod.weight.float().T).reshape(-1, a.E)
        p = lg.softmax(-1); w, idx = p.topk(a.k, -1); w = w / w.sum(-1, keepdim=True)
        keep = mask["m"].reshape(-1)[: lg.shape[0]]
        w, idx = w[keep], idx[keep]
        mass[i].index_add_(0, idx.reshape(-1).cpu(), w.reshape(-1).double().cpu())
        cnt[i].index_add_(0, idx.reshape(-1).cpu(), torch.ones(idx.numel(), dtype=torch.float64))
    return h
for i, (_, mod) in enumerate(gates): mod.register_forward_hook(hook(i))
import time
for IT, AN, OUT in zip(a.items, a.answers, a.out):
  mass.zero_(); cnt.zero_(); t0 = time.time()
  items = {json.loads(l)["id"]: json.loads(l) for l in open(IT)}
  ans = [json.loads(l) for l in open(AN)][: a.n]
  ntok = 0
  with torch.no_grad():
    for r in ans:
        it = items[r["id"]]
        p = tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(p + r["text"], return_tensors="pt").input_ids[:, :4096].cuda()
        mask["m"] = torch.ones_like(ids, dtype=torch.bool)
        m(input_ids=ids); ntok += ids.numel()
  print(OUT, "tokens", ntok, f"{time.time()-t0:.0f}s", flush=True)
  torch.save({"mass": mass.clone(), "cnt": cnt.clone(), "names": [n for n, _ in gates]}, OUT)
