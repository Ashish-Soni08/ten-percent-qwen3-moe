"""Reconstruction-based expert selection. For each layer, greedily drop the expert whose removal (with the router
re-picking top-8 among the rest) least increases the MoE output error on calibration tokens; then split a global
expert budget across layers by marginal relative error. Writes a plan like make_plan.py."""
import argparse, json, random, time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
for k in ["model", "items", "answers", "out"]: ap.add_argument("--" + k, required=True)
ap.add_argument("--total", type=int, required=True); ap.add_argument("--ntok", type=int, default=6144)
ap.add_argument("--minper", type=int, default=24); ap.add_argument("--min-prompt", type=int, default=0)
ap.add_argument("--bits", default='{"expert": 3, "attn": 4, "embed": 4, "head": 6}')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", experts_implementation="grouped_mm"); m.eval()
L = m.config.num_hidden_layers; E = m.config.num_experts; K = m.config.num_experts_per_tok
items = {json.loads(x)["id"]: json.loads(x) for x in open(a.items)}
ans = [json.loads(x) for x in open(a.answers)]; random.Random(1).shuffle(ans)
X = [[] for _ in range(L)]; state = {}
def hook(l):
    def h(mod, args):
        x = args[0].reshape(-1, args[0].shape[-1]); X[l].append(x[state["sel"]].detach())
    return h
hs = [layer.mlp.register_forward_pre_hook(hook(l)) for l, layer in enumerate(m.model.layers)]
n = 0; g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for r in ans:
        if n >= a.ntok: break
        it = items[r["id"]]
        p = tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False)
        pi = tok(p, add_special_tokens=False).input_ids; ai = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
        T = len(pi) + len(ai)
        # all answer tokens + an equal number of random prompt tokens
        pr = torch.randperm(len(pi), generator=g)[: max(len(ai), a.min_prompt)].tolist()
        sel = torch.tensor(sorted(pr + list(range(len(pi), T))), device="cuda"); state["sel"] = sel
        m(input_ids=torch.tensor([pi + ai], device="cuda")); n += len(sel)
for h in hs: h.remove()
print("tokens", n, flush=True)
order, curves = [], []
t0 = time.time()
with torch.no_grad():
    for l, layer in enumerate(m.model.layers):
        x = torch.cat(X[l]); X[l] = None
        moe = layer.mlp; ex = moe.experts
        logits = (x.float() @ moe.gate.weight.float().T)
        Y = torch.empty(E, x.shape[0], x.shape[1], device="cuda", dtype=torch.bfloat16)
        for e in range(E):
            gg, u = F.linear(x, ex.gate_up_proj[e]).chunk(2, -1); Y[e] = F.linear(ex.act_fn(gg) * u, ex.down_proj[e])
        def out(alive):
            lg = logits.masked_fill(~alive, float("-inf"))
            tv, ti = lg.softmax(-1).topk(K, -1); tv = tv / tv.sum(-1, keepdim=True)
            idx = torch.arange(x.shape[0], device="cuda")
            return sum(Y[ti[:, j], idx].float() * tv[:, j, None] for j in range(K))
        alive = torch.ones(E, dtype=torch.bool, device="cuda"); y0 = out(alive); nrm = y0.pow(2).sum()
        removed, errs = [], []
        while alive.sum() > a.minper:
            best = None
            for e in torch.where(alive)[0].tolist():
                al = alive.clone(); al[e] = False
                err = (out(al) - y0).pow(2).sum() / nrm
                if best is None or err < best[1]: best = (e, err.item())
            alive[best[0]] = False; removed.append(best[0]); errs.append(best[1])
        order.append(removed); curves.append(errs)
        print(l, f"{time.time()-t0:.0f}s", "err at keep 64/48/32:", [round(errs[E - k - 1], 4) for k in (64, 48, 32) if E - k - 1 < len(errs)], flush=True)
        del Y
# allocate: start from minper per layer, add back experts (reverse removal order) by smallest error reduction cost
keepn = [a.minper] * L
def err_at(l, k):  # error when keeping k experts
    i = E - k - 1
    return 0.0 if k >= E else curves[l][i] if i < len(curves[l]) else curves[l][-1]
while sum(keepn) < a.total:
    gains = [err_at(l, keepn[l]) - err_at(l, keepn[l] + 1) for l in range(L)]
    l = max(range(L), key=lambda i: gains[i]); keepn[l] += 1
keep = []
for l in range(L):
    dropped = order[l][: E - keepn[l]]
    keep.append(sorted(set(range(E)) - set(dropped)))
json.dump({"keep": keep, "bits": json.loads(a.bits), "group": 128, "expert_bits": {}}, open(a.out, "w"))
json.dump({"order": order, "curves": curves}, open(a.out.replace(".json", "_curves.json"), "w"))
print("experts per layer:", keepn)
