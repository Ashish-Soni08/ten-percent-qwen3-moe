"""Write a compression plan: keep the top-K experts per layer by a saliency criterion from reap stats."""
import argparse, json, torch
ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True); ap.add_argument("--k", type=int, required=True)
ap.add_argument("--crit", default="sum", choices=["sum", "mean", "freq"])
ap.add_argument("--expert", type=int, default=4); ap.add_argument("--attn", type=int, default=4)
ap.add_argument("--embed", type=int, default=4); ap.add_argument("--head", type=int, default=6)
ap.add_argument("--group", type=int, default=128); ap.add_argument("--imp", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--total", type=int, default=0, help="global expert budget (non-uniform per layer)")
ap.add_argument("--hi", type=int, default=0, help="number of top experts (global) quantized at --hibits")
ap.add_argument("--hibits", type=int, default=4); ap.add_argument("--minper", type=int, default=16)
a = ap.parse_args()
st = torch.load(a.stats)
S, C = st["S_all"], st["C_all"]
score = {"sum": S, "mean": S / C.clamp(min=1), "freq": C}[a.crit]
L, E = S.shape
ebits = {}
if a.total:
    frac = score / score.sum(1, keepdim=True)
    order = frac.flatten().argsort(descending=True).tolist()
    keep = [[] for _ in range(L)]
    for l in range(L): keep[l] = frac[l].topk(a.minper).indices.tolist()
    n = sum(len(k) for k in keep)
    for i in order:
        if n >= a.total: break
        l, e = divmod(i, E)
        if e not in keep[l]: keep[l].append(e); n += 1
    keep = [sorted(k) for k in keep]
    for i in order[: a.hi]:
        l, e = divmod(i, E)
        if e in keep[l]: ebits[f"{l}.{e}"] = a.hibits
else:
    keep = [sorted(score[l].topk(a.k).indices.tolist()) for l in range(L)]
json.dump({"keep": keep, "bits": {"expert": a.expert, "attn": a.attn, "embed": a.embed, "head": a.head},
           "group": a.group, "imp": a.imp, "expert_bits": ebits}, open(a.out, "w"))
print("experts per layer:", [len(k) for k in keep])
cov = torch.stack([S[l, keep[l]].sum() / S[l].sum() for l in range(L)])
print(f"plan k={a.k} crit={a.crit} saliency kept: mean {cov.mean():.3f} min {cov.min():.3f}")
