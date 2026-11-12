"""Locked comparison: learn from the locked feedback set (real / shuffled labels / empty) with both methods,
check on the locked probes and nearby items (teacher-forced exact check), and save what each run learned."""
import argparse, json, random, time, torch, sys, os
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from feedback import FB
ap = argparse.ArgumentParser()
for k in ["ckpt", "fb", "probe", "guard", "nearby", "calib", "out"]: ap.add_argument("--" + k, required=True)
ap.add_argument("--only-switch", action="store_true"); ap.add_argument("--guard-tol", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
def load(p): return [json.loads(l) for l in open(p)]
fb = FB(a.ckpt)
F, P, G, N = load(a.fb), load(a.probe), load(a.guard), load(a.nearby)
enc = lambda its: [fb.encode(it, it["annotation"]) for it in its]
Fp, Pp, Gp, Np = enc(F), enc(P), enc(G), enc(N)
perm = list(range(len(F))); rng = random.Random(0)
while any(i == j for i, j in enumerate(perm)): rng.shuffle(perm)
Fs = [fb.encode(F[i], F[perm[i]]["annotation"]) for i in range(len(F))]
def ev(tag, extra=None):
    r = {"tag": tag, "probe_fixed": len(Pp) - fb.n_wrong(Pp), "nearby_broken": fb.n_wrong(Np), "guard_broken": fb.n_wrong(Gp)}
    r.update(extra or {}); print(json.dumps(r), flush=True); return r
res = [ev("empty")]
for nm, data in [("real", Fp), ("shuffled", Fs)]:
    for mo in (24, 48):
        off, log = fb.learn_switch(data, Gp, max_off=mo, rounds=mo // 4 + 2, guard_tol=a.guard_tol)
        res.append(ev(f"switch{mo}_{nm}", {"n_off": len(off)})); fb.set_bias([])
        json.dump(off, open(f"{a.out}/switch{mo}_{nm}.json", "w"))
if a.only_switch:
    json.dump(res, open(f"{a.out}/results.json", "w")); sys.exit(0)
layers = list(range(20, 36))
calp = [fb.encode(it, it["text"]) for it in load(a.calib)[:120]]
C = fb.key_cov(calp, layers)
for nm, data in [("real", Fp), ("shuffled", Fs)]:
    W0 = {l: fb.layers[l].self_attn.o_proj.weight.data.clone() for l in layers}
    fac, log = fb.learn_edit(data, layers, C, lam=100, tau=3, rounds=12)
    rank = sum(f[0].shape[1] for f in fac[layers[0]])
    res.append(ev(f"edit_{nm}", {"rank": rank, "log": log[-1]}))
    torch.save({l: (torch.cat([d for d, _ in fac[l]], 1), torch.cat([A for _, A in fac[l]], 1)) for l in layers if fac[l]}, f"{a.out}/edit_{nm}.pt")
    fb.reset_edit(W0)
json.dump(res, open(f"{a.out}/results.json", "w"))
