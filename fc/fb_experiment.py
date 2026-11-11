"""Tune / compare the two feedback methods with teacher-forced exact checks (greedy output == annotation)."""
import argparse, json, random, time, torch, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from feedback import FB

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--fb", required=True); ap.add_argument("--probe", required=True)
ap.add_argument("--guard", required=True); ap.add_argument("--nearby", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--configs", default="all"); ap.add_argument("--max-off", type=int, default=24)
ap.add_argument("--layer-sets", default="36-43,28-43"); ap.add_argument("--lams", default="1000,10000"); ap.add_argument("--tau", type=float, default=2.0); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--out", default=None)
a = ap.parse_args()
def load(p): return [json.loads(l) for l in open(p)]
fb = FB(a.ckpt)
def pairs(items, key="annotation"): return [fb.encode(it, it[key]) for it in items]
F, P, G, N = load(a.fb), load(a.probe), load(a.guard), load(a.nearby)
Fp, Pp, Gp, Np = pairs(F), pairs(P), pairs(G), pairs(N)
perm = list(range(len(F))); rng = random.Random(0)
while any(i == j for i, j in enumerate(perm)): rng.shuffle(perm)
Fs = [fb.encode(F[i], F[perm[i]]["annotation"]) for i in range(len(F))]
def evaluate(tag, t0):
    r = {"tag": tag, "fb_fixed": len(Fp) - fb.n_wrong(Fp), "probe_fixed": len(Pp) - fb.n_wrong(Pp),
         "guard_broken": fb.n_wrong(Gp), "nearby_broken": fb.n_wrong(Np), "secs": round(time.time() - t0)}
    print(json.dumps(r), flush=True); return r
res = []
fb.set_bias([]); res.append(evaluate("none", time.time()))
cal = [json.loads(l) for l in open(a.calib)][:120]
calp = [fb.encode(it, it["text"]) for it in cal]
cfgs = a.configs.split(",")
# ----- method A
for name, data in [("switch_real", Fp), ("switch_shuffled", Fs)]:
    if a.configs != "all" and name not in cfgs: continue
    t0 = time.time(); off, log = fb.learn_switch(data, Gp, max_off=a.max_off, rounds=a.max_off // 4 + 2)
    print(name, "off:", off, "log:", log, flush=True)
    r = evaluate(name, t0); r["off"] = off; res.append(r); fb.set_bias([])
# ----- method B
for layers_s in a.layer_sets.split(","):
    lo, hi = map(int, layers_s.split("-")); layers = list(range(lo, hi + 1))
    C = fb.key_cov(calp, layers)
    for lam in [float(x) for x in a.lams.split(",")]:
        for name, data in [("edit_real", Fp), ("edit_shuffled", Fs)]:
            tag = f"{name}_L{layers_s}_lam{lam:g}"
            if a.configs != "all" and name not in cfgs and tag not in cfgs: continue
            W0 = {l: fb.layers[l].self_attn.o_proj.weight.data.clone() for l in layers}
            t0 = time.time(); fac, log = fb.learn_edit(data, layers, C, lam=lam, tau=a.tau, rounds=a.rounds)
            print(tag, "log:", log, flush=True)
            r = evaluate(tag, t0); r["rank"] = sum(f[0].shape[1] for f in fac[layers[0]]); res.append(r)
            fb.reset_edit(W0)
if a.out: json.dump(res, open(a.out, "w"))
