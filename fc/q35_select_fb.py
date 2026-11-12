"""Pick feedback from the build's real mistakes on a task's feedback pool (never used to calibrate or build).
Class = mistakes on the gold labels where the build errs most (at least 90 mistakes). 25 become feedback, the rest
locked probes. Nearby = pool items of the class labels and of the wrongly predicted labels that the build answers
right (300). Guard = 160 other items the build answers right (annotation = the build's own answer).
Usage: q35_select_fb.py pool.jsonl base_out.jsonl build_out.jsonl outdir"""
import json, random, collections, hashlib, os, sys
from score_label import norm
pool_p, base_p, q_p, od = sys.argv[1:5]
pool = {json.loads(l)["id"]: json.loads(l) for l in open(pool_p)}
B = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(base_p)}
Q = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(q_p)}
G = lambda k: pool[k]["gold"].lower()
rng = random.Random(20261112)
wrong = [k for k in pool if norm(Q[k]) != G(k)]
print("build wrong", len(wrong), "of", len(pool), "; base wrong", sum(norm(B[k]) != G(k) for k in pool))
byg = collections.defaultdict(list)
for k in wrong: byg[G(k)].append(k)
S, n = [], 0
for g in sorted(byg, key=lambda g: -len(byg[g])):
    S.append(g); n += len(byg[g])
    if n >= 90: break
print("class labels", [(g, len(byg[g])) for g in S])
cls = [k for g in S for k in byg[g]]; rng.shuffle(cls)
fb, probe = cls[:25], cls[25:]
labs = set(S) | {norm(Q[k]) for k in cls}
right = [k for k in pool if norm(Q[k]) == G(k)]; rng.shuffle(right)
near = [k for k in right if G(k) in labs][:300]
used = set(near)
other = [k for k in right if G(k) not in labs]
if len(other) < 160: other += [k for k in right if k not in used and k not in other]
guard = other[:160]
os.makedirs(od, exist_ok=True)
def w(name, ks, ann):
    with open(f"{od}/{name}.jsonl", "w") as f:
        for k in ks: f.write(json.dumps(dict(pool[k], annotation=ann(k))) + "\n")
    print(name, len(ks), hashlib.sha256(open(f"{od}/{name}.jsonl", "rb").read()).hexdigest()[:16])
w("feedback", fb, G); w("probe_locked", probe, G); w("nearby_test", near, G); w("guard", guard, lambda k: norm(Q[k]))
