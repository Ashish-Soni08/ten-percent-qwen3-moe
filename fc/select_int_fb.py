"""Pick intent feedback from the build's real mistakes on the CLINC pool.
Class = mistakes on the gold intents where the build errs most. Split: 25 feedback, rest locked probes;
nearby = pool items of those intents and of the wrongly-predicted labels that the build answers right;
guard = 160 build-right items of other intents (annotation = the build's own answer)."""
import json, random, collections, hashlib, os, sys
def norm(s): return s.strip().strip("`'\".").strip().lower().split("\n")[0].strip()
pool = {json.loads(l)["id"]: json.loads(l) for l in open("data/int_pool.jsonl")}
o = json.load(open("runs/pool_outs.json")); B, Q = o["base"], o["q"]
rng = random.Random(20261111)
wrong = [k for k in pool if norm(Q[k]) != pool[k]["gold"]]
print("build wrong", len(wrong), "of", len(pool), "; base wrong", sum(norm(B[k]) != pool[k]["gold"] for k in pool))
byg = collections.defaultdict(list)
for k in wrong: byg[pool[k]["gold"]].append(k)
top = sorted(byg, key=lambda g: -len(byg[g]))
S, n = [], 0
for g in top:
    S.append(g); n += len(byg[g])
    if n >= 90 and len(S) >= 3: break
print("class intents", [(g, len(byg[g])) for g in S])
cls = [k for g in S for k in byg[g]]; rng.shuffle(cls)
fb, probe = cls[:25], cls[25:]
preds = {norm(Q[k]) for k in cls}
near = [k for k in pool if pool[k]["gold"] in set(S) | preds and norm(Q[k]) == pool[k]["gold"] and k not in cls]
rng.shuffle(near); near = near[:300]
other = [k for k in pool if pool[k]["gold"] not in set(S) | preds and norm(Q[k]) == pool[k]["gold"]]
rng.shuffle(other); guard = other[:160]
os.makedirs("data/ifb", exist_ok=True)
def w(name, ks, ann):
    with open(f"data/ifb/{name}.jsonl", "w") as f:
        for k in ks: f.write(json.dumps(dict(pool[k], annotation=ann(k))) + "\n")
    print(name, len(ks), hashlib.sha256(open(f"data/ifb/{name}.jsonl", "rb").read()).hexdigest()[:16])
w("feedback", fb, lambda k: pool[k]["gold"]); w("probe_locked", probe, lambda k: pool[k]["gold"])
w("nearby_test", near, lambda k: pool[k]["gold"]); w("guard", guard, lambda k: norm(Q[k]))
