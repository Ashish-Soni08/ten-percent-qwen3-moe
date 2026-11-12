"""Score the locked exam: base accuracy per category, faithfulness of each variant (share of base-correct items kept,
overall and per category), and probe / nearby results."""
import json, sys, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from score import score_item
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
exam = load(sys.argv[1]); R = sys.argv[2]; variants = sys.argv[3].split(",")
base = load(f"{R}/base/exam.jsonl")
bok = {k: score_item(exam[k], base[k]["text"]) for k in exam}
rep = {"base": {}}
cats = sorted({it["category"] for it in exam.values()})
for c in cats:
    ks = [k for k in exam if exam[k]["category"] == c]; rep["base"][c] = sum(bok[k] for k in ks) / len(ks)
rep["base"]["all"] = sum(bok.values()) / len(bok)
for v in variants:
    out = load(f"{R}/{v}/exam.jsonl"); vok = {k: score_item(exam[k], out[k]["text"]) for k in exam}
    r = {}
    for c in cats + ["all"]:
        ks = [k for k in exam if c == "all" or exam[k]["category"] == c]
        kb = [k for k in ks if bok[k]]
        r[c] = {"acc": round(sum(vok[k] for k in ks) / len(ks), 4), "kept": round(sum(vok[k] for k in kb) / len(kb), 4)}
    for s in ["probe_locked", "nearby_test"]:
        try:
            its = load(f"data/fb/{s}.jsonl"); o = load(f"{R}/{v}/{s}.jsonl")
            r[s] = sum(score_item(its[k], o[k]["text"]) for k in its) / len(its)
        except FileNotFoundError: pass
    rep[v] = r
print(json.dumps(rep, indent=1))
