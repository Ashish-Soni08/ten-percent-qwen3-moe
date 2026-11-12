"""Faithfulness of a variant vs the base: share of base-correct items the variant keeps (gold sets), and
agreement of parsed calls with the base (no-gold sets)."""
import json, sys, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from score import score_item, parse_calls

def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
def canon(t):
    c = parse_calls(t)
    return None if c is None else sorted(json.dumps(x, sort_keys=True) for x in c)

def report(base_out, var_out, items_path, tag=""):
    items, b, v = load(items_path), load(base_out), load(var_out)
    per = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # base_ok, kept, var_ok, agree, n
    for k, it in items.items():
        if k not in b or k not in v: continue
        r = per[it["category"]]; r[4] += 1
        r[3] += canon(b[k]["text"]) == canon(v[k]["text"])
        if it.get("gold") is not None:
            bo = score_item(it, b[k]["text"]); vo = score_item(it, v[k]["text"])
            r[0] += bo; r[1] += bo and vo; r[2] += vo
    res = {}
    for c, (bo, kept, vo, ag, n) in sorted(per.items()):
        res[c] = {"n": n, "base_acc": bo / n, "var_acc": vo / n, "kept": kept / max(bo, 1), "agree": ag / n}
        print(f"{tag:22s} {c:14s} n={n:4d} base={bo/n:.3f} var={vo/n:.3f} kept={kept/max(bo,1):.3f} agree={ag/n:.3f}", flush=True)
    return res

if __name__ == "__main__":
    report(*sys.argv[1:4], tag=sys.argv[4] if len(sys.argv) > 4 else "")
