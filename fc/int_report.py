"""Exam report: int_report.py exam.jsonl base.jsonl build.jsonl [build2.jsonl ...] -> kept per category."""
import json, sys, collections
def norm(s): return s.strip().strip("`'\".").strip().lower().split("\n")[0].strip()
ex = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
def load(p): return {json.loads(l)["id"]: norm(json.loads(l)["text"]) for l in open(p)}
b = load(sys.argv[2])
for p in sys.argv[3:]:
    q = load(p); st = collections.defaultdict(lambda: [0, 0, 0, 0])
    for k, it in ex.items():
        g = it["gold"].lower()
        for c in ("ALL", it["cat"], "clinc_all" if it["cat"].startswith("clinc") else "unseen_all"):
            r = st[c]; r[0] += 1; r[1] += b[k] == g; r[2] += q[k] == g; r[3] += b[k] == g and q[k] == g
    print("==", p)
    for c, (n, bo, qo, kept) in sorted(st.items()):
        print(f"{c:28s} n={n:5d} base={bo/n:.4f} build={qo/n:.4f} kept={kept}/{bo}={kept/max(bo,1):.4f}")
