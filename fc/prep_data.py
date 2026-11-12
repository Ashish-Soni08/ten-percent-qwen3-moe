"""Build the dev, calibration, feedback-pool and LOCKED exam splits.

Sources
  xLAM function-calling 60k (CC-BY-4.0)  -> in-distribution: calibration, dev, feedback pool, locked exam (xlam_*)
  BFCL v3 (Apache-2.0)                   -> unseen source: locked exam only (bfcl_*). Never used to build, tune or calibrate.

Every item: {id, category, source, messages, tools (OpenAI/Qwen JSON-schema), gold}
  xLAM gold: list of {name, arguments} (+ tool defaults kept in tools)
  BFCL gold: BFCL possible_answer list of {fname: {param: [acceptable values]}}
"""
import hashlib, json, random, sys
from pathlib import Path

RAW = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261110

def xlam_type(t):
    t = (t or "str").split(",")[0].strip().lower()
    if t.startswith(("list", "tuple", "set")): return "array"
    if t.startswith("dict"): return "object"
    return {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(t, "string")

def xlam_tools(tools):
    out = []
    for t in tools:
        props, req = {}, []
        for p, v in t["parameters"].items():
            pd = {"type": xlam_type(v.get("type")), "description": v.get("description", "")}
            if "default" in v and v["default"] not in (None, ""): pd["default"] = v["default"]
            props[p] = pd
            if "optional" not in (v.get("type") or "") and "default" not in v: req.append(p)
        out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                    "parameters": {"type": "object", "properties": props, "required": req}}})
    return out

def fix_schema(s):
    if isinstance(s, dict):
        s = {k: fix_schema(v) for k, v in s.items()}
        if s.get("type") == "dict": s["type"] = "object"
        if s.get("type") == "float": s["type"] = "number"
        if s.get("type") == "tuple": s["type"] = "array"
        if s.get("type") == "any": s["type"] = "string"
        return s
    if isinstance(s, list): return [fix_schema(x) for x in s]
    return s

def bfcl_tools(funcs):
    return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
             "parameters": fix_schema(f["parameters"])}} for f in funcs]

# ---------- xLAM ----------
xl = json.load(open(RAW / "xlam60k.json"))
items = []
for x in xl:
    ans = json.loads(x["answers"])
    cat = "xlam_single" if len(ans) == 1 else "xlam_parallel"
    items.append({"id": f"xlam_{x['id']}", "category": cat, "source": "xlam",
                  "messages": [{"role": "user", "content": x["query"]}],
                  "tools": xlam_tools(json.loads(x["tools"])), "gold": ans})
rng = random.Random(SEED); rng.shuffle(items)
by = {"xlam_single": [], "xlam_parallel": []}
for it in items: by[it["category"]].append(it)
splits = {"exam": [], "dev": [], "calib": [], "pool": []}
for cat, lst in by.items():
    splits["exam"] += lst[:600]; splits["dev"] += lst[600:1000]
    splits["calib"] += lst[1000:1750]; splits["pool"] += lst[1750:]

# ---------- BFCL (unseen source: exam only) ----------
def load(f): return [json.loads(l) for l in open(RAW / "bfcl" / f)]
def bfcl(name, cat, n=None):
    qs = load(f"BFCL_v3_{name}.json"); ans = {a["id"]: a["ground_truth"] for a in load(f"possible_answer/BFCL_v3_{name}.json")}
    out = [{"id": f"bfcl_{q['id']}", "category": cat, "source": "bfcl", "messages": q["question"][0],
            "tools": bfcl_tools(q["function"]), "gold": ans[q["id"]]} for q in qs if q["id"] in ans]
    if n is not None: random.Random(SEED).shuffle(out); out = out[:n]
    return out
bf = (bfcl("simple", "bfcl_simple") + bfcl("live_simple", "bfcl_simple") +
      bfcl("multiple", "bfcl_multiple") + bfcl("live_multiple", "bfcl_multiple", 300) +
      bfcl("parallel", "bfcl_parallel") + bfcl("parallel_multiple", "bfcl_parallel") +
      bfcl("live_parallel", "bfcl_parallel") + bfcl("live_parallel_multiple", "bfcl_parallel"))
splits["exam"] += bf

for k, v in splits.items():
    p = OUT / f"{k}.jsonl"
    with open(p, "w") as f:
        for it in v: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(k, len(v), dict(Counter(i["category"] for i in v)), hashlib.sha256(p.read_bytes()).hexdigest()[:16])
