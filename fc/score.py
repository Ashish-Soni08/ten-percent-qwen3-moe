"""Parse Qwen3 tool calls and score them.

score_item(item, text) -> bool. Rules:
  xLAM: the set of calls must match the gold calls one to one (any order). Within a call the name must match and each
        argument value must match; an argument that only one side has is fine when it equals the tool's default.
  BFCL: AST-style check against BFCL's possible answers: every predicted argument must be one of the acceptable values,
        every required argument (no "" in its acceptable list) must be present, calls matched one to one in any order.
"""
import json, re

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)

def parse_calls(text):
    calls = []
    for m in CALL_RE.finditer(text or ""):
        s = m.group(1).strip()
        if not s: continue
        try: obj = json.loads(s)
        except Exception:
            return None  # malformed output
        if not isinstance(obj, dict) or "name" not in obj: return None
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: return None
        if not isinstance(args, dict): return None
        calls.append({"name": obj["name"], "arguments": args})
    return calls

def _num(x):
    if isinstance(x, bool): return None
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        try: return float(x.strip())
        except Exception: return None
    return None

def veq(a, b):
    """Lenient value equality used for xLAM."""
    if isinstance(a, bool) or isinstance(b, bool):
        sa = str(a).strip().lower(); sb = str(b).strip().lower(); return sa == sb
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None and (not isinstance(a, str) or not isinstance(b, str)):
        return abs(na - nb) <= 1e-6 * max(1.0, abs(na), abs(nb))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(veq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(veq(a[k], b[k]) for k in a)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, str) and isinstance(b, (list, dict)) or isinstance(b, str) and isinstance(a, (list, dict)):
        s, o = (a, b) if isinstance(a, str) else (b, a)
        try: return veq(json.loads(s), o)
        except Exception: return False
    return a == b

def _bijection(preds, golds, match):
    if len(preds) != len(golds): return False
    used = [False] * len(golds)
    def rec(i):
        if i == len(preds): return True
        for j in range(len(golds)):
            if not used[j] and match(preds[i], golds[j]):
                used[j] = True
                if rec(i + 1): return True
                used[j] = False
        return False
    return rec(0)

def _defaults(item):
    d = {}
    for t in item["tools"]:
        f = t["function"]; d[f["name"]] = {p: v.get("default", None) for p, v in f["parameters"].get("properties", {}).items()}
    return d

def xlam_call_match(p, g, defaults):
    if p["name"] != g["name"]: return False
    dflt = defaults.get(g["name"], {})
    pa, ga = p["arguments"], g["arguments"]
    for k in set(pa) | set(ga):
        if k in pa and k in ga:
            if not veq(pa[k], ga[k]): return False
        else:
            v = pa[k] if k in pa else ga[k]
            if k not in dflt or dflt[k] is None or not veq(v, dflt[k]): return False
    return True

_STD = re.compile(r"[ ,./\-_*^]")
def bstd(s): return _STD.sub("", s.lower())

def bfcl_val(v, acc):
    """v predicted value, acc list of acceptable values."""
    for a in acc:
        if bfcl_eq(v, a): return True
    return False

def bfcl_eq(v, a):
    if isinstance(a, str) and isinstance(v, str): return bstd(v) == bstd(a)
    if isinstance(a, bool) or isinstance(v, bool): return v is a or (isinstance(v, str) and v.lower() == str(a).lower())
    if isinstance(a, (int, float)) and isinstance(v, (int, float)): return abs(float(a) - float(v)) < 1e-9
    if isinstance(a, list) and isinstance(v, list):
        return len(a) == len(v) and all(bfcl_eq(x, y) for x, y in zip(v, a))
    if isinstance(a, dict) and isinstance(v, dict):
        for k, accs in a.items():
            accs = accs if isinstance(accs, list) else [accs]
            if k in v:
                if not bfcl_val(v[k], accs): return False
            elif "" not in accs: return False
        return all(k in a for k in v)
    return False

def bfcl_call_match(p, g):
    (gname, gargs), = g.items()
    if p["name"] != gname: return False
    for k, v in p["arguments"].items():
        if k not in gargs or not bfcl_val(v, gargs[k]): return False
    for k, acc in gargs.items():
        if k not in p["arguments"] and "" not in acc: return False
    return True

def score_item(item, text):
    calls = parse_calls(text)
    if calls is None: return False
    if item["source"] == "xlam":
        d = _defaults(item)
        return _bijection(calls, item["gold"], lambda p, g: xlam_call_match(p, g, d))
    return _bijection(calls, item["gold"], bfcl_call_match)

if __name__ == "__main__":
    import sys, collections
    items = {json.loads(l)["id"]: json.loads(l) for f in sys.argv[2:] for l in open(f)}
    per = collections.defaultdict(lambda: [0, 0])
    for l in open(sys.argv[1]):
        r = json.loads(l); it = items.get(r["id"])
        if it is None: continue
        ok = score_item(it, r["text"]); per[it["category"]][0] += ok; per[it["category"]][1] += 1
    for c, (a, n) in sorted(per.items()): print(f"{c:16s} {a}/{n} = {a/n:.3f}")
