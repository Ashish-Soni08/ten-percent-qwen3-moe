"""Score label outputs: score_label.py gold.jsonl out.jsonl [out2.jsonl] -> acc (and kept vs out1)."""
import json, sys
def norm(s): return s.strip().strip("`'\".").strip().lower().split("\n")[0].strip()
if __name__ == "__main__":
    g = {json.loads(l)["id"]: json.loads(l)["gold"].lower() for l in open(sys.argv[1])}
    def load(p): return {json.loads(l)["id"]: norm(json.loads(l)["text"]) for l in open(p)}
    a = load(sys.argv[2]); ok = {k for k in a if a[k] == g[k]}
    print(f"{sys.argv[2]}: acc {len(ok)/len(a):.4f} ({len(ok)}/{len(a)})")
    if len(sys.argv) > 3:
        b = load(sys.argv[3]); kept = sum(b[k] == g[k] for k in ok)
        print(f"  kept {kept}/{len(ok)} = {kept/len(ok):.4f}; agree {sum(a[k]==b[k] for k in a)/len(a):.4f}")
