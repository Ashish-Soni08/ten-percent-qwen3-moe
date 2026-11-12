"""The two commands of the entry.

  answer: python fc/cli.py answer --artifact ART [--state state.json] --in questions.jsonl --out answers.jsonl
          questions: one JSON object per line: {"id", "messages": [chat messages], "tools": [OpenAI-style tool specs]}
          answers:   {"id", "text" (raw model output), "calls": [{"name", "arguments"}] or null if unparsable}
  learn:  python fc/cli.py learn --artifact ART [--state old.json] --in feedback.jsonl --out new.json
          feedback: {"messages", "tools", "annotation"} where annotation is the corrected answer, either a list of
          {"name", "arguments"} calls or the corrected raw text. At most 25 items.
          Learning changes no weights: it picks experts to switch off (a list of (layer, slot) pairs) so the annotated
          answers become the model's own, and refuses any switch that changes answers on fc/guard.jsonl
          (questions the model already answers right).
The artifact is unpacked once into a bf16 cache (--cache, default /tmp/fc_cache) that SGLang serves.
"""
import argparse, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def cache_dir(art, state, root):
    key = hashlib.sha1((os.path.abspath(art) + json.dumps(state.get("switched_off", []))).encode()).hexdigest()[:12]
    return os.path.join(root, key)

def materialize(art, state, root):
    d = cache_dir(art, state, root)
    if not os.path.exists(os.path.join(d, "model.safetensors.index.json")):
        from unpack import unpack_artifact
        unpack_artifact(art, d, router_off=[tuple(x) for x in state.get("switched_off", [])])
    return d

def load_state(p): return json.load(open(p)) if p else {"switched_off": [], "history": []}

def cmd_answer(a):
    st = load_state(a.state); ck = materialize(a.artifact, st, a.cache)
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "sglpatch") + os.pathsep + os.environ.get("PYTHONPATH", ""), FC_ROUTER_BIAS="1")
    tmp = a.out + ".tmpdir"
    subprocess.run([sys.executable, os.path.join(HERE, "gen_sglang.py"), "--model", ck, "--out", tmp, "--split", f"{a.inp}::q"], env=env, check=True)
    from score import parse_calls
    with open(a.out, "w") as f:
        for l in open(os.path.join(tmp, "q.jsonl")):
            r = json.loads(l); f.write(json.dumps({"id": r["id"], "text": r["text"], "calls": parse_calls(r["text"])}, ensure_ascii=False) + "\n")

def cmd_learn(a):
    from feedback import FB
    from feedback_render import render
    st = load_state(a.state)
    items = [json.loads(l) for l in open(a.inp)]
    assert len(items) <= 25, "at most 25 feedback items"
    ck = materialize(a.artifact, {"switched_off": []}, a.cache)
    fb = FB(ck)
    pairs = [fb.encode(it, it["annotation"] if isinstance(it["annotation"], str) else render(it["annotation"])) for it in items]
    guard = [json.loads(l) for l in open(a.guard)]
    gp = [fb.encode(it, it["annotation"]) for it in guard]
    off, log = fb.learn_switch(pairs, gp, max_off=a.max_off, rounds=a.max_off // 4 + 2, init=[tuple(x) for x in st["switched_off"]])
    st = {"switched_off": [list(x) for x in off], "history": st.get("history", []) + [{"n_items": len(items), "added": len(off) - len(st["switched_off"])}]}
    json.dump(st, open(a.out, "w"))
    print(f"switched off {len(off)} experts in total ({st['history'][-1]['added']} new)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("answer", "learn"):
        p = sub.add_parser(n); p.add_argument("--artifact", required=True); p.add_argument("--state")
        p.add_argument("--in", dest="inp", required=True); p.add_argument("--out", required=True)
        p.add_argument("--cache", default="/tmp/fc_cache")
        if n == "learn":
            p.add_argument("--guard", default=os.path.join(HERE, "guard.jsonl")); p.add_argument("--max-off", type=int, default=48)
    a = ap.parse_args()
    {"answer": cmd_answer, "learn": cmd_learn}[a.cmd](a)
