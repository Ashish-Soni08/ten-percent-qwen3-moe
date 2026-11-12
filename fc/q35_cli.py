"""The two commands of a Qwen3.5 entry (short-label tasks: the model writes one label).

  answer: python fc/q35_cli.py answer --artifact ART [--state state.json] --in questions.jsonl --out answers.jsonl
          questions: {"id", "messages": [chat messages]}; answers: {"id", "text"} (free greedy generation)
  learn:  python fc/q35_cli.py learn --artifact ART [--state old.json] --in feedback.jsonl --guard guard.jsonl --out new.json
          feedback: {"messages", "annotation"} where annotation is the corrected label. At most 25 items.
          Learning changes no weights: it picks experts to switch off (a list of (layer, slot) pairs) so the annotated
          answers become the model's own, and refuses any switch that changes answers on the guard set.
The artifact is unpacked once into a bf16 cache (--cache) that SGLang serves; switched-off experts get router
offset -1e4 there, every other expert 0.
"""
import argparse, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def load_state(p): return json.load(open(p)) if p else {"switched_off": [], "history": []}

def materialize(art, state, root):
    key = hashlib.sha1((os.path.abspath(art) + json.dumps(state.get("switched_off", []))).encode()).hexdigest()[:12]
    d = os.path.join(root, key)
    if not os.path.exists(os.path.join(d, "model.safetensors.index.json")):
        from q35_unpack import unpack_artifact
        unpack_artifact(art, d, router_off=[tuple(x) for x in state.get("switched_off", [])])
    return d

def cmd_answer(a):
    ck = materialize(a.artifact, load_state(a.state), a.cache)
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "sglpatch") + os.pathsep + os.environ.get("PYTHONPATH", ""), FC_ROUTER_BIAS="1")
    tmp = a.out + ".tmpdir"
    subprocess.run([sys.executable, os.path.join(HERE, "gen_sglang.py"), "--model", ck, "--out", tmp, "--max-new", str(a.max_new), "--split", f"{a.inp}::q"], env=env, check=True)
    with open(a.out, "w") as f:
        for l in open(os.path.join(tmp, "q.jsonl")):
            r = json.loads(l); f.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")

def cmd_learn(a):
    from q35_feedback import FB
    st = load_state(a.state)
    items = [json.loads(l) for l in open(a.inp)]
    assert len(items) <= 25, "at most 25 feedback items"
    fb = FB(materialize(a.artifact, {"switched_off": []}, a.cache))
    pairs = [fb.encode(it, it["annotation"]) for it in items]
    gp = [fb.encode(it, it["annotation"]) for it in (json.loads(l) for l in open(a.guard))]
    off, log = fb.learn_switch(pairs, gp, max_off=a.max_off, rounds=a.max_off // 4 + 2, init=[tuple(x) for x in st["switched_off"]])
    st = {"switched_off": [list(x) for x in off], "history": st.get("history", []) + [{"n_items": len(items), "added": len(off) - len(st["switched_off"])}]}
    json.dump(st, open(a.out, "w"))
    print(f"switched off {len(off)} experts in total ({st['history'][-1]['added']} new)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("answer", "learn"):
        p = sub.add_parser(n); p.add_argument("--artifact", required=True); p.add_argument("--state")
        p.add_argument("--in", dest="inp", required=True); p.add_argument("--out", required=True)
        p.add_argument("--cache", default="/tmp/q35_cache")
        if n == "answer": p.add_argument("--max-new", type=int, default=16)
        if n == "learn": p.add_argument("--guard", required=True); p.add_argument("--max-off", type=int, default=24)
    a = ap.parse_args()
    {"answer": cmd_answer, "learn": cmd_learn}[a.cmd](a)
