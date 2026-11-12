"""Greedy generation with SGLang for a list of (split file, n) inputs. Writes {id, text} jsonl per split."""
import argparse, json, os, time

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", action="append", required=True, help="path[:n][:name]")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    import sglang as sgl
    kw = dict(model_path=a.model, mem_fraction_static=a.mem, context_length=32768, log_level="warning", random_seed=0)
    t0 = time.time()
    try:
        eng = sgl.Engine(**kw, enable_deterministic_inference=True)
        print("deterministic inference: on", flush=True)
    except TypeError as e:
        print("deterministic flag unsupported:", e, flush=True)
        eng = sgl.Engine(**kw)
    print(f"engine up in {time.time()-t0:.0f}s", flush=True)
    for spec in a.split:
        parts = spec.split(":"); path = parts[0]; n = int(parts[1]) if len(parts) > 1 and parts[1] else None
        name = parts[2] if len(parts) > 2 else os.path.basename(path).replace(".jsonl", "")
        items = [json.loads(l) for l in open(path)]
        if n: items = items[:n]
        if a.reverse: items = items[::-1]
        prompts = [tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False) for it in items]
        t1 = time.time()
        outs = eng.generate(prompt=prompts, sampling_params={"temperature": 0.0, "max_new_tokens": a.max_new})
        with open(os.path.join(a.out, name + ".jsonl"), "w") as f:
            for it, o in zip(items, outs):
                f.write(json.dumps({"id": it["id"], "text": o["text"], "ntok": o["meta_info"]["completion_tokens"]}) + "\n")
        print(f"{name}: {len(items)} items in {time.time()-t1:.0f}s", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
