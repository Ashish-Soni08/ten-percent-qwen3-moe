"""Build intent-routing items: {id, messages, tools, gold, src}. Usage: intent_prep.py probe"""
import json, random, sys, collections
import pyarrow.parquet as pq
R = "data_raw/intent/"
def massive(split):
    return [json.loads(l) for l in open(R + f"massive_{split}.jsonl")]
def pq_rows(f, col, names):
    t = pq.read_table(R + f).to_pydict()
    return [{"text": x, "label_text": names[y]} for x, y in zip(t["text"], t[col])]
CL = json.load(open(R + "clinc_labels.json")); BL = json.load(open(R + "bank_labels.json"))
MI = sorted({r["label_text"] for r in massive("train")})
MS = sorted({l.split("_")[0] for l in MI})
SYS = ("You route requests for {what}. Choose the single label that best matches the user's request.\n"
       "Labels:\n{labels}\n\nReply with the label only, exactly as written above.")
def item(i, src, what, labels, text, gold):
    return {"id": f"{src}-{i}", "src": src, "gold": gold, "tools": None,
            "messages": [{"role": "system", "content": SYS.format(what=what, labels="\n".join(labels))},
                         {"role": "user", "content": text}]}
def build(src, rows):
    if src == "massive_scn":
        return [item(i, src, "a voice assistant", MS, r["text"], r["label_text"].split("_")[0]) for i, r in enumerate(rows)]
    if src == "massive_int":
        return [item(i, src, "a voice assistant", MI, r["text"], r["label_text"]) for i, r in enumerate(rows)]
    if src == "clinc":
        return [item(i, src, "a virtual assistant", [l for l in CL if l != "oos"], r["text"], r["label_text"]) for i, r in enumerate(rows) if r["label_text"] != "oos"]
    if src == "bank":
        return [item(i, src, "a bank's customer support", BL, r["text"], r["label_text"]) for i, r in enumerate(rows)]
if __name__ == "__main__":
    rng = random.Random(20261111)
    srcs = {"massive_scn": massive("validation"), "massive_int": massive("validation"),
            "clinc": pq_rows("clinc_validation.parquet", "intent", CL), "bank": pq_rows("bank_train.parquet", "label", BL)}
    for s, rows in srcs.items():
        its = build(s, rows); rng.shuffle(its); its = its[:500]
        with open(f"data/int_probe_{s}.jsonl", "w") as f:
            for it in its: f.write(json.dumps(it) + "\n")
        print(s, len(its), collections.Counter(i["gold"] for i in its).most_common(2))

def clinc_splits():
    rng = random.Random(20261111)
    tr = build("clinc", pq_rows("clinc_train.parquet", "intent", CL)); rng.shuffle(tr)
    va = build("clinc", pq_rows("clinc_validation.parquet", "intent", CL)); rng.shuffle(va)
    for it in tr: it["id"] = "tr-" + it["id"]
    for it in va: it["id"] = "va-" + it["id"]
    for name, its in [("int_clinc_calib", tr[:1500]), ("int_clinc_dev", va[:1500])]:
        with open(f"data/{name}.jsonl", "w") as f:
            for it in its: f.write(json.dumps(it) + "\n")
        print(name, len(its))
