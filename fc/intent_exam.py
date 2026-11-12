"""Lock the exam: CLINC150 test (in-scope, category = CLINC domain) + Banking77 test (unseen source). Also the feedback pool (CLINC train minus calib)."""
import json, random, hashlib, sys
sys.path.insert(0, "fc")
import intent_prep as P
from clinc_domains import I2D
ex = []
for i, it in enumerate(P.build("clinc", P.pq_rows("clinc_test.parquet", "intent", P.CL))):
    it["id"] = "te-" + it["id"]; it["cat"] = "clinc/" + I2D[it["gold"]]; ex.append(it)
for it in P.build("bank", P.pq_rows("bank_test.parquet", "label", P.BL)):
    it["id"] = "bte-" + it["id"]; it["cat"] = "banking77"; ex.append(it)
with open("data/int_exam.jsonl", "w") as f:
    for it in ex: f.write(json.dumps(it) + "\n")
print("exam", len(ex), hashlib.sha256(open("data/int_exam.jsonl", "rb").read()).hexdigest())
calib = {json.loads(l)["id"] for l in open("data/int_clinc_calib.jsonl")}
rng = random.Random(20261111)
tr = P.build("clinc", P.pq_rows("clinc_train.parquet", "intent", P.CL))
for it in tr: it["id"] = "tr-" + it["id"]
pool = [it for it in tr if it["id"] not in calib]
with open("data/int_pool.jsonl", "w") as f:
    for it in pool: f.write(json.dumps(it) + "\n")
print("pool", len(pool))
