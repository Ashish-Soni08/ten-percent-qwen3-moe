M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; B=/data/fc/base; name=${1:-gq56}; P=${2:-/data/fc/art/g56q3.plan.json}; R=/data/fc/var/$name; mkdir -p $R
python3 -c "
import json; p=json.load(open('$P')); q=dict(p, bits={'expert':16,'attn':16,'embed':16,'head':16}, expert_bits={}); json.dump(q, open('/tmp/planbf.json','w'))"
cp $P $R/plan.json
(cd fc && python3 compress.py --base $M --plan /tmp/planbf.json --out /tmp/artbf) > $R/c1.log 2>&1; tail -1 $R/c1.log
(cd fc && python3 unpack.py /tmp/artbf /tmp/ckbf) > $R/u1.log 2>&1 || tail $R/u1.log; rm -rf /tmp/artbf
cp /data/fc/saf/saf_calib.jsonl /tmp/ci.jsonl; cp /data/fc/saf/base/saf_calib.jsonl /tmp/ca.jsonl
(cd fc && python3 gptq_compress.py --base $M --plan $R/plan.json --prunedbf /tmp/ckbf --calib /tmp/ci.jsonl --answers /tmp/ca.jsonl --out /data/fc/art/$name) > $R/gptq.log 2>&1; grep -vE "Warning|warn|UNEXPECTED|LOAD|Key|^---|Notes|ignored" $R/gptq.log | tail -12
rm -rf /tmp/ckbf
(cd fc && python3 unpack.py /data/fc/art/$name /tmp/ck) > $R/u2.log 2>&1 || tail $R/u2.log
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $R --max-new 8 --split /data/fc/saf/saf_dev.jsonl > $R/gen.log 2>&1; grep -E "items in|rror" $R/gen.log
python3 fc/compare_labels.py /data/fc/saf/base/saf_dev.jsonl $R/saf_dev.jsonl /data/fc/saf/saf_dev.jsonl $name
