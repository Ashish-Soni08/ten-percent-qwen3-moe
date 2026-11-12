M=/data/models/Qwen3-30B-A3B-Instruct-2507; I=/data/fc/int; name=int3000; R=/data/fc/var/$name; mkdir -p $R
python3 fc/make_plan.py --stats $I/reap_calib.pt --k 56 --crit sum --total 3000 --minper 32 --expert 2 --hi 2000 --hibits 3 --attn 4 --embed 4 --head 6 --out $R/plan.json | tail -1
python3 -c "
import json; p=json.load(open('$R/plan.json')); print('3-bit experts', len(p['expert_bits'])); q=dict(p, bits={'expert':16,'attn':16,'embed':16,'head':16}, expert_bits={}); json.dump(q, open('/tmp/planbf.json','w'))"
(cd fc && python3 compress.py --base $M --plan /tmp/planbf.json --out /tmp/artbf) > $R/c1.log 2>&1
(cd fc && python3 unpack.py /tmp/artbf /tmp/ckbf) > $R/u1.log 2>&1 || tail $R/u1.log; rm -rf /tmp/artbf
(cd fc && python3 gptq_compress.py --base $M --plan $R/plan.json --prunedbf /tmp/ckbf --calib $I/data/int_clinc_calib.jsonl --answers $I/base/int_clinc_calib.jsonl --out /data/fc/art/$name) > $R/gptq.log 2>&1; grep -vE "Warning|warn|UNEXPECTED|LOAD|Key|^---|Notes|ignored" $R/gptq.log | tail -3
rm -rf /tmp/ckbf; du -sb /data/fc/art/$name
(cd fc && python3 unpack.py /data/fc/art/$name /tmp/ck) > $R/u2.log 2>&1 || tail $R/u2.log
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $R/q --max-new 16 --split $I/data/int_clinc_dev.jsonl > $R/gen.log 2>&1; grep -E "items in|rror" $R/gen.log
echo "== $name final"; python3 fc/score_label.py $I/data/int_clinc_dev.jsonl $I/base/int_clinc_dev.jsonl $R/q/int_clinc_dev.jsonl
