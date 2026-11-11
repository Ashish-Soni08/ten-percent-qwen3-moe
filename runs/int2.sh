M=/data/models/Qwen3-30B-A3B-Instruct-2507; I=/data/fc/int; mkdir -p $I/data; cp data/int_clinc_*.jsonl $I/data/
python3 fc/gen_sglang.py --model $M --out $I/base --max-new 16 --split $I/data/int_clinc_calib.jsonl --split $I/data/int_clinc_dev.jsonl > $I/base.log 2>&1; grep -E "items in|rror" $I/base.log
python3 fc/score_label.py $I/data/int_clinc_dev.jsonl $I/base/int_clinc_dev.jsonl
python3 fc/recon_prune.py --model $M --items $I/data/int_clinc_calib.jsonl --answers $I/base/int_clinc_calib.jsonl --total 2690 --ntok 10000 --min-prompt 24 --out /data/fc/art/int56.plan.json 2>&1 | grep -E "experts per layer|tokens"
name=int56; P=/data/fc/art/int56.plan.json; R=/data/fc/var/$name; mkdir -p $R
python3 -c "
import json; p=json.load(open('$P')); q=dict(p, bits={'expert':16,'attn':16,'embed':16,'head':16}, expert_bits={}); json.dump(q, open('/tmp/planbf.json','w'))"
cp $P $R/plan.json
(cd fc && python3 compress.py --base $M --plan /tmp/planbf.json --out /tmp/artbf) > $R/c1.log 2>&1; tail -1 $R/c1.log
(cd fc && python3 unpack.py /tmp/artbf /tmp/ckbf) > $R/u1.log 2>&1 || tail $R/u1.log; rm -rf /tmp/artbf
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ckbf --out $R/bf --max-new 16 --split $I/data/int_clinc_dev.jsonl > $R/genbf.log 2>&1; grep -E "items in|rror" $R/genbf.log
echo "== pruned bf16"; python3 fc/score_label.py $I/data/int_clinc_dev.jsonl $I/base/int_clinc_dev.jsonl $R/bf/int_clinc_dev.jsonl
(cd fc && python3 gptq_compress.py --base $M --plan $R/plan.json --prunedbf /tmp/ckbf --calib $I/data/int_clinc_calib.jsonl --answers $I/base/int_clinc_calib.jsonl --out /data/fc/art/$name) > $R/gptq.log 2>&1; grep -vE "Warning|warn|UNEXPECTED|LOAD|Key|^---|Notes|ignored" $R/gptq.log | tail -5
rm -rf /tmp/ckbf; du -sb /data/fc/art/$name
(cd fc && python3 unpack.py /data/fc/art/$name /tmp/ck) > $R/u2.log 2>&1 || tail $R/u2.log
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $R/q --max-new 16 --split $I/data/int_clinc_dev.jsonl > $R/gen.log 2>&1; grep -E "items in|rror" $R/gen.log
echo "== int56 final"; python3 fc/score_label.py $I/data/int_clinc_dev.jsonl $I/base/int_clinc_dev.jsonl $R/q/int_clinc_dev.jsonl
