M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; A=/data/fc/art/g56q3; O=/data/fc/g56q3; mkdir -p $A $O
python3 -c "
import json
L=open('$D/pool.jsonl').readlines()
sel=L[3000:15000]+L[26711:32711]
open('$D/mine.jsonl','w').writelines(sel); print(len(sel))"
cp /data/fc/var/g56q3/plan.json $A.plan.json
(cd fc && python3 compress.py --base $M --plan $A.plan.json --out $A) > $O/compress.log 2>&1; grep "artifact bytes" $O/compress.log
(cd fc && python3 unpack.py $A /tmp/ck) > $O/unpack.log 2>&1 || tail $O/unpack.log
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $O --split $D/mine.jsonl --split $D/dev.jsonl > $O/gen.log 2>&1; grep -E "items in|engine up|rror" $O/gen.log
python3 fc/score.py $O/mine.jsonl $D/mine.jsonl
gzip -c $O/mine.jsonl | base64 -w0 > $O/mine.b64; wc -c $O/mine.b64
