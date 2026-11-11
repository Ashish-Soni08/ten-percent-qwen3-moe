M=/data/models/Qwen3-30B-A3B-Instruct-2507; I=/data/fc/int; cp data/int_pool.jsonl $I/data/
python3 fc/gen_sglang.py --model $M --out $I/base --max-new 16 --split $I/data/int_pool.jsonl > $I/base_pool.log 2>&1; grep -E "items in|rror" $I/base_pool.log
(cd fc && python3 unpack.py /data/fc/art/int3300 /tmp/ck) > /dev/null 2>&1
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out /data/fc/var/int3300/q --max-new 16 --split $I/data/int_pool.jsonl > /tmp/g.log 2>&1; grep -E "items in|rror" /tmp/g.log
python3 fc/score_label.py $I/data/int_pool.jsonl $I/base/int_pool.jsonl /data/fc/var/int3300/q/int_pool.jsonl
python3 - <<'PY' | gzip | base64 -w0 > /tmp/o.b64
import json
b={json.loads(l)['id']:json.loads(l)['text'] for l in open('/data/fc/int/base/int_pool.jsonl')}
q={json.loads(l)['id']:json.loads(l)['text'] for l in open('/data/fc/var/int3300/q/int_pool.jsonl')}
print(json.dumps({'base':b,'q':q}))
PY
wc -c /tmp/o.b64; echo B64START; cat /tmp/o.b64; echo; echo B64END
