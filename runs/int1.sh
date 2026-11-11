M=/data/models/Qwen3-30B-A3B-Instruct-2507; O=/data/fc/int/probe; mkdir -p $O /data/fc/int/data; cp data/int_probe_*.jsonl /data/fc/int/data/
S=""; for s in massive_scn massive_int clinc bank; do S="$S --split data/int_probe_$s.jsonl"; done
python3 fc/gen_sglang.py --model $M --out $O/base --max-new 16 $S > $O/base.log 2>&1; grep -E "items in|rror" $O/base.log
(cd fc && python3 unpack.py /data/fc/art/rp56 /tmp/ck) > /dev/null 2>&1
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $O/rp56 --max-new 16 $S > $O/rp56.log 2>&1; grep -E "items in|rror" $O/rp56.log
for s in massive_scn massive_int clinc bank; do python3 fc/score_label.py data/int_probe_$s.jsonl $O/base/int_probe_$s.jsonl $O/rp56/int_probe_$s.jsonl; done
python3 - <<'PY'
import json
for s in ['massive_int','clinc','bank']:
    g={json.loads(l)['id']:json.loads(l)['gold'] for l in open(f'data/int_probe_{s}.jsonl')}
    n=0
    for l in open(f'/data/fc/int/probe/base/int_probe_{s}.jsonl'):
        o=json.loads(l)
        if o['text'].strip().lower()!=g[o['id']] and n<6: print(s,'|',repr(o['text'][:40]),'| gold',g[o['id']]); n+=1
PY
