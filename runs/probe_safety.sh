M=/data/models/Qwen3-30B-A3B-Instruct-2507; O=/data/fc/safety_probe; mkdir -p $O; cp data/safety_probe.jsonl /data/fc/data/
python3 fc/gen_sglang.py --model $M --out $O/base --max-new 8 --split data/safety_probe.jsonl > $O/base.log 2>&1; grep -E "items in|rror" $O/base.log
(cd fc && python3 unpack.py /data/fc/art/rp56 /tmp/ck) > /dev/null 2>&1
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $O/rp56 --max-new 8 --split data/safety_probe.jsonl > $O/rp56.log 2>&1; grep -E "items in|rror" $O/rp56.log
python3 - <<'PY'
import json, collections
it={json.loads(l)['id']:json.loads(l) for l in open('data/safety_probe.jsonl')}
def lab(t):
    w=t.strip().lower().split()
    return w[0].strip('.:,') if w else ''
b={json.loads(l)['id']:lab(json.loads(l)['text']) for l in open('/data/fc/safety_probe/base/safety_probe.jsonl')}
v={json.loads(l)['id']:lab(json.loads(l)['text']) for l in open('/data/fc/safety_probe/rp56/safety_probe.jsonl')}
print('base label set', collections.Counter(b.values()).most_common(5))
per=collections.defaultdict(lambda:[0,0,0,0,0])
for k,x in it.items():
    for c in (x['src'], x['src']+'/'+x['gold'], 'ALL'):
        r=per[c]; r[0]+=1; bo=b[k]==x['gold']; vo=v[k]==x['gold']; r[1]+=bo; r[2]+=vo; r[3]+=bo and vo; r[4]+=b[k]==v[k]
for c,(n,bo,vo,kept,ag) in sorted(per.items()):
    print(f"{c:24s} n={n:5d} base={bo/n:.3f} build={vo/n:.3f} kept={kept/max(bo,1):.4f} agree={ag/n:.4f}")
PY
