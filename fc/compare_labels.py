import json, sys, collections
def lab(t):
    w = t.strip().lower().split(); return w[0].strip('.:,!') if w else ''
it = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[3])}
b = {json.loads(l)['id']: lab(json.loads(l)['text']) for l in open(sys.argv[1])}
v = {json.loads(l)['id']: lab(json.loads(l)['text']) for l in open(sys.argv[2])}
per = collections.defaultdict(lambda: [0, 0, 0, 0])
for k, x in it.items():
    if k not in v: continue
    for c in (x['category'], 'ALL'):
        r = per[c]; bo = b[k] == x['gold']; vo = v[k] == x['gold']; r[0] += 1; r[1] += bo; r[2] += vo; r[3] += bo and vo
for c, (n, bo, vo, kept) in sorted(per.items()):
    print(f"{sys.argv[4] if len(sys.argv)>4 else ''} {c:24s} n={n:5d} base={bo/n:.3f} build={vo/n:.3f} kept={kept/max(bo,1):.4f}")
