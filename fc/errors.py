import json, sys, collections
sys.path.insert(0, 'fc'); from score import *; from score import _defaults
def load(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
items = {}
for f in ['data/calib.jsonl', 'data/dev.jsonl']: items.update(load(f))
pool = {}
for l in open('data/pool.jsonl'):
    x = json.loads(l); pool[x['id']] = x
    if len(pool) >= 3000: break
items.update(pool)
outs = {}
for f in ['base_out/calib.jsonl', 'base_out/pool3k.jsonl']: outs.update(load(f))
def classify(it, text):
    calls = parse_calls(text)
    if calls is None: return 'malformed', None
    g = it['gold']
    if len(calls) != len(g): return 'ncalls', None
    gn = sorted(c['name'] for c in g); pn = sorted(c['name'] for c in calls)
    if gn != pn: return 'fname', None
    d = _defaults(it)
    for gc in g:
        cands = [p for p in calls if p['name'] == gc['name']]
        if any(xlam_call_match(p, gc, d) for p in cands): continue
        p = cands[0]; pa, ga = p['arguments'], gc['arguments']
        miss = [k for k in ga if k not in pa]; extra = [k for k in pa if k not in ga]
        if miss: return 'missing_arg', (gc['name'], miss, ga, pa)
        if extra: return 'extra_arg', (gc['name'], extra, ga, pa)
        bad = [k for k in ga if not veq(pa[k], ga[k])]
        return 'value', (gc['name'], bad, {k: ga[k] for k in bad}, {k: pa[k] for k in bad})
    return 'ok?', None
cnt = collections.Counter(); ex = collections.defaultdict(list)
n = 0
for k, o in outs.items():
    it = items[k]; n += 1
    if score_item(it, o['text']): continue
    c, info = classify(it, o['text']); cnt[c] += 1; ex[c].append((k, info, o['text'][:300]))
print(n, cnt.most_common())
for c in ['extra_arg', 'missing_arg', 'value']:
    print('=====', c)
    for e in ex[c][:int(sys.argv[1]) if len(sys.argv) > 1 else 12]: print(json.dumps(e[1], ensure_ascii=False)[:400])

# --- size of candidate feedback classes across all mined items
def extra_only(it, text):
    calls = parse_calls(text)
    if not calls or len(calls) != len(it['gold']): return False
    d = _defaults(it); stripped = []
    for p in calls:
        gc = [g for g in it['gold'] if g['name'] == p['name']]
        if not gc: return False
        stripped.append({'name': p['name'], 'arguments': {k: v for k, v in p['arguments'].items() if any(k in g['arguments'] for g in gc)}})
    return any(len(p['arguments']) != len(s['arguments']) for p, s in zip(calls, stripped)) and \
        score_item(it, ''.join('<tool_call>' + json.dumps(s) + '</tool_call>' for s in stripped))
eo = [k for k, o in outs.items() if not score_item(items[k], o['text']) and extra_only(items[k], o['text'])]
lam = [k for k, o in outs.items() if any('lambda' in json.dumps(g['arguments']) for g in items[k]['gold'])]
lam_wrong = [k for k in lam if not score_item(items[k], outs[k]['text'])]
print('extra-only errors', len(eo), 'lambda items', len(lam), 'lambda wrong', len(lam_wrong))
