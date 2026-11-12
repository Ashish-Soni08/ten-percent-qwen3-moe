"""Pick the feedback class data from the shipped model's own mistakes on the mining split.

Class: the model adds optional arguments the user never asked for. Annotation = the model's own answer with those
arguments removed (what a user marking the mistake would write). Tools are kept disjoint across splits, so probes
test the behaviour, not a memorised tool.
"""
import json, random, sys, collections
sys.path.insert(0, 'fc'); from score import score_item, parse_calls
from feedback_render import render
items = {json.loads(l)['id']: json.loads(l) for l in open('data/mine.jsonl')}
outs = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[1])}
def strip_extra(it, calls):
    out = []
    for p in calls:
        gc = [g for g in it['gold'] if g['name'] == p['name']]
        if not gc: return None
        out.append({'name': p['name'], 'arguments': {k: v for k, v in p['arguments'].items() if any(k in g['arguments'] for g in gc)}})
    return out
def optional_names(it):
    s = set()
    for t in it['tools']:
        f = t['function']; req = set(f['parameters'].get('required', []))
        s |= {(f['name'], k) for k in f['parameters'].get('properties', {}) if k not in req}
    return s
cls, nearby = [], []
for k, o in outs.items():
    it = items[k]; calls = parse_calls(o['text'])
    if calls is None or len(calls) != len(it['gold']): continue
    ok = score_item(it, o['text'])
    if not ok:
        s = strip_extra(it, calls)
        if s and s != calls and score_item(it, render(s)):
            cls.append(dict(it, annotation=render(s), model_answer=o['text']))
    else:
        opt = optional_names(it)
        if any((g['name'], a) in opt for g in it['gold'] for a in g['arguments']):
            nearby.append(dict(it, annotation=o['text']))
print('class', len(cls), 'nearby', len(nearby))
def tool(it): return it['gold'][0]['name']
rng = random.Random(7)
tools = sorted({tool(x) for x in cls}); rng.shuffle(tools)
by = collections.defaultdict(list)
for x in cls: by[tool(x)].append(x)
splits = {'feedback': [], 'probe_locked': [], 'tune_feedback': [], 'tune_probe': []}
want = {'feedback': 25, 'probe_locked': 50, 'tune_feedback': 25, 'tune_probe': 50}
order = ['probe_locked', 'feedback', 'tune_probe', 'tune_feedback']
i = 0
for t in tools:
    open_ = [s for s in order if len(splits[s]) < want[s]]
    if not open_: break
    s = open_[i % len(open_)]; i += 1
    take = by[t][: min(3, want[s] - len(splits[s]))] if s.endswith('probe') or s == 'probe_locked' else by[t][:1]
    splits[s] += take
rng.shuffle(nearby)
splits['guard'] = nearby[:40]; splits['nearby_test'] = nearby[40:340]
import os; os.makedirs('data/fb', exist_ok=True)
for s, v in splits.items():
    open(f'data/fb/{s}.jsonl', 'w').write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in v))
    print(s, len(v), len({tool(x) for x in v}))
