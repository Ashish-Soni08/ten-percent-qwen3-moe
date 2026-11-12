# final: guard regen, learn feedback (real/shuffled/empty), locked exam for base and all variants
ART=$1; M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; R=/data/fc/final/$(basename $ART); mkdir -p $R
export PYTHONPATH_SGL=$PWD/fc/sglpatch
python3 fc/gen_sglang.py --model $M --out $R/base --split $D/exam.jsonl > $R/base.log 2>&1; grep -E "items in|rror" $R/base.log
(cd fc && python3 unpack.py $ART /tmp/ck) > /dev/null 2>&1
PYTHONPATH=$PYTHONPATH_SGL FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $R/nofb --split $D/exam.jsonl --split data/fb/probe_locked.jsonl --split data/fb/nearby_test.jsonl --split data/fb/guard_big.jsonl --split data/fb/feedback.jsonl > $R/nofb.log 2>&1; grep -E "items in|rror" $R/nofb.log
# guard = guard_big items the final model still answers right, annotated with its own answer
python3 -c "
import json,sys; sys.path.insert(0,'fc'); from score import score_item
its={json.loads(l)['id']:json.loads(l) for l in open('data/fb/guard_big.jsonl')}
out={json.loads(l)['id']:json.loads(l) for l in open('$R/nofb/guard_big.jsonl')}
g=[dict(its[k],annotation=out[k]['text']) for k in its if score_item(its[k],out[k]['text'])]
open('$R/guard.jsonl','w').writelines(json.dumps(x,ensure_ascii=False)+'\n' for x in g); print('guard', len(g))
fb=[json.loads(l) for l in open('data/fb/feedback.jsonl')]
fo={json.loads(l)['id']:json.loads(l) for l in open('$R/nofb/feedback.jsonl')}
print('feedback items the final model gets wrong:', sum(not score_item(x,fo[x['id']]['text']) for x in fb), '/', len(fb))"
python3 -c "
import json,random
F=[json.loads(l) for l in open('data/fb/feedback.jsonl')]
perm=list(range(len(F))); rng=random.Random(0)
while any(i==j for i,j in enumerate(perm)): rng.shuffle(perm)
open('/tmp/fb_shuf.jsonl','w').writelines(json.dumps(dict(F[i],annotation=F[perm[i]]['annotation']),ensure_ascii=False)+'\n' for i in range(len(F)))"
for kind in real shuf; do
  f=data/fb/feedback.jsonl; [ $kind = shuf ] && f=/tmp/fb_shuf.jsonl
  python3 fc/cli.py learn --artifact $ART --cache /tmp/fcc --in $f --guard $R/guard.jsonl --out $R/state_$kind.json 2>&1 | grep -E "switched|rror"
done
echo '{"switched_off": [], "history": []}' > $R/state_empty.json
rm -rf /tmp/ck
for kind in real shuf; do
  python3 -c "
import sys,json; sys.path.insert(0,'fc'); from unpack import unpack_artifact
unpack_artifact('$ART','/tmp/ck_$kind',router_off=[tuple(x) for x in json.load(open('$R/state_$kind.json'))['switched_off']])"
  PYTHONPATH=$PYTHONPATH_SGL FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck_$kind --out $R/fb_$kind --split $D/exam.jsonl --split data/fb/probe_locked.jsonl --split data/fb/nearby_test.jsonl > $R/fb_$kind.log 2>&1; grep -E "items in|rror" $R/fb_$kind.log
  rm -rf /tmp/ck_$kind
done
python3 fc/final_report.py $D/exam.jsonl $R nofb,fb_real,fb_shuf | tee $R/report.json
cp $R/guard.jsonl $R/state_*.json /data/fc/final/ 2>/dev/null; true
