ART=/data/fc/art/rp56; R=/data/fc/final/rp56/fb2; mkdir -p $R; G=/data/fc/final/rp56/guard.jsonl
python3 -c "
import json,random
F=[json.loads(l) for l in open('data/fb2/feedback.jsonl')]
perm=list(range(len(F))); rng=random.Random(0)
while any(i==j for i,j in enumerate(perm)): rng.shuffle(perm)
open('/tmp/fb_shuf.jsonl','w').writelines(json.dumps(dict(F[i],annotation=F[perm[i]]['annotation']),ensure_ascii=False)+'\n' for i in range(len(F)))"
for kind in real shuf; do
  f=data/fb2/feedback.jsonl; [ $kind = shuf ] && f=/tmp/fb_shuf.jsonl
  python3 fc/cli.py learn --artifact $ART --cache /tmp/fcc --in $f --guard $G --out $R/state_$kind.json 2>&1 | grep -E "switched|rror"
done
rm -rf /tmp/fcc
echo '{"switched_off": []}' > $R/state_empty.json
for kind in empty real shuf; do
  python3 -c "
import sys,json; sys.path.insert(0,'fc'); from unpack import unpack_artifact
unpack_artifact('$ART','/tmp/ck_$kind',router_off=[tuple(x) for x in json.load(open('$R/state_$kind.json'))['switched_off']])"
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck_$kind --out $R/$kind --split data/fb2/probe_locked.jsonl --split data/fb2/nearby_test.jsonl --split data/fb2/feedback.jsonl > $R/$kind.log 2>&1; grep -E "rror" $R/$kind.log
  echo "== $kind"; for s in probe_locked nearby_test feedback; do python3 fc/score.py $R/$kind/$s.jsonl data/fb2/$s.jsonl | sed "s/^/$s /"; done
  rm -rf /tmp/ck_$kind
done
cat $R/state_real.json
