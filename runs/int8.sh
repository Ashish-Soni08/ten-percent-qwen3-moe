M=/data/models/Qwen3-30B-A3B-Instruct-2507; ART=/data/fc/art/int3300; R=/data/fc/final/int3300; mkdir -p $R; cp -r data/ifb $R/; cp data/int_exam.jsonl $R/
sha256sum data/int_exam.jsonl data/ifb/*.jsonl
python3 -c "
import json,random
F=[json.loads(l) for l in open('data/ifb/feedback.jsonl')]
perm=list(range(len(F))); rng=random.Random(0)
while any(F[i]['annotation']==F[perm[i]]['annotation'] for i in range(len(F))): rng.shuffle(perm)
open('/tmp/fb_shuf.jsonl','w').writelines(json.dumps(dict(F[i],annotation=F[perm[i]]['annotation']))+'\n' for i in range(len(F)))"
for kind in real shuf; do
  f=data/ifb/feedback.jsonl; [ $kind = shuf ] && f=/tmp/fb_shuf.jsonl
  python3 fc/cli.py learn --artifact $ART --cache /tmp/fcc --in $f --guard data/ifb/guard.jsonl --out $R/state_$kind.json 2>&1 | grep -E "switched|rror"
done
rm -rf /tmp/fcc; echo '{"switched_off": []}' > $R/state_empty.json; cat $R/state_real.json; echo
for kind in empty real shuf; do
  python3 -c "
import sys,json; sys.path.insert(0,'fc'); from unpack import unpack_artifact
unpack_artifact('$ART','/tmp/ck_$kind',router_off=[tuple(x) for x in json.load(open('$R/state_$kind.json'))['switched_off']])"
  X=""; [ $kind != shuf ] && X="--split data/int_exam.jsonl"
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck_$kind --out $R/$kind --max-new 16 --split data/ifb/probe_locked.jsonl --split data/ifb/nearby_test.jsonl --split data/ifb/feedback.jsonl $X > $R/$kind.log 2>&1; grep -E "rror" $R/$kind.log
  echo "== $kind"; for s in probe_locked nearby_test feedback; do echo -n "$s "; python3 fc/score_label.py data/ifb/$s.jsonl $R/$kind/$s.jsonl | head -1; done
  rm -rf /tmp/ck_$kind
done
python3 fc/gen_sglang.py --model $M --out $R/base --max-new 16 --split data/int_exam.jsonl --split data/ifb/probe_locked.jsonl --split data/ifb/nearby_test.jsonl > $R/base.log 2>&1; grep -E "items in|rror" $R/base.log
echo "== base probes"; for s in probe_locked nearby_test; do echo -n "$s "; python3 fc/score_label.py data/ifb/$s.jsonl $R/base/$s.jsonl | head -1; done
echo "== nearby kept vs empty"; python3 fc/score_label.py data/ifb/nearby_test.jsonl $R/empty/nearby_test.jsonl $R/real/nearby_test.jsonl
python3 fc/int_report.py data/int_exam.jsonl $R/base/int_exam.jsonl $R/empty/int_exam.jsonl $R/real/int_exam.jsonl | tee $R/report.txt
