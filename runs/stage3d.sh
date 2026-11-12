A=/data/fc/art/g56q3; O=/data/fc/g56q3/final; mkdir -p $O; D=/data/fc/data
cp data/fb/*.jsonl $D/ 2>/dev/null; mkdir -p $D/fb; cp data/fb/*.jsonl $D/fb/
(cd fc && python3 unpack.py $A /tmp/ck) > /tmp/unpack.log 2>&1 || tail /tmp/unpack.log
python3 -c "
import json
it={json.loads(l)['id']:json.loads(l) for l in open('$D/calib.jsonl')}
out=[json.loads(l) for l in open('/data/fc/base/calib.jsonl')]
open('/tmp/calib_text.jsonl','w').writelines(json.dumps(dict(it[o['id']],text=o['text']))+'\n' for o in out[:200])"
python3 fc/fb_final.py --ckpt /tmp/ck --fb data/fb/feedback.jsonl --probe data/fb/probe_locked.jsonl --guard data/fb/guard.jsonl --nearby data/fb/nearby_test.jsonl --calib /tmp/calib_text.jsonl --out $O 2>&1 | grep '^{'
rm -rf /tmp/ck
# SGLang confirmation (what the shipped system answers)
for st in none switch24_real.json switch48_real.json edit_real.pt switch48_shuffled.json edit_shuffled.pt; do
  if [ $st = none ]; then (cd fc && python3 unpack.py $A /tmp/ck_$st) > /dev/null 2>&1; else python3 fc/apply_and_gen.py $A $O/$st /tmp/ck_$st > /dev/null 2>&1; fi
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck_$st --out $O/gen_$st --split data/fb/probe_locked.jsonl --split data/fb/nearby_test.jsonl --split $D/dev.jsonl > $O/gen_$st.log 2>&1; grep -E "rror" $O/gen_$st.log | head -3
  echo "== $st"; python3 fc/score.py $O/gen_$st/probe_locked.jsonl data/fb/probe_locked.jsonl; python3 fc/score.py $O/gen_$st/nearby_test.jsonl data/fb/nearby_test.jsonl
  python3 fc/compare.py /data/fc/g56q3/dev.jsonl $O/gen_$st/dev.jsonl $D/dev.jsonl "$st vs nofb"
  rm -rf /tmp/ck_$st
done
