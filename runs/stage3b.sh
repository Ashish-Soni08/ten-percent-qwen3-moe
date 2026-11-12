FBARGS="$1"
A=/data/fc/art/g56q3; O=/data/fc/g56q3
(cd fc && python3 unpack.py $A /tmp/ck) > /tmp/unpack.log 2>&1 || tail /tmp/unpack.log
cat /data/fc/data/calib.jsonl > /tmp/x
python3 -c "
import json
it={json.loads(l)['id']:json.loads(l) for l in open('/data/fc/data/calib.jsonl')}
out=[json.loads(l) for l in open('/data/fc/base/calib.jsonl')]
open('/tmp/calib_text.jsonl','w').writelines(json.dumps(dict(it[o['id']],text=o['text']))+'\n' for o in out[:200])"
python3 fc/fb_experiment.py --ckpt /tmp/ck --fb data/fb/tune_feedback.jsonl --probe data/fb/tune_probe.jsonl --guard data/fb/guard.jsonl --nearby data/fb/nearby_test.jsonl --calib /tmp/calib_text.jsonl --out $O/fb_tune2.json $FBARGS 2>&1 | grep -vE "Warning|warn|Loading weights"
