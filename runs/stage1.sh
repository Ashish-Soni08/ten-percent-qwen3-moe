M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; O=/data/fc/base; mkdir -p $O
cp data/*toolace.jsonl $D/
python3 fc/gen_sglang.py --model $M --out $O --split $D/dev.jsonl --split $D/calib.jsonl --split $D/pool.jsonl:3000:pool3k --split $D/calib_toolace.jsonl --split $D/dev_toolace.jsonl > $O/gen.log 2>&1; grep -E "items in|engine up|determin|Error" $O/gen.log
python3 fc/gen_sglang.py --model $M --out $O/rev --reverse --split $D/dev.jsonl > $O/gen_rev.log 2>&1; grep -E "items in|Error" $O/gen_rev.log
python3 fc/score.py $O/dev.jsonl $D/dev.jsonl; python3 fc/score.py $O/rev/dev.jsonl $D/dev.jsonl
python3 -c "
import json; a={json.loads(l)['id']:json.loads(l)['text'] for l in open('$O/dev.jsonl')}; b={json.loads(l)['id']:json.loads(l)['text'] for l in open('$O/rev/dev.jsonl')}
print('dev text identical across orders:', sum(a[k]==b[k] for k in a), '/', len(a))"
python3 fc/score.py $O/calib.jsonl $D/calib.jsonl; python3 fc/score.py $O/pool3k.jsonl $D/pool.jsonl
cat $D/calib.jsonl $D/calib_toolace.jsonl > /tmp/ci.jsonl; cat $O/calib.jsonl $O/calib_toolace.jsonl > /tmp/ca.jsonl
python3 fc/reap_stats.py --model $M --items /tmp/ci.jsonl --answers /tmp/ca.jsonl --out $O/reap_calib.pt > $O/reap.log 2>&1; grep -vi warn $O/reap.log | tail -8
