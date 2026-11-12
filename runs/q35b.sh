M=/data/models/Qwen3.5-35B-A3B; Q=/data/fc/q35; cp data/pi_*.jsonl $Q/
head -c 600 $Q/base/dev.jsonl; echo; sed -n 5p $Q/base/dev.jsonl | head -c 600; echo
python3 fc/gen_sglang.py --model $M --out $Q/base --max-new 4 --split $Q/pi_dev.jsonl --split $Q/pi_unseen_probe.jsonl --split $Q/pi_calib.jsonl > $Q/base_pi.log 2>&1; grep -E "items in|rror" $Q/base_pi.log | head -5
for s in pi_dev pi_unseen_probe; do python3 fc/score_label.py $Q/$s.jsonl $Q/base/$s.jsonl; python3 fc/int_report.py $Q/$s.jsonl $Q/base/$s.jsonl $Q/base/$s.jsonl | grep -v "==" | cut -c1-80; done
python3 fc/q35_route_stats.py --model $M --items $Q/int_clinc_calib.jsonl --answers $Q/base/int_clinc_calib.jsonl --out $Q/rs_clinc.pt \
  --items $Q/pi_calib.jsonl --answers $Q/base/pi_calib.jsonl --out $Q/rs_pi.pt \
  --items /data/fc/data/calib.jsonl --answers $Q/base/calib.jsonl --out $Q/rs_fc.pt > $Q/rs.log 2>&1; grep -vE "Warning|warn" $Q/rs.log | tail -8
for t in clinc pi fc; do
  python3 fc/q35_slice.py --model $M --stats $Q/rs_$t.pt --k 104 --bits 3 --out /tmp/s_$t 2>&1 | grep -E "coverage|rror"
  case $t in clinc) S="--split $Q/int_clinc_dev.jsonl"; MN=16;; pi) S="--split $Q/pi_dev.jsonl --split $Q/pi_unseen_probe.jsonl"; MN=4;; fc) S="--split /data/fc/data/dev.jsonl"; MN=768;; esac
  python3 fc/gen_sglang.py --model /tmp/s_$t --out $Q/k104_$t --max-new $MN $S > $Q/k104_$t.log 2>&1; grep -E "items in|rror" $Q/k104_$t.log | head -3
  rm -rf /tmp/s_$t
done
echo "== CLINC k104"; python3 fc/score_label.py $Q/int_clinc_dev.jsonl $Q/base/int_clinc_dev.jsonl $Q/k104_clinc/int_clinc_dev.jsonl
for s in pi_dev pi_unseen_probe; do echo "== PI $s k104"; python3 fc/int_report.py $Q/$s.jsonl $Q/base/$s.jsonl $Q/k104_pi/$s.jsonl | grep -v "==" | cut -c1-90; done
echo "== FC k104 agree with base"; python3 -c "
import json
b={json.loads(l)['id']:json.loads(l)['text'] for l in open('$Q/base/dev.jsonl')}
q={json.loads(l)['id']:json.loads(l)['text'] for l in open('$Q/k104_fc/dev.jsonl')}
print('identical text', sum(b[k]==q[k] for k in b)/len(b))"
