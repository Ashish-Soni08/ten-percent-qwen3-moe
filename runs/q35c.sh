M=/data/models/Qwen3.5-35B-A3B; Q=/data/fc/q35; mkdir -p $Q; cp data/lid_*.jsonl data/sent_*.jsonl $Q/
python3 fc/gen_sglang.py --model $M --out $Q/base --max-new 6 --split $Q/lid_dev.jsonl --split $Q/lid_unseen_probe.jsonl --split $Q/lid_calib.jsonl --split $Q/sent_dev.jsonl --split $Q/sent_unseen_probe.jsonl --split $Q/sent_calib.jsonl > $Q/base_ls.log 2>&1; grep -E "items in|rror" $Q/base_ls.log | head
RUN=""
for t in lid sent; do
  for s in ${t}_dev ${t}_unseen_probe; do echo "== base $s"; python3 fc/int_report.py $Q/$s.jsonl $Q/base/$s.jsonl $Q/base/$s.jsonl | grep -v "==" | awk '{print $1,$2,$3}'; done
  acc=$(python3 fc/score_label.py $Q/${t}_dev.jsonl $Q/base/${t}_dev.jsonl | grep -o 'acc [0-9.]*' | cut -d' ' -f2)
  python3 -c "import sys; sys.exit(0 if 0.90 <= $acc <= 0.985 else 1)" && RUN="$RUN $t"
done
echo "tasks to prune: $RUN"
[ -z "$RUN" ] && exit 0
A=""; for t in $RUN; do A="$A --items $Q/${t}_calib.jsonl --answers $Q/base/${t}_calib.jsonl --out $Q/rs_$t.pt"; done
python3 fc/q35_route_stats.py --model $M $A > $Q/rs_ls.log 2>&1; grep -vE "Warning|warn" $Q/rs_ls.log | tail -5
for t in $RUN; do
  python3 fc/q35_slice.py --model $M --stats $Q/rs_$t.pt --k 104 --bits 3 --out /tmp/s_$t 2>&1 | grep -E "coverage|rror"
  python3 fc/gen_sglang.py --model /tmp/s_$t --out $Q/k104_$t --max-new 6 --split $Q/${t}_dev.jsonl --split $Q/${t}_unseen_probe.jsonl > $Q/k104_$t.log 2>&1; grep -E "items in|rror" $Q/k104_$t.log | head -3
  for s in ${t}_dev ${t}_unseen_probe; do echo "== k104 $s"; python3 fc/int_report.py $Q/$s.jsonl $Q/base/$s.jsonl $Q/k104_$t/$s.jsonl | grep -v "==" | cut -c1-90; done
  rm -rf /tmp/s_$t
done
