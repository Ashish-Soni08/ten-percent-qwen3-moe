M=/data/models/Qwen3.5-35B-A3B; Q=/data/fc/q35; cp data/saf_calib.jsonl $Q/
[ -s $Q/base/saf_calib.jsonl ] || python3 fc/gen_sglang.py --model $M --out $Q/base --max-new 4 --split $Q/saf_calib.jsonl > $Q/base_saf.log 2>&1
python3 fc/score_label.py $Q/saf_calib.jsonl $Q/base/saf_calib.jsonl
[ -s $Q/rs_saf.pt ] || python3 fc/q35_route_stats.py --model $M --items $Q/saf_calib.jsonl --answers $Q/base/saf_calib.jsonl --out $Q/rs_saf.pt --n 1000 > $Q/rs_saf.log 2>&1; grep tokens $Q/rs_saf.log
bash runs/q35_task.sh saf 108 4 data/saf_exam.jsonl data/saf_pool.jsonl
