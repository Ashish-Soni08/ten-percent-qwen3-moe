M=/data/models/Qwen3-30B-A3B-Instruct-2507; S=/data/fc/saf; mkdir -p $S; cp data/saf_calib.jsonl data/saf_dev.jsonl $S/
python3 fc/gen_sglang.py --model $M --out $S/base --max-new 8 --split $S/saf_calib.jsonl --split $S/saf_dev.jsonl > $S/base.log 2>&1; grep -E "items in|rror" $S/base.log
python3 fc/compare_labels.py $S/base/saf_dev.jsonl $S/base/saf_dev.jsonl $S/saf_dev.jsonl base
python3 fc/recon_prune.py --model $M --items $S/saf_calib.jsonl --answers $S/base/saf_calib.jsonl --total 2690 --ntok 10000 --min-prompt 24 --out /data/fc/art/saf56.plan.json 2>&1 | grep -E "experts per layer|tokens"
