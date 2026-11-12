M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data
python3 fc/gen_sglang.py --model $M --out /tmp/o --split $D/dev.jsonl:40 > log.txt 2>&1
grep -vE "Warning|warn\(" log.txt | grep -iE -B2 -A8 "error|Traceback" | head -80
tail -5 log.txt
python3 fc/score.py /tmp/o/dev.jsonl $D/dev.jsonl; head -c 1500 /tmp/o/dev.jsonl
