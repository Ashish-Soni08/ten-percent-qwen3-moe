M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; B=/data/fc/base
cat $D/calib.jsonl $D/calib_toolace.jsonl > /tmp/ci.jsonl; cat $B/calib.jsonl $B/calib_toolace.jsonl > /tmp/ca.jsonl
python3 fc/recon_prune.py --model $M --items /tmp/ci.jsonl --answers /tmp/ca.jsonl --total 2690 --out /data/fc/art/rp56.plan.json 2>&1 | grep -vE "Warning|warn|Loading" | tail -55
bash stage4.sh rp56 /data/fc/art/rp56.plan.json
