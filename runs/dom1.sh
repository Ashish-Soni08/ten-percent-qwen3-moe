M=/data/models/Qwen3-30B-A3B-Instruct-2507; I=/data/fc/int; cp data/int_dom_*.jsonl $I/data/
python3 fc/gen_sglang.py --model $M --out $I/base --max-new 12 --split $I/data/int_dom_calib.jsonl --split $I/data/int_dom_dev.jsonl > $I/base_dom.log 2>&1; grep -E "items in|rror" $I/base_dom.log
(cd fc && python3 unpack.py /data/fc/art/int3300 /tmp/ck) > /dev/null 2>&1
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out /data/fc/var/int3300/dom --max-new 12 --split $I/data/int_dom_dev.jsonl > /tmp/g.log 2>&1; grep -E "items in|rror" /tmp/g.log
echo "== int3300 on domains"; python3 fc/score_label.py $I/data/int_dom_dev.jsonl $I/base/int_dom_dev.jsonl /data/fc/var/int3300/dom/int_dom_dev.jsonl
python3 fc/score_label.py $I/data/int_dom_calib.jsonl $I/base/int_dom_calib.jsonl
