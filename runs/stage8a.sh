D=/data/fc/data; O=/data/fc/rp56mine; mkdir -p $O
(cd fc && python3 unpack.py /data/fc/art/rp56 /tmp/ck) > /dev/null 2>&1
PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck --out $O --split $D/mine.jsonl > $O/gen.log 2>&1; grep -E "items in|rror" $O/gen.log
gzip -c $O/mine.jsonl | base64 -w0 > $O/mine.b64; wc -c $O/mine.b64
