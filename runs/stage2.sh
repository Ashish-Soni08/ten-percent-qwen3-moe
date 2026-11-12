# usage: bash stage2.sh "name|make_plan args" ...
M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; B=/data/fc/base; R=/data/fc/var; mkdir -p $R
for spec in "$@"; do
  name=${spec%%|*}; args=${spec#*|}; t0=$(date +%s)
  mkdir -p $R/$name
  python3 fc/make_plan.py --stats $B/reap_calib.pt $args --out $R/$name/plan.json
  (cd fc && python3 compress.py --base $M --plan $R/$name/plan.json --out /tmp/art_$name) > $R/$name/compress.log 2>&1; grep -E "artifact bytes|Error|error" $R/$name/compress.log
  (cd fc && python3 unpack.py /tmp/art_$name /tmp/ck_$name) > $R/$name/unpack.log 2>&1 || tail -5 $R/$name/unpack.log
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck_$name --out $R/$name --split $D/dev.jsonl --split $D/dev_toolace.jsonl > $R/$name/gen.log 2>&1; grep -E "items in|engine up|Error|patch failed" $R/$name/gen.log
  python3 fc/compare.py $B/dev.jsonl $R/$name/dev.jsonl $D/dev.jsonl $name
  python3 fc/compare.py $B/dev_toolace.jsonl $R/$name/dev_toolace.jsonl $D/dev_toolace.jsonl $name
  rm -rf /tmp/ck_$name /tmp/art_$name; echo "$name took $(( $(date +%s) - t0 ))s"
done
