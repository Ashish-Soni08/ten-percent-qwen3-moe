A=/data/fc/art/g56q3; D=/data/fc/data; O0=/data/fc/g56q3/final; O=/data/fc/g56q3/final_gb; mkdir -p $O
(cd fc && python3 unpack.py $A /tmp/ck) > /tmp/unpack.log 2>&1 || tail /tmp/unpack.log
python3 fc/fb_final.py --only-switch --ckpt /tmp/ck --fb data/fb/feedback.jsonl --probe data/fb/probe_locked.jsonl --guard data/fb/guard_big.jsonl --nearby data/fb/nearby_test.jsonl --calib x --out $O 2>&1 | grep '^{'
rm -rf /tmp/ck
for st in $O0/switch24_real.json $O0/switch48_real.json $O0/switch48_shuffled.json $O/switch24_real.json $O/switch48_real.json $O/switch48_shuffled.json; do
  tag=$(basename $(dirname $st))_$(basename $st .json)
  python3 fc/apply_and_gen.py $A $st /tmp/ck_$tag > /dev/null 2>&1
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ck_$tag --out $O/gen_$tag --split data/fb/probe_locked.jsonl --split data/fb/nearby_test.jsonl --split $D/dev.jsonl > $O/gen_$tag.log 2>&1; grep -E "rror" $O/gen_$tag.log | head -3
  echo "== $tag $(python3 -c "import json;print(len(json.load(open('$st'))))") switches"; python3 fc/score.py $O/gen_$tag/probe_locked.jsonl data/fb/probe_locked.jsonl; python3 fc/score.py $O/gen_$tag/nearby_test.jsonl data/fb/nearby_test.jsonl
  python3 fc/compare.py /data/fc/g56q3/dev.jsonl $O/gen_$tag/dev.jsonl $D/dev.jsonl "$tag"
  rm -rf /tmp/ck_$tag
done
