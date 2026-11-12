pip install -q hf_transfer 2>&1 | tail -1; export HF_HUB_ENABLE_HF_TRANSFER=1
M=/data/models/Qwen3.5-35B-A3B; Q=/data/fc/q35; mkdir -p $Q
[ -f $M/config.json ] || (time hf download Qwen/Qwen3.5-35B-A3B --local-dir $M 2>&1 | tail -2)
du -sh $M; python3 -c "import transformers,sglang;print('tf',transformers.__version__,'sgl',sglang.__version__)"
python3 - <<'PY'
import json; w=json.load(open('/data/models/Qwen3.5-35B-A3B/model.safetensors.index.json'))['weight_map']
ks=sorted(w); print(len(ks)); [print(k) for k in ks if '.layers.0.' in k or '.layers.3.' in k][:80]
print([k for k in ks if 'layers' not in k][:30])
PY
cp data/int_clinc_calib.jsonl data/int_clinc_dev.jsonl $Q/; ls /data/fc/data | head
python3 fc/gen_sglang.py --model $M --out $Q/base --max-new 16 --split $Q/int_clinc_dev.jsonl --split $Q/int_clinc_calib.jsonl > $Q/base_int.log 2>&1; grep -E "items in|rror|deterministic" $Q/base_int.log | head; tail -5 $Q/base_int.log
python3 fc/score_label.py $Q/int_clinc_dev.jsonl $Q/base/int_clinc_dev.jsonl
head -3 $Q/base/int_clinc_dev.jsonl
python3 fc/gen_sglang.py --model $M --out $Q/base --max-new 768 --split /data/fc/data/dev.jsonl --split /data/fc/data/calib.jsonl > $Q/base_fc.log 2>&1; grep -E "items in|rror" $Q/base_fc.log | head
python3 fc/score.py $Q/base/dev.jsonl /data/fc/data/dev.jsonl 2>&1 | tail -4
