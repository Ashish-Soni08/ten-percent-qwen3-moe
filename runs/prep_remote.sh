set -e
mkdir -p /data/fc/raw/bfcl/possible_answer /data/fc/data
cd /data/fc/raw
H="Authorization: Bearer $HF_TOKEN"
[ -f xlam60k.json ] || curl -sSL -H "$H" -o xlam60k.json https://huggingface.co/datasets/lockon/xlam-function-calling-60k/resolve/main/xlam_function_calling_60k.json
for f in simple multiple parallel parallel_multiple live_simple live_multiple live_parallel live_parallel_multiple; do
  curl -sSL -H "$H" -o bfcl/BFCL_v3_$f.json https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_$f.json
  curl -sSL -H "$H" -o bfcl/possible_answer/BFCL_v3_$f.json https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_$f.json
done
cd -; python3 fc/prep_data.py /data/fc/raw /data/fc/data
sha256sum /data/fc/data/*.jsonl
