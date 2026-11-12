set -x
python3 -c "import sys;print(sys.version)"; nproc; free -g; df -h /tmp / | cat
pip list 2>/dev/null | grep -iE "^(torch|transformers|sglang|vllm|flashinfer|accelerate|safetensors|hf.transfer|huggingface.hub|numpy|triton|datasets) "
ls /data/models
pip install -q hf_transfer 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
time huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --local-dir /data/models/Qwen3-30B-A3B-Instruct-2507 2>&1 | tail -3 || time hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --local-dir /data/models/Qwen3-30B-A3B-Instruct-2507 2>&1 | tail -3
du -sh /data/models/Qwen3-30B-A3B-Instruct-2507; ls /data/models/Qwen3-30B-A3B-Instruct-2507 | head -50
