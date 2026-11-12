M=/data/models/Qwen3-30B-A3B-Instruct-2507; D=/data/fc/data; O=/data/fc/base
cat $D/calib.jsonl $D/calib_toolace.jsonl > /tmp/ci.jsonl; cat $O/calib.jsonl $O/calib_toolace.jsonl > /tmp/ca.jsonl
timeout 1500 python3 fc/reap_stats.py --model $M --items /tmp/ci.jsonl --answers /tmp/ca.jsonl --out $O/reap_calib.pt > $O/reap.log 2>&1; grep -vi warn $O/reap.log | tail -15
python3 -c "
import torch; s=torch.load('$O/reap_calib.pt'); S=s['S_all']; C=s['C_all']
import sys
for l in [0,12,24,36,47]:
  v,_=S[l].sort(descending=True); c=v.cumsum(0)/v.sum(); print(l, 'experts for 80/90/95/99% mass:', [(c<q).sum().item()+1 for q in (0.8,0.9,0.95,0.99)], 'never routed', (C[l]==0).sum().item())
"
