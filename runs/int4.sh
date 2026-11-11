M=/data/models/Qwen3-30B-A3B-Instruct-2507; I=/data/fc/int
ev() { n=$1; shift; R=/data/fc/var/$n; mkdir -p $R
  python3 fc/make_plan.py --stats $I/reap_calib.pt --expert 16 --attn 16 --embed 16 --head 16 --out $R/plan.json "$@" | tail -1
  (cd fc && python3 compress.py --base $M --plan $R/plan.json --out /tmp/artbf) > $R/c.log 2>&1
  (cd fc && python3 unpack.py /tmp/artbf /tmp/ckbf) > $R/u.log 2>&1 || tail $R/u.log; rm -rf /tmp/artbf
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ckbf --out $R/bf --max-new 16 --split $I/data/int_clinc_dev.jsonl > $R/gen.log 2>&1; grep -E "rror" $R/gen.log
  echo "== $n"; python3 fc/score_label.py $I/data/int_clinc_dev.jsonl $I/base/int_clinc_dev.jsonl $R/bf/int_clinc_dev.jsonl | tail -1; rm -rf /tmp/ckbf; }
ev sum_g3000 --k 56 --crit sum --total 3000 --minper 32
ev sum_g3300 --k 56 --crit sum --total 3300 --minper 36
ev sum_g3600 --k 56 --crit sum --total 3600 --minper 40
