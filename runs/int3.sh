M=/data/models/Qwen3-30B-A3B-Instruct-2507; I=/data/fc/int
python3 fc/reap_stats.py --model $M --items $I/data/int_clinc_calib.jsonl --answers $I/base/int_clinc_calib.jsonl --out $I/reap_calib.pt > $I/reap.log 2>&1; tail -2 $I/reap.log
ev() { # name, plan args
  n=$1; shift; R=/data/fc/var/$n; mkdir -p $R
  python3 fc/make_plan.py --stats $I/reap_calib.pt --expert 16 --attn 16 --embed 16 --head 16 --out $R/plan.json "$@" | tail -1
  (cd fc && python3 compress.py --base $M --plan $R/plan.json --out /tmp/artbf) > $R/c.log 2>&1
  (cd fc && python3 unpack.py /tmp/artbf /tmp/ckbf) > $R/u.log 2>&1 || tail $R/u.log; rm -rf /tmp/artbf
  PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1 python3 fc/gen_sglang.py --model /tmp/ckbf --out $R/bf --max-new 16 --split $I/data/int_clinc_dev.jsonl > $R/gen.log 2>&1; grep -E "rror" $R/gen.log
  echo "== $n"; python3 fc/score_label.py $I/data/int_clinc_dev.jsonl $I/base/int_clinc_dev.jsonl $R/bf/int_clinc_dev.jsonl | tail -1
  rm -rf /tmp/ckbf
}
ev freq_u56 --k 56 --crit freq
ev freq_g --k 56 --crit freq --total 2690 --minper 32
ev sum_g --k 56 --crit sum --total 2690 --minper 32
