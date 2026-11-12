# Full Qwen3.5 entry for one short-label task: base answers, packed build, locked exam, feedback + controls.
# Usage: bash runs/q35_task.sh TASK K MAXNEW EXAM POOL   (EXAM/POOL relative to the repo, e.g. data/sent_exam.jsonl)
set -u
exec > >(tee -a /data/fc/q35final/$1.progress.txt) 2>&1
# Jobs can share one GPU container: wait until the GPU is free before starting.
for i in $(seq 1 300); do mkdir /tmp/fc_gpu_lock 2>/dev/null && break; sleep 10; done; trap 'rmdir /tmp/fc_gpu_lock' EXIT
for i in $(seq 1 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 4000 ] && break; sleep 10; done; echo "gpu wait $((i*10))s used=${u}MiB"
T=$1; K=$2; MN=$3; EXAM=$4; POOL=$5
M=/data/models/Qwen3.5-35B-A3B; Q=/data/fc/q35; R=/data/fc/q35final/$T; ART=/data/fc/art/q35_$T
mkdir -p $R; cp $EXAM $POOL $R/
EN=$(basename $EXAM .jsonl); PN=$(basename $POOL .jsonl)
echo "== locked exam sha256"; sha256sum $EXAM
GEN="python3 fc/gen_sglang.py --max-new $MN"
SGP="env PYTHONPATH=$PWD/fc/sglpatch FC_ROUTER_BIAS=1"
# 1. base model on the exam and the feedback pool
[ -s $R/base/$PN.jsonl ] || $GEN --model $M --out $R/base --split $EXAM --split $POOL > $R/base.log 2>&1; grep -E "items in|rror" $R/base.log | head -4
# 2. build
[ -s $ART/qmeta.json ] || python3 fc/q35_build.py --model $M --stats $Q/rs_$T.pt --k $K --out $ART 2>&1 | grep -E "coverage|ARTIFACT|rror|Traceback"
ls -la $ART
# 2b. (CHECK=1) the expert switch must really act: switch off 100 of K experts in every layer and expect garbage
if [ "${CHECK:-0}" = 1 ]; then
  python3 -c "import sys; sys.path.insert(0,'fc'); from q35_unpack import unpack_artifact; unpack_artifact('$ART','/tmp/ck_chk',router_off=[(l,e) for l in range(40) for e in range(8,$K)])"
  $SGP $GEN --model /tmp/ck_chk --out $R/chk --split $EXAM:40 > $R/chk.log 2>&1
  python3 -c "
import json; t=[json.loads(l)['text'] for l in open('$R/chk/$EN.jsonl')]; print('switch check outputs', t[:6])"
  rm -rf /tmp/ck_chk
fi
# 3. build answers (empty feedback state) on the exam and pool
if [ ! -s $R/empty/$PN.jsonl ]; then
python3 -c "import sys; sys.path.insert(0,'fc'); from q35_unpack import unpack_artifact; unpack_artifact('$ART','/tmp/ck_empty')"
$SGP $GEN --model /tmp/ck_empty --out $R/empty --split $EXAM --split $POOL > $R/empty.log 2>&1; grep -E "items in|rror|router-bias" $R/empty.log | head -4
fi
rm -rf /tmp/ck_empty
echo "== EXAM (empty feedback)"; python3 fc/int_report.py $EXAM $R/base/$EN.jsonl $R/empty/$EN.jsonl | tee $R/report_empty.txt
# 4. feedback class from the build's real mistakes on the pool
[ -s $R/fb/guard.jsonl ] || python3 fc/q35_select_fb.py $POOL $R/base/$PN.jsonl $R/empty/$PN.jsonl $R/fb
python3 -c "
import json,random
F=[json.loads(l) for l in open('$R/fb/feedback.jsonl')]
labs=sorted({x['annotation'] for x in F}|{x['gold'].lower() for x in [json.loads(l) for l in open('$POOL')]})
rng=random.Random(0); out=[]
for x in F:
    c=[l for l in labs if l!=x['annotation']]; out.append(dict(x,annotation=rng.choice(c)))
open('$R/fb/feedback_shuf.jsonl','w').writelines(json.dumps(o)+'\n' for o in out)"
# 5. learn (real and shuffled annotations)
for kind in real shuf; do
  f=$R/fb/feedback.jsonl; [ $kind = shuf ] && f=$R/fb/feedback_shuf.jsonl
  [ -s $R/state_$kind.json ] || python3 fc/q35_cli.py learn --artifact $ART --cache /tmp/fcc --in $f --guard $R/fb/guard.jsonl --out $R/state_$kind.json > $R/learn_$kind.log 2>&1; grep -E "switched|rror|Traceback" $R/learn_$kind.log | head -3
done
rm -rf /tmp/fcc; echo '{"switched_off": []}' > $R/state_empty.json; cat $R/state_real.json; echo
# 6. probes, nearby, feedback items for empty / real / shuffled; locked exam again with real feedback
for kind in empty real shuf; do
  X=""; [ $kind = real ] && X="--split $EXAM"
  if [ ! -s $R/$kind/feedback.jsonl ] || { [ $kind = real ] && [ ! -s $R/$kind/$EN.jsonl ]; }; then
  python3 -c "
import sys,json; sys.path.insert(0,'fc'); from q35_unpack import unpack_artifact
unpack_artifact('$ART','/tmp/ck_$kind',router_off=[tuple(x) for x in json.load(open('$R/state_$kind.json'))['switched_off']])"
  $SGP $GEN --model /tmp/ck_$kind --out $R/$kind --split $R/fb/probe_locked.jsonl --split $R/fb/nearby_test.jsonl --split $R/fb/feedback.jsonl $X > $R/fb_$kind.log 2>&1; grep -E "rror" $R/fb_$kind.log | head -3
  fi
  echo "== $kind"; for s in probe_locked nearby_test feedback; do echo -n "$s "; python3 fc/score_label.py $R/fb/$s.jsonl $R/$kind/$s.jsonl | head -1; done
  rm -rf /tmp/ck_$kind
done
[ -s $R/base/nearby_test.jsonl ] || $GEN --model $M --out $R/base --split $R/fb/probe_locked.jsonl --split $R/fb/nearby_test.jsonl > $R/base_fb.log 2>&1
echo "== base probes"; for s in probe_locked nearby_test; do echo -n "$s "; python3 fc/score_label.py $R/fb/$s.jsonl $R/base/$s.jsonl | head -1; done
echo "== nearby kept vs empty"; python3 fc/score_label.py $R/fb/nearby_test.jsonl $R/empty/nearby_test.jsonl $R/real/nearby_test.jsonl
echo "== EXAM after real feedback"; python3 fc/int_report.py $EXAM $R/base/$EN.jsonl $R/real/$EN.jsonl | tee $R/report_real.txt
sha256sum $R/fb/*.jsonl
# 7. ship the small result files back (feedback sets, states, reports) as base64 tar between markers
echo "@@TAR_BEGIN@@"; tar czf - -C $R fb state_empty.json state_real.json state_shuf.json report_empty.txt report_real.txt | base64 -w0; echo; echo "@@TAR_END@@"
