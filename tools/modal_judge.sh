#!/usr/bin/env bash
# Run a command on a Modal runner with repo-relative file layout preserved.
# Usage: modal_judge.sh <gpu> "<cmd>" <timeout_s> <out_json> key=localpath [key=localpath ...]
set -euo pipefail
gpu=$1; cmd=$2; timeout_s=$3; out=$4; shift 4
base="https://${MODAL_WORKSPACE}--okc"; dom="${MODAL_REGION:-us-east}.modal.run"
auth=(-H "Modal-Key: ${MODAL_KEY}" -H "Modal-Secret: ${MODAL_SECRET}")
bodyf=$(mktemp); trap 'rm -f "$bodyf"' EXIT
python3 - "$gpu" "$cmd" "$timeout_s" "$@" > "$bodyf" <<'PY'
import json, sys
gpu, cmd, t, *pairs = sys.argv[1:]
files = {}
for p in pairs:
    k, v = p.split("=", 1)
    files[k] = open(v).read()
print(json.dumps({"gpu": gpu, "cmd": cmd, "timeout_s": int(t), "files": files}))
PY
call_id=$(curl -sS -f "${auth[@]}" -H 'Content-Type: application/json' --data-binary @"$bodyf" "${base}-submit.${dom}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("call_id") or sys.exit(json.dumps(d)))')
echo "$(date -u +%H:%M:%S) call_id=$call_id gpu=$gpu" >&2
while :; do
  r=$(curl -sS -f "${auth[@]}" "${base}-result.${dom}?call_id=${call_id}" || echo '{"status":"running"}')
  st=$(python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])' <<<"$r")
  if [ "$st" = running ]; then sleep 15; else printf '%s' "$r" > "$out"; python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("status", d.get("status"), "exit", d.get("exit_code"), "seconds", d.get("seconds"), d.get("error",""))
print("--- stdout (tail)"); print((d.get("stdout") or "")[-8000:])
print("--- stderr (tail)"); print((d.get("stderr") or "")[-3000:])
PY
    break; fi
done
