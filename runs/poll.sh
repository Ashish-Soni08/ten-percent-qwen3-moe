# poll an existing call id until done; write result json
cid=$1; out=$2; base="https://${MODAL_WORKSPACE}--okc"; dom="${MODAL_REGION:-us-east}.modal.run"
while :; do r=$(curl -sS -f -H "Modal-Key: ${MODAL_KEY}" -H "Modal-Secret: ${MODAL_SECRET}" "${base}-result.${dom}?call_id=${cid}" || echo '{"status":"running"}')
  st=$(python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])' <<<"$r"); [ "$st" = running ] && sleep 20 || { printf '%s' "$r" > $out; break; }; done
