#!/usr/bin/env bash
# Queued after S1 on the same L4: waits for the Full FT / L2-SP chain to exit 0, then trains LoRA-4, LoRA-16
# and OPLoRA-16 in that order. usage: bash scripts/s1b_task1_adapters.sh /content/repo/outputs/logs/s1.log
set -u
prev=$1
STORE=hub:hafeez381/llm-adaptation
while ! grep -q '^EXIT=' "$prev"; do sleep 30; done
grep -q '^EXIT=0' "$prev" || { echo "previous chain failed: $(grep '^EXIT=' "$prev")"; exit 1; }
for m in lora4 lora16 oplora16; do
  echo "START $m $(date -u +%FT%TZ)"
  python -m task1.train --method $m --mode full --store $STORE > "outputs/logs/$m.log" 2>&1
  rc=$?
  echo "END $m rc=$rc $(date -u +%FT%TZ)"; tail -n 2 "outputs/logs/$m.log"
  [ $rc -eq 0 ] || exit $rc
done
