#!/usr/bin/env bash
# Colab L4 chain for the three adapter runs while Kaggle trains Full FT and L2-SP. Snapshots, logs, weights and
# results go to the private Hub repo at every snapshot (sha256-verified), so the VM never holds the only copy.
set -u
STORE=hub:hafeez381/llm-adaptation-t4
for m in lora4 lora16 oplora16; do
  echo "START $m $(date -u +%FT%TZ)"
  python -m task1.train --method $m --mode full --store $STORE --snapshot-store $STORE > "outputs/logs/$m.log" 2>&1
  rc=$?
  echo "END $m rc=$rc $(date -u +%FT%TZ)"; tail -n 2 "outputs/logs/$m.log"
  [ $rc -eq 0 ] || exit $rc
done
