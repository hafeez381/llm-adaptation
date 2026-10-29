#!/usr/bin/env bash
# GPU session S1 (Colab L4), from the repo root: a 20-update Full FT dry run (seconds/update, peak memory),
# then the Full FT and L2-SP runs. Each stage must exit 0 before the next starts.
set -u
STORE=hub:hafeez381/llm-adaptation
stage() {  # stage NAME ARGS...
  local name=$1; shift
  echo "START $name $(date -u +%FT%TZ)"
  python -m task1.train "$@" > "outputs/logs/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)"; tail -n 2 "outputs/logs/$name.log"
  [ $rc -eq 0 ] || exit $rc
}
stage dryrun --method full_ft --mode full --store $STORE --stop-after 20
mkdir -p outputs/dryrun && mv results/task1/full_ft/* outputs/dryrun/ && cat outputs/dryrun/resources.json
stage full_ft --method full_ft --mode full --store $STORE
stage l2sp --method l2sp --mode full --store $STORE
