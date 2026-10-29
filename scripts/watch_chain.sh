#!/usr/bin/env bash
# Laptop-side watcher: polls the VM every INTERVAL s and exits (so the agent is woken) on a failed stage,
# a lost session, or the DONE pattern. Writes each poll to outputs/watch.log.
# usage: bash scripts/watch_chain.sh "<remote shell poll cmd>" "<done regex>" [INTERVAL]
poll=$1; done_re=$2; interval=${3:-600}
while true; do
  out=$(bash scripts/colab_run.sh sh "$poll" 2>&1 | grep -v "colab\] ")
  { date "+%F %T"; echo "$out" | tail -n 12; } >> outputs/watch.log
  if echo "$out" | grep -qi "session not found\|no such session"; then echo "SESSION_LOST"; echo "$out" | tail -20; exit 2; fi
  if echo "$out" | grep -qE "rc=[1-9]|EXIT=[1-9]|failed|Traceback|Error"; then echo "STAGE_FAILED"; echo "$out" | tail -30; exit 1; fi
  if echo "$out" | grep -qE "$done_re"; then echo "DONE"; echo "$out" | tail -30; exit 0; fi
  sleep "$interval"
done
