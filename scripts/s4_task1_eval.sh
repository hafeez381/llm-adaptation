#!/usr/bin/env bash
# GPU session S4 on the same L4, queued after the adapter chain: evaluate the seven Task 1 conditions (C7)
# and generate the 168 qualitative responses (C9). usage: bash scripts/s4_task1_eval.sh <previous chain log>
set -u
prev=$1
STORE=hub:hafeez381/llm-adaptation
while ! grep -q '^EXIT=' "$prev"; do sleep 30; done
grep -q '^EXIT=0' "$prev" || { echo "previous chain failed: $(grep '^EXIT=' "$prev")"; exit 1; }
rev=$(python -c 'import json; print(json.load(open("data/manifest.json"))["revisions"]["github:emergent-misalignment"])')
mkdir -p data/cache && curl -sL "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/$rev/evaluation/first_plot_questions.yaml" -o data/cache/first_plot_questions.yaml
python -c 'import json,hashlib; m=json.load(open("data/manifest.json")); h=hashlib.sha256(open("data/cache/first_plot_questions.yaml","rb").read()).hexdigest(); assert h==m["qual_prompts"]["sha256"], h; print("yaml sha ok")' || exit 1
for c in baseline full_ft@625 full_ft l2sp lora4 lora16 oplora16; do
  echo "START $c $(date -u +%FT%TZ)"
  python -m task1.evaluate --condition "$c" --store $STORE --mode full > "outputs/logs/eval_$c.log" 2>&1
  rc=$?
  echo "END $c rc=$rc $(date -u +%FT%TZ)"; tail -n 3 "outputs/logs/eval_$c.log"
  [ $rc -eq 0 ] || exit $rc
  rm -rf outputs/models/src   # downloaded weights; the FP16 eval copy stays until the next condition
done
