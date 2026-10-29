#!/usr/bin/env bash
# Recovery chain after the Hub private-storage quota stopped Full FT's final upload (update 1,250 was trained and its
# resume snapshot is on the VM). Frees the Hub copy of that snapshot, writes Full FT's final weights from it, then
# trains L2-SP and the adapters with VM-local resume snapshots, then evaluates all seven conditions.
set -u
STORE=hub:hafeez381/llm-adaptation
SNAP=/content/snapshots
OLD=/content/repo
stage() {  # stage NAME CMD...
  local name=$1; shift
  echo "START $name $(date -u +%FT%TZ)"
  "$@" > "outputs/logs/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)"; tail -n 3 "outputs/logs/$name.log"
  [ $rc -eq 0 ] || exit $rc
}
mkdir -p results/task1/full_ft data outputs/logs && ln -sfn $OLD/data/cache data/cache
rm -f $OLD/results/task1/full_ft/resume-0937.pt   # superseded by resume-1250
cp $OLD/results/task1/full_ft/train_log.jsonl results/task1/full_ft/ && cp -r $OLD/results/task1/full_ft/step-0625 results/task1/full_ft/
stage free_hub python -c "
from common.hub_sync import HubStore
s = HubStore('hafeez381/llm-adaptation')
for f in ('task1/full_ft/resume/resume-1250.pt', 'task1/full_ft/resume/LATEST'):
    s.delete(f)
print(sorted(s.api.list_repo_files(s.repo_id)))"
stage finalize_full_ft python -m task1.finalize_run --method full_ft --snapshot $OLD/results/task1/full_ft/resume-1250.pt \
  --store $STORE --peak-from $OLD/outputs/dryrun/resources.json
rm -f $OLD/results/task1/full_ft/resume-1250.pt  # final weights are on the Hub, sha-verified
for m in l2sp lora4 lora16 oplora16; do
  stage "$m" python -m task1.train --method "$m" --mode full --store $STORE --snapshot-store $SNAP
done
rev=$(python -c 'import json; print(json.load(open("data/manifest.json"))["revisions"]["github:emergent-misalignment"])')
curl -sL "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/$rev/evaluation/first_plot_questions.yaml" -o /tmp/fpq.yaml
python -c 'import json,hashlib,shutil; m=json.load(open("data/manifest.json")); h=hashlib.sha256(open("/tmp/fpq.yaml","rb").read()).hexdigest(); assert h==m["qual_prompts"]["sha256"], h; shutil.copy("/tmp/fpq.yaml","data/cache/first_plot_questions.yaml"); print("yaml sha ok")' || exit 1
for c in baseline full_ft@625 full_ft l2sp lora4 lora16 oplora16; do
  stage "eval_$c" python -m task1.evaluate --condition "$c" --store $STORE --mode full
  rm -rf outputs/models/src
done
