#!/usr/bin/env bash
# Runs detached on the Colab VM from the repo root: install deps without touching the image's torch/numpy,
# record the environment, rebuild the Task 1 training data and check it against the committed manifest.
set -euo pipefail
grep -v -E '^(torch|numpy|pandas|pyarrow|matplotlib|pytest)==' requirements.txt > /tmp/req.txt
pip -q install -r /tmp/req.txt
python -c 'from task1.setup_inputs import env_info; from common.results import write_json, ROOT; write_json(ROOT/"results/env.json", env_info())'
cat results/env.json GIT_SHA
want=$(python -c 'import json; print(json.load(open("data/manifest.json"))["task1_data"]["ordered_ids_sha256"])')
cp data/manifest.json /tmp/manifest.committed.json
python -m task1.build_data --mode full | tail -c 400
got=$(python -c 'import json; print(json.load(open("data/manifest.json"))["task1_data"]["ordered_ids_sha256"])')
cp /tmp/manifest.committed.json data/manifest.json   # the committed manifest stays the reference
[ "$want" = "$got" ] && echo "DATA_MATCH $got" || { echo "DATA_MISMATCH want=$want got=$got"; exit 1; }
