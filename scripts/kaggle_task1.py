"""Kaggle GPU kernel for one Task 1 training run (C3-C6) on a T4. The push script prepends `METHOD = "<name>"`.

Two copies of everything, one off the machine (global rule): resume snapshots, the training log, weights and
results go to the private Hub repo (sha256-verified on every upload), and the weights and results are also kept
in /kaggle/working, which Kaggle saves as the kernel output. The token comes from the private dataset
abdul381/pa3-hf-token.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

METHOD = globals().get("METHOD")  # set by scripts/kaggle_task1.sh
REPO = Path("/tmp/repo")
LOG = Path("/kaggle/working/logs")
STORE = "hub:hafeez381/llm-adaptation-t4"
SNAP = STORE


def sh(name, cmd):
    LOG.mkdir(parents=True, exist_ok=True)
    print(f"== {name}: {cmd}", flush=True)
    with open(LOG / f"{name}.log", "w") as f:
        rc = subprocess.run(cmd, shell=True, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                            env=os.environ | {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
                                              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).returncode
    print(f"== {name} rc={rc}", flush=True)
    print((LOG / f"{name}.log").read_text()[-3000:], flush=True)
    if rc:
        sys.exit(rc)


src = next(p for p in [Path("/kaggle/input/pa3-task1-code"), *Path("/kaggle/input").rglob("*")]
           if (p / "GIT_SHA").exists())
if (src / "repo.tar.gz").exists():
    REPO.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(src / "repo.tar.gz"), "-C", str(REPO)], check=True)
else:  # Kaggle unpacks uploaded archives into a folder
    shutil.copytree(src / "repo", REPO, dirs_exist_ok=True)
shutil.copy(src / "GIT_SHA", REPO / "GIT_SHA")
tok = next(Path("/kaggle/input").rglob("hf_token"))
Path("/root/.cache/huggingface").mkdir(parents=True, exist_ok=True)
shutil.copy(tok, "/root/.cache/huggingface/token")

sh("setup", "pip uninstall -q -y torchao; grep -v -E '^(torch|numpy|pandas|pyarrow|matplotlib|pytest)==' requirements.txt > /tmp/req.txt && "
            "pip install -q -r /tmp/req.txt && pip install -q pytest && nvidia-smi && "
            "python -c 'from task1.setup_inputs import env_info; from common.results import write_json; "
            "write_json(\"/kaggle/working/env.json\", env_info()); print(env_info())' && cat GIT_SHA")
sh("data", "python -c 'import json; print(json.load(open(\"data/manifest.json\"))[\"task1_data\"][\"ordered_ids_sha256\"])' > /tmp/want && "
           "cp data/manifest.json /tmp/manifest.json && python -m task1.build_data --mode full | tail -c 300 && "
           "python -c 'import json; print(json.load(open(\"data/manifest.json\"))[\"task1_data\"][\"ordered_ids_sha256\"])' > /tmp/got && "
           "cp /tmp/manifest.json data/manifest.json && diff /tmp/want /tmp/got && echo DATA_MATCH")
sh("tests", "python -m pytest -q tests/test_task1_train.py tests/test_task1_adapters.py")
sh(f"train_{METHOD}", f"python -m task1.train --method {METHOD} --mode full --store {STORE} --snapshot-store {SNAP}")
shutil.copytree(REPO / "results/task1" / METHOD, f"/kaggle/working/results/{METHOD}",
                ignore=shutil.ignore_patterns("resume-*.pt"), dirs_exist_ok=True)   # weights, logs, resources
print("ALL_DONE", flush=True)
