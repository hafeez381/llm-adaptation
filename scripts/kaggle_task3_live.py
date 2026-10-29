"""Kaggle GPU kernel for the Task 3 live experiment (C24).

Unpacks the committed repo tarball from the attached dataset, installs the pinned packages (keeping the image's
torch), runs every smoke mode first, then feature caching, the 4,500-update run and the final evaluation.
Checkpoints go to a local store folder in /kaggle/working; everything needed afterwards stays in the output.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/kaggle/input/pa3-task3-live")
REPO = Path("/kaggle/working/repo")
LOG = Path("/kaggle/working/logs")
STORE = "/kaggle/working/store"


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


src = next(p for p in [SRC, *Path("/kaggle/input").rglob("*")] if (p / "task3_live_problems.json").exists())
if (src / "repo.tar.gz").exists():
    REPO.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(src / "repo.tar.gz"), "-C", str(REPO)], check=True)
else:  # Kaggle unpacks uploaded archives into a folder
    shutil.copytree(src / "repo", REPO, dirs_exist_ok=True)
(REPO / "data/cache").mkdir(parents=True, exist_ok=True)
shutil.copy(src / "task3_live_problems.json", REPO / "data/cache/task3_live_problems.json")
shutil.copy(src / "GIT_SHA", REPO / "GIT_SHA")

sh("setup", "grep -v -E '^(torch|numpy|pandas|pyarrow|matplotlib|pytest)==' requirements.txt > /tmp/req.txt && "
            "pip install -q -r /tmp/req.txt && pip install -q pytest && nvidia-smi && "
            "python -c 'from task1.setup_inputs import env_info; from common.results import write_json, ROOT; "
            "write_json(ROOT/\"results/task3/live_env.json\", env_info()); print(env_info())' && cat GIT_SHA")
sh("live", f"python -m task3.live --mode full --store {STORE}")
print("ALL_DONE", flush=True)
