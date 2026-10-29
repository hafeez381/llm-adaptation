"""Kaggle GPU kernel for Task 2 (C11-C17). Pushed as the kernel's code file by scripts/kaggle_task2.sh.

Unpacks the committed repo tarball from the attached dataset, installs the pinned packages (keeping the image's
torch), runs every smoke mode first, then feature caching, the 4,500-update run and the final evaluation.
Checkpoints go to a local store folder in /kaggle/working; everything needed afterwards stays in the output.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/kaggle/input/pa3-task2-code")
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


src = next(p for p in [SRC, *Path("/kaggle/input").rglob("*")] if (p / "task2_items.json").exists())
if (src / "repo.tar.gz").exists():
    REPO.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(src / "repo.tar.gz"), "-C", str(REPO)], check=True)
else:  # Kaggle unpacks uploaded archives into a folder
    shutil.copytree(src / "repo", REPO, dirs_exist_ok=True)
(REPO / "data/cache").mkdir(parents=True, exist_ok=True)
shutil.copy(src / "task2_items.json", REPO / "data/cache/task2_items.json")
shutil.copy(src / "GIT_SHA", REPO / "GIT_SHA")

sh("setup", "grep -v -E '^(torch|numpy|pandas|pyarrow|matplotlib|pytest)==' requirements.txt > /tmp/req.txt && "
            "pip install -q -r /tmp/req.txt && pip install -q pytest && nvidia-smi && "
            "python -c 'from task1.setup_inputs import env_info; from common.results import write_json, ROOT; "
            "write_json(ROOT/\"results/task2/env.json\", env_info()); print(env_info())' && cat GIT_SHA")
sh("build_smoke", "python -c \"import json; d=json.load(open('data/cache/task2_items.json')); "
                  "s={k:(v[:8] if k.startswith('caption') else v[:4]) for k,v in d.items()}; "
                  "json.dump(s, open('data/cache/task2_items_smoke.json','w'))\"")
sh("tests", "python -m pytest -q tests/test_task2.py")
sh("smoke_train", f"python -m task2.train --mode smoke --store {STORE}_smoke")
sh("smoke_final", f"python -m task2.final_eval --mode smoke --store {STORE}_smoke")
sh("cache_features", "python -m task2.cache_features --mode full")
sh("train", f"python -m task2.train --mode full --store {STORE}")
sh("final_eval", f"python -m task2.final_eval --mode full --store {STORE}")
shutil.rmtree(REPO / "data/cache/task2_features", ignore_errors=True)   # 2.7 GB, rebuilt from the image ids
shutil.rmtree(REPO / "data/cache/task2_images", ignore_errors=True)
print("ALL_DONE", flush=True)
