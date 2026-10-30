"""Kaggle GPU kernel for Task 1 evaluation (C7, C9) and, optionally, diagnostics (C8). The push script prepends
`CONDITIONS = [...]` and `DIAGNOSTICS = True/False`. The five training kernels are attached as inputs; their
store folders are linked into one local store so merge/evaluate/diagnostics read them unchanged.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONDITIONS = globals().get("CONDITIONS", [])
DIAGNOSTICS = globals().get("DIAGNOSTICS", False)
REPO = Path("/tmp/repo")
LOG = Path("/kaggle/working/logs")
STORE = Path("/tmp/store")


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


src = next(p for p in Path("/kaggle/input").rglob("*") if (p / "GIT_SHA").exists() and "pa3-task1-code" in str(p))
if (src / "repo.tar.gz").exists():
    REPO.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(src / "repo.tar.gz"), "-C", str(REPO)], check=True)
else:
    shutil.copytree(src / "repo", REPO, dirs_exist_ok=True)
shutil.copy(src / "GIT_SHA", REPO / "GIT_SHA")

# one store view over the training kernels' outputs: <input>/store/task1/<method>/... -> /tmp/store/task1/<method>
for run in Path("/kaggle/input").rglob("store/task1/*"):
    if run.is_dir() and not (STORE / "task1" / run.name).exists():
        (STORE / "task1").mkdir(parents=True, exist_ok=True)
        os.symlink(run, STORE / "task1" / run.name)
print("store runs:", sorted(p.name for p in (STORE / "task1").iterdir()), flush=True)

sh("setup", "grep -v -E '^(torch|numpy|pandas|pyarrow|matplotlib|pytest)==' requirements.txt > /tmp/req.txt && "
            "pip install -q -r /tmp/req.txt && nvidia-smi && cat GIT_SHA && "
            "python -c 'from task1.setup_inputs import env_info; from common.results import write_json; "
            "write_json(\"/kaggle/working/env.json\", env_info())'")
sh("yaml", "mkdir -p data/cache && rev=$(python -c 'import json; print(json.load(open(\"data/manifest.json\"))[\"revisions\"][\"github:emergent-misalignment\"])') && "
           "curl -sL https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/$rev/evaluation/first_plot_questions.yaml "
           "-o data/cache/first_plot_questions.yaml && python -c 'import json,hashlib; m=json.load(open(\"data/manifest.json\")); "
           "h=hashlib.sha256(open(\"data/cache/first_plot_questions.yaml\",\"rb\").read()).hexdigest(); "
           "assert h==m[\"qual_prompts\"][\"sha256\"], h; print(\"yaml sha ok\")'")
for c in CONDITIONS:
    sh(f"eval_{c}", f"python -m task1.evaluate --condition '{c}' --store {STORE} --mode full")
    shutil.rmtree(REPO / "outputs/models", ignore_errors=True)
if DIAGNOSTICS:
    sh("diagnostics", f"python -m task1.diagnostics --store {STORE}")
    shutil.rmtree(REPO / "outputs/diag", ignore_errors=True)
shutil.copytree(REPO / "results/task1", "/kaggle/working/results/task1", dirs_exist_ok=True)
print("ALL_DONE", flush=True)
