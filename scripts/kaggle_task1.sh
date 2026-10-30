#!/usr/bin/env bash
# Push one Kaggle T4 kernel per Task 1 method, keeping at most two GPU sessions busy: a push that hits the session
# limit is retried every 2 minutes. usage: bash scripts/kaggle_task1.sh full_ft l2sp lora4 lora16 oplora16
set -u
cd "$(dirname "$0")/.."
for m in "$@"; do
  slug="pa3-t1-${m//_/-}"
  dir="outputs/kaggle_t1/$m"
  mkdir -p "$dir"
  printf 'METHOD = "%s"\n' "$m" > "$dir/run.py"
  cat scripts/kaggle_task1.py >> "$dir/run.py"
  cat > "$dir/kernel-metadata.json" <<EOF
{"id": "abdul381/$slug", "title": "$slug", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "",
 "dataset_sources": ["abdul381/pa3-task1-code"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
EOF
  until out=$(kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4 2>&1 | grep -v outdated); echo "$(date +%T) $m: $out"; \
        echo "$out" | grep -q "successfully pushed"; do
    python3 -c "import time; time.sleep(120)"
  done
done
