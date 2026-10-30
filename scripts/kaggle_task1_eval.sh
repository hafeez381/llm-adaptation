#!/usr/bin/env bash
# Push the two Task 1 evaluation kernels once all five training kernels have finished. Kernel A evaluates the
# baseline and the full-model conditions; kernel B evaluates the adapters and runs the diagnostics.
set -u
cd "$(dirname "$0")/.."
SOURCES='["abdul381/pa3-t1-full-ft", "abdul381/pa3-t1-l2sp", "abdul381/pa3-t1-lora4", "abdul381/pa3-t1-lora16", "abdul381/pa3-t1-oplora16"]'
push() {  # push SLUG CONDITIONS_PY DIAGNOSTICS_PY
  local dir="outputs/kaggle_t1/$1"
  mkdir -p "$dir"
  printf 'CONDITIONS = %s\nDIAGNOSTICS = %s\n' "$2" "$3" > "$dir/run.py"
  cat scripts/kaggle_task1_eval.py >> "$dir/run.py"
  cat > "$dir/kernel-metadata.json" <<EOF
{"id": "abdul381/$1", "title": "$1", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "",
 "dataset_sources": ["abdul381/pa3-task1-code", "abdul381/pa3-hf-token"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
EOF
  until out=$(kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4 2>&1 | grep -v outdated); echo "$(date +%T) $1: $out"; \
        echo "$out" | grep -q "successfully pushed"; do
    python3 -c "import time; time.sleep(120)"
  done
}
push pa3-t1-eval-a '["baseline", "full_ft@625", "full_ft", "l2sp"]' False
push pa3-t1-eval-b '["lora4", "lora16", "oplora16"]' True
