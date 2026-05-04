#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash src/jobs/probe_base.bash <model_name> [task1,task2,...] [precision] [processing]" >&2
  exit 1
fi

MODEL_NAME="$1"
TASKS="${2:-blimp,olmpics,stereoset,tomi}"
PRECISION="${3:-full}"
PROCESSING="${4:-parallel}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENCODINGS_DIR="$ROOT_DIR/encodings"
RESULTS_DIR="$ROOT_DIR/results"
CACHE_DIR="$ROOT_DIR/cache"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

for task in "${TASK_ARRAY[@]}"; do
  python3 "$ROOT_DIR/src/probe.py" \
    --task "$task" \
    --probe_type linear \
    --processing "$PROCESSING" \
    --project_prefix lift-pair \
    --chat_template_model_name "$MODEL_NAME" \
    --model_name "$MODEL_NAME" \
    --template_index 0 \
    --limit 5000 \
    --num_hidden_layers 0 \
    --encoding_folder "$ENCODINGS_DIR" \
    --model_precision "$PRECISION" \
    --control_task NONE \
    --result_folder "$RESULTS_DIR" \
    --probing_labels label \
    --questions original \
    --logging local \
    --cache_folder "$CACHE_DIR"
done
