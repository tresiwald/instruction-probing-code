#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: bash src/jobs/encode_tasks.bash <model_name> <task1,task2,...> <device> <precision> [encoding_batch_size]" >&2
  exit 1
fi

MODEL_NAME="$1"
TASKS="$2"
DEVICE="$3"
PRECISION="$4"
ENCODING_BATCH_SIZE="${5:-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
ENCODINGS_DIR="$ROOT_DIR/encodings"

run_encode() {
  local task="$1"
  shift

  python3 "$SRC_DIR/encode.py" \
    --model_name "$MODEL_NAME" \
    --chat_template_model_name "$MODEL_NAME" \
    --task "$task" \
    --device "$DEVICE" \
    --model_precision "$PRECISION" \
    --encoding_folder "$ENCODINGS_DIR" \
    "$@"
}

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

for task in "${TASK_ARRAY[@]}"; do
  run_encode "$task" --upfront --questions original --k 0
  run_encode "$task" --upfront --questions original --k 4 --encoding_batch_size 1

  run_encode "$task" --questions original --k 0
  run_encode "$task" --questions original --k 0 --attention_limitation_question_right --attention_limitation_output --attention_limitation_layer_from 0
  run_encode "$task" --questions original --k 0 --attention_limitation_question_right --attention_limitation_layer_from 0
  run_encode "$task" --questions original --k 0 --attention_limitation_input_output --attention_limitation_layer_from 0
  run_encode "$task" --questions original --k 0 --attention_limitation_instruction_output --attention_limitation_layer_from 0

  run_encode "$task" --questions original --k 4 --encoding_batch_size "$ENCODING_BATCH_SIZE"
  run_encode "$task" --zero --questions original --k 4 --encoding_batch_size "$ENCODING_BATCH_SIZE"
done
