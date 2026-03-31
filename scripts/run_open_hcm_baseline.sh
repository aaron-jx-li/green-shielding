#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"

INPUT_PATH=./results/HCM-3k/exp_4/responses_gpt-5-nano.json
PXHX_PATH=./results/HCM-3k/truth/merged_truth_new.json
OUT_DIR=./results/HCM-3k/exp_4
MODEL_NAME=gpt-5-nano

echo "Evaluating raw baseline responses"
python open_eval/evaluate.py \
  --input_path "${INPUT_PATH}" \
  --pxhx_path "${PXHX_PATH}" \
  --output_path "${OUT_DIR}/eval_raw_1_${MODEL_NAME}.json" \
  --col_name "model_response_raw_1"
