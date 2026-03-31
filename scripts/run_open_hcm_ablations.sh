#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"

PXHX_PATH=./results/HCM-3k/truth/merged_truth_new.json
OUT_DIR=./results/HCM-3k/exp_5
MODEL_NAME=4.1-mini

declare -a CONDITIONS=(
  content
  format
  tone
  content_format
  content_tone
  format_tone
  all
)

for condition in "${CONDITIONS[@]}"; do
  input_path="${OUT_DIR}/remove_${condition}_4.1-mini.json"
  output_path="${OUT_DIR}/eval_${condition}_1_${MODEL_NAME}.json"

  echo "Evaluating ablation: ${condition}"
  python open_eval/evaluate.py \
    --input_path "${input_path}" \
    --pxhx_path "${PXHX_PATH}" \
    --output_path "${output_path}" \
    --col_name "model_response_neutralized_1" \
    --resume_path "${output_path}"
done
