#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

enter_repo_root
require_env OPENAI_API_KEY

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
  echo "Evaluating ablation: ${condition}"
  evaluate_open_eval \
    "${OUT_DIR}/remove_${condition}_4.1-mini.json" \
    "${PXHX_PATH}" \
    "${OUT_DIR}/eval_${condition}_1_${MODEL_NAME}.json" \
    "model_response_neutralized_1"
done
