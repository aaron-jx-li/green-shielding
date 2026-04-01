#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

enter_repo_root
require_env OPENAI_API_KEY

INPUT_PATH=./results/HCM-3k/exp_4/responses_gpt-5-nano.json
PXHX_PATH=./results/HCM-3k/truth/merged_truth_new.json
OUT_DIR=./results/HCM-3k/exp_4
MODEL_NAME=gpt-5-nano

echo "Evaluating raw baseline responses"
evaluate_open_eval \
  "${INPUT_PATH}" \
  "${PXHX_PATH}" \
  "${OUT_DIR}/eval_raw_1_${MODEL_NAME}.json" \
  "model_response_raw_1"
