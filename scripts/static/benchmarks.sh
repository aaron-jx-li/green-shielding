#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

enter_repo_root
require_env OPENAI_API_KEY

python static_eval/main.py \
  --task medxpertqa_diag \
  --model gpt-4.1-mini \
  --perturbation format_mc \
  --out_csv ./results/static/medxpertqa_mc_gpt-4.1-mini.csv \
  --include_raw \
  --judge_template with_Q \
  --num_runs 5 \
  --temperature 0.7

python static_eval/main.py \
  --task medxpertqa_diag \
  --model gpt-4.1-mini \
  --perturbation format_binary \
  --out_csv ./results/static/medxpertqa_binary_gpt-4.1-mini.csv \
  --include_raw \
  --judge_template with_Q \
  --num_runs 5 \
  --temperature 0.7
