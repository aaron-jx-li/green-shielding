#!/usr/bin/env bash
# Pilot A: n=50 raw, concurrency=8, gemini-3.1-pro-preview @ global.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-./results/management_reasoning/data/hcm_full_inputs.json}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
ARM="${ARM:-raw}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-49}"
CONCURRENCY="${CONCURRENCY:-8}"
OUT_DIR="${OUT_DIR:-./results/management_reasoning/responses/vertex/${MODEL}/${ARM}}"

if [[ ! -f "$INPUT" ]]; then
  echo "Preparing inputs..."
  python3 management_reasoning/prepare_data.py --out_path "$INPUT"
fi

export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

python3 management_reasoning/run_async.py \
  --input_path "$INPUT" \
  --out_dir "$OUT_DIR" \
  --model "$MODEL" \
  --arm "$ARM" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --concurrency "$CONCURRENCY" \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --skip_existing
