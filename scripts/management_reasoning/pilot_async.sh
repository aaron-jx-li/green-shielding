#!/usr/bin/env bash
# Pilot A/B helper: n=50, concurrency=8. Set PROVIDER, ARM, OUT_DIR as needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-./results/management_reasoning/data/hcm_full_inputs.json}"
PROVIDER="${PROVIDER:-gemini}"
ARM="${ARM:-raw}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-49}"
CONCURRENCY="${CONCURRENCY:-8}"

if [[ "$PROVIDER" == "claude" ]]; then
  MODEL="${MODEL:-claude-opus-4-5@20251101}"
else
  MODEL="${MODEL:-gemini-3.1-pro-preview}"
fi
MODEL_DIR="${MODEL//@/_}"
OUT_DIR="${OUT_DIR:-./results/management_reasoning/responses/vertex/${MODEL_DIR}/${ARM}}"

if [[ ! -f "$INPUT" ]]; then
  echo "Preparing inputs..."
  python3 management_reasoning/prepare_data.py --out_path "$INPUT"
fi

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

python3 management_reasoning/run_async.py \
  --input_path "$INPUT" \
  --out_dir "$OUT_DIR" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --arm "$ARM" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --concurrency "$CONCURRENCY" \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --skip_existing
