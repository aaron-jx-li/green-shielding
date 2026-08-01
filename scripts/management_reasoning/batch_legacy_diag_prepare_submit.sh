#!/usr/bin/env bash
# Prepare + submit Claude legacy_diag Batch (paper diagnosis-only, free-form).
# Usage:
#   bash ./scripts/management_reasoning/batch_legacy_diag_prepare_submit.sh
#   END_IDX=19 bash ./scripts/management_reasoning/batch_legacy_diag_prepare_submit.sh   # smoke n=20
#   ARM=raw bash ./scripts/management_reasoning/batch_legacy_diag_prepare_submit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
# Claude Batch uses us-east5 inside submit_job; global is fine for ADC default.
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

SUITE="${SUITE:-legacy_diag}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
INPUT="${INPUT:-./results/management_reasoning/data/hcm_legacy_diag_inputs.json}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
PROVIDER="${PROVIDER:-claude}"

if [[ ! -f "$INPUT" ]]; then
  echo "Building legacy_diag inputs…"
  python3 -m management_reasoning.prepare_legacy_diag_data
fi

PROVIDER_ARGS=(--provider "$PROVIDER")
ARM_ARGS=()
[[ -n "${ARM:-}" ]] && ARM_ARGS=(--arm "$ARM")

PREPARE_EXTRA=(--start_idx "$START_IDX")
[[ -n "$END_IDX" ]] && PREPARE_EXTRA+=(--end_idx "$END_IDX")

python3 -m management_reasoning.batch prepare \
  --suite "$SUITE" \
  --bucket "$BUCKET" \
  --input_path "$INPUT" \
  "${PREPARE_EXTRA[@]}" \
  "${PROVIDER_ARGS[@]}" \
  "${ARM_ARGS[@]}"

python3 -m management_reasoning.batch submit \
  --suite "$SUITE" \
  --bucket "$BUCKET" \
  "${PROVIDER_ARGS[@]}" \
  "${ARM_ARGS[@]}"

echo "Submitted. Poll with:"
echo "  SUITE=$SUITE PROVIDER=$PROVIDER bash ./scripts/management_reasoning/batch_status_collect.sh status"
echo "Collect with:"
echo "  SUITE=$SUITE PROVIDER=$PROVIDER bash ./scripts/management_reasoning/batch_status_collect.sh collect"
