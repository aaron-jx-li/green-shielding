#!/usr/bin/env bash
# Prepare + submit Vertex Batch jobs (primary suite by default).
# Usage:
#   bash ./scripts/management_reasoning/batch_prepare_submit.sh
#   SUITE=smoke_n5 END_IDX=4 PROVIDER=gemini ARM=raw bash ./scripts/management_reasoning/batch_prepare_submit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

SUITE="${SUITE:-primary}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
INPUT="${INPUT:-./results/management_reasoning/data/hcm_full_inputs.json}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
PROVIDER_ARGS=()
ARM_ARGS=()
[[ -n "${PROVIDER:-}" ]] && PROVIDER_ARGS=(--provider "$PROVIDER")
[[ -n "${ARM:-}" ]] && ARM_ARGS=(--arm "$ARM")

PREPARE_EXTRA=()
[[ -n "$END_IDX" ]] && PREPARE_EXTRA+=(--end_idx "$END_IDX")
PREPARE_EXTRA+=(--start_idx "$START_IDX")

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
