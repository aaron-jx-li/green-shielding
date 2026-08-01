#!/usr/bin/env bash
# Smoke: prepare+submit Flash-Lite judge Batch stages for n=3, one target/arm.
# Default: gemini/raw. Stages extract+unc first (parallel-ready), then wait/collect,
# then sem+ground, then aggregate.
#
# Usage:
#   bash ./scripts/management_reasoning/smoke_eval_batch.sh
#   TARGET=claude ARM=neutralized bash ./scripts/management_reasoning/smoke_eval_batch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

SUITE="${SUITE:-smoke_n3}"
TARGET="${TARGET:-gemini}"
ARM="${ARM:-raw}"
END_IDX="${END_IDX:-2}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
POLL_SEC="${POLL_SEC:-30}"

common=(--suite "$SUITE" --target "$TARGET" --arm "$ARM" --bucket "$BUCKET" --location "$GOOGLE_CLOUD_LOCATION")

echo "=== prepare extract+unc (n=0..$END_IDX) ==="
python3 -m management_reasoning.eval.batch prepare --stage extract "${common[@]}" --end_idx "$END_IDX"
python3 -m management_reasoning.eval.batch prepare --stage unc "${common[@]}" --end_idx "$END_IDX"

echo "=== submit extract+unc ==="
python3 -m management_reasoning.eval.batch submit --stage extract "${common[@]}"
python3 -m management_reasoning.eval.batch submit --stage unc "${common[@]}"

wait_stage() {
  local stage="$1"
  while true; do
    out=$(python3 -m management_reasoning.eval.batch status --stage "$stage" "${common[@]}" 2>&1 || true)
    echo "$out"
    if echo "$out" | grep -Eq 'JOB_STATE_SUCCEEDED|SKIPPED_EMPTY'; then
      return 0
    fi
    if echo "$out" | grep -Eq 'JOB_STATE_FAILED|JOB_STATE_CANCELLED'; then
      echo "Job failed for stage=$stage" >&2
      return 1
    fi
    sleep "$POLL_SEC"
  done
}

echo "=== wait extract+unc ==="
wait_stage extract
wait_stage unc

echo "=== collect extract+unc ==="
python3 -m management_reasoning.eval.batch collect --stage extract "${common[@]}"
python3 -m management_reasoning.eval.batch collect --stage unc "${common[@]}"

echo "=== prepare+submit sem+ground ==="
python3 -m management_reasoning.eval.batch prepare --stage sem "${common[@]}" --end_idx "$END_IDX"
python3 -m management_reasoning.eval.batch prepare --stage ground "${common[@]}" --end_idx "$END_IDX"
python3 -m management_reasoning.eval.batch submit --stage sem "${common[@]}"
python3 -m management_reasoning.eval.batch submit --stage ground "${common[@]}"

echo "=== wait sem+ground ==="
wait_stage sem
wait_stage ground

echo "=== collect sem+ground ==="
python3 -m management_reasoning.eval.batch collect --stage sem "${common[@]}"
python3 -m management_reasoning.eval.batch collect --stage ground "${common[@]}"

echo "=== aggregate ==="
python3 -m management_reasoning.eval.batch aggregate "${common[@]}" --end_idx "$END_IDX"

echo "Done smoke_eval_batch suite=$SUITE target=$TARGET arm=$ARM"
