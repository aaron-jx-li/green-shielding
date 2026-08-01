#!/usr/bin/env bash
# Status / collect Vertex Batch jobs.
# Usage:
#   bash ./scripts/management_reasoning/batch_status_collect.sh
#   SUITE=order_ord1 PROVIDER=claude ARM=raw bash ./scripts/management_reasoning/batch_status_collect.sh status
#   SUITE=order_ord1 PROVIDER=claude ARM=raw bash ./scripts/management_reasoning/batch_status_collect.sh collect
#
# Collect tag defaults from suite (primary → primary_batch, order_ord1 → order_ord1_batch).
# Only set TAG=... to override; do NOT leave TAG=primary_batch when collecting other suites.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

SUITE="${SUITE:-primary}"
CMD="${1:-status}"
PROVIDER_ARGS=()
ARM_ARGS=()
TAG_ARGS=()
[[ -n "${PROVIDER:-}" ]] && PROVIDER_ARGS=(--provider "$PROVIDER")
[[ -n "${ARM:-}" ]] && ARM_ARGS=(--arm "$ARM")
# Only pass --tag when explicitly set; otherwise Python uses suite_tag(suite).
if [[ -n "${TAG:-}" ]]; then
  TAG_ARGS=(--tag "$TAG")
fi

case "$CMD" in
  status)
    python3 -m management_reasoning.batch status \
      --suite "$SUITE" \
      "${PROVIDER_ARGS[@]}" \
      "${ARM_ARGS[@]}"
    ;;
  collect)
    python3 -m management_reasoning.batch collect \
      --suite "$SUITE" \
      "${TAG_ARGS[@]}" \
      "${PROVIDER_ARGS[@]}" \
      "${ARM_ARGS[@]}"
    ;;
  *)
    echo "Usage: $0 [status|collect]" >&2
    exit 2
    ;;
esac
