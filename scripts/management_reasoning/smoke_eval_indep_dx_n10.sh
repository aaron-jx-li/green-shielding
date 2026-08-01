#!/usr/bin/env bash
# Smoke: Flash-Lite diagnosis eval on independent dx rows (n=10).
# Jobs: Claude raw/remove_all/ct_old/ct_new + Gemini raw.
#
# Usage:
#   bash ./scripts/management_reasoning/smoke_eval_indep_dx_n10.sh
#   END_IDX=4 bash ./scripts/management_reasoning/smoke_eval_indep_dx_n10.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

SUITE="${SUITE:-indep_dx_smoke_n10}"
END_IDX="${END_IDX:-9}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
POLL_SEC="${POLL_SEC:-20}"

mapfile -t JOBS < <(python3 - <<'PY'
from management_reasoning.eval.batch.paths import resolve_jobs
for t, a in resolve_jobs(None, None, suite="indep_dx_smoke_n10"):
    print(f"{t} {a}")
PY
)

common_base=(--suite "$SUITE" --bucket "$BUCKET" --location "$GOOGLE_CLOUD_LOCATION")

echo "=== suite=$SUITE end_idx=$END_IDX jobs=${#JOBS[@]} ==="
for ja in "${JOBS[@]}"; do echo "  $ja"; done

for stage in extract unc; do
  echo "=== prepare $stage ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.eval.batch prepare \
      --stage "$stage" "${common_base[@]}" \
      --target "$TARGET" --arm "$ARM" --end_idx "$END_IDX"
  done
done

for stage in extract unc; do
  echo "=== submit $stage ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.eval.batch submit \
      --stage "$stage" "${common_base[@]}" \
      --target "$TARGET" --arm "$ARM"
  done
done

wait_all_stage() {
  local stage="$1"
  while true; do
    all_ok=1
    for ja in "${JOBS[@]}"; do
      read -r TARGET ARM <<<"$ja"
      out=$(python3 -m management_reasoning.eval.batch status \
        --stage "$stage" "${common_base[@]}" \
        --target "$TARGET" --arm "$ARM" 2>&1 || true)
      echo "$out"
      if echo "$out" | grep -Eq 'JOB_STATE_FAILED|JOB_STATE_CANCELLED'; then
        echo "Job failed stage=$stage $TARGET/$ARM" >&2
        return 1
      fi
      if ! echo "$out" | grep -Eq 'JOB_STATE_SUCCEEDED|SKIPPED_EMPTY'; then
        all_ok=0
      fi
    done
    if [[ "$all_ok" -eq 1 ]]; then
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

echo "=== wait extract+unc ==="
wait_all_stage extract
wait_all_stage unc

for stage in extract unc; do
  echo "=== collect $stage ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.eval.batch collect \
      --stage "$stage" "${common_base[@]}" \
      --target "$TARGET" --arm "$ARM"
  done
done

for stage in sem ground; do
  echo "=== prepare $stage ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.eval.batch prepare \
      --stage "$stage" "${common_base[@]}" \
      --target "$TARGET" --arm "$ARM" --end_idx "$END_IDX"
  done
done

for stage in sem ground; do
  echo "=== submit $stage ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.eval.batch submit \
      --stage "$stage" "${common_base[@]}" \
      --target "$TARGET" --arm "$ARM"
  done
done

echo "=== wait sem+ground ==="
wait_all_stage sem
wait_all_stage ground

for stage in sem ground; do
  echo "=== collect $stage ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.eval.batch collect \
      --stage "$stage" "${common_base[@]}" \
      --target "$TARGET" --arm "$ARM"
  done
done

echo "=== aggregate ==="
for ja in "${JOBS[@]}"; do
  read -r TARGET ARM <<<"$ja"
  python3 -m management_reasoning.eval.batch aggregate \
    "${common_base[@]}" \
    --target "$TARGET" --arm "$ARM" --end_idx "$END_IDX"
done

echo "Done smoke_eval_indep_dx_n10 suite=$SUITE"
