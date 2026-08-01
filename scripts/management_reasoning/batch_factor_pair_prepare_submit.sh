#!/usr/bin/env bash
# Full factor-pair runs: format_tone + content_format × Claude+Gemini.
# 8 Batch jobs: 4 legacy free-form dx + 4 independent MR.
# Set END_IDX for a partial run; omit for full n=2697.
#
# Usage:
#   bash ./scripts/management_reasoning/batch_factor_pair_prepare_submit.sh
#   END_IDX=99 bash ./scripts/management_reasoning/batch_factor_pair_prepare_submit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
DX_SUITE="${DX_SUITE:-legacy_dx_factor}"
MR_SUITE="${MR_SUITE:-independent_factor}"
END_IDX="${END_IDX:-}"

echo "Building factor-pair inputs…"
python3 -m management_reasoning.prepare_factor_pair_inputs

mapfile -t JOBS < <(python3 - <<'PY'
from management_reasoning.batch.paths import FACTOR_PAIR_JOBS
for t, a in FACTOR_PAIR_JOBS:
    print(f"{t} {a}")
PY
)

submit_suite() {
  local suite="$1"
  echo "=== prepare+submit suite=$suite end_idx=${END_IDX:-FULL} ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    if [[ -n "$END_IDX" ]]; then
      python3 -m management_reasoning.batch prepare \
        --suite "$suite" --provider "$TARGET" --arm "$ARM" \
        --bucket "$BUCKET" --end_idx "$END_IDX"
    else
      python3 -m management_reasoning.batch prepare \
        --suite "$suite" --provider "$TARGET" --arm "$ARM" \
        --bucket "$BUCKET"
    fi
  done
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.batch submit \
      --suite "$suite" --provider "$TARGET" --arm "$ARM" \
      --bucket "$BUCKET"
  done
}

submit_suite "$DX_SUITE"
submit_suite "$MR_SUITE"

echo "=== manifests ==="
python3 - <<PY
import json
from management_reasoning.batch.paths import FACTOR_PAIR_JOBS, local_manifest_path
for suite in ("$DX_SUITE", "$MR_SUITE"):
    for t, a in FACTOR_PAIR_JOBS:
        p = local_manifest_path(t, a, suite=suite)
        m = json.load(open(p))
        print(f"{suite} {t}/{a}: {m.get('job_name')}  {m.get('job_state')}")
PY

cat <<EOF

Next:
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh status
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh status
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect

Flash-Lite (diagnosis arms):
  SUITE=legacy_dx_factor bash ./scripts/management_reasoning/smoke_eval_factor_dx_n10.sh

Flip plots (after MR collect):
  python3 -m management_reasoning.analysis.plot_independent_factor

EOF
