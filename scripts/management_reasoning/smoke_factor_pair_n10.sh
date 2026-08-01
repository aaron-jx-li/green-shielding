#!/usr/bin/env bash
# Smoke n=10: factor-pair arms (format_tone, content_format) × Claude+Gemini.
# Submits 8 Batch jobs: 4 legacy free-form dx + 4 independent MR.
# Reuses existing Claude/Gemini raw baselines (not submitted).
#
# Usage:
#   bash ./scripts/management_reasoning/smoke_factor_pair_n10.sh
#   END_IDX=4 bash ./scripts/management_reasoning/smoke_factor_pair_n10.sh
#
# After SUCCEEDED:
#   SUITE=legacy_dx_factor_smoke_n10 bash ./scripts/management_reasoning/batch_status_collect.sh collect
#   SUITE=independent_factor_smoke_n10 bash ./scripts/management_reasoning/batch_status_collect.sh collect
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

END_IDX="${END_IDX:-9}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
DX_SUITE="${DX_SUITE:-legacy_dx_factor_smoke_n10}"
MR_SUITE="${MR_SUITE:-independent_factor_smoke_n10}"

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
  echo "=== prepare+submit suite=$suite end_idx=$END_IDX ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.batch prepare \
      --suite "$suite" --provider "$TARGET" --arm "$ARM" \
      --bucket "$BUCKET" --end_idx "$END_IDX"
  done
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.batch submit \
      --suite "$suite" --provider "$TARGET" --arm "$ARM" \
      --bucket "$BUCKET"
  done
  python3 - <<PY
import json, os
from management_reasoning.batch.paths import FACTOR_PAIR_JOBS, local_manifest_path
suite = os.environ["SUITE_PRINT"]
for t, a in FACTOR_PAIR_JOBS:
    p = local_manifest_path(t, a, suite=suite)
    m = json.load(open(p))
    print(f"  {suite} {t}/{a}: {m.get('job_name')}  {m.get('job_state')}")
PY
}

export SUITE_PRINT="$DX_SUITE"
submit_suite "$DX_SUITE"
export SUITE_PRINT="$MR_SUITE"
submit_suite "$MR_SUITE"

cat <<EOF

Submitted 8 smoke jobs (4 dx + 4 MR), end_idx=$END_IDX.

Next:
  export GOOGLE_CLOUD_PROJECT=bin-yu-green-shield GOOGLE_CLOUD_LOCATION=global
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh status
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh status
  # when SUCCEEDED:
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect

Full cohort (no END_IDX):
  bash ./scripts/management_reasoning/batch_factor_pair_prepare_submit.sh

EOF
