#!/usr/bin/env bash
# Smoke: legacy free-form diagnosis (LEGACY_DIAG_INSTRUCTION, temp 0.7, n=10).
# Skips Claude raw/remove_all (reuse existing legacy_diag collects).
# Generates: Claude ct_old/ct_new + Gemini raw/remove_all/ct_old/ct_new.
#
# Usage:
#   bash ./scripts/management_reasoning/smoke_legacy_dx_n10.sh
#   END_IDX=4 bash ./scripts/management_reasoning/smoke_legacy_dx_n10.sh
#
# After SUCCEEDED:
#   SUITE=legacy_dx_smoke_n10 bash ./scripts/management_reasoning/batch_status_collect.sh status
#   SUITE=legacy_dx_smoke_n10 bash ./scripts/management_reasoning/batch_status_collect.sh collect
#
# Full (same 6 arms, all samples):
#   SUITE=legacy_dx END_IDX= bash ./scripts/management_reasoning/…  # or omit END_IDX
#   See management_reasoning/README or comments below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

SUITE="${SUITE:-legacy_dx_smoke_n10}"
END_IDX="${END_IDX:-9}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"

# Ensure input cohorts exist.
if [[ ! -f ./results/management_reasoning/data/hcm_legacy_diag_inputs.json ]]; then
  python3 -m management_reasoning.prepare_legacy_diag_data
fi
if [[ ! -f ./results/management_reasoning/data/hcm_new_neu_ct_old_inputs.json ]]; then
  python3 -m management_reasoning.prepare_new_neu_inputs
fi

mapfile -t JOBS < <(python3 - <<'PY'
from management_reasoning.batch.paths import LEGACY_DX_GENERATE_JOBS
for t, a in LEGACY_DX_GENERATE_JOBS:
    print(f"{t} {a}")
PY
)

echo "=== suite=$SUITE end_idx=$END_IDX (skip Claude raw/remove_all) ==="
for ja in "${JOBS[@]}"; do echo "  $ja"; done

for ja in "${JOBS[@]}"; do
  read -r TARGET ARM <<<"$ja"
  python3 -m management_reasoning.batch prepare \
    --suite "$SUITE" --provider "$TARGET" --arm "$ARM" \
    --bucket "$BUCKET" --end_idx "$END_IDX"
done

for ja in "${JOBS[@]}"; do
  read -r TARGET ARM <<<"$ja"
  python3 -m management_reasoning.batch submit \
    --suite "$SUITE" --provider "$TARGET" --arm "$ARM" \
    --bucket "$BUCKET"
done

echo "=== submitted; manifests ==="
python3 <<'PY'
import json
from management_reasoning.batch.paths import LEGACY_DX_GENERATE_JOBS, local_manifest_path
import os
suite = os.environ.get("SUITE", "legacy_dx_smoke_n10")
for t, a in LEGACY_DX_GENERATE_JOBS:
    p = local_manifest_path(t, a, suite=suite)
    m = json.load(open(p))
    print(f"{t}/{a}: {m.get('job_name')}  {m.get('job_state')}")
PY

cat <<'EOF'

Next steps (after jobs SUCCEEDED):
  export GOOGLE_CLOUD_PROJECT=bin-yu-green-shield GOOGLE_CLOUD_LOCATION=global
  SUITE=legacy_dx_smoke_n10 bash ./scripts/management_reasoning/batch_status_collect.sh status
  SUITE=legacy_dx_smoke_n10 bash ./scripts/management_reasoning/batch_status_collect.sh collect

Full cohort (6 generate arms only):
  # prepare+submit with SUITE=legacy_dx and no END_IDX, looping LEGACY_DX_GENERATE_JOBS
  # Claude raw/remove_all: reuse results/HCM-3k/exp_frontier/.../{raw,remove_all}_legacy_batch/
  # Flash-Lite: full_response; reuse claude_{raw,remove_all}_legacy_diag eval for those two arms

EOF
