#!/usr/bin/env python3
"""Join HCM cohort with paper remove_all neutralization for legacy_diag Batch.

Writes ``results/management_reasoning/data/hcm_legacy_diag_inputs.json`` with:
  sample_id, raw_input, neutralized_prompt (= remove_all text), reference_diagnosis

``neutralized_prompt`` here is remove_all (format collapse), NOT MR content+tone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.batch.paths import LEGACY_DIAG_INPUT_PATH

DEFAULT_HCM = "./data/HCM-3k.json"
DEFAULT_REMOVE_ALL = "./results/HCM-3k/neutralized_prompts/remove_all.json"
DEFAULT_COHORT = "./results/management_reasoning/data/hcm_full_inputs.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hcm_path", default=DEFAULT_HCM)
    ap.add_argument("--remove_all_path", default=DEFAULT_REMOVE_ALL)
    ap.add_argument("--cohort_path", default=DEFAULT_COHORT, help="Optional MR cohort for reference_diagnosis")
    ap.add_argument("--out_path", default=LEGACY_DIAG_INPUT_PATH)
    args = ap.parse_args()

    with open(args.remove_all_path, encoding="utf-8") as f:
        remove_all = json.load(f)
    if not isinstance(remove_all, list):
        raise SystemExit(f"Expected list in {args.remove_all_path}")

    ref_by_id = {}
    if os.path.isfile(args.cohort_path):
        with open(args.cohort_path, encoding="utf-8") as f:
            cohort = json.load(f)
        for i, s in enumerate(cohort):
            sid = int(s.get("sample_id", i))
            ref_by_id[sid] = s.get("reference_diagnosis")

    out = []
    for i, row in enumerate(remove_all):
        raw = (row.get("raw_input") or "").strip()
        neut = (row.get("neutralized_prompt") or "").strip()
        if not raw or not neut:
            raise SystemExit(f"Empty raw/neutralized at index {i}")
        out.append(
            {
                "sample_id": i,
                "raw_input": raw,
                "neutralized_prompt": neut,
                "neutralization": {
                    "categories": ["content", "format", "tone"],
                    "artifact": "remove_all",
                },
                "reference_diagnosis": ref_by_id.get(i),
            }
        )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote n={len(out)} → {args.out_path}")


if __name__ == "__main__":
    main()
