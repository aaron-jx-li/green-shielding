#!/usr/bin/env python3
"""Build MR Batch inputs from results/new_neu neutralize artifacts.

Writes:
  hcm_new_neu_ct_old_inputs.json  ← gpt-5.2_old_remove_content_tone.json
  hcm_new_neu_ct_new_inputs.json  ← gpt-5.2_new_remove_content_tone.json
  hcm_new_neu_ra_new_inputs.json  ← gpt-5.2_old_remove_content_format_tone.json

Each row: sample_id, raw_input, neutralized_prompt, neutralization meta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.batch.paths import (
    NEW_NEU_CATEGORIES,
    NEW_NEU_INPUT_BY_ARM,
    NEW_NEU_SOURCE_BY_ARM,
)


def _build(source_path: str, out_path: str, arm: str) -> int:
    with open(source_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list in {source_path}")

    cats = list(NEW_NEU_CATEGORIES[arm])
    out = []
    for i, row in enumerate(rows):
        raw = (row.get("raw_input") or "").strip()
        neut = (row.get("neutralized_prompt") or "").strip()
        if not raw or not neut:
            raise SystemExit(f"Empty raw/neutralized at index {i} in {source_path}")
        out.append(
            {
                "sample_id": i,
                "raw_input": raw,
                "neutralized_prompt": neut,
                "neutralization": {
                    "categories": cats,
                    "artifact": f"new_neu_{arm}",
                },
            }
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote n={len(out)} → {out_path}")
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        choices=("ct_old", "ct_new", "ra_new", "all"),
        default="all",
        help="Which new_neu arm input(s) to build (default: all)",
    )
    args = ap.parse_args()

    arms = ("ct_old", "ct_new", "ra_new") if args.arm == "all" else (args.arm,)
    for arm in arms:
        _build(
            NEW_NEU_SOURCE_BY_ARM[arm],
            NEW_NEU_INPUT_BY_ARM[arm],
            arm=arm,
        )


if __name__ == "__main__":
    main()
