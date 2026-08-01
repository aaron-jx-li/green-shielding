#!/usr/bin/env python3
"""Build Batch inputs for paper factor-pair neutralize artifacts.

Writes:
  hcm_format_tone_inputs.json     ← remove_format_tone.json
  hcm_content_format_inputs.json  ← remove_content_format.json
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
    FACTOR_PAIR_INPUT_BY_ARM,
    FACTOR_PAIR_SOURCE_BY_ARM,
    FACTOR_PAIR_CATEGORIES,
)


def _build(source_path: str, out_path: str, arm: str) -> int:
    with open(source_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list in {source_path}")

    cats = FACTOR_PAIR_CATEGORIES[arm]
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
                    "categories": list(cats),
                    "artifact": f"remove_{arm}",
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
        choices=("format_tone", "content_format", "all"),
        default="all",
    )
    args = ap.parse_args()
    arms = ("format_tone", "content_format") if args.arm == "all" else (args.arm,)
    for arm in arms:
        _build(
            FACTOR_PAIR_SOURCE_BY_ARM[arm],
            FACTOR_PAIR_INPUT_BY_ARM[arm],
            arm=arm,
        )


if __name__ == "__main__":
    main()
