#!/usr/bin/env python3
"""Join full HCM-3k raw inputs with management-safe (content+tone) neutralized prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.prompts import NEUTRALIZATION_RECIPE
from normalization.io import atomic_json_dump


def _load_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def prepare(
    hcm_path: str,
    neutralized_path: str,
    out_path: str,
) -> List[Dict[str, Any]]:
    hcm = _load_list(hcm_path)
    neut = _load_list(neutralized_path)

    if len(hcm) != len(neut):
        raise ValueError(
            f"Length mismatch: HCM has {len(hcm)} samples, neutralized has {len(neut)}"
        )

    categories = list(NEUTRALIZATION_RECIPE["categories"])  # type: ignore[arg-type]
    artifact = str(NEUTRALIZATION_RECIPE["artifact_relpath"])

    out: List[Dict[str, Any]] = []
    for i, (h, n) in enumerate(zip(hcm, neut)):
        raw_h = h.get("raw_input", "")
        raw_n = n.get("raw_input", "")
        if raw_h != raw_n:
            raise ValueError(
                f"raw_input mismatch at index {i}: HCM and neutralized file are out of order"
            )
        neut_prompt = n.get("neutralized_prompt", "")
        if not isinstance(neut_prompt, str) or not neut_prompt.strip():
            raise ValueError(f"empty neutralized_prompt at index {i}")

        out.append(
            {
                "sample_id": i,
                "raw_input": raw_h,
                "neutralized_prompt": neut_prompt,
                "reference_diagnosis": h.get("reference_diagnosis"),
                "factors": h.get("factors") or {},
                "neutralization": {
                    "categories": categories,
                    "artifact": artifact,
                },
            }
        )

    atomic_json_dump(out, out_path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hcm_path", default="./data/HCM-3k.json")
    ap.add_argument(
        "--neutralized_path",
        default="./results/HCM-3k/neutralized_prompts/remove_content_tone.json",
    )
    ap.add_argument(
        "--out_path",
        default="./results/management_reasoning/data/hcm_full_inputs.json",
    )
    args = ap.parse_args()

    records = prepare(args.hcm_path, args.neutralized_path, args.out_path)
    print(f"Wrote {len(records)} records to {args.out_path}")


if __name__ == "__main__":
    main()
