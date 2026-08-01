#!/usr/bin/env python3
"""Print radar-metric comparison table: paper mini vs Claude legacy vs MR primary.

Judge stacks differ (paper = gpt-4.1-mini; Claude legacy/MR = Flash-Lite HIGH).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

METRICS = [
    ("mean_plausibility", "Plaus"),
    ("mean_h_coverage", "H-cov"),
    ("mean_c_coverage", "S-cov"),
    ("mean_normalized_breadth", "Breadth"),
    ("mean_support_rate", "Evid"),
    ("mean_indirect_inference_rate", "Infer"),
    ("uncertainty_rate", "Unc"),
]

ROWS: List[Tuple[str, str, str]] = [
    (
        "paper mini raw (gpt-4.1-mini judge)",
        "results/HCM-3k/exp_4/eval_raw_1_gpt-4.1-mini.json",
        "paper",
    ),
    (
        "paper mini remove_all (gpt-4.1-mini judge)",
        "results/HCM-3k/exp_5/eval_all_1_4.1-mini.json",
        "paper",
    ),
    (
        "Claude legacy raw (Flash-Lite judge)",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_raw_legacy_diag/eval.json",
        "legacy",
    ),
    (
        "Claude legacy remove_all (Flash-Lite judge)",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_remove_all_legacy_diag/eval.json",
        "legacy",
    ),
    (
        "Claude MR primary raw (Flash-Lite, dx field)",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_raw_primary/eval.json",
        "mr",
    ),
    (
        "Claude MR primary neut content+tone (Flash-Lite, dx field)",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_neutralized_primary/eval.json",
        "mr",
    ),
]


def load_summary(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", default=".")
    args = ap.parse_args()

    header = f"{'condition':<58} " + " ".join(f"{lab:>7}" for _, lab in METRICS)
    print(header)
    print("-" * len(header))
    for label, rel, _ in ROWS:
        path = os.path.join(args.base_dir, rel)
        s = load_summary(path)
        if s is None:
            print(f"{label:<58} {'MISSING':>7}")
            continue
        vals = " ".join(f"{float(s[k]):7.3f}" for k, _ in METRICS)
        print(f"{label:<58} {vals}")
    print(
        "\nNote: paper rows use gpt-4.1-mini judges; Claude legacy/MR use "
        "gemini-3.1-flash-lite (thinking HIGH). MR primary scores only "
        "parsed.diagnosis; legacy scores full free-form answers. "
        "MR neutralized = content+tone, not remove_all."
    )


if __name__ == "__main__":
    main()
