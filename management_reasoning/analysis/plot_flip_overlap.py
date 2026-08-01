#!/usr/bin/env python3
"""Overlap of flipped sample IDs across independent MR ablations.

For each model, builds flip sets (vs raw) for:
  diagnostic_consensus, next_steps_consensus, care_seeking (|Δ|≥1),
  and a structured union of those three.

Reports pairwise Jaccard, intersection sizes, and how many ablations
touch each flipped sample. Helps choose a neutralized arm for clinician
annotation when rate summaries alone are ambiguous.

Usage:
  python -m management_reasoning.analysis.plot_flip_overlap
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from management_reasoning.analysis.compare import flip_sample_ids
from management_reasoning.analysis.load_responses import (
    load_independent_factor_grid,
    load_independent_new_neu_grid,
    load_independent_remove_all_grid,
)
from management_reasoning.analysis.plot_changes import configure_matplotlib, save_fig

ABLATIONS = (
    "format_tone",
    "content_format",
    "remove_all",
    "ct_old",
    "ct_new",
)

ABLATION_DISPLAY = {
    "format_tone": "Format+tone",
    "content_format": "Content+format",
    "remove_all": "Remove all",
    "ct_old": "CT old",
    "ct_new": "CT new",
}

FIELDS = (
    "diagnostic_consensus",
    "next_steps_consensus",
    "care_seeking",
    "structured_any",
)

FIELD_DISPLAY = {
    "diagnostic_consensus": "Diagnostic consensus",
    "next_steps_consensus": "Next-steps consensus",
    "care_seeking": "Care-seeking (|Δ|≥1)",
    "structured_any": "Any structured flip",
}

MODEL_DIRS = {
    "claude": "claude-opus-4-5_20251101",
    "gemini": "gemini-3.1-pro-preview",
}

RAW_REUSE_TAG = {
    "claude": "independent_batch",
    "gemini": "independent_remove_all_batch",
}


def load_full_independent_grid(model_key: str) -> Dict[str, Dict[int, dict]]:
    model_dir = MODEL_DIRS[model_key]
    raw_tag = RAW_REUSE_TAG[model_key]
    grid: Dict[str, Dict[int, dict]] = {}
    rem = load_independent_remove_all_grid(
        model_dir, reuse_raw_independent_tag=raw_tag
    )
    grid["raw"] = rem["raw"]
    grid["remove_all"] = rem["remove_all"]
    neu = load_independent_new_neu_grid(model_dir, reuse_raw_independent_tag=raw_tag)
    grid["ct_old"] = neu["ct_old"]
    grid["ct_new"] = neu["ct_new"]
    fac = load_independent_factor_grid(model_dir, reuse_raw_independent_tag=raw_tag)
    grid["format_tone"] = fac["format_tone"]
    grid["content_format"] = fac["content_format"]
    return grid


def build_flip_sets(
    grid: Dict[str, Dict[int, dict]],
) -> Dict[str, Dict[str, Set[int]]]:
    """ablation → field → sample_id set."""
    baseline = grid["raw"]
    out: Dict[str, Dict[str, Set[int]]] = {}
    for ab in ABLATIONS:
        sets: Dict[str, Set[int]] = {}
        for field in ("diagnostic_consensus", "next_steps_consensus", "care_seeking"):
            sets[field] = flip_sample_ids(baseline, grid[ab], field_name=field)
        sets["structured_any"] = (
            sets["diagnostic_consensus"]
            | sets["next_steps_consensus"]
            | sets["care_seeking"]
        )
        out[ab] = sets
    return out


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else float("nan")


def pairwise_rows(
    flip_sets: Dict[str, Dict[str, Set[int]]],
    *,
    model: str,
    field: str,
) -> List[dict]:
    rows: List[dict] = []
    for i, a in enumerate(ABLATIONS):
        for b in ABLATIONS[i:]:
            sa, sb = flip_sets[a][field], flip_sets[b][field]
            inter = len(sa & sb)
            only_a = len(sa - sb)
            only_b = len(sb - sa)
            rows.append(
                {
                    "model": model,
                    "field": field,
                    "arm_a": a,
                    "arm_b": b,
                    "n_a": len(sa),
                    "n_b": len(sb),
                    "n_intersection": inter,
                    "n_only_a": only_a,
                    "n_only_b": only_b,
                    "n_union": len(sa | sb),
                    "jaccard": round(jaccard(sa, sb), 4) if (sa or sb) else "",
                    "pct_a_in_b": round(100.0 * inter / len(sa), 2) if sa else "",
                    "pct_b_in_a": round(100.0 * inter / len(sb), 2) if sb else "",
                }
            )
    return rows


def coverage_vs_remove_all(
    flip_sets: Dict[str, Dict[str, Set[int]]],
    *,
    model: str,
    field: str,
) -> List[dict]:
    """How much of each arm's flips are shared with remove_all (and vice versa)."""
    ref = flip_sets["remove_all"][field]
    rows: List[dict] = []
    for ab in ABLATIONS:
        s = flip_sets[ab][field]
        inter = len(s & ref)
        rows.append(
            {
                "model": model,
                "field": field,
                "arm": ab,
                "n_arm": len(s),
                "n_remove_all": len(ref),
                "n_intersection": inter,
                "pct_arm_also_in_remove_all": round(100.0 * inter / len(s), 2)
                if s
                else "",
                "pct_remove_all_also_in_arm": round(100.0 * inter / len(ref), 2)
                if ref
                else "",
                "n_unique_to_arm": len(s - ref),
                "jaccard_vs_remove_all": round(jaccard(s, ref), 4) if (s or ref) else "",
            }
        )
    return rows


def multi_hit_counts(
    flip_sets: Dict[str, Dict[str, Set[int]]],
    field: str,
) -> Counter:
    """How many ablations flip each sample (among samples flipped by ≥1)."""
    hits: Counter = Counter()
    for ab in ABLATIONS:
        for sid in flip_sets[ab][field]:
            hits[sid] += 1
    return hits


def write_csv(path: str, rows: List[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote  {path}")


def plot_jaccard_heatmap(
    flip_sets: Dict[str, Dict[str, Set[int]]],
    *,
    model: str,
    field: str,
    stem: str,
) -> None:
    n = len(ABLATIONS)
    mat = np.zeros((n, n))
    for i, a in enumerate(ABLATIONS):
        for j, b in enumerate(ABLATIONS):
            mat[i, j] = jaccard(flip_sets[a][field], flip_sets[b][field])
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [ABLATION_DISPLAY[a] for a in ABLATIONS]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.55 else "black",
                fontsize=8.5,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    ax.set_title(
        f"{model.capitalize()} — flip-set overlap ({FIELD_DISPLAY[field]})"
    )
    fig.tight_layout()
    save_fig(fig, stem)


def plot_multi_hit_hist(
    flip_sets: Dict[str, Dict[str, Set[int]]],
    *,
    model: str,
    field: str,
    stem: str,
) -> None:
    hits = multi_hit_counts(flip_sets, field)
    counts = Counter(hits.values())
    xs = list(range(1, len(ABLATIONS) + 1))
    ys = [counts.get(k, 0) for k in xs]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(xs, ys, color="#1b9e77" if model == "claude" else "#d95f02", edgecolor="white")
    for bar, v in zip(bars, ys):
        if v:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ys) * 0.01,
                str(v),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(xs)
    ax.set_xlabel("# of ablations that flip the sample")
    ax.set_ylabel("# of samples")
    ax.set_title(
        f"{model.capitalize()} — shared vs unique flips ({FIELD_DISPLAY[field]})"
    )
    n_any = sum(ys)
    n_all = counts.get(len(ABLATIONS), 0)
    n_unique = counts.get(1, 0)
    ax.text(
        0.98,
        0.95,
        f"any={n_any}\nunique-to-1={n_unique}\nin-all-5={n_all}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_remove_all_coverage(
    cov_rows: List[dict],
    *,
    model: str,
    field: str,
    stem: str,
) -> None:
    arms = [r for r in cov_rows if r["arm"] != "remove_all"]
    labels = [ABLATION_DISPLAY[r["arm"]] for r in arms]
    pct_in_ra = [float(r["pct_arm_also_in_remove_all"]) for r in arms]
    pct_ra_in = [float(r["pct_remove_all_also_in_arm"]) for r in arms]
    x = np.arange(len(arms))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(
        x - width / 2,
        pct_in_ra,
        width,
        label="% of arm flips also in remove_all",
        color="#7570b3",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        pct_ra_in,
        width,
        label="% of remove_all flips also in arm",
        color="#e6ab02",
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"{model.capitalize()} — overlap with remove_all ({FIELD_DISPLAY[field]})"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    save_fig(fig, stem)


def run_model(
    model_key: str,
    *,
    fig_dir: str,
    csv_dir: str,
    fields: Sequence[str],
) -> Dict[str, Dict[str, Set[int]]]:
    print(f"Loading independent grid for {model_key} …")
    grid = load_full_independent_grid(model_key)
    for k, v in grid.items():
        print(f"  {k}: {len(v)} samples")
    flip_sets = build_flip_sets(grid)

    pair_all: List[dict] = []
    cov_all: List[dict] = []
    size_rows: List[dict] = []
    multi_rows: List[dict] = []

    for field in fields:
        sizes = {ab: len(flip_sets[ab][field]) for ab in ABLATIONS}
        print(f"  [{field}] sizes: {sizes}")
        pair_all.extend(pairwise_rows(flip_sets, model=model_key, field=field))
        cov = coverage_vs_remove_all(flip_sets, model=model_key, field=field)
        cov_all.extend(cov)

        hits = multi_hit_counts(flip_sets, field)
        hist = Counter(hits.values())
        for k in range(1, len(ABLATIONS) + 1):
            multi_rows.append(
                {
                    "model": model_key,
                    "field": field,
                    "n_ablations": k,
                    "n_samples": hist.get(k, 0),
                }
            )
        for ab in ABLATIONS:
            size_rows.append(
                {
                    "model": model_key,
                    "field": field,
                    "arm": ab,
                    "n_flipped": len(flip_sets[ab][field]),
                }
            )

        stem_base = os.path.join(fig_dir, f"flip_overlap_{model_key}_{field}")
        plot_jaccard_heatmap(
            flip_sets, model=model_key, field=field, stem=f"{stem_base}_jaccard"
        )
        plot_multi_hit_hist(
            flip_sets, model=model_key, field=field, stem=f"{stem_base}_multi_hit"
        )
        plot_remove_all_coverage(
            cov, model=model_key, field=field, stem=f"{stem_base}_vs_remove_all"
        )

    write_csv(
        os.path.join(csv_dir, f"flip_overlap_pairwise_{model_key}.csv"),
        pair_all,
        [
            "model",
            "field",
            "arm_a",
            "arm_b",
            "n_a",
            "n_b",
            "n_intersection",
            "n_only_a",
            "n_only_b",
            "n_union",
            "jaccard",
            "pct_a_in_b",
            "pct_b_in_a",
        ],
    )
    write_csv(
        os.path.join(csv_dir, f"flip_overlap_vs_remove_all_{model_key}.csv"),
        cov_all,
        [
            "model",
            "field",
            "arm",
            "n_arm",
            "n_remove_all",
            "n_intersection",
            "pct_arm_also_in_remove_all",
            "pct_remove_all_also_in_arm",
            "n_unique_to_arm",
            "jaccard_vs_remove_all",
        ],
    )
    write_csv(
        os.path.join(csv_dir, f"flip_overlap_sizes_{model_key}.csv"),
        size_rows,
        ["model", "field", "arm", "n_flipped"],
    )
    write_csv(
        os.path.join(csv_dir, f"flip_overlap_multi_hit_{model_key}.csv"),
        multi_rows,
        ["model", "field", "n_ablations", "n_samples"],
    )
    return flip_sets


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_matplotlib()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("claude", "gemini", "both"), default="both")
    p.add_argument("--fig-dir", default="./plotting/management_reasoning")
    p.add_argument("--csv-dir", default="./results/management_reasoning/analysis")
    p.add_argument(
        "--fields",
        nargs="+",
        default=["diagnostic_consensus", "structured_any"],
        choices=list(FIELDS),
        help="Fields to analyze (default: diag consensus + structured union)",
    )
    args = p.parse_args(argv)
    models = ("claude", "gemini") if args.model == "both" else (args.model,)
    for m in models:
        run_model(m, fig_dir=args.fig_dir, csv_dir=args.csv_dir, fields=args.fields)
    print("Done flip overlap analysis.")


if __name__ == "__main__":
    main()
