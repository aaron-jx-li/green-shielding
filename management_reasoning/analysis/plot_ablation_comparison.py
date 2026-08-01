#!/usr/bin/env python3
"""Compare all neutralization ablations: Flash-Lite diagnosis metrics + MR flips.

Legacy free-form diagnosis grid (Flash-Lite ``full_response``):
  raw, format_tone, content_format, remove_all, ct_old, ct_new × Claude/Gemini

Independent MR flips vs raw (structured consensus / acuity):
  remove_all, format_tone, content_format, ct_old, ct_new × Claude/Gemini

Usage:
  python -m management_reasoning.analysis.plot_ablation_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from management_reasoning.analysis.plot_changes import configure_matplotlib, save_fig

EVAL_ROOT = Path("./results/management_reasoning/eval/gemini-3.1-flash-lite")
ANALYSIS_ROOT = Path("./results/management_reasoning/analysis")

# Scientific arm order: baseline → 2-factor paper → 3-factor → CT (new neutralize model).
ARM_ORDER = (
    "raw",
    "format_tone",
    "content_format",
    "remove_all",
    "ct_old",
    "ct_new",
)

ARM_DISPLAY = {
    "raw": "Raw",
    "format_tone": "Format+tone",
    "content_format": "Content+format",
    "remove_all": "Remove all\n(C+F+T)",
    "ct_old": "CT old\n(C+T)",
    "ct_new": "CT new\n(C+T)",
}

ARM_DISPLAY_ONE = {
    "raw": "Raw",
    "format_tone": "Format+tone",
    "content_format": "Content+format",
    "remove_all": "Remove all (C+F+T)",
    "ct_old": "CT old (C+T)",
    "ct_new": "CT new (C+T)",
}

# Eval directory stems for legacy free-form diagnosis Flash-Lite.
LEGACY_DX_EVAL: Dict[Tuple[str, str], str] = {
    ("claude", "raw"): "claude_raw_legacy_diag",
    ("claude", "remove_all"): "claude_remove_all_legacy_diag",
    ("claude", "format_tone"): "claude_format_tone_legacy_dx_factor",
    ("claude", "content_format"): "claude_content_format_legacy_dx_factor",
    ("claude", "ct_old"): "claude_ct_old_legacy_dx",
    ("claude", "ct_new"): "claude_ct_new_legacy_dx",
    ("gemini", "raw"): "gemini_raw_legacy_dx",
    ("gemini", "remove_all"): "gemini_remove_all_legacy_dx",
    ("gemini", "format_tone"): "gemini_format_tone_legacy_dx_factor",
    ("gemini", "content_format"): "gemini_content_format_legacy_dx_factor",
    ("gemini", "ct_old"): "gemini_ct_old_legacy_dx",
    ("gemini", "ct_new"): "gemini_ct_new_legacy_dx",
}

INDEP_DX_EVAL: Dict[Tuple[str, str], str] = {
    ("claude", "raw"): "claude_raw_indep_dx",
    ("claude", "remove_all"): "claude_remove_all_indep_dx",
    ("claude", "ct_old"): "claude_ct_old_indep_dx",
    ("claude", "ct_new"): "claude_ct_new_indep_dx",
    ("gemini", "raw"): "gemini_raw_indep_dx",
}

# Flip CSVs: (model, ablation) → path relative to ANALYSIS_ROOT
FLIP_SOURCES: List[Tuple[str, str, str]] = [
    ("claude", "remove_all", "indep_remove_all_claude_flips.csv"),
    ("gemini", "remove_all", "indep_remove_all_gemini_flips.csv"),
    ("claude", "ct_old", "indep_new_neu_claude_flips.csv"),
    ("claude", "ct_new", "indep_new_neu_claude_flips.csv"),
    ("gemini", "ct_old", "indep_new_neu_gemini_flips.csv"),
    ("gemini", "ct_new", "indep_new_neu_gemini_flips.csv"),
    ("claude", "format_tone", "indep_factor_claude_flips.csv"),
    ("claude", "content_format", "indep_factor_claude_flips.csv"),
    ("gemini", "format_tone", "indep_factor_gemini_flips.csv"),
    ("gemini", "content_format", "indep_factor_gemini_flips.csv"),
]

FLIP_FIELDS = (
    "diagnostic_consensus",
    "next_steps_consensus",
    "care_seeking",
)

FLIP_FIELD_DISPLAY = {
    "diagnostic_consensus": "Diagnostic\nconsensus",
    "next_steps_consensus": "Next-steps\nconsensus",
    "care_seeking": "Care-seeking\n(|Δ|≥1)",
}

MODEL_COLORS = {
    "claude": "#1b9e77",
    "gemini": "#d95f02",
}

# Distinct colors per arm for flip grouped bars (shared across models via hatch).
ARM_COLORS = {
    "remove_all": "#7570b3",
    "format_tone": "#e7298a",
    "content_format": "#66a61e",
    "ct_old": "#e6ab02",
    "ct_new": "#a6761d",
}

METRIC_KEYS = (
    "mean_normalized_breadth",
    "mean_plausibility",
    "mean_h_coverage",
    "mean_c_coverage",
    "uncertainty_rate",
    "mean_support_rate",
)

METRIC_LABELS = {
    "mean_normalized_breadth": "Normalized breadth",
    "mean_plausibility": "Plausibility",
    "mean_h_coverage": "H coverage",
    "mean_c_coverage": "C coverage",
    "uncertainty_rate": "Uncertainty rate",
    "mean_support_rate": "Support rate",
}


def _load_summary(stem: str) -> Dict[str, Any]:
    path = EVAL_ROOT / stem / "eval.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary"]


def load_legacy_dx_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (model, arm), stem in LEGACY_DX_EVAL.items():
        s = _load_summary(stem)
        row = {
            "protocol": "legacy_freeform",
            "model": model,
            "arm": arm,
            "n": int(s.get("num_total_processed") or 0),
            "eval_stem": stem,
        }
        for k in METRIC_KEYS:
            row[k] = float(s[k])
        rows.append(row)
    # Δ vs raw within model
    raw = {
        r["model"]: r
        for r in rows
        if r["arm"] == "raw"
    }
    for r in rows:
        base = raw[r["model"]]
        r["delta_breadth"] = r["mean_normalized_breadth"] - base["mean_normalized_breadth"]
        r["delta_plausibility"] = r["mean_plausibility"] - base["mean_plausibility"]
        r["delta_h"] = r["mean_h_coverage"] - base["mean_h_coverage"]
        r["delta_c"] = r["mean_c_coverage"] - base["mean_c_coverage"]
    return rows


def load_indep_dx_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (model, arm), stem in INDEP_DX_EVAL.items():
        s = _load_summary(stem)
        row = {
            "protocol": "indep_dx",
            "model": model,
            "arm": arm,
            "n": int(s.get("num_total_processed") or 0),
            "eval_stem": stem,
        }
        for k in METRIC_KEYS:
            row[k] = float(s[k])
        rows.append(row)
    raw = {r["model"]: r for r in rows if r["arm"] == "raw"}
    for r in rows:
        if r["model"] not in raw:
            continue
        base = raw[r["model"]]
        r["delta_breadth"] = r["mean_normalized_breadth"] - base["mean_normalized_breadth"]
        r["delta_plausibility"] = r["mean_plausibility"] - base["mean_plausibility"]
    return rows


def load_flip_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for model, ablation, fname in FLIP_SOURCES:
        key = (model, ablation)
        if key in seen:
            continue
        seen.add(key)
        path = ANALYSIS_ROOT / fname
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["ablation"] != ablation:
                    continue
                field = row["field"]
                if field not in FLIP_FIELDS:
                    continue
                if field == "care_seeking":
                    pct = float(row["pct_changed"]) if row["pct_changed"] else float("nan")
                    n = int(float(row["n_ordinal"])) if row["n_ordinal"] else 0
                else:
                    pct = float(row["pct_flip"]) if row["pct_flip"] else float("nan")
                    n = int(float(row["n_paired"])) if row["n_paired"] else 0
                rows.append(
                    {
                        "model": model,
                        "arm": ablation,
                        "field": field,
                        "pct": pct,
                        "n": n,
                        "source": fname,
                    }
                )
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote  {path}")


def _metric_panel(
    rows: List[Dict[str, Any]],
    *,
    metrics: Sequence[str],
    arms: Sequence[str],
    title: str,
    stem: str,
    ylabel: str = "Score",
) -> None:
    models = ("claude", "gemini")
    by = {(r["model"], r["arm"]): r for r in rows}
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.4 * n_metrics, 4.6), sharey=False)
    if n_metrics == 1:
        axes = [axes]
    x = np.arange(len(arms))
    width = 0.36
    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            vals = []
            for arm in arms:
                r = by.get((model, arm))
                vals.append(r[metric] if r else np.nan)
            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=model.capitalize() if metric == metrics[0] else None,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_DISPLAY[a] for a in arms], fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_ylabel(ylabel if metric == metrics[0] else "")
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="#888", linewidth=0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08, fontsize=13)
    fig.tight_layout()
    save_fig(fig, stem)


def _delta_breadth_panel(
    rows: List[Dict[str, Any]],
    *,
    arms: Sequence[str],
    title: str,
    stem: str,
) -> None:
    """Δ normalized breadth vs raw (raw omitted)."""
    ablations = [a for a in arms if a != "raw"]
    models = ("claude", "gemini")
    by = {(r["model"], r["arm"]): r for r in rows}
    x = np.arange(len(ablations))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, model in enumerate(models):
        vals = [by[(model, a)]["delta_breadth"] for a in ablations]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model.capitalize(),
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (0.008 if v >= 0 else -0.018),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5,
            )
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_DISPLAY_ONE[a] for a in ablations], fontsize=9)
    ax.set_ylabel("Δ normalized breadth vs raw")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_legacy_dx(rows: List[Dict[str, Any]], fig_dir: str) -> None:
    arms = ARM_ORDER
    _metric_panel(
        rows,
        metrics=(
            "mean_normalized_breadth",
            "mean_plausibility",
            "mean_h_coverage",
            "mean_c_coverage",
        ),
        arms=arms,
        title="Legacy free-form diagnosis (Flash-Lite) — all ablations",
        stem=os.path.join(fig_dir, "ablation_legacy_dx_metrics"),
    )
    _delta_breadth_panel(
        rows,
        arms=arms,
        title="Legacy free-form diagnosis — Δ breadth vs raw",
        stem=os.path.join(fig_dir, "ablation_legacy_dx_delta_breadth"),
    )


def plot_indep_dx(rows: List[Dict[str, Any]], fig_dir: str) -> None:
    # Only arms that exist for at least one model; order preserved.
    present = {r["arm"] for r in rows}
    arms = tuple(a for a in ARM_ORDER if a in present)
    if not arms:
        return
    # Claude-focused grid (Gemini only has raw); still plot both where present.
    _metric_panel(
        rows,
        metrics=(
            "mean_normalized_breadth",
            "mean_plausibility",
            "mean_h_coverage",
            "mean_c_coverage",
        ),
        arms=arms,
        title="Independent MR diagnosis field (Flash-Lite) — available arms",
        stem=os.path.join(fig_dir, "ablation_indep_dx_metrics"),
    )


def plot_flips(flip_rows: List[Dict[str, Any]], fig_dir: str) -> None:
    ablations = [a for a in ARM_ORDER if a != "raw"]
    models = ("claude", "gemini")
    by = {(r["model"], r["arm"], r["field"]): r for r in flip_rows}

    # One figure per model: grouped bars by field × ablation
    for model in models:
        x = np.arange(len(FLIP_FIELDS))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        for i, arm in enumerate(ablations):
            vals = []
            for field in FLIP_FIELDS:
                r = by.get((model, arm, field))
                vals.append(r["pct"] if r else np.nan)
            offset = (i - (len(ablations) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=ARM_DISPLAY_ONE[arm],
                color=ARM_COLORS[arm],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([FLIP_FIELD_DISPLAY[f] for f in FLIP_FIELDS], fontsize=10)
        ax.set_ylabel("% changed vs raw independent")
        ax.set_title(f"Independent MR flips — {model.capitalize()}")
        ax.set_ylim(0, 40)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        save_fig(fig, os.path.join(fig_dir, f"ablation_indep_flips_{model}"))

    # Combined: diagnostic consensus only, Claude vs Gemini side-by-side
    field = "diagnostic_consensus"
    x = np.arange(len(ablations))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, model in enumerate(models):
        vals = [
            by[(model, arm, field)]["pct"] if (model, arm, field) in by else np.nan
            for arm in ablations
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model.capitalize(),
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_DISPLAY_ONE[a] for a in ablations], fontsize=9)
    ax.set_ylabel("% diagnostic-consensus flip vs raw")
    ax.set_title("Independent MR — diagnostic consensus flips (all ablations)")
    ax.set_ylim(0, 35)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, os.path.join(fig_dir, "ablation_indep_flips_diag_consensus"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_matplotlib()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fig-dir", default="./plotting/management_reasoning")
    p.add_argument("--csv-dir", default="./results/management_reasoning/analysis")
    args = p.parse_args(argv)

    legacy = load_legacy_dx_table()
    indep_dx = load_indep_dx_table()
    flips = load_flip_table()

    write_csv(
        os.path.join(args.csv_dir, "ablation_legacy_dx_metrics.csv"),
        legacy,
        [
            "protocol",
            "model",
            "arm",
            "n",
            *METRIC_KEYS,
            "delta_breadth",
            "delta_plausibility",
            "delta_h",
            "delta_c",
            "eval_stem",
        ],
    )
    write_csv(
        os.path.join(args.csv_dir, "ablation_indep_dx_metrics.csv"),
        indep_dx,
        [
            "protocol",
            "model",
            "arm",
            "n",
            *METRIC_KEYS,
            "delta_breadth",
            "delta_plausibility",
            "eval_stem",
        ],
    )
    write_csv(
        os.path.join(args.csv_dir, "ablation_indep_flips_summary.csv"),
        flips,
        ["model", "arm", "field", "pct", "n", "source"],
    )

    plot_legacy_dx(legacy, args.fig_dir)
    plot_indep_dx(indep_dx, args.fig_dir)
    plot_flips(flips, args.fig_dir)
    print("Done ablation comparison plots.")


if __name__ == "__main__":
    main()
