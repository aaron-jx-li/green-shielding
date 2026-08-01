"""Radar charts for management-reasoning diagnosis eval (raw vs neutralized).

Mirrors the visual style of ``plotting/ablation.py`` (paper HCM ablation radars)
but reads primary-suite judge summaries and writes under
``plotting/management_reasoning/`` so paper figures are never overwritten.

Usage:
  python plotting/management_reasoning_radar.py
  python plotting/management_reasoning_radar.py --out_dir plotting/management_reasoning
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RAW_COLOR = "#1a1a1a"
FONT_FAMILY = "DejaVu Sans"
FIG_SIZE = (9.6, 7.2)
TITLE_FONT_SIZE = 17
LABEL_FONT_SIZE = 15
RADIAL_TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12.5
LEGEND_TITLE_FONT_SIZE = 12

COLORS = {
    "Raw (baseline)": RAW_COLOR,
    "Neutralized (content+tone)": "#7570b3",  # same as paper "Remove Content+Tone"
    "Remove All": "#d62728",  # same as paper ablation_raw_vs_all
}

METRICS = [
    ("mean_plausibility", "Plausibility"),
    ("mean_h_coverage", "H-coverage"),
    ("mean_c_coverage", "S-coverage"),
    ("mean_normalized_breadth", "Breadth"),
    ("mean_support_rate", "Evidence"),
    ("mean_indirect_inference_rate", "Inference"),
    ("uncertainty_rate", "Uncertainty"),
]

# (stem, title, raw eval path, other eval path, other_label) relative to --base_dir
MODELS = (
    (
        "ablation_raw_vs_neutralized_gemini",
        "Gemini 3.1 Pro: Raw vs. Neutralized",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/gemini_raw_primary/eval.json",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/gemini_neutralized_primary/eval.json",
        "Neutralized (content+tone)",
    ),
    (
        "ablation_raw_vs_neutralized_claude",
        "Claude Opus 4.5: Raw vs. Neutralized",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_raw_primary/eval.json",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_neutralized_primary/eval.json",
        "Neutralized (content+tone)",
    ),
)

LEGACY_MODELS = (
    (
        "ablation_raw_vs_remove_all_claude_legacy",
        "Claude Opus 4.5 (legacy diag): Raw vs. Remove All",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_raw_legacy_diag/eval.json",
        "results/management_reasoning/eval/gemini-3.1-flash-lite/claude_remove_all_legacy_diag/eval.json",
        "Remove All",
    ),
)



def load_summary(path: str) -> list[float]:
    with open(path) as f:
        s = json.load(f)["summary"]
    return [s[k] for k, _ in METRICS]


def save_fig(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to overwrite existing figure: {path}"
            )
        fig.savefig(path, format=ext, dpi=300)
        print(f"Saved  {path}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "axes.linewidth": 0.8,
        }
    )


def make_radar(
    data_subset,
    title,
    y_min=0.0,
    y_max=1.0,
    ring_step=0.2,
    dominant=None,
    legend_title="",
):
    """Return a (fig, ax) radar chart matching ``plotting/ablation.py`` style."""
    dominant = dominant or set(data_subset.keys())
    metric_labels = [m[1] for m in METRICS]
    n = len(METRICS)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang_c = angles + angles[:1]

    fig = plt.figure(figsize=FIG_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.72),
        left=0.06,
        right=0.96,
        bottom=0.08,
        top=0.84,
        wspace=0.22,
    )
    ax = fig.add_subplot(grid[0, 0], polar=True)
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.axis("off")
    ax.set_facecolor("#f9f9f9")

    bg_labels = [l for l in data_subset if l not in dominant]
    dom_labels = [l for l in data_subset if l in dominant]

    for label in bg_labels + dom_labels:
        vals = data_subset[label] + data_subset[label][:1]
        is_dom = label in dominant
        lw = 2.6 if is_dom else 1.2
        alpha_line = 1.0 if is_dom else 0.45
        alpha_fill = 0.13 if is_dom else 0.04
        zorder_l = 4 if is_dom else 3
        zorder_f = 3 if is_dom else 2
        ax.plot(
            ang_c,
            vals,
            color=COLORS[label],
            linewidth=lw,
            alpha=alpha_line,
            linestyle="-",
            solid_capstyle="round",
            label=label,
            zorder=zorder_l,
        )
        ax.fill(
            ang_c,
            vals,
            color=COLORS[label],
            alpha=alpha_fill,
            zorder=zorder_f,
        )

    rings = np.arange(
        ring_step * np.ceil(y_min / ring_step),
        y_max + 1e-9,
        ring_step,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(rings)
    ax.set_yticklabels(
        [f"{int(round(v * 100))}%" for v in rings],
        size=RADIAL_TICK_FONT_SIZE,
        color="#555555",
    )
    ax.yaxis.grid(True, color="#999999", linestyle="-", linewidth=0.9, alpha=0.9)
    ax.xaxis.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.7, alpha=0.8)

    theta_full = np.linspace(0, 2 * np.pi, 360)
    ax.plot(
        theta_full,
        np.full(360, y_max),
        color="#999999",
        linewidth=1.0,
        zorder=1,
    )
    ax.spines["polar"].set_visible(False)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        metric_labels,
        size=LABEL_FONT_SIZE,
        fontweight="bold",
        color="#222222",
    )
    ax.tick_params(axis="x", pad=14)
    ax.set_rlabel_position(20)

    fig.suptitle(
        title,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color="#222222",
        y=0.94,
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        title=legend_title,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.0,
        labelspacing=0.5,
        borderpad=0.8,
    )
    if legend_title:
        legend.get_title().set_fontweight("bold")

    return fig, ax


def main() -> None:
    configure_matplotlib()
    parser = argparse.ArgumentParser(
        description="Management-reasoning raw vs neutralized radar charts"
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plotting/management_reasoning",
        help="Output directory (defaults away from paper ablation_*.pdf).",
    )
    parser.add_argument(
        "--panel",
        choices=("primary", "legacy", "all"),
        default="all",
        help="Which figure set to render (default: all available).",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Overwrite existing png/pdf stems if present.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    panels = []
    if args.panel in ("primary", "all"):
        panels.extend(MODELS)
    if args.panel in ("legacy", "all"):
        panels.extend(LEGACY_MODELS)

    for stem, title, raw_rel, other_rel, other_label in panels:
        raw_path = os.path.join(args.base_dir, raw_rel)
        other_path = os.path.join(args.base_dir, other_rel)
        if not os.path.isfile(raw_path) or not os.path.isfile(other_path):
            print(f"Skip {stem}: missing eval JSON")
            continue
        subset = {
            "Raw (baseline)": load_summary(raw_path),
            other_label: load_summary(other_path),
        }
        fig, _ = make_radar(
            subset,
            title=title,
            y_min=0.0,
            y_max=1.0,
            ring_step=0.2,
            dominant=set(subset),
        )
        out_stem = os.path.join(args.out_dir, stem)
        if args.allow_overwrite:
            for ext in ("png", "pdf"):
                path = f"{out_stem}.{ext}"
                fig.savefig(path, format=ext, dpi=300)
                print(f"Saved  {path}")
        else:
            save_fig(fig, out_stem)
        plt.close(fig)


if __name__ == "__main__":
    main()
