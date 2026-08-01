"""Plot answer-change rates across management-reasoning ablations.

Usage:
  python -m management_reasoning.analysis.plot_changes
  python -m management_reasoning.analysis.plot_changes --model-dir claude-opus-4-5_20251101
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from management_reasoning.analysis.compare import (
    ABLATION_LABELS,
    ACUITY_SHORT,
    FIELD_DISPLAY,
    PairStats,
    compare_all,
    transition_matrix,
    write_summary_csv,
)
from management_reasoning.analysis.load_responses import load_claude_ablation_grid
FONT_FAMILY = "DejaVu Sans"
ABLATION_DISPLAY = {
    "neut": "Neutralized",
    "indep": "Independent",
    "ord1": "Order ord1",
    "ord2": "Order ord2",
    "ord3": "Order ord3",
}
# Structured flip-rate questions (free-text dx/c omitted — near-ceiling exact-match noise)
BAR_FIELDS = (
    "diagnostic_consensus",
    "next_steps_consensus",
)
COLORS = {
    "neut": "#0571b0",
    "indep": "#d95f02",
    "ord1": "#1b9e77",
    "ord2": "#7570b3",
    "ord3": "#e7298a",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=300, bbox_inches="tight")
        print(f"Saved  {path}")
    plt.close(fig)


def plot_flip_bars(
    all_stats: Dict[str, Dict[str, PairStats]],
    ablations: Sequence[str],
    stem: str,
) -> None:
    fields = list(BAR_FIELDS) + ["care_seeking"]
    x = np.arange(len(fields))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, ab in enumerate(ablations):
        vals = []
        for fname in fields:
            st = all_stats[ab][fname]
            if fname == "care_seeking":
                vals.append(st.pct_changed if st.n_ordinal else 0.0)
            else:
                vals.append(st.pct_flip if st.n_paired else 0.0)
        offset = (i - (len(ablations) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=ABLATION_DISPLAY[ab],
            color=COLORS[ab],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, v in zip(bars, vals):
            if v >= 1.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    labels = [
        FIELD_DISPLAY[f] if f != "care_seeking" else "b: acuity |Δ|≥1"
        for f in fields
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("% samples changed vs raw primary")
    ax.set_xlabel("")
    ax.set_title("Answer change rate by question and ablation (Claude)")
    ax.set_ylim(0, max(5, ax.get_ylim()[1] * 1.08))
    ax.legend(frameon=False, ncol=min(5, len(ablations)), loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_directional_consensus(
    all_stats: Dict[str, Dict[str, PairStats]],
    ablations: Sequence[str],
    stem: str,
) -> None:
    fields = ("diagnostic_consensus", "next_steps_consensus")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    x = np.arange(len(ablations))
    width = 0.35
    for ax, fname in zip(axes, fields):
        h2l = [all_stats[ab][fname].pct_high_to_low for ab in ablations]
        l2h = [all_stats[ab][fname].pct_low_to_high for ab in ablations]
        ax.bar(x - width / 2, h2l, width, label="high → low", color="#b2182b")
        ax.bar(x + width / 2, l2h, width, label="low → high", color="#2166ac")
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_DISPLAY[a] for a in ablations], rotation=20, ha="right")
        ax.set_title(FIELD_DISPLAY[fname])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("% of paired samples")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Directional consensus flips vs raw primary", y=1.02)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_acuity_histograms(
    all_stats: Dict[str, Dict[str, PairStats]],
    ablations: Sequence[str],
    stem: str,
) -> None:
    n = len(ablations)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    bins = np.arange(-5.5, 6.5, 1.0)
    for ax, ab in zip(axes_flat, ablations):
        st = all_stats[ab]["care_seeking"]
        nonzero = [d for d in st.deltas if d != 0]
        ax.hist(
            nonzero,
            bins=bins,
            color=COLORS[ab],
            edgecolor="white",
            linewidth=0.5,
        )
        pct = st.pct_changed if st.n_ordinal else 0.0
        ax.set_title(
            f"{ABLATION_DISPLAY[ab]}\n"
            f"changed {pct:.1f}% (n_ord={st.n_ordinal}; Δ≠0 shown)"
        )
        ax.set_xlim(-5.5, 5.5)
        ax.set_xticks(range(-5, 6))
        ax.set_xlabel("Δ acuity (ablation − raw)")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_flat[0].set_ylabel("Count")
    for ax in axes_flat[len(ablations) :]:
        ax.set_visible(False)
    fig.suptitle("Acuity shift distribution (Δ=0 omitted)", y=1.01)
    fig.tight_layout()
    save_fig(fig, stem)


def _short_labels(labels: List[str]) -> List[str]:
    return [ACUITY_SHORT.get(x, x[:8]) for x in labels]


def plot_acuity_heatmaps(
    all_stats: Dict[str, Dict[str, PairStats]],
    ablations: Sequence[str],
    stem: str,
) -> None:
    n = len(ablations)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, ab in zip(axes_flat, ablations):
        labels, mat = transition_matrix(all_stats[ab]["care_seeking"])
        arr = np.array(mat)
        im = ax.imshow(arr, cmap="YlOrRd", vmin=0, vmax=max(40, float(arr.max()) if arr.size else 40))
        shorts = _short_labels(labels)
        ax.set_xticks(range(len(shorts)))
        ax.set_yticks(range(len(shorts)))
        ax.set_xticklabels(shorts, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(shorts, fontsize=8)
        ax.set_xlabel("Ablation")
        ax.set_ylabel("Raw")
        ax.set_title(ABLATION_DISPLAY[ab])
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if v >= 8:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of raw row")
    for ax in axes_flat[len(ablations) :]:
        ax.set_visible(False)
    fig.suptitle("Care-seeking transitions (row-normalized %; NMI excluded)", y=1.01)
    fig.tight_layout()
    save_fig(fig, stem)


def print_summary(all_stats: Dict[str, Dict[str, PairStats]], ablations: Sequence[str]) -> None:
    print("\n=== Flip / change summary (% vs raw primary) ===")
    header = f"{'ablation':<8} {'field':<32} {'n':>6} {'%flip':>7} {'%|Δ|≥1':>8} {'meanΔ':>7}"
    print(header)
    for ab in ablations:
        for fname in BAR_FIELDS + ("care_seeking",):
            st = all_stats[ab][fname]
            if fname == "care_seeking":
                print(
                    f"{ab:<8} {fname:<32} {st.n_paired:6d} {st.pct_flip:7.2f} "
                    f"{st.pct_changed:8.2f} {st.mean_delta:7.3f}"
                )
            else:
                print(
                    f"{ab:<8} {fname:<32} {st.n_paired:6d} {st.pct_flip:7.2f} "
                    f"{'':>8} {'':>7}"
                )


def run(
    *,
    model_dir: str,
    out_fig_dir: str,
    out_csv: str,
) -> None:
    configure_matplotlib()
    print(f"Loading responses for {model_dir} …")
    grid = load_claude_ablation_grid(model_dir)
    for k, v in grid.items():
        print(f"  {k}: {len(v)} samples")
    ablations = [a for a in ABLATION_LABELS if a in grid]
    all_stats = compare_all(grid, ablation_keys=ablations)
    write_summary_csv(out_csv, all_stats)
    print(f"Wrote  {out_csv}")
    print_summary(all_stats, ablations)

    os.makedirs(out_fig_dir, exist_ok=True)
    plot_flip_bars(all_stats, ablations, os.path.join(out_fig_dir, "flip_rates"))
    plot_directional_consensus(
        all_stats, ablations, os.path.join(out_fig_dir, "consensus_directional")
    )
    plot_acuity_histograms(all_stats, ablations, os.path.join(out_fig_dir, "acuity_delta_hist"))
    plot_acuity_heatmaps(all_stats, ablations, os.path.join(out_fig_dir, "acuity_transitions"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Plot management-reasoning ablation answer changes")
    p.add_argument(
        "--model-dir",
        default="claude-opus-4-5_20251101",
        help="Directory under results/.../responses/vertex/",
    )
    p.add_argument(
        "--fig-dir",
        default="./plotting/management_reasoning",
        help="Output directory for png/pdf figures",
    )
    p.add_argument(
        "--csv",
        default="./results/management_reasoning/analysis/claude_ablation_flips.csv",
        help="Summary CSV path",
    )
    args = p.parse_args(argv)
    run(model_dir=args.model_dir, out_fig_dir=args.fig_dir, out_csv=args.csv)


if __name__ == "__main__":
    main()
