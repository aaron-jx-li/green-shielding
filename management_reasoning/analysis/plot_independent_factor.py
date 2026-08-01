#!/usr/bin/env python3
"""Flip/acuity plots: independent raw vs factor-pair arms (format_tone, content_format).

Claude reuses prior ``raw_independent_batch``; Gemini reuses
``raw_independent_remove_all_batch``.

Usage:
  python -m management_reasoning.analysis.plot_independent_factor
  python -m management_reasoning.analysis.plot_independent_factor --model gemini
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Sequence

from management_reasoning.analysis.compare import (
    PairStats,
    compare_all,
    write_summary_csv,
)
from management_reasoning.analysis.load_responses import load_independent_factor_grid
from management_reasoning.analysis.plot_changes import (
    configure_matplotlib,
    plot_acuity_heatmaps,
    plot_acuity_histograms,
    plot_directional_consensus,
    plot_flip_bars,
    print_summary,
)

ABLATIONS = ("format_tone", "content_format")
ABLATION_DISPLAY = {
    "format_tone": "Format+tone (indep)",
    "content_format": "Content+format (indep)",
}
ABLATION_COLORS = {
    "format_tone": "#d95f02",
    "content_format": "#7570b3",
}

MODEL_DIRS = {
    "claude": "claude-opus-4-5_20251101",
    "gemini": "gemini-3.1-pro-preview",
}

RAW_REUSE_TAG = {
    "claude": "independent_batch",
    "gemini": "independent_remove_all_batch",
}


def run_model(
    *,
    model_key: str,
    out_fig_dir: str,
    out_csv: str,
    factor_tag: str,
) -> None:
    from management_reasoning.analysis import plot_changes as pc

    pc.ABLATION_DISPLAY.update(ABLATION_DISPLAY)
    pc.COLORS.update(ABLATION_COLORS)

    model_dir = MODEL_DIRS[model_key]
    reuse = RAW_REUSE_TAG[model_key]
    print(
        f"Loading independent factor grid for {model_dir} "
        f"(raw tag={reuse}, factor tag={factor_tag}) …"
    )
    grid = load_independent_factor_grid(
        model_dir,
        reuse_raw_independent_tag=reuse,
        factor_tag=factor_tag,
    )
    for k, v in grid.items():
        print(f"  {k}: {len(v)} samples")
    if "raw" not in grid or not grid["raw"]:
        raise SystemExit(f"Missing raw independent for {model_dir}")
    for ab in ABLATIONS:
        if ab not in grid or not grid[ab]:
            raise SystemExit(f"Missing {ab} independent for {model_dir}")

    ablations: Sequence[str] = ABLATIONS
    all_stats: Dict[str, Dict[str, PairStats]] = compare_all(
        grid, ablation_keys=ablations
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    write_summary_csv(out_csv, all_stats)
    print(f"Wrote  {out_csv}")
    print_summary(all_stats, ablations)

    os.makedirs(out_fig_dir, exist_ok=True)
    stem = f"indep_factor_{model_key}"
    plot_flip_bars(all_stats, ablations, os.path.join(out_fig_dir, f"{stem}_flip_rates"))
    plot_directional_consensus(
        all_stats, ablations, os.path.join(out_fig_dir, f"{stem}_consensus_directional")
    )
    plot_acuity_histograms(
        all_stats, ablations, os.path.join(out_fig_dir, f"{stem}_acuity_delta_hist")
    )
    plot_acuity_heatmaps(
        all_stats, ablations, os.path.join(out_fig_dir, f"{stem}_acuity_transitions")
    )


def main(argv=None) -> None:
    configure_matplotlib()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=("claude", "gemini", "both"),
        default="both",
    )
    p.add_argument("--fig-dir", default="./plotting/management_reasoning")
    p.add_argument(
        "--csv-dir",
        default="./results/management_reasoning/analysis",
    )
    p.add_argument(
        "--factor-tag",
        default="independent_factor_batch",
        help="Collect tag for factor-pair independent arms",
    )
    args = p.parse_args(argv)
    models = ("claude", "gemini") if args.model == "both" else (args.model,)
    for m in models:
        run_model(
            model_key=m,
            out_fig_dir=args.fig_dir,
            out_csv=os.path.join(args.csv_dir, f"indep_factor_{m}_flips.csv"),
            factor_tag=args.factor_tag,
        )


if __name__ == "__main__":
    main()
