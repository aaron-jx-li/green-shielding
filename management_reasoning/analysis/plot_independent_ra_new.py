#!/usr/bin/env python3
"""Flip plots: independent raw vs ra_new (gpt-5.2 content+format+tone).

Usage:
  python -m management_reasoning.analysis.plot_independent_ra_new
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
from management_reasoning.analysis.load_responses import load_independent_ra_new_grid
from management_reasoning.analysis.plot_changes import (
    configure_matplotlib,
    plot_acuity_heatmaps,
    plot_acuity_histograms,
    plot_directional_consensus,
    plot_flip_bars,
    print_summary,
)

ABLATIONS = ("ra_new",)
ABLATION_DISPLAY = {"ra_new": "RA new (gpt-5.2 C+F+T)"}
ABLATION_COLORS = {"ra_new": "#e7298a"}

MODEL_DIRS = {
    "claude": "claude-opus-4-5_20251101",
    "gemini": "gemini-3.1-pro-preview",
}
RAW_REUSE_TAG = {
    "claude": "independent_batch",
    "gemini": "independent_remove_all_batch",
}


def run_model(*, model_key: str, out_fig_dir: str, out_csv: str, ra_tag: str) -> None:
    from management_reasoning.analysis import plot_changes as pc

    pc.ABLATION_DISPLAY.update(ABLATION_DISPLAY)
    pc.COLORS.update(ABLATION_COLORS)

    model_dir = MODEL_DIRS[model_key]
    reuse = RAW_REUSE_TAG[model_key]
    print(f"Loading ra_new grid for {model_dir} (raw={reuse}, ra={ra_tag}) …")
    grid = load_independent_ra_new_grid(
        model_dir, reuse_raw_independent_tag=reuse, ra_tag=ra_tag
    )
    for k, v in grid.items():
        print(f"  {k}: {len(v)} samples")
    ablations: Sequence[str] = ABLATIONS
    all_stats: Dict[str, Dict[str, PairStats]] = compare_all(
        grid, ablation_keys=ablations
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    write_summary_csv(out_csv, all_stats)
    print(f"Wrote  {out_csv}")
    print_summary(all_stats, ablations)
    os.makedirs(out_fig_dir, exist_ok=True)
    stem = f"indep_ra_new_{model_key}"
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
    p.add_argument("--model", choices=("claude", "gemini", "both"), default="both")
    p.add_argument("--fig-dir", default="./plotting/management_reasoning")
    p.add_argument("--csv-dir", default="./results/management_reasoning/analysis")
    p.add_argument("--ra-tag", default="independent_ra_new_batch")
    args = p.parse_args(argv)
    models = ("claude", "gemini") if args.model == "both" else (args.model,)
    for m in models:
        run_model(
            model_key=m,
            out_fig_dir=args.fig_dir,
            out_csv=os.path.join(args.csv_dir, f"indep_ra_new_{m}_flips.csv"),
            ra_tag=args.ra_tag,
        )


if __name__ == "__main__":
    main()
