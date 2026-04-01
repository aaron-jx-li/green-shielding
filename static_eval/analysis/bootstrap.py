"""Bootstrap confidence intervals for static-eval metrics."""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

from static_eval.analysis.csv_metrics import (
    collect_csv_paths,
    coerce_boolish,
    metric_columns,
    run_id_from_col,
    run_means,
)


def question_run_triples(df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    default_cols = metric_columns(df, "default_correct")
    perturb_cols = metric_columns(df, "perturbation_success")
    perturbed_cols = metric_columns(df, "perturbed_correct")
    if not default_cols or not perturb_cols or not perturbed_cols:
        raise ValueError("Missing metric columns for default_correct, perturbation_success, or perturbed_correct")
    dc_by_run = {run_id_from_col("default_correct", c): c for c in default_cols}
    ps_by_run = {run_id_from_col("perturbation_success", c): c for c in perturb_cols}
    pc_by_run = {run_id_from_col("perturbed_correct", c): c for c in perturbed_cols}
    run_ids = sorted(set(dc_by_run) & set(ps_by_run) & set(pc_by_run))
    if not run_ids:
        raise ValueError("No run IDs common to all three metrics.")
    triples: List[Tuple[float, float, float]] = []
    for i in range(len(df)):
        for run_id in run_ids:
            dc = df[dc_by_run[run_id]].iloc[i]
            ps = df[ps_by_run[run_id]].iloc[i]
            pc = df[pc_by_run[run_id]].iloc[i]
            dc = coerce_boolish(pd.Series([dc])).iloc[0]
            ps = coerce_boolish(pd.Series([ps])).iloc[0]
            pc = coerce_boolish(pd.Series([pc])).iloc[0]
            dc_f = pd.to_numeric(dc, errors="coerce")
            ps_f = pd.to_numeric(ps, errors="coerce")
            pc_f = pd.to_numeric(pc, errors="coerce")
            if pd.notna(dc_f) and pd.notna(ps_f) and pd.notna(pc_f):
                triples.append((float(dc_f), float(ps_f), float(pc_f)))
    return triples


def bootstrap_ci(
    triples: List[Tuple[float, float, float]],
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
]:
    arr = np.array(triples, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("No (question, run) observations.")
    point = (float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
    low = (1 - confidence) / 2
    high = 1 - low
    boot_means = np.zeros((n_bootstrap, 3), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b, 0] = arr[idx, 0].mean()
        boot_means[b, 1] = arr[idx, 1].mean()
        boot_means[b, 2] = arr[idx, 2].mean()
    ci_default = (float(np.percentile(boot_means[:, 0], low * 100)), float(np.percentile(boot_means[:, 0], high * 100)))
    ci_perturb = (float(np.percentile(boot_means[:, 1], low * 100)), float(np.percentile(boot_means[:, 1], high * 100)))
    ci_acc = (float(np.percentile(boot_means[:, 2], low * 100)), float(np.percentile(boot_means[:, 2], high * 100)))
    return point, (ci_default, ci_perturb, ci_acc)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="static-eval-analyze",
        description="Aggregate perturbation metrics across runs (bootstrap CIs).",
    )
    ap.add_argument(
        "--paths",
        nargs="+",
        required=True,
        type=str,
        help="CSV files, directories of CSVs, or glob patterns",
    )
    ap.add_argument(
        "--per_run",
        action="store_true",
        help="Print per-run metrics before the aggregate.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = collect_csv_paths(args.paths)
    if not csv_paths:
        raise SystemExit("No CSV files found.")

    all_triples: List[Tuple[float, float, float]] = []
    per_run: List[Tuple[str, float, float, float]] = []

    for path in csv_paths:
        df = pd.read_csv(path)
        default_runs = run_means(df, "default_correct")
        perturb_runs = run_means(df, "perturbation_success")
        perturbed_runs = run_means(df, "perturbed_correct")
        default_ids = set(default_runs)
        perturb_ids = set(perturb_runs)
        perturbed_ids = set(perturbed_runs)
        if default_ids != perturb_ids or default_ids != perturbed_ids:
            raise ValueError(
                "Run columns do not align across metrics for "
                f"{path}. default={sorted(default_ids)} "
                f"perturb={sorted(perturb_ids)} perturbed={sorted(perturbed_ids)}"
            )
        for run_id in sorted(default_ids):
            per_run.append(
                (
                    f"{path}:{run_id}",
                    default_runs[run_id],
                    perturb_runs[run_id],
                    perturbed_runs[run_id],
                )
            )
        triples = question_run_triples(df)
        all_triples.extend(triples)

    if args.per_run:
        for name, default_acc, perturb_rate, global_acc in per_run:
            print(
                f"{name}\tdefault_acc={default_acc:.4f}"
                f"\tperturb_success={perturb_rate:.4f}"
                f"\tperturbed_acc={global_acc:.4f}"
            )

    n_bootstrap = 2000
    confidence = 0.95
    rng = np.random.default_rng()
    point, (ci_default, ci_perturb, ci_acc) = bootstrap_ci(all_triples, n_bootstrap, confidence, rng)
    level_pct = confidence * 100

    default_err = (ci_default[1] - ci_default[0]) / 2
    perturb_err = (ci_perturb[1] - ci_perturb[0]) / 2
    acc_err = (ci_acc[1] - ci_acc[0]) / 2

    print(f"Runs: {len(per_run)}")
    print(f"Default accuracy: {point[0]:.4f} ± {default_err:.4f} (bootstrap {level_pct:.0f}% CI/2)")
    print(f"Perturbation success rate: {point[1]:.4f} ± {perturb_err:.4f} (bootstrap {level_pct:.0f}% CI/2)")
    print(f"Global accuracy after perturbation: {point[2]:.4f} ± {acc_err:.4f} (bootstrap {level_pct:.0f}% CI/2)")


if __name__ == "__main__":
    main()
