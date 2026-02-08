"""Bootstrap confidence intervals for static-eval metrics.

Assumes (question, run) pairs are exchangeable: each observation is treated as
drawn from the same distribution, and we resample (question, run) pairs with
replacement to build the bootstrap distribution of the mean for each metric.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def _collect_csv_paths(paths: Iterable[str]) -> List[Path]:
    csvs: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            csvs.extend(sorted(p.glob("*.csv")))
        elif p.exists():
            csvs.append(p)
        else:
            csvs.extend(sorted(Path().glob(raw)))
    return [p for p in csvs if p.suffix.lower() == ".csv"]


def _metric_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    numbered = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if numbered:
        return numbered
    return [prefix] if prefix in df.columns else []


def _coerce_boolish(series: pd.Series) -> pd.Series:
    mapped = series.map({"True": 1, "False": 0, True: 1, False: 0})
    return series.where(mapped.isna(), mapped)


def _run_means(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    cols = _metric_columns(df, prefix)
    if not cols:
        raise ValueError(f"Missing columns for '{prefix}'")
    values = df[cols].copy()
    values = values.apply(_coerce_boolish)
    means: dict[str, float] = {}
    for col in cols:
        series = pd.to_numeric(values[col], errors="coerce").dropna()
        if len(series):
            if col == prefix:
                run_id = "0"
            else:
                run_id = col[len(prefix) + 1 :]
            means[run_id] = series.mean()
    return means


def _run_id_from_col(prefix: str, col: str) -> str:
    if col == prefix:
        return "0"
    return col[len(prefix) + 1 :]


def _question_run_triples(df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """Extract (default_correct, perturbation_success, perturbed_correct) for each (question, run).
    Only includes (question, run) pairs where all three metrics are valid.
    Assumes questions and runs are exchangeable (iid from the same distribution).
    """
    default_cols = _metric_columns(df, "default_correct")
    perturb_cols = _metric_columns(df, "perturbation_success")
    perturbed_cols = _metric_columns(df, "perturbed_correct")
    if not default_cols or not perturb_cols or not perturbed_cols:
        raise ValueError("Missing metric columns for default_correct, perturbation_success, or perturbed_correct")
    # Build run_id -> (dc_col, ps_col, pc_col)
    dc_by_run = {_run_id_from_col("default_correct", c): c for c in default_cols}
    ps_by_run = {_run_id_from_col("perturbation_success", c): c for c in perturb_cols}
    pc_by_run = {_run_id_from_col("perturbed_correct", c): c for c in perturbed_cols}
    run_ids = sorted(set(dc_by_run) & set(ps_by_run) & set(pc_by_run))
    if not run_ids:
        raise ValueError("No run IDs common to all three metrics.")
    triples: List[Tuple[float, float, float]] = []
    for i in range(len(df)):
        for run_id in run_ids:
            dc = df[dc_by_run[run_id]].iloc[i]
            ps = df[ps_by_run[run_id]].iloc[i]
            pc = df[pc_by_run[run_id]].iloc[i]
            dc = _coerce_boolish(pd.Series([dc])).iloc[0]
            ps = _coerce_boolish(pd.Series([ps])).iloc[0]
            pc = _coerce_boolish(pd.Series([pc])).iloc[0]
            dc_f = pd.to_numeric(dc, errors="coerce")
            ps_f = pd.to_numeric(ps, errors="coerce")
            pc_f = pd.to_numeric(pc, errors="coerce")
            if pd.notna(dc_f) and pd.notna(ps_f) and pd.notna(pc_f):
                triples.append((float(dc_f), float(ps_f), float(pc_f)))
    return triples


def _summary_se(values: List[float]) -> Tuple[float, float]:
    series = pd.Series(values, dtype="float64")
    mean = series.mean()
    if len(series) <= 1:
        return mean, 0.0
    sem = series.std(ddof=1) / (len(series) ** 0.5)
    return mean, sem


def _bootstrap_ci(
    triples: List[Tuple[float, float, float]],
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
]:
    """Bootstrap CI assuming (question, run) pairs are exchangeable.
    Resamples pairs with replacement, then computes mean of each metric per bootstrap sample.
    Returns (point_estimates, ((lo_d, hi_d), (lo_p, hi_p), (lo_a, hi_a))).
    """
    arr = np.array(triples, dtype=np.float64)  # (N, 3)
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
        prog="static-eval-analyze-bootstrap",
        description="Bootstrap confidence intervals for perturbation metrics (questions and runs exchangeable).",
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
    ap.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples (default: 2000).",
    )
    ap.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = _collect_csv_paths(args.paths)
    if not csv_paths:
        raise SystemExit("No CSV files found.")

    all_triples: List[Tuple[float, float, float]] = []
    per_run: List[Tuple[str, float, float, float]] = []

    for path in csv_paths:
        df = pd.read_csv(path)
        default_runs = _run_means(df, "default_correct")
        perturb_runs = _run_means(df, "perturbation_success")
        perturbed_runs = _run_means(df, "perturbed_correct")
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
        triples = _question_run_triples(df)
        all_triples.extend(triples)

    if args.per_run:
        for name, default_acc, perturb_rate, global_acc in per_run:
            print(
                f"{name}\tdefault_acc={default_acc:.4f}"
                f"\tperturb_success={perturb_rate:.4f}"
                f"\tperturbed_acc={global_acc:.4f}"
            )

    rng = np.random.default_rng(args.seed)
    point, (ci_default, ci_perturb, ci_acc) = _bootstrap_ci(
        all_triples, args.n_bootstrap, args.confidence, rng
    )
    level_pct = args.confidence * 100

    n_obs = len(all_triples)
    n_runs = len(per_run)
    print(f"(Question, run) pairs (exchangeable units): {n_obs}")
    print(f"Runs: {n_runs}")
    print(
        f"Default accuracy: {point[0]:.4f}  [{level_pct:.0f}% CI: {ci_default[0]:.4f} -- {ci_default[1]:.4f}]"
    )
    print(
        f"Perturbation success rate: {point[1]:.4f}  [{level_pct:.0f}% CI: {ci_perturb[0]:.4f} -- {ci_perturb[1]:.4f}]"
    )
    print(
        f"Global accuracy after perturbation: {point[2]:.4f}  [{level_pct:.0f}% CI: {ci_acc[0]:.4f} -- {ci_acc[1]:.4f}]"
    )


if __name__ == "__main__":
    main()
