from __future__ import annotations

import argparse
from typing import List, Tuple

import pandas as pd

from static_eval.analysis.csv_metrics import collect_csv_paths, run_means


def _summary_se(values: List[float]) -> Tuple[float, float]:
    series = pd.Series(values, dtype="float64")
    mean = series.mean()
    if len(series) <= 1:
        return mean, 0.0
    sem = series.std(ddof=1) / (len(series) ** 0.5)
    return mean, sem


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="static-eval-analyze",
        description="Aggregate perturbation metrics across runs.",
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

    if args.per_run:
        for name, default_acc, perturb_rate, global_acc in per_run:
            print(
                f"{name}\tdefault_acc={default_acc:.4f}"
                f"\tperturb_success={perturb_rate:.4f}"
                f"\tperturbed_acc={global_acc:.4f}"
            )

    default_values = [d for _, d, _, _ in per_run]
    perturb_values = [p for _, _, p, _ in per_run]
    acc_values = [a for _, _, _, a in per_run]

    default_mean, default_se = _summary_se(default_values)
    perturb_mean, perturb_se = _summary_se(perturb_values)
    acc_mean, acc_se = _summary_se(acc_values)

    print(f"Runs: {len(per_run)}")
    print(f"Default accuracy: {default_mean:.4f} ± {default_se:.4f} (SE)")
    print(f"Perturbation success rate: {perturb_mean:.4f} ± {perturb_se:.4f} (SE)")
    print(f"Global accuracy after perturbation: {acc_mean:.4f} ± {acc_se:.4f} (SE)")


if __name__ == "__main__":
    main()
