from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple

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


def _flatten_metric_values(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = _metric_columns(df, prefix)
    if not cols:
        raise ValueError(f"Missing columns for '{prefix}'")
    values = df[cols].copy()

    def _coerce_boolish(series: pd.Series) -> pd.Series:
        mapped = series.map({"True": 1, "False": 0, True: 1, False: 0})
        return series.where(mapped.isna(), mapped)

    values = values.apply(_coerce_boolish)
    flat = values.stack(future_stack=True)
    return pd.to_numeric(flat, errors="coerce").dropna()


def _compute_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    default_correct = _flatten_metric_values(df, "default_correct")
    perturb_success = _flatten_metric_values(df, "perturbation_success")
    perturbed_correct = _flatten_metric_values(df, "perturbed_correct")
    return default_correct.mean(), perturb_success.mean(), perturbed_correct.mean()


def _generation_means(df: pd.DataFrame, prefix: str) -> List[float]:
    cols = _metric_columns(df, prefix)
    if not cols:
        raise ValueError(f"Missing columns for '{prefix}'")
    values = df[cols].copy()

    def _coerce_boolish(series: pd.Series) -> pd.Series:
        mapped = series.map({"True": 1, "False": 0, True: 1, False: 0})
        return series.where(mapped.isna(), mapped)

    values = values.apply(_coerce_boolish)
    means: List[float] = []
    for col in cols:
        series = pd.to_numeric(values[col], errors="coerce").dropna()
        if len(series):
            means.append(series.mean())
    return means


def _item_means(df: pd.DataFrame, prefix: str) -> List[float]:
    cols = _metric_columns(df, prefix)
    if not cols:
        raise ValueError(f"Missing columns for '{prefix}'")
    values = df[cols].copy()

    def _coerce_boolish(series: pd.Series) -> pd.Series:
        mapped = series.map({"True": 1, "False": 0, True: 1, False: 0})
        return series.where(mapped.isna(), mapped)

    values = values.apply(_coerce_boolish)
    numeric = values.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=1, skipna=True)
    return means.dropna().tolist()


def _summary(values: List[float], error: str) -> Tuple[float, float]:
    series = pd.Series(values, dtype="float64")
    mean = series.mean()
    if len(series) <= 1:
        return mean, 0.0
    if error == "std":
        return mean, series.std(ddof=1)
    sem = series.std(ddof=1) / (len(series) ** 0.5) if len(series) > 1 else 0.0
    return mean, sem


def _bootstrap_ci(
    values: List[float],
    *,
    iters: int,
    seed: int | None,
    alpha: float,
) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(values) / n
    low_idx = int((alpha / 2) * iters)
    high_idx = int((1 - alpha / 2) * iters) - 1
    low = means[max(0, min(low_idx, iters - 1))]
    high = means[max(0, min(high_idx, iters - 1))]
    return mean, low, high


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
        "--error",
        choices=["sem", "std"],
        default="sem",
        help="Error bar type across runs (default: sem)",
    )
    ap.add_argument(
        "--error_over",
        choices=["runs", "generations", "items"],
        default="items",
        help="Compute error bars over runs, within-run generations, or items (default: generations)",
    )
    ap.add_argument(
        "--per_run",
        action="store_true",
        help="Print per-run metrics before the aggregate.",
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="Report bootstrap CIs over items.",
    )
    ap.add_argument(
        "--bootstrap_iters",
        type=int,
        default=2000,
        help="Bootstrap iterations (default: 2000).",
    )
    ap.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed (default: 0).",
    )
    ap.add_argument(
        "--bootstrap_ci",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = _collect_csv_paths(args.paths)
    if not csv_paths:
        raise SystemExit("No CSV files found.")

    per_run: List[Tuple[str, float, float, float]] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        default_acc, perturb_rate, global_acc = _compute_metrics(df)
        per_run.append((str(path), default_acc, perturb_rate, global_acc))

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

    if args.error_over == "generations":
        default_samples: List[float] = []
        perturb_samples: List[float] = []
        acc_samples: List[float] = []
        for path in csv_paths:
            df = pd.read_csv(path)
            default_samples.extend(_generation_means(df, "default_correct"))
            perturb_samples.extend(_generation_means(df, "perturbation_success"))
            acc_samples.extend(_generation_means(df, "perturbed_correct"))
        default_mean, default_err = _summary(default_samples, args.error)
        perturb_mean, perturb_err = _summary(perturb_samples, args.error)
        acc_mean, acc_err = _summary(acc_samples, args.error)
    elif args.error_over == "items":
        default_samples = []
        perturb_samples = []
        acc_samples = []
        for path in csv_paths:
            df = pd.read_csv(path)
            default_samples.extend(_item_means(df, "default_correct"))
            perturb_samples.extend(_item_means(df, "perturbation_success"))
            acc_samples.extend(_item_means(df, "perturbed_correct"))
        default_mean, default_err = _summary(default_samples, args.error)
        perturb_mean, perturb_err = _summary(perturb_samples, args.error)
        acc_mean, acc_err = _summary(acc_samples, args.error)
    else:
        default_mean, default_err = _summary(default_values, args.error)
        perturb_mean, perturb_err = _summary(perturb_values, args.error)
        acc_mean, acc_err = _summary(acc_values, args.error)

    print(f"Runs: {len(per_run)}")
    if args.error_over == "generations":
        print("Error bars computed over generations.")
    if args.error_over == "items":
        print("Error bars computed over items.")
    print(f"Default accuracy: {default_mean:.4f} ± {default_err:.4f} ({args.error})")
    print(f"Perturbation success rate: {perturb_mean:.4f} ± {perturb_err:.4f} ({args.error})")
    print(f"Global accuracy after perturbation: {acc_mean:.4f} ± {acc_err:.4f} ({args.error})")

    if args.bootstrap:
        alpha = 1 - args.bootstrap_ci
        default_items = []
        perturb_items = []
        acc_items = []
        for path in csv_paths:
            df = pd.read_csv(path)
            default_items.extend(_item_means(df, "default_correct"))
            perturb_items.extend(_item_means(df, "perturbation_success"))
            acc_items.extend(_item_means(df, "perturbed_correct"))
        d_mean, d_lo, d_hi = _bootstrap_ci(
            default_items,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            alpha=alpha,
        )
        p_mean, p_lo, p_hi = _bootstrap_ci(
            perturb_items,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            alpha=alpha,
        )
        a_mean, a_lo, a_hi = _bootstrap_ci(
            acc_items,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            alpha=alpha,
        )
        level = int(args.bootstrap_ci * 100)
        print(f"Bootstrap {level}% CI over items:")
        print(f"  Default accuracy: {d_mean:.4f} [{d_lo:.4f}, {d_hi:.4f}]")
        print(f"  Perturbation success rate: {p_mean:.4f} [{p_lo:.4f}, {p_hi:.4f}]")
        print(f"  Global accuracy after perturbation: {a_mean:.4f} [{a_lo:.4f}, {a_hi:.4f}]")


if __name__ == "__main__":
    main()

