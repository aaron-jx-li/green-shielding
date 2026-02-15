import argparse
import glob
import json
import os
import random
import re
import sys
import statistics
from typing import Dict, Iterable, List, Optional, Tuple


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        raise ValueError("percentile requires non-empty list")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def bootstrap_ci(
    values: List[float],
    n_boot: int,
    ci: float,
    seed: Optional[int],
) -> Tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_ci requires non-empty values")
    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    alpha = (1.0 - ci) / 2.0
    low = _percentile(means, alpha * 100.0)
    high = _percentile(means, (1.0 - alpha) * 100.0)
    return low, high


def load_run_summaries(
    results_dir: str,
    pattern: str,
    run_regex: str,
) -> List[Dict[str, object]]:
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' in '{results_dir}'"
        )

    run_re = re.compile(run_regex)
    runs = []
    for path in files:
        match = run_re.search(os.path.basename(path))
        run_idx = int(match.group(1)) if match else None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        metrics = {k: v for k, v in summary.items() if _is_number(v)}
        runs.append({"run": run_idx, "path": path, "metrics": metrics})

    return runs


def intersect_metric_keys(runs: Iterable[Dict[str, object]]) -> List[str]:
    keys: List[str] = []
    for i, run in enumerate(runs):
        run_keys = set(run["metrics"].keys())
        if i == 0:
            keys = sorted(run_keys)
        else:
            keys = sorted(set(keys).intersection(run_keys))
    return keys


def compute_summary(
    runs: List[Dict[str, object]],
    n_boot: int,
    ci: float,
    seed: Optional[int],
) -> List[Tuple[str, float, float, float]]:
    keys = intersect_metric_keys(runs)
    results = []
    for key in keys:
        values = [run["metrics"][key] for run in runs]
        mean_val = statistics.mean(values)
        if len(values) < 2:
            ci_low, ci_high = mean_val, mean_val
        else:
            ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, ci=ci, seed=seed)
        results.append((key, mean_val, ci_low, ci_high))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average summary metrics and CIs "
            "from multiple runs."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join("..", "results", "HCM-3k", "exp_4"),
        help="Directory containing eval_converted_*_*.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="eval_converted_*_gpt-4.1-mini.json",
        help="Glob pattern to select run files.",
    )
    parser.add_argument(
        "--run-regex",
        default=r"eval_converted_(\d+)_",
        help="Regex with a single capture group for run index.",
    )
    args = parser.parse_args()

    runs = load_run_summaries(args.results_dir, args.pattern, args.run_regex)
    if len(runs) < 1:
        raise ValueError("Need at least 1 run to compute metrics.")
    if len(runs) == 1:
        print(
            "Only one run found; CI will be degenerate at the mean.",
            file=sys.stderr,
        )

    summary = compute_summary(
        runs,
        n_boot=2000,
        ci=0.95,
        seed=None,
    )

    print("metric\tmean\tci_low\tci_high")
    for key, mean_val, ci_low, ci_high in summary:
        print(f"{key}\t{mean_val:.6f}\t{ci_low:.6f}\t{ci_high:.6f}")


if __name__ == "__main__":
    main()
