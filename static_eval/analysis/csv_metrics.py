from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def collect_csv_paths(paths: Iterable[str]) -> List[Path]:
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


def metric_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    numbered = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if numbered:
        return numbered
    return [prefix] if prefix in df.columns else []


def coerce_boolish(series: pd.Series) -> pd.Series:
    mapped = series.map({"True": 1, "False": 0, True: 1, False: 0})
    return series.where(mapped.isna(), mapped)


def run_means(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    cols = metric_columns(df, prefix)
    if not cols:
        raise ValueError(f"Missing columns for '{prefix}'")
    values = df[cols].copy()
    values = values.apply(coerce_boolish)
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


def run_id_from_col(prefix: str, col: str) -> str:
    if col == prefix:
        return "0"
    return col[len(prefix) + 1 :]
