#!/usr/bin/env python3
"""Analyze worried sentence experiment results.

Calculates flip rate and accuracy drop with confidence intervals across multiple runs.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def analyze_csv(csv_path: str):
    """Analyze a single CSV file.
    
    Returns:
        (flip_rate, accuracy_drop)
        - flip_rate: percentage of questions that flipped (correct↔incorrect)
        - accuracy_drop: default_accuracy - worried_accuracy
    """
    df = pd.read_csv(csv_path)
    
    # Convert boolean columns (handle string "True"/"False" if present)
    df['default_correct'] = df['default_correct'].astype(str).map({'True': True, 'False': False, 'true': True, 'false': False}).fillna(False)
    df['worried_correct'] = df['worried_correct'].astype(str).map({'True': True, 'False': False, 'true': True, 'false': False}).fillna(False)
    
    # Filter out rows with None/NaN
    valid = df['default_correct'].notna() & df['worried_correct'].notna()
    df = df[valid]
    
    if len(df) == 0:
        return 0.0, 0.0
    
    # Calculate flip rate: questions where correctness changed
    flips = (df['default_correct'] != df['worried_correct']).sum()
    flip_rate = (flips / len(df)) * 100
    
    # Calculate accuracy drop
    default_acc = df['default_correct'].mean() * 100
    worried_acc = df['worried_correct'].mean() * 100
    accuracy_drop = default_acc - worried_acc
    
    return flip_rate, accuracy_drop


def compute_ci(values, confidence: float = 0.95):
    """Compute mean and confidence interval using t-distribution.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(n)  # Standard error
    
    # t-value for (1-confidence)/2 tail (e.g., 0.025 for 95% CI)
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    
    margin = t_val * std_err
    return mean, mean - margin, mean + margin


def analyze_dataset(dataset_name: str, csv_pattern: str, results_dir: str = "results") -> None:
    """Analyze results for a dataset across multiple CSV files.
    
    Args:
        dataset_name: Name for display (e.g., "medxpertqa_diag")
        csv_pattern: Pattern to match CSV files (e.g., "worried_medxpertqa_open_*.csv")
        results_dir: Directory containing CSV files
    """
    results_path = Path(results_dir)
    csv_files = sorted(results_path.glob(csv_pattern))
    
    if not csv_files:
        print(f"⚠️  No files found matching: {csv_pattern}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name}")
    print(f"Found {len(csv_files)} CSV files")
    print(f"{'='*60}")
    
    # Analyze each CSV
    flip_rates = []
    accuracy_drops = []
    
    for csv_file in csv_files:
        flip_rate, accuracy_drop = analyze_csv(str(csv_file))
        flip_rates.append(flip_rate)
        accuracy_drops.append(accuracy_drop)
        print(f"  {csv_file.name}: flip_rate={flip_rate:.2f}%, accuracy_drop={accuracy_drop:.2f}%")
    
    # Compute statistics across runs
    flip_mean, flip_lower, flip_upper = compute_ci(flip_rates)
    drop_mean, drop_lower, drop_upper = compute_ci(accuracy_drops)
    
    print(f"\n📊 Results across {len(csv_files)} runs (95% CI):")
    print(f"  Flip Rate:     {flip_mean:.2f}% [{flip_lower:.2f}%, {flip_upper:.2f}%]")
    print(f"  Accuracy Drop: {drop_mean:.2f}% [{drop_lower:.2f}%, {drop_upper:.2f}%]")


def main():
    """Main analysis function."""
    print("Analyzing worried sentence experiment results...")
    
    # Analyze medxpertqa_diag (files 2-5)
    analyze_dataset(
        "medxpertqa_diag",
        "worried_medxpertqa_open_*.csv",
        results_dir="../results"
    )
    
    # Analyze medqa_diag (files 1-5)
    analyze_dataset(
        "medqa_diag",
        "worried_medqa_open_*.csv",
        results_dir="../results"
    )
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

