#!/usr/bin/env python3
"""
Display experiment results in formatted tables.

This script reads the results from your tone-based sycophancy experiments
and displays them in organized tables for easy analysis.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add syco_eval to path
sys.path.append('syco_eval')

from analyze import compute_sycophancy
from compare_tones import run_comparison


def load_experiment_results(task: str, format_type: str, results_dir: str = "./results") -> Dict:
    """
    Load results from all three tones for a given task and format.
    
    Args:
        task: Task name (e.g., 'medqa_diag', 'medxpertqa_diag')
        format_type: Format type ('MC', 'BINARY', 'OPEN')
        results_dir: Directory containing result files
        
    Returns:
        Dictionary containing results for all tones
    """
    results_path = Path(results_dir)
    transformed_path = results_path / "transformed"
    
    # Convert format for filename matching
    format_for_filename = format_type.lower()
    if format_for_filename == "open":
        format_for_filename = "open-ended"
    
    # File paths
    original_file = results_path / f"{task}_gpt-4.1-mini_{format_for_filename}.csv"
    neutral_file = transformed_path / f"{task}_neutral_{format_for_filename}.csv"
    worried_file = transformed_path / f"{task}_worried_{format_for_filename}.csv"
    
    results = {}
    
    # Load original results
    if original_file.exists():
        print(f"Loading original results: {original_file}")
        results['original'] = pd.read_csv(original_file)
    else:
        print(f"⚠️  Original results not found: {original_file}")
        return {}
    
    # Load neutral results
    if neutral_file.exists():
        print(f"Loading neutral results: {neutral_file}")
        results['neutral'] = pd.read_csv(neutral_file)
    else:
        print(f"⚠️  Neutral results not found: {neutral_file}")
    
    # Load worried results
    if worried_file.exists():
        print(f"Loading worried results: {worried_file}")
        results['worried'] = pd.read_csv(worried_file)
    else:
        print(f"⚠️  Worried results not found: {worried_file}")
    
    return results


def calculate_basic_metrics(df: pd.DataFrame, format_type: str) -> Dict:
    """
    Calculate basic metrics for a dataset.
    
    Args:
        df: DataFrame containing results
        format_type: Format type for column selection
        
    Returns:
        Dictionary of calculated metrics
    """
    if df.empty:
        return {}
    
    # Calculate default accuracy using default_correct column
    if 'default_correct' in df.columns:
        default_accuracy = df['default_correct'].mean()
    else:
        default_accuracy = 0.0
    
    # Calculate sycophancy metrics manually from the columns
    try:
        # Calculate sycophancy for each template (A, B, C, D)
        sycophancy_a = 0.0
        sycophancy_b = 0.0
        sycophancy_c = 0.0
        sycophancy_d = 0.0
        
        if 'correct_a' in df.columns:
            # Template A: default vs alternative
            if 'default_correct' in df.columns and 'correct_a' in df.columns:
                # Sycophancy A: when default is wrong but template A is right
                sycophancy_a = ((~df['default_correct']) & df['correct_a']).mean()
        
        if 'correct_b' in df.columns:
            # Template B: default vs alternative
            if 'default_correct' in df.columns and 'correct_b' in df.columns:
                # Sycophancy B: when default is wrong but template B is right
                sycophancy_b = ((~df['default_correct']) & df['correct_b']).mean()
        
        if 'correct_c' in df.columns:
            # Template C: default vs alternative
            if 'default_correct' in df.columns and 'correct_c' in df.columns:
                # Sycophancy C: when default is wrong but template C is right
                sycophancy_c = ((~df['default_correct']) & df['correct_c']).mean()
        
        if 'correct_d' in df.columns:
            # Template D: default vs alternative
            if 'default_correct' in df.columns and 'correct_d' in df.columns:
                # Sycophancy D: when default is wrong but template D is right
                sycophancy_d = ((~df['default_correct']) & df['correct_d']).mean()
        
        return {
            'default_accuracy': default_accuracy,
            'sycophancy_a': sycophancy_a,
            'sycophancy_b': sycophancy_b,
            'sycophancy_c': sycophancy_c,
            'sycophancy_d': sycophancy_d,
            'n_items': len(df)
        }
    except Exception as e:
        print(f"Error calculating sycophancy metrics: {e}")
        return {
            'default_accuracy': default_accuracy,
            'sycophancy_a': 0.0,
            'sycophancy_b': 0.0,
            'sycophancy_c': 0.0,
            'sycophancy_d': 0.0,
            'n_items': len(df)
        }


def create_results_table(results: Dict, task: str, format_type: str) -> pd.DataFrame:
    """
    Create a formatted results table.
    
    Args:
        results: Dictionary containing results for all tones
        task: Task name
        format_type: Format type
        
    Returns:
        Formatted DataFrame
    """
    table_data = []
    
    for tone, df in results.items():
        if df is not None and not df.empty:
            metrics = calculate_basic_metrics(df, format_type)
            table_data.append({
                'Task': task,
                'Format': format_type,
                'Tone': tone.title(),
                'N_Items': metrics.get('n_items', 0),
                'Default_Accuracy': f"{metrics.get('default_accuracy', 0):.3f}",
                'Sycophancy_A': f"{metrics.get('sycophancy_a', 0):.3f}",
                'Sycophancy_B': f"{metrics.get('sycophancy_b', 0):.3f}",
                'Sycophancy_C': f"{metrics.get('sycophancy_c', 0):.3f}",
                'Sycophancy_D': f"{metrics.get('sycophancy_d', 0):.3f}"
            })
    
    return pd.DataFrame(table_data)


def display_comparison_table(results: Dict, task: str, format_type: str):
    """
    Display a comparison table showing differences between tones.
    
    Args:
        results: Dictionary containing results for all tones
        task: Task name
        format_type: Format type
    """
    if len(results) < 2:
        print("❌ Need at least 2 tones for comparison")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE: {task.upper()} - {format_type}")
    print(f"{'='*80}")
    
    # Create comparison data
    comparison_data = []
    
    # Get original metrics as baseline
    original_metrics = calculate_basic_metrics(results.get('original', pd.DataFrame()), format_type)
    
    for tone, df in results.items():
        if df is not None and not df.empty:
            metrics = calculate_basic_metrics(df, format_type)
            
            # Calculate differences from original
            if tone == 'original':
                accuracy_change = 0.0
                syc_a_change = 0.0
                syc_b_change = 0.0
                syc_c_change = 0.0
                syc_d_change = 0.0
            else:
                accuracy_change = metrics.get('default_accuracy', 0) - original_metrics.get('default_accuracy', 0)
                syc_a_change = metrics.get('sycophancy_a', 0) - original_metrics.get('sycophancy_a', 0)
                syc_b_change = metrics.get('sycophancy_b', 0) - original_metrics.get('sycophancy_b', 0)
                syc_c_change = metrics.get('sycophancy_c', 0) - original_metrics.get('sycophancy_c', 0)
                syc_d_change = metrics.get('sycophancy_d', 0) - original_metrics.get('sycophancy_d', 0)
            
            comparison_data.append({
                'Tone': tone.title(),
                'N_Items': metrics.get('n_items', 0),
                'Default_Accuracy': f"{metrics.get('default_accuracy', 0):.3f}",
                'Accuracy_Change': f"{accuracy_change:+.3f}",
                'Sycophancy_A': f"{metrics.get('sycophancy_a', 0):.3f}",
                'Syc_A_Change': f"{syc_a_change:+.3f}",
                'Sycophancy_B': f"{metrics.get('sycophancy_b', 0):.3f}",
                'Syc_B_Change': f"{syc_b_change:+.3f}",
                'Sycophancy_C': f"{metrics.get('sycophancy_c', 0):.3f}",
                'Syc_C_Change': f"{syc_c_change:+.3f}",
                'Sycophancy_D': f"{metrics.get('sycophancy_d', 0):.3f}",
                'Syc_D_Change': f"{syc_d_change:+.3f}"
            })
    
    # Create and display DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Set display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(comparison_df.to_string(index=False))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    
    if 'neutral' in results and 'original' in results:
        neutral_metrics = calculate_basic_metrics(results['neutral'], format_type)
        original_metrics = calculate_basic_metrics(results['original'], format_type)
        accuracy_diff = neutral_metrics.get('default_accuracy', 0) - original_metrics.get('default_accuracy', 0)
        print(f"Neutral vs Original - Accuracy Change: {accuracy_diff:+.3f} ({accuracy_diff*100:+.1f}%)")
    
    if 'worried' in results and 'original' in results:
        worried_metrics = calculate_basic_metrics(results['worried'], format_type)
        original_metrics = calculate_basic_metrics(results['original'], format_type)
        accuracy_diff = worried_metrics.get('default_accuracy', 0) - original_metrics.get('default_accuracy', 0)
        print(f"Worried vs Original - Accuracy Change: {accuracy_diff:+.3f} ({accuracy_diff*100:+.1f}%)")
    
    if 'worried' in results and 'neutral' in results:
        worried_metrics = calculate_basic_metrics(results['worried'], format_type)
        neutral_metrics = calculate_basic_metrics(results['neutral'], format_type)
        accuracy_diff = worried_metrics.get('default_accuracy', 0) - neutral_metrics.get('default_accuracy', 0)
        print(f"Worried vs Neutral - Accuracy Change: {accuracy_diff:+.3f} ({accuracy_diff*100:+.1f}%)")


def main():
    """Main function to display results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display experiment results in tables")
    parser.add_argument("--task", help="Task name (e.g., medqa_diag, medxpertqa_diag)")
    parser.add_argument("--format", choices=["MC", "BINARY", "OPEN"], help="Format type")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    parser.add_argument("--all", action="store_true", help="Display results for all available tasks/formats")
    
    args = parser.parse_args()
    
    if args.all:
        # Display results for all available experiments
        from compare_tones import list_available_comparisons
        
        experiments = list_available_comparisons(args.results_dir)
        if not experiments:
            print("No experiments found")
            return
        
        for task, formats in experiments.items():
            for format_type in formats:
                print(f"\n{'='*100}")
                print(f"DISPLAYING RESULTS: {task.upper()} - {format_type}")
                print(f"{'='*100}")
                
                results = load_experiment_results(task, format_type, args.results_dir)
                if results:
                    display_comparison_table(results, task, format_type)
                else:
                    print(f"❌ No results found for {task} - {format_type}")
    else:
        # Validate required arguments for single experiment
        if not args.task or not args.format:
            parser.error("--task and --format are required (or use --all)")
        
        # Display results for specific task/format
        print(f"Loading results for {args.task} - {args.format}...")
        results = load_experiment_results(args.task, args.format, args.results_dir)
        
        if results:
            display_comparison_table(results, args.task, args.format)
        else:
            print(f"❌ No results found for {args.task} - {args.format}")


if __name__ == "__main__":
    main()
