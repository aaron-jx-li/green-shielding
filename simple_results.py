#!/usr/bin/env python3
"""
Simple results display script for tone-based sycophancy experiments.
"""

import pandas as pd
import sys
from pathlib import Path

def display_simple_results(task: str, format_type: str, results_dir: str = "./results"):
    """Display results in a simple, readable format."""
    
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
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {task.upper()} - {format_type}")
    print(f"{'='*60}")
    
    results = {}
    
    # Load and analyze original results
    if original_file.exists():
        df_orig = pd.read_csv(original_file)
        orig_accuracy = df_orig['default_correct'].mean()
        orig_syc_a = ((~df_orig['default_correct']) & df_orig['correct_a']).mean()
        orig_syc_b = ((~df_orig['default_correct']) & df_orig['correct_b']).mean()
        orig_syc_c = ((~df_orig['default_correct']) & df_orig['correct_c']).mean()
        orig_syc_d = ((~df_orig['default_correct']) & df_orig['correct_d']).mean()
        
        print(f"📊 ORIGINAL TONE:")
        print(f"   Items: {len(df_orig)}")
        print(f"   Default Accuracy: {orig_accuracy:.3f} ({orig_accuracy*100:.1f}%)")
        print(f"   Sycophancy A: {orig_syc_a:.3f}")
        print(f"   Sycophancy B: {orig_syc_b:.3f}")
        print(f"   Sycophancy C: {orig_syc_c:.3f}")
        print(f"   Sycophancy D: {orig_syc_d:.3f}")
        
        results['original'] = {
            'accuracy': orig_accuracy,
            'syc_a': orig_syc_a,
            'syc_b': orig_syc_b,
            'syc_c': orig_syc_c,
            'syc_d': orig_syc_d,
            'n': len(df_orig)
        }
    else:
        print(f"❌ Original results not found: {original_file}")
        return
    
    # Load and analyze neutral results
    if neutral_file.exists():
        df_neutral = pd.read_csv(neutral_file)
        neutral_accuracy = df_neutral['default_correct'].mean()
        neutral_syc_a = ((~df_neutral['default_correct']) & df_neutral['correct_a']).mean()
        neutral_syc_b = ((~df_neutral['default_correct']) & df_neutral['correct_b']).mean()
        neutral_syc_c = ((~df_neutral['default_correct']) & df_neutral['correct_c']).mean()
        neutral_syc_d = ((~df_neutral['default_correct']) & df_neutral['correct_d']).mean()
        
        print(f"\n📊 NEUTRAL TONE:")
        print(f"   Items: {len(df_neutral)}")
        print(f"   Default Accuracy: {neutral_accuracy:.3f} ({neutral_accuracy*100:.1f}%)")
        print(f"   Sycophancy A: {neutral_syc_a:.3f}")
        print(f"   Sycophancy B: {neutral_syc_b:.3f}")
        print(f"   Sycophancy C: {neutral_syc_c:.3f}")
        print(f"   Sycophancy D: {neutral_syc_d:.3f}")
        
        results['neutral'] = {
            'accuracy': neutral_accuracy,
            'syc_a': neutral_syc_a,
            'syc_b': neutral_syc_b,
            'syc_c': neutral_syc_c,
            'syc_d': neutral_syc_d,
            'n': len(df_neutral)
        }
    else:
        print(f"⚠️  Neutral results not found: {neutral_file}")
    
    # Load and analyze worried results
    if worried_file.exists():
        df_worried = pd.read_csv(worried_file)
        worried_accuracy = df_worried['default_correct'].mean()
        worried_syc_a = ((~df_worried['default_correct']) & df_worried['correct_a']).mean()
        worried_syc_b = ((~df_worried['default_correct']) & df_worried['correct_b']).mean()
        worried_syc_c = ((~df_worried['default_correct']) & df_worried['correct_c']).mean()
        worried_syc_d = ((~df_worried['default_correct']) & df_worried['correct_d']).mean()
        
        print(f"\n📊 WORRIED TONE:")
        print(f"   Items: {len(df_worried)}")
        print(f"   Default Accuracy: {worried_accuracy:.3f} ({worried_accuracy*100:.1f}%)")
        print(f"   Sycophancy A: {worried_syc_a:.3f}")
        print(f"   Sycophancy B: {worried_syc_b:.3f}")
        print(f"   Sycophancy C: {worried_syc_c:.3f}")
        print(f"   Sycophancy D: {worried_syc_d:.3f}")
        
        results['worried'] = {
            'accuracy': worried_accuracy,
            'syc_a': worried_syc_a,
            'syc_b': worried_syc_b,
            'syc_c': worried_syc_c,
            'syc_d': worried_syc_d,
            'n': len(df_worried)
        }
    else:
        print(f"⚠️  Worried results not found: {worried_file}")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY:")
    print(f"{'='*60}")
    
    if 'neutral' in results:
        acc_diff = results['neutral']['accuracy'] - results['original']['accuracy']
        print(f"🔄 Neutral vs Original:")
        print(f"   Accuracy Change: {acc_diff:+.3f} ({acc_diff*100:+.1f}%)")
        print(f"   Sycophancy A Change: {results['neutral']['syc_a'] - results['original']['syc_a']:+.3f}")
        print(f"   Sycophancy B Change: {results['neutral']['syc_b'] - results['original']['syc_b']:+.3f}")
        print(f"   Sycophancy C Change: {results['neutral']['syc_c'] - results['original']['syc_c']:+.3f}")
        print(f"   Sycophancy D Change: {results['neutral']['syc_d'] - results['original']['syc_d']:+.3f}")
    
    if 'worried' in results:
        acc_diff = results['worried']['accuracy'] - results['original']['accuracy']
        print(f"\n🔄 Worried vs Original:")
        print(f"   Accuracy Change: {acc_diff:+.3f} ({acc_diff*100:+.1f}%)")
        print(f"   Sycophancy A Change: {results['worried']['syc_a'] - results['original']['syc_a']:+.3f}")
        print(f"   Sycophancy B Change: {results['worried']['syc_b'] - results['original']['syc_b']:+.3f}")
        print(f"   Sycophancy C Change: {results['worried']['syc_c'] - results['original']['syc_c']:+.3f}")
        print(f"   Sycophancy D Change: {results['worried']['syc_d'] - results['original']['syc_d']:+.3f}")
    
    if 'neutral' in results and 'worried' in results:
        acc_diff = results['worried']['accuracy'] - results['neutral']['accuracy']
        print(f"\n🔄 Worried vs Neutral:")
        print(f"   Accuracy Change: {acc_diff:+.3f} ({acc_diff*100:+.1f}%)")
        print(f"   Sycophancy A Change: {results['worried']['syc_a'] - results['neutral']['syc_a']:+.3f}")
        print(f"   Sycophancy B Change: {results['worried']['syc_b'] - results['neutral']['syc_b']:+.3f}")
        print(f"   Sycophancy C Change: {results['worried']['syc_c'] - results['neutral']['syc_c']:+.3f}")
        print(f"   Sycophancy D Change: {results['worried']['syc_d'] - results['neutral']['syc_d']:+.3f}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display simple experiment results")
    parser.add_argument("--task", required=True, help="Task name (e.g., medqa_diag, medxpertqa_diag)")
    parser.add_argument("--format", required=True, choices=["MC", "BINARY", "OPEN"], help="Format type")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    
    args = parser.parse_args()
    
    display_simple_results(args.task, args.format, args.results_dir)


if __name__ == "__main__":
    main()
