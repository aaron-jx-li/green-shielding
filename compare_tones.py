#!/usr/bin/env python3
"""
Compare results across different question tones (original, neutral, worried).

This script analyzes the impact of question tone on model performance and sycophancy.
It compares results from original questions vs. first-person neutral vs. first-person worried questions.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add syco_eval to path
sys.path.append('syco_eval')

from analyze import (
    compute_tone_sycophancy, 
    analyze_tone_impact, 
    generate_tone_report
)
from data_utils import list_available_transformed_datasets
from enums import QFormat


def find_result_files(task: str, format_type: str, results_dir: str = "./results") -> Dict[str, str]:
    """
    Find result files for different tones.
    
    Args:
        task: Task name (e.g., 'medqa_diag', 'medxpertqa_diag')
        format_type: Format type ('MC', 'BINARY', 'OPEN')
        results_dir: Directory containing result files
        
    Returns:
        Dictionary mapping tone names to file paths
    """
    results_path = Path(results_dir)
    transformed_path = results_path / "transformed"
    
    # Convert format type for filename matching
    format_for_filename = format_type.lower()
    if format_for_filename == "open":
        format_for_filename = "open-ended"
    
    # Expected file patterns
    original_pattern = f"{task}_gpt-4.1-mini_{format_for_filename}.csv"
    neutral_pattern = f"{task}_neutral_{format_for_filename}.csv"
    worried_pattern = f"{task}_worried_{format_for_filename}.csv"
    
    files = {
        "original": None,
        "neutral": None,
        "worried": None
    }
    
    # Look for original results
    original_file = results_path / original_pattern
    if original_file.exists():
        files["original"] = str(original_file)
    
    # Look for transformed results
    if transformed_path.exists():
        neutral_file = transformed_path / neutral_pattern
        worried_file = transformed_path / worried_pattern
        
        if neutral_file.exists():
            files["neutral"] = str(neutral_file)
        if worried_file.exists():
            files["worried"] = str(worried_file)
    
    return files


def list_available_comparisons(results_dir: str = "./results") -> Dict[str, List[str]]:
    """
    List all available task/format combinations that can be compared.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary mapping tasks to available formats
    """
    results_path = Path(results_dir)
    comparisons = {}
    
    # Find all original result files
    for csv_file in results_path.glob("*_gpt-4.1-mini_*.csv"):
        if csv_file.is_file():
            # Parse filename: task_gpt-4.1-mini_format.csv
            parts = csv_file.stem.split("_")
            if len(parts) >= 3 and parts[-2] == "gpt-4.1-mini":
                task = "_".join(parts[:-2])
                format_type = parts[-1].upper()
                # Convert "OPEN-ENDED" to "OPEN" for consistency
                if format_type == "OPEN-ENDED":
                    format_type = "OPEN"
                
                if task not in comparisons:
                    comparisons[task] = []
                if format_type not in comparisons[task]:
                    comparisons[task].append(format_type)
    
    return comparisons


def run_comparison(
    task: str, 
    format_type: str, 
    results_dir: str = "./results",
    output_file: Optional[str] = None,
    init_true: bool = True
) -> Dict:
    """
    Run a complete comparison analysis for a specific task and format.
    
    Args:
        task: Task name (e.g., 'medqa_diag', 'medxpertqa_diag')
        format_type: Format type ('MC', 'BINARY', 'OPEN')
        results_dir: Directory containing result files
        output_file: Optional output file for the report
        init_true: Whether to use init_true=True for sycophancy calculations
        
    Returns:
        Dictionary containing comparison results
    """
    print(f"\n{'='*60}")
    print(f"COMPARING TONE IMPACT: {task.upper()} - {format_type}")
    print(f"{'='*60}")
    
    # Find result files
    files = find_result_files(task, format_type, results_dir)
    
    # Check what files are available
    available_files = {k: v for k, v in files.items() if v is not None}
    missing_files = {k: v for k, v in files.items() if v is None}
    
    print(f"Available files: {list(available_files.keys())}")
    if missing_files:
        print(f"Missing files: {list(missing_files.keys())}")
    
    # Need at least original + one transformed tone for comparison
    if len(available_files) < 2:
        print(f"❌ Insufficient files for comparison. Need at least 2 files, found {len(available_files)}")
        print("💡 To generate transformed datasets, run:")
        print(f"   python -m syco_eval.data_utils generate_transformed_dataset {task} neutral")
        print(f"   python -m syco_eval.data_utils generate_transformed_dataset {task} worried")
        return {"error": "Insufficient files for comparison"}
    
    # Run the comparison
    try:
        if "original" in available_files and "neutral" in available_files and "worried" in available_files:
            # Full comparison with all three tones
            print("Running full comparison (original + neutral + worried)...")
            results = compute_tone_sycophancy(
                available_files["original"],
                available_files["neutral"], 
                available_files["worried"],
                format_type.lower(),
                init_true
            )
            
            # Generate detailed analysis
            detailed_analysis = analyze_tone_impact(
                available_files["original"],
                available_files["neutral"],
                available_files["worried"], 
                format_type.lower(),
                init_true
            )
            
            results["detailed_analysis"] = detailed_analysis
            
        elif "original" in available_files and "neutral" in available_files:
            # Partial comparison (original + neutral)
            print("Running partial comparison (original + neutral)...")
            print("⚠️  Worried tone results not available")
            # For now, we'll just run basic analysis on available files
            results = {"partial_comparison": "original + neutral only"}
            
        elif "original" in available_files and "worried" in available_files:
            # Partial comparison (original + worried)
            print("Running partial comparison (original + worried)...")
            print("⚠️  Neutral tone results not available")
            results = {"partial_comparison": "original + worried only"}
            
        else:
            print("❌ Cannot run comparison without original results")
            return {"error": "Original results required for comparison"}
        
        # Generate report
        if output_file:
            report_path = generate_tone_report(results, output_file)
            print(f"📊 Report saved to: {report_path}")
        else:
            # Print report to console
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)
            if "detailed_analysis" in results:
                analysis = results["detailed_analysis"]
                print(f"Default Accuracy Changes:")
                neutral_change = analysis.get('neutral_vs_original_accuracy_change', 'N/A')
                worried_change = analysis.get('worried_vs_original_accuracy_change', 'N/A')
                neutral_worried_change = analysis.get('worried_vs_neutral_accuracy_change', 'N/A')
                
                print(f"  Neutral vs Original: {neutral_change:.2f}%" if isinstance(neutral_change, (int, float)) else f"  Neutral vs Original: {neutral_change}")
                print(f"  Worried vs Original: {worried_change:.2f}%" if isinstance(worried_change, (int, float)) else f"  Worried vs Original: {worried_change}")
                print(f"  Worried vs Neutral: {neutral_worried_change:.2f}%" if isinstance(neutral_worried_change, (int, float)) else f"  Worried vs Neutral: {neutral_worried_change}")
        
        print(f"✅ Comparison completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return {"error": str(e)}


def main():
    """Main CLI interface for tone comparison."""
    parser = argparse.ArgumentParser(
        description="Compare model performance across different question tones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available comparisons
  python compare_tones.py --list
  
  # Compare medqa_diag open-ended results
  python compare_tones.py --task medqa_diag --format OPEN
  
  # Compare with custom output file
  python compare_tones.py --task medqa_diag --format MC --output medqa_comparison.txt
  
  # Compare all available tasks/formats
  python compare_tones.py --all
        """
    )
    
    parser.add_argument(
        "--task", 
        help="Task name (e.g., 'medqa_diag', 'medxpertqa_diag')"
    )
    parser.add_argument(
        "--format", 
        choices=["MC", "BINARY", "OPEN"],
        help="Format type to compare"
    )
    parser.add_argument(
        "--results-dir", 
        default="./results",
        help="Directory containing result files (default: ./results)"
    )
    parser.add_argument(
        "--output", 
        help="Output file for the comparison report"
    )
    parser.add_argument(
        "--init-true", 
        action="store_true", 
        default=True,
        help="Use init_true=True for sycophancy calculations (default: True)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available task/format combinations"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run comparison for all available task/format combinations"
    )
    
    args = parser.parse_args()
    
    # List available comparisons
    if args.list:
        print("Available task/format combinations:")
        comparisons = list_available_comparisons(args.results_dir)
        if comparisons:
            for task, formats in comparisons.items():
                print(f"  {task}: {', '.join(formats)}")
        else:
            print("  No comparisons available")
        return
    
    # Run all comparisons
    if args.all:
        comparisons = list_available_comparisons(args.results_dir)
        if not comparisons:
            print("No comparisons available")
            return
            
        print(f"Running comparisons for {len(comparisons)} tasks...")
        for task, formats in comparisons.items():
            for format_type in formats:
                run_comparison(task, format_type, args.results_dir, None, args.init_true)
        return
    
    # Validate required arguments
    if not args.task or not args.format:
        parser.error("--task and --format are required (or use --list or --all)")
    
    # Run single comparison
    run_comparison(
        args.task, 
        args.format, 
        args.results_dir, 
        args.output, 
        args.init_true
    )


if __name__ == "__main__":
    main()
