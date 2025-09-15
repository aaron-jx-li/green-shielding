#!/usr/bin/env python3
"""
Complete workflow script for running tone-based sycophancy experiments.

This script orchestrates the entire process:
1. Generate transformed datasets (neutral and worried tones)
2. Run evaluations for all three tones (original, neutral, worried)
3. Perform comparative analysis
4. Generate comprehensive reports

Usage:
    python run_tone_experiment.py --task medqa_diag --format OPEN
    python run_tone_experiment.py --all
    python run_tone_experiment.py --task medqa_diag --format OPEN --skip-generation
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add syco_eval to path
sys.path.append('syco_eval')

from data_utils import (
    generate_transformed_dataset, 
    check_transformed_dataset_exists,
    list_available_transformed_datasets
)
from compare_tones import (
    list_available_comparisons,
    run_comparison
)
from enums import QFormat, QuestionTone


class ToneExperimentRunner:
    """Main class for running tone-based experiments."""
    
    def __init__(self, results_dir: str = "./results", model: str = "gpt-4o-mini"):
        self.results_dir = Path(results_dir)
        self.model = model
        self.transformed_dir = self.results_dir / "transformed"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.transformed_dir.mkdir(exist_ok=True)
    
    def list_available_experiments(self) -> Dict[str, List[str]]:
        """List all available task/format combinations for experiments."""
        return list_available_comparisons(str(self.results_dir))
    
    def check_prerequisites(self, task: str, format_type: str) -> Dict[str, bool]:
        """Check what components are available for an experiment."""
        # Convert format for filename matching
        format_for_filename = format_type.lower()
        if format_for_filename == "open":
            format_for_filename = "open-ended"
        
        original_file = self.results_dir / f"{task}_gpt-4.1-mini_{format_for_filename}.csv"
        neutral_file = self.transformed_dir / f"{task}_neutral_{format_for_filename}.csv"
        worried_file = self.transformed_dir / f"{task}_worried_{format_for_filename}.csv"
        
        return {
            "original_results": original_file.exists(),
            "neutral_dataset": check_transformed_dataset_exists(task, "neutral"),
            "worried_dataset": check_transformed_dataset_exists(task, "worried"),
            "neutral_results": neutral_file.exists(),
            "worried_results": worried_file.exists()
        }
    
    def generate_datasets(self, task: str, tones: List[str] = ["neutral", "worried"], 
                         max_items: Optional[int] = None, delay: float = 0.5) -> Dict[str, bool]:
        """Generate transformed datasets for specified tones."""
        results = {}
        
        for tone in tones:
            print(f"\n{'='*60}")
            print(f"GENERATING {tone.upper()} DATASET: {task}")
            print(f"{'='*60}")
            
            if check_transformed_dataset_exists(task, tone):
                print(f"✅ {tone} dataset already exists, skipping generation")
                results[tone] = True
                continue
            
            try:
                print(f"🔄 Generating {tone} dataset...")
                output_path = generate_transformed_dataset(
                    task=task,
                    tone=tone,
                    model=self.model,
                    max_items=max_items,
                    delay=delay
                )
                print(f"✅ {tone} dataset generated: {output_path}")
                results[tone] = True
                
            except Exception as e:
                print(f"❌ Failed to generate {tone} dataset: {e}")
                results[tone] = False
        
        return results
    
    def run_evaluations(self, task: str, format_type: str, tones: List[str] = ["original", "neutral", "worried"]) -> Dict[str, bool]:
        """Run evaluations for specified tones."""
        results = {}
        
        for tone in tones:
            print(f"\n{'='*60}")
            print(f"RUNNING EVALUATION: {task} - {format_type} - {tone.upper()}")
            print(f"{'='*60}")
            
            # Convert format for filename matching
            format_for_filename = format_type.lower()
            if format_for_filename == "open":
                format_for_filename = "open-ended"
            
            # Check if results already exist
            if tone == "original":
                result_file = self.results_dir / f"{task}_gpt-4.1-mini_{format_for_filename}.csv"
            else:
                result_file = self.transformed_dir / f"{task}_{tone}_{format_for_filename}.csv"
            
            if result_file.exists():
                print(f"✅ {tone} results already exist: {result_file}")
                results[tone] = True
                continue
            
            # Check if dataset exists for transformed tones
            if tone != "original" and not check_transformed_dataset_exists(task, tone):
                print(f"❌ {tone} dataset not found. Run dataset generation first.")
                results[tone] = False
                continue
            
            try:
                print(f"🔄 Running evaluation for {tone} tone...")
                
                # Import and run the evaluation
                from main import main as run_evaluation
                
                # Convert format type for the main script
                format_for_main = format_type.upper()
                if format_for_main == "OPEN":
                    format_for_main = "open-ended"
                
                # Determine output CSV path (use lowercase for filename)
                format_for_filename = format_for_main.lower()
                if format_for_filename == "open-ended":
                    format_for_filename = "open-ended"
                
                if tone == "original":
                    out_csv = str(self.results_dir / f"{task}_gpt-4.1-mini_{format_for_filename}.csv")
                else:
                    out_csv = str(self.transformed_dir / f"{task}_{tone}_{format_for_filename}.csv")
                
                # Prepare arguments for the evaluation
                sys.argv = [
                    "main.py",
                    "--task", task,
                    "--format", format_for_main,
                    "--tone", tone,
                    "--model", self.model,
                    "--out_csv", out_csv
                ]
                
                # Run the evaluation
                run_evaluation()
                
                # Check if the result file was created
                if result_file.exists():
                    print(f"✅ {tone} evaluation completed: {result_file}")
                    results[tone] = True
                else:
                    print(f"❌ {tone} evaluation failed - no result file created")
                    results[tone] = False
                
            except Exception as e:
                print(f"❌ Failed to run {tone} evaluation: {e}")
                results[tone] = False
        
        return results
    
    def run_comparison(self, task: str, format_type: str, output_file: Optional[str] = None) -> Dict:
        """Run comparative analysis for the experiment."""
        print(f"\n{'='*60}")
        print(f"RUNNING COMPARISON: {task} - {format_type}")
        print(f"{'='*60}")
        
        try:
            result = run_comparison(
                task=task,
                format_type=format_type,
                results_dir=str(self.results_dir),
                output_file=output_file,
                init_true=True
            )
            
            if "error" in result:
                print(f"⚠️  Comparison completed with warnings: {result['error']}")
            else:
                print("✅ Comparison completed successfully")
            
            return result
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            return {"error": str(e)}
    
    def run_full_experiment(self, task: str, format_type: str, 
                           skip_generation: bool = False,
                           max_items: Optional[int] = None,
                           delay: float = 0.5) -> Dict:
        """Run a complete experiment for a specific task and format."""
        print(f"\n{'='*80}")
        print(f"STARTING FULL EXPERIMENT: {task.upper()} - {format_type}")
        print(f"{'='*80}")
        
        start_time = time.time()
        experiment_results = {
            "task": task,
            "format": format_type,
            "start_time": start_time,
            "steps": {}
        }
        
        # Step 1: Check prerequisites
        print("\n📋 Checking prerequisites...")
        prerequisites = self.check_prerequisites(task, format_type)
        experiment_results["prerequisites"] = prerequisites
        
        print("Prerequisites status:")
        for component, available in prerequisites.items():
            status = "✅" if available else "❌"
            print(f"  {status} {component}")
        
        # Step 2: Generate datasets (if not skipping)
        if not skip_generation:
            print("\n🔄 Step 1: Generating transformed datasets...")
            dataset_results = self.generate_datasets(task, ["neutral", "worried"], max_items, delay)
            experiment_results["steps"]["dataset_generation"] = dataset_results
            
            if not all(dataset_results.values()):
                print("❌ Dataset generation failed for some tones")
                return experiment_results
        else:
            print("\n⏭️  Skipping dataset generation")
            experiment_results["steps"]["dataset_generation"] = {"skipped": True}
        
        # Step 3: Run evaluations
        print("\n🔄 Step 2: Running evaluations...")
        evaluation_results = self.run_evaluations(task, format_type, ["original", "neutral", "worried"])
        experiment_results["steps"]["evaluations"] = evaluation_results
        
        if not all(evaluation_results.values()):
            print("❌ Some evaluations failed")
            return experiment_results
        
        # Step 4: Run comparison
        print("\n🔄 Step 3: Running comparison...")
        comparison_result = self.run_comparison(task, format_type)
        experiment_results["steps"]["comparison"] = comparison_result
        
        # Calculate total time
        end_time = time.time()
        experiment_results["end_time"] = end_time
        experiment_results["total_time"] = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED: {task.upper()} - {format_type}")
        print(f"Total time: {experiment_results['total_time']:.2f} seconds")
        print(f"{'='*80}")
        
        return experiment_results
    
    def run_all_experiments(self, skip_generation: bool = False, 
                           max_items: Optional[int] = None,
                           delay: float = 0.5) -> Dict:
        """Run experiments for all available task/format combinations."""
        print(f"\n{'='*80}")
        print("STARTING ALL EXPERIMENTS")
        print(f"{'='*80}")
        
        available_experiments = self.list_available_experiments()
        if not available_experiments:
            print("❌ No experiments available")
            return {"error": "No experiments available"}
        
        print(f"Found {len(available_experiments)} tasks with experiments:")
        for task, formats in available_experiments.items():
            print(f"  {task}: {formats}")
        
        all_results = {}
        total_experiments = sum(len(formats) for formats in available_experiments.values())
        current_experiment = 0
        
        for task, formats in available_experiments.items():
            for format_type in formats:
                current_experiment += 1
                print(f"\n🔄 Running experiment {current_experiment}/{total_experiments}")
                
                try:
                    result = self.run_full_experiment(
                        task=task,
                        format_type=format_type,
                        skip_generation=skip_generation,
                        max_items=max_items,
                        delay=delay
                    )
                    all_results[f"{task}_{format_type}"] = result
                    
                except Exception as e:
                    print(f"❌ Experiment {task}_{format_type} failed: {e}")
                    all_results[f"{task}_{format_type}"] = {"error": str(e)}
        
        print(f"\n{'='*80}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
        
        successful = sum(1 for r in all_results.values() if "error" not in r)
        print(f"Successful experiments: {successful}/{total_experiments}")
        
        return all_results


def main():
    """Main CLI interface for the tone experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run complete tone-based sycophancy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment for specific task/format
  python run_tone_experiment.py --task medqa_diag --format OPEN
  
  # Run all available experiments
  python run_tone_experiment.py --all
  
  # Skip dataset generation (use existing datasets)
  python run_tone_experiment.py --task medqa_diag --format OPEN --skip-generation
  
  # Limit items for testing
  python run_tone_experiment.py --task medqa_diag --format OPEN --max-items 10
  
  # List available experiments
  python run_tone_experiment.py --list
        """
    )
    
    parser.add_argument(
        "--task", 
        help="Task name (e.g., 'medqa_diag', 'medxpertqa_diag')"
    )
    parser.add_argument(
        "--format", 
        choices=["MC", "BINARY", "OPEN"],
        help="Format type to run experiment for"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run experiments for all available task/format combinations"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available experiments"
    )
    parser.add_argument(
        "--skip-generation", 
        action="store_true",
        help="Skip dataset generation (use existing datasets)"
    )
    parser.add_argument(
        "--max-items", 
        type=int,
        help="Maximum number of items to process (for testing)"
    )
    parser.add_argument(
        "--model", 
        default="gpt-4o-mini",
        help="Model to use for transformations and evaluations (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--delay", 
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--results-dir", 
        default="./results",
        help="Directory for results (default: ./results)"
    )
    
    args = parser.parse_args()
    
    # Initialize the experiment runner
    runner = ToneExperimentRunner(
        results_dir=args.results_dir,
        model=args.model
    )
    
    # List available experiments
    if args.list:
        print("Available experiments:")
        experiments = runner.list_available_experiments()
        if experiments:
            for task, formats in experiments.items():
                print(f"  {task}: {', '.join(formats)}")
        else:
            print("  No experiments available")
        return
    
    # Run all experiments
    if args.all:
        results = runner.run_all_experiments(
            skip_generation=args.skip_generation,
            max_items=args.max_items,
            delay=args.delay
        )
        return
    
    # Validate required arguments for single experiment
    if not args.task or not args.format:
        parser.error("--task and --format are required (or use --list or --all)")
    
    # Run single experiment
    result = runner.run_full_experiment(
        task=args.task,
        format_type=args.format,
        skip_generation=args.skip_generation,
        max_items=args.max_items,
        delay=args.delay
    )
    
    # Print summary
    if "error" in result:
        print(f"\n❌ Experiment failed: {result['error']}")
    else:
        print(f"\n✅ Experiment completed successfully")
        print(f"Total time: {result.get('total_time', 0):.2f} seconds")


if __name__ == "__main__":
    main()
