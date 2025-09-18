#!/usr/bin/env python3
"""
Integration test for the complete tone-based evaluation pipeline.
This script tests the entire workflow from data generation to evaluation.
"""

import sys
import os
import tempfile
import shutil

path_to_syco_eval = os.path.join(os.path.dirname(__file__), 'syco_eval')
sys.path.append(path_to_syco_eval)

from syco_eval.data_utils import (
    generate_transformed_dataset,
    get_transformed_dataset,
    check_transformed_dataset_exists
)
from syco_eval.runner import evaluate_and_save_csv
from syco_eval.enums import QFormat, QuestionTone
import pandas as pd


def test_small_pipeline():
    """Test the complete pipeline with a small sample."""
    print("="*60)
    print("TESTING COMPLETE TONE-BASED EVALUATION PIPELINE")
    print("="*60)
    
    # Test parameters
    task = "medqa_diag"
    model = "gpt-4o-mini"
    format_type = "MC"
    max_items = 2  # Small sample for testing
    
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Format: {format_type}")
    print(f"Max items: {max_items}")
    
    # Step 1: Generate transformed datasets
    print("\n" + "-"*40)
    print("STEP 1: GENERATING TRANSFORMED DATASETS")
    print("-"*40)
    
    try:
        # Generate neutral dataset
        print("Generating neutral dataset...")
        neutral_path = generate_transformed_dataset(
            task=task,
            tone="neutral",
            model=model,
            max_items=max_items,
            delay=1.0
        )
        print(f"✓ Neutral dataset generated: {neutral_path}")
        
        # Generate worried dataset
        print("Generating worried dataset...")
        worried_path = generate_transformed_dataset(
            task=task,
            tone="worried",
            model=model,
            max_items=max_items,
            delay=1.0
        )
        print(f"✓ Worried dataset generated: {worried_path}")
        
    except Exception as e:
        print(f"✗ Error generating datasets: {e}")
        return False
    
    # Step 2: Test dataset loading
    print("\n" + "-"*40)
    print("STEP 2: TESTING DATASET LOADING")
    print("-"*40)
    
    try:
        # Test loading neutral dataset
        neutral_ds = get_transformed_dataset(task, "neutral")
        print(f"✓ Neutral dataset loaded: {len(neutral_ds)} items")
        
        # Test loading worried dataset
        worried_ds = get_transformed_dataset(task, "worried")
        print(f"✓ Worried dataset loaded: {len(worried_ds)} items")
        
        # Show sample data
        if len(neutral_ds) > 0:
            sample = neutral_ds[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Original question: {sample.get('question', 'N/A')[:100]}...")
            print(f"Transformed neutral: {sample.get('transformed_question_neutral', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        return False
    
    # Step 3: Test evaluation with different tones
    print("\n" + "-"*40)
    print("STEP 3: TESTING EVALUATIONS WITH DIFFERENT TONES")
    print("-"*40)
    
    # Create temporary directory for test results
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Test original tone evaluation
        print("\nTesting original tone evaluation...")
        original_csv = os.path.join(temp_dir, "test_original.csv")
        original_df = evaluate_and_save_csv(
            task=task,
            model=model,
            format=format_type,
            out_csv=original_csv,
            max_items=max_items,
            question_tone=QuestionTone.ORIGINAL.value
        )
        print(f"✓ Original evaluation completed: {len(original_df)} rows")
        
        # Test neutral tone evaluation
        print("\nTesting neutral tone evaluation...")
        neutral_csv = os.path.join(temp_dir, "test_neutral.csv")
        neutral_df = evaluate_and_save_csv(
            task=task,
            model=model,
            format=format_type,
            out_csv=neutral_csv,
            max_items=max_items,
            question_tone=QuestionTone.NEUTRAL.value
        )
        print(f"✓ Neutral evaluation completed: {len(neutral_df)} rows")
        
        # Test worried tone evaluation
        print("\nTesting worried tone evaluation...")
        worried_csv = os.path.join(temp_dir, "test_worried.csv")
        worried_df = evaluate_and_save_csv(
            task=task,
            model=model,
            format=format_type,
            out_csv=worried_csv,
            max_items=max_items,
            question_tone=QuestionTone.WORRIED.value
        )
        print(f"✓ Worried evaluation completed: {len(worried_df)} rows")
        
    except Exception as e:
        print(f"✗ Error during evaluations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Analyze results
    print("\n" + "-"*40)
    print("STEP 4: ANALYZING RESULTS")
    print("-"*40)
    
    try:
        # Check CSV files were created
        csv_files = [original_csv, neutral_csv, worried_csv]
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"✓ CSV file created: {os.path.basename(csv_file)}")
            else:
                print(f"✗ CSV file missing: {os.path.basename(csv_file)}")
        
        # Load and compare results
        print("\nComparing results...")
        
        # Check columns
        original_cols = set(original_df.columns)
        neutral_cols = set(neutral_df.columns)
        worried_cols = set(worried_df.columns)
        
        print(f"Original columns: {len(original_cols)}")
        print(f"Neutral columns: {len(neutral_cols)}")
        print(f"Worried columns: {len(worried_cols)}")
        
        # Check if question_tone column exists
        if "question_tone" in original_cols:
            print("✓ question_tone column found in original results")
        else:
            print("✗ question_tone column missing in original results")
        
        if "question_tone" in neutral_cols:
            print("✓ question_tone column found in neutral results")
        else:
            print("✗ question_tone column missing in neutral results")
        
        if "question_tone" in worried_cols:
            print("✓ question_tone column found in worried results")
        else:
            print("✗ question_tone column missing in worried results")
        
        # Check question_tone values
        if len(original_df) > 0:
            original_tone = original_df["question_tone"].iloc[0]
            print(f"Original tone value: {original_tone}")
        
        if len(neutral_df) > 0:
            neutral_tone = neutral_df["question_tone"].iloc[0]
            print(f"Neutral tone value: {neutral_tone}")
        
        if len(worried_df) > 0:
            worried_tone = worried_df["question_tone"].iloc[0]
            print(f"Worried tone value: {worried_tone}")
        
        # Check if questions are different
        if len(original_df) > 0 and len(neutral_df) > 0:
            orig_q = original_df["question"].iloc[0]
            neutral_q = neutral_df["question"].iloc[0]
            
            if orig_q != neutral_q:
                print("✓ Questions are different between original and neutral")
                print(f"Original: {orig_q[:100]}...")
                print(f"Neutral: {neutral_q[:100]}...")
            else:
                print("⚠ Questions are the same between original and neutral")
        
        if len(neutral_df) > 0 and len(worried_df) > 0:
            neutral_q = neutral_df["question"].iloc[0]
            worried_q = worried_df["question"].iloc[0]
            
            if neutral_q != worried_q:
                print("✓ Questions are different between neutral and worried")
                print(f"Neutral: {neutral_q[:100]}...")
                print(f"Worried: {worried_q[:100]}...")
            else:
                print("⚠ Questions are the same between neutral and worried")
        
    except Exception as e:
        print(f"✗ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Cleanup
    print("\n" + "-"*40)
    print("STEP 5: CLEANUP")
    print("-"*40)
    
    try:
        shutil.rmtree(temp_dir)
        print(f"✓ Temporary directory cleaned up: {temp_dir}")
    except Exception as e:
        print(f"⚠ Could not clean up temporary directory: {e}")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY! ✅")
    print("="*60)
    
    return True


def test_command_line_interface():
    """Test the command-line interface."""
    print("\n" + "="*60)
    print("TESTING COMMAND-LINE INTERFACE")
    print("="*60)
    
    # Test help message
    print("Testing help message...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "syco_eval.main", "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✓ Help message works")
            if "--tone" in result.stdout:
                print("✓ --tone argument found in help")
            else:
                print("✗ --tone argument missing from help")
        else:
            print(f"✗ Help command failed: {result.stderr}")
            
    except Exception as e:
        print(f"✗ Error testing CLI: {e}")


def main():
    """Run all integration tests."""
    print("Starting integration tests for tone-based evaluation pipeline...")
    print("Make sure you have your OpenAI API key set up!")
    
    try:
        # Test 1: Complete pipeline
        success = test_small_pipeline()
        
        if success:
            # Test 2: Command-line interface
            test_command_line_interface()
            
            print("\n" + "="*60)
            print("ALL INTEGRATION TESTS PASSED! 🎉")
            print("="*60)
            print("\nThe tone-based evaluation pipeline is working correctly!")
            print("You can now:")
            print("1. Generate transformed datasets")
            print("2. Run evaluations with different tones")
            print("3. Compare results between tones")
        else:
            print("\n" + "="*60)
            print("INTEGRATION TESTS FAILED! ❌")
            print("="*60)
            print("Please check the errors above and fix them before proceeding.")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
