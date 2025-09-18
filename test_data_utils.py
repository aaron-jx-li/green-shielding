#!/usr/bin/env python3
"""
Test script for the extended data_utils module.
This script tests the new transformed dataset functions.
"""

import sys
import os

path_to_syco_eval = os.path.join(os.path.dirname(__file__), 'syco_eval')
sys.path.append(path_to_syco_eval)

from syco_eval.data_utils import (
    get_transformed_dataset_path,
    get_transformed_dataset,
    save_transformed_dataset,
    generate_transformed_dataset,
    list_available_transformed_datasets,
    check_transformed_dataset_exists
)


def test_path_functions():
    """Test path-related functions."""
    print("="*60)
    print("TESTING PATH FUNCTIONS")
    print("="*60)
    
    # Test path generation
    neutral_path = get_transformed_dataset_path("medqa_diag", "neutral")
    worried_path = get_transformed_dataset_path("medqa_diag", "worried")
    original_path = get_transformed_dataset_path("medqa_diag", "original")
    
    print(f"Neutral path: {neutral_path}")
    print(f"Worried path: {worried_path}")
    print(f"Original path: {original_path}")
    
    # Verify paths are in results directory
    if neutral_path and "results/transformed" in neutral_path:
        print("✓ Neutral path correctly points to results/transformed")
    if worried_path and "results/transformed" in worried_path:
        print("✓ Worried path correctly points to results/transformed")
    
    # Test existence checks
    print(f"\nExistence checks:")
    print(f"medqa_diag neutral exists: {check_transformed_dataset_exists('medqa_diag', 'neutral')}")
    print(f"medqa_diag worried exists: {check_transformed_dataset_exists('medqa_diag', 'worried')}")
    print(f"medqa_diag original exists: {check_transformed_dataset_exists('medqa_diag', 'original')}")


def test_list_available():
    """Test listing available transformed datasets."""
    print("\n" + "="*60)
    print("TESTING LIST AVAILABLE DATASETS")
    print("="*60)
    
    available = list_available_transformed_datasets()
    print(f"Available transformed datasets: {available}")
    
    if available:
        for task, tones in available.items():
            print(f"  {task}: {tones}")
    else:
        print("No transformed datasets found yet.")


def test_generate_small_dataset():
    """Test generating a small transformed dataset."""
    print("\n" + "="*60)
    print("TESTING DATASET GENERATION (SMALL SAMPLE)")
    print("="*60)
    
    try:
        # Generate a small neutral dataset for testing
        print("Generating neutral dataset for medqa_diag (2 items)...")
        neutral_path = generate_transformed_dataset(
            task="medqa_diag",
            tone="neutral",
            model="gpt-4o-mini",
            max_items=2,
            delay=1.0
        )
        print(f"Generated: {neutral_path}")
        
        # Check if it exists now
        exists = check_transformed_dataset_exists("medqa_diag", "neutral")
        print(f"Dataset now exists: {exists}")
        
        # List available datasets again
        available = list_available_transformed_datasets()
        print(f"Updated available datasets: {available}")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()


def test_load_transformed_dataset():
    """Test loading a transformed dataset."""
    print("\n" + "="*60)
    print("TESTING LOAD TRANSFORMED DATASET")
    print("="*60)
    
    try:
        # Try to load the neutral dataset we just generated
        if check_transformed_dataset_exists("medqa_diag", "neutral"):
            print("Loading neutral transformed dataset...")
            dataset = get_transformed_dataset("medqa_diag", "neutral")
            print(f"Loaded dataset with {len(dataset)} items")
            
            # Show sample data
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample keys: {list(sample.keys())}")
                print(f"Original question: {sample.get('question', 'N/A')[:100]}...")
                print(f"Transformed question: {sample.get('transformed_question_neutral', 'N/A')[:100]}...")
        else:
            print("No neutral dataset found to load")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


def test_original_dataset_loading():
    """Test that original dataset loading still works."""
    print("\n" + "="*60)
    print("TESTING ORIGINAL DATASET LOADING")
    print("="*60)
    
    try:
        # Test loading original dataset
        print("Loading original medqa_diag dataset...")
        from syco_eval.data_utils import get_dataset
        original_dataset = get_dataset("medqa_diag")
        print(f"Original dataset loaded with {len(original_dataset)} items")
        
        # Test loading via transformed function with "original" tone
        print("Loading via get_transformed_dataset with 'original' tone...")
        original_via_transformed = get_transformed_dataset("medqa_diag", "original")
        print(f"Original via transformed function: {len(original_via_transformed)} items")
        
        # They should be the same
        print(f"Datasets match: {len(original_dataset) == len(original_via_transformed)}")
        
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Testing extended data_utils functions...")
    print("Make sure you have your OpenAI API key set up!")
    
    try:
        # Test 1: Path functions
        test_path_functions()
        
        # Test 2: List available datasets
        test_list_available()
        
        # Test 3: Generate small dataset
        test_generate_small_dataset()
        
        # Test 4: Load transformed dataset
        test_load_transformed_dataset()
        
        # Test 5: Original dataset loading
        test_original_dataset_loading()
        
        print("\n" + "="*60)
        print("ALL DATA_UTILS TESTS COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
