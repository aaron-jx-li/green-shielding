#!/usr/bin/env python3
"""
Test script for the extended analyze.py functions.
This script tests the new tone-specific analysis functions.
"""

import sys
import os
import tempfile
import shutil

path_to_syco_eval = os.path.join(os.path.dirname(__file__), 'syco_eval')
sys.path.append(path_to_syco_eval)

from syco_eval.analyze import (
    compute_sycophancy,
    compute_tone_sycophancy,
    analyze_tone_impact,
    generate_tone_report
)


def test_backward_compatibility():
    """Test that the original compute_sycophancy function still works."""
    print("="*60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("="*60)
    
    # Test with existing CSV file if available
    existing_csv = "./results/medqa_diag_gpt-4.1-mini_MC.csv"
    
    if os.path.exists(existing_csv):
        print(f"Testing with existing CSV: {existing_csv}")
        try:
            # This should work exactly as before
            compute_sycophancy(existing_csv, format="MC")
            print("✓ Original compute_sycophancy function works")
        except Exception as e:
            print(f"✗ Error with original function: {e}")
            return False
    else:
        print("⚠ No existing CSV found, skipping backward compatibility test")
    
    return True


def test_tone_analysis_functions():
    """Test the new tone analysis functions with sample data."""
    print("\n" + "="*60)
    print("TESTING TONE ANALYSIS FUNCTIONS")
    print("="*60)
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create sample CSV files with different tones
        print("Creating sample CSV files...")
        
        # Sample data structure
        sample_data = {
            "index": [0, 1, 2],
            "question": ["Sample question 1", "Sample question 2", "Sample question 3"],
            "options": ["A. Option1 B. Option2 C. Option3 D. Option4"] * 3,
            "solution": ["A", "B", "C"],
            "question_tone": ["original"] * 3,
            "default": ["A", "B", "C"],
            "alternative": ["B", "C", "D"],
            "default_correct": [True, True, False],
            "pred_a": ["A", "B", "C"],
            "correct_a": [True, False, True],
            "pred_b": ["B", "B", "C"],
            "correct_b": [False, True, True],
            "pred_c": ["A", "B", "C"],
            "correct_c": [True, False, True],
            "pred_d": ["A", "B", "D"],
            "correct_d": [True, False, False]
        }
        
        # Create original tone CSV
        import pandas as pd
        df_original = pd.DataFrame(sample_data)
        original_csv = os.path.join(temp_dir, "test_original.csv")
        df_original.to_csv(original_csv, index=False)
        
        # Create neutral tone CSV (same data, different tone)
        sample_data["question_tone"] = ["neutral"] * 3
        df_neutral = pd.DataFrame(sample_data)
        neutral_csv = os.path.join(temp_dir, "test_neutral.csv")
        df_neutral.to_csv(neutral_csv, index=False)
        
        # Create worried tone CSV (same data, different tone)
        sample_data["question_tone"] = ["worried"] * 3
        df_worried = pd.DataFrame(sample_data)
        worried_csv = os.path.join(temp_dir, "test_worried.csv")
        df_worried.to_csv(worried_csv, index=False)
        
        print("✓ Sample CSV files created")
        
        # Test compute_tone_sycophancy
        print("\nTesting compute_tone_sycophancy...")
        try:
            results = compute_tone_sycophancy(
                original_csv=original_csv,
                neutral_csv=neutral_csv,
                worried_csv=worried_csv,
                format="MC"
            )
            
            if results and "comparison" in results:
                print("✓ compute_tone_sycophancy works")
                print(f"  Results keys: {list(results.keys())}")
            else:
                print("✗ compute_tone_sycophancy returned empty results")
                return False
                
        except Exception as e:
            print(f"✗ Error in compute_tone_sycophancy: {e}")
            return False
        
        # Test analyze_tone_impact
        print("\nTesting analyze_tone_impact...")
        try:
            analysis = analyze_tone_impact(
                original_csv=original_csv,
                neutral_csv=neutral_csv,
                worried_csv=worried_csv,
                format="MC"
            )
            
            if analysis:
                print("✓ analyze_tone_impact works")
                print(f"  Analysis keys: {list(analysis.keys())}")
            else:
                print("✗ analyze_tone_impact returned empty results")
                return False
                
        except Exception as e:
            print(f"✗ Error in analyze_tone_impact: {e}")
            return False
        
        # Test generate_tone_report
        print("\nTesting generate_tone_report...")
        try:
            report_file = os.path.join(temp_dir, "test_report.txt")
            report_text = generate_tone_report(results, output_file=report_file)
            
            if report_text and len(report_text) > 100:
                print("✓ generate_tone_report works")
                print(f"  Report length: {len(report_text)} characters")
                
                if os.path.exists(report_file):
                    print(f"  Report file created: {report_file}")
                else:
                    print("⚠ Report file not created")
            else:
                print("✗ generate_tone_report returned empty report")
                return False
                
        except Exception as e:
            print(f"✗ Error in generate_tone_report: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in test setup: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print(f"\n✓ Temporary directory cleaned up: {temp_dir}")
        except Exception as e:
            print(f"⚠ Could not clean up temporary directory: {e}")


def test_with_existing_data():
    """Test with existing CSV files if available."""
    print("\n" + "="*60)
    print("TESTING WITH EXISTING DATA")
    print("="*60)
    
    # Look for existing CSV files
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print("⚠ No results directory found")
        return True
    
    # Find CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        print("⚠ No CSV files found in results directory")
        return True
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Test with the first CSV file
    test_csv = os.path.join(results_dir, csv_files[0])
    print(f"\nTesting with: {test_csv}")
    
    try:
        # Test original function
        print("Testing original compute_sycophancy...")
        compute_sycophancy(test_csv, format="MC")
        print("✓ Original function works with existing data")
        
    except Exception as e:
        print(f"✗ Error with existing data: {e}")
        return False
    
    return True


def main():
    """Run all analysis tests."""
    print("Testing extended analyze.py functions...")
    
    try:
        # Test 1: Backward compatibility
        compat_success = test_backward_compatibility()
        
        # Test 2: New tone analysis functions
        tone_success = test_tone_analysis_functions()
        
        # Test 3: With existing data
        existing_success = test_with_existing_data()
        
        if compat_success and tone_success and existing_success:
            print("\n" + "="*60)
            print("ALL ANALYSIS TESTS PASSED! ✅")
            print("="*60)
            print("The extended analyze.py functions are working correctly!")
            print("\nYou can now use:")
            print("- compute_tone_sycophancy() to compare tones")
            print("- analyze_tone_impact() for statistical analysis")
            print("- generate_tone_report() for comprehensive reports")
        else:
            print("\n" + "="*60)
            print("ANALYSIS TESTS FAILED! ❌")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
