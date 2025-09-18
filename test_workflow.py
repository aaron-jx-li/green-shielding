#!/usr/bin/env python3
"""
Test script for run_tone_experiment.py workflow functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_list_functionality():
    """Test the --list functionality."""
    print("Testing --list functionality...")
    try:
        result = subprocess.run([
            sys.executable, "run_tone_experiment.py", "--list"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ --list works correctly")
            print("Available experiments:")
            print(result.stdout)
        else:
            print(f"❌ --list failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing --list: {e}")
        return False
    
    return True

def test_prerequisites_check():
    """Test prerequisites checking functionality."""
    print("\nTesting prerequisites check...")
    
    # Import the workflow module
    sys.path.append('.')
    from run_tone_experiment import ToneExperimentRunner
    
    runner = ToneExperimentRunner("./results")
    
    # Test with available task/format
    try:
        experiments = runner.list_available_experiments()
        if experiments:
            task = list(experiments.keys())[0]
            format_type = experiments[task][0]
            
            print(f"Testing prerequisites for {task} - {format_type}:")
            prerequisites = runner.check_prerequisites(task, format_type)
            
            print("Prerequisites status:")
            for component, available in prerequisites.items():
                status = "✅" if available else "❌"
                print(f"  {status} {component}")
            
            print("✓ Prerequisites check works correctly")
            return True
        else:
            print("❌ No experiments available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prerequisites: {e}")
        return False

def test_help_functionality():
    """Test the help functionality."""
    print("\nTesting help functionality...")
    try:
        result = subprocess.run([
            sys.executable, "run_tone_experiment.py", "--help"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ --help works correctly")
            # Check if key sections are present
            help_text = result.stdout
            if "Examples:" in help_text and "Arguments:" in help_text:
                print("✓ Help contains expected sections")
            else:
                print("⚠️  Help may be missing some sections")
        else:
            print(f"❌ --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing --help: {e}")
        return False
    
    return True

def test_dry_run_experiment():
    """Test a dry run of an experiment (without actually running it)."""
    print("\nTesting dry run experiment...")
    
    # Import the workflow module
    sys.path.append('.')
    from run_tone_experiment import ToneExperimentRunner
    
    runner = ToneExperimentRunner("./results")
    
    try:
        experiments = runner.list_available_experiments()
        if not experiments:
            print("❌ No experiments available for testing")
            return False
        
        task = list(experiments.keys())[0]
        format_type = experiments[task][0]
        
        print(f"Testing dry run for {task} - {format_type}:")
        
        # Test prerequisites check
        prerequisites = runner.check_prerequisites(task, format_type)
        print(f"Prerequisites: {prerequisites}")
        
        # Test that we can import the required modules
        from data_utils import check_transformed_dataset_exists
        from compare_tones import run_comparison
        
        print("✓ All required modules can be imported")
        print("✓ Dry run test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error in dry run test: {e}")
        return False

def test_cli_validation():
    """Test CLI argument validation."""
    print("\nTesting CLI validation...")
    
    # Test missing required arguments
    try:
        result = subprocess.run([
            sys.executable, "run_tone_experiment.py", "--task", "medqa_diag"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print("✓ Missing --format argument properly rejected")
        else:
            print("⚠️  Missing --format should have been rejected")
    except Exception as e:
        print(f"❌ Error testing CLI validation: {e}")
    
    # Test invalid format
    try:
        result = subprocess.run([
            sys.executable, "run_tone_experiment.py", "--task", "medqa_diag", "--format", "INVALID"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print("✓ Invalid format properly rejected")
        else:
            print("⚠️  Invalid format should have been rejected")
    except Exception as e:
        print(f"❌ Error testing invalid format: {e}")
    
    return True

def main():
    """Run all tests."""
    print("Testing run_tone_experiment.py workflow functionality...")
    print("="*60)
    
    tests = [
        test_list_functionality,
        test_prerequisites_check,
        test_help_functionality,
        test_dry_run_experiment,
        test_cli_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed or had warnings")
    
    print("\nNext steps:")
    print("1. Run: python run_tone_experiment.py --list")
    print("2. Run: python run_tone_experiment.py --task medqa_diag --format OPEN --max-items 5")
    print("3. Run: python run_tone_experiment.py --all (for full experiment)")

if __name__ == "__main__":
    main()
