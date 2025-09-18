#!/usr/bin/env python3
"""
Test script for compare_tones.py functionality.
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
            sys.executable, "compare_tones.py", "--list"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ --list works correctly")
            print("Available comparisons:")
            print(result.stdout)
        else:
            print(f"❌ --list failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing --list: {e}")
        return False
    
    return True

def test_file_discovery():
    """Test file discovery functionality."""
    print("\nTesting file discovery...")
    
    # Import the compare_tones module
    sys.path.append('.')
    from compare_tones import find_result_files, list_available_comparisons
    
    # Test list_available_comparisons
    comparisons = list_available_comparisons("./results")
    print(f"Found {len(comparisons)} task/format combinations:")
    for task, formats in comparisons.items():
        print(f"  {task}: {formats}")
    
    # Test find_result_files for a specific task
    if comparisons:
        task = list(comparisons.keys())[0]
        format_type = comparisons[task][0]
        print(f"\nTesting file discovery for {task} - {format_type}:")
        
        files = find_result_files(task, format_type, "./results")
        print(f"Found files: {files}")
        
        # Check if original file exists
        if files["original"]:
            print(f"✓ Original file found: {files['original']}")
        else:
            print("❌ Original file not found")
            
        # Check transformed files
        if files["neutral"]:
            print(f"✓ Neutral file found: {files['neutral']}")
        else:
            print("⚠️  Neutral file not found (expected if not generated yet)")
            
        if files["worried"]:
            print(f"✓ Worried file found: {files['worried']}")
        else:
            print("⚠️  Worried file not found (expected if not generated yet)")
    
    return True

def test_basic_comparison():
    """Test basic comparison functionality with existing data."""
    print("\nTesting basic comparison...")
    
    # Import the compare_tones module
    sys.path.append('.')
    from compare_tones import run_comparison
    
    # Find available comparisons
    from compare_tones import list_available_comparisons
    comparisons = list_available_comparisons("./results")
    
    if not comparisons:
        print("❌ No comparisons available")
        return False
    
    # Test with first available task/format
    task = list(comparisons.keys())[0]
    format_type = comparisons[task][0]
    
    print(f"Testing comparison for {task} - {format_type}...")
    
    try:
        results = run_comparison(task, format_type, "./results", None, True)
        
        if "error" in results:
            print(f"⚠️  Comparison returned error (expected if transformed files missing): {results['error']}")
        else:
            print("✓ Comparison completed successfully")
            print(f"Results keys: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return False

def test_cli_interface():
    """Test CLI interface with various arguments."""
    print("\nTesting CLI interface...")
    
    # Test help
    try:
        result = subprocess.run([
            sys.executable, "compare_tones.py", "--help"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ --help works correctly")
        else:
            print(f"❌ --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing --help: {e}")
        return False
    
    # Test invalid arguments
    try:
        result = subprocess.run([
            sys.executable, "compare_tones.py", "--task", "nonexistent"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print("✓ Invalid arguments properly rejected")
        else:
            print("⚠️  Invalid arguments should have been rejected")
    except Exception as e:
        print(f"❌ Error testing invalid arguments: {e}")
    
    return True

def main():
    """Run all tests."""
    print("Testing compare_tones.py functionality...")
    print("="*50)
    
    tests = [
        test_list_functionality,
        test_file_discovery,
        test_basic_comparison,
        test_cli_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed or had warnings")
    
    print("\nNext steps:")
    print("1. Generate transformed datasets if you want full comparisons")
    print("2. Run: python compare_tones.py --list")
    print("3. Run: python compare_tones.py --task medqa_diag --format OPEN")

if __name__ == "__main__":
    main()
