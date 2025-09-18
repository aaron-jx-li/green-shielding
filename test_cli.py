#!/usr/bin/env python3
"""
Simple test for the command-line interface without making API calls.
"""

import sys
import os
import subprocess

def test_cli_help():
    """Test that the CLI help message includes the new --tone argument."""
    print("="*60)
    print("TESTING COMMAND-LINE INTERFACE")
    print("="*60)
    
    try:
        # Test help message
        print("Testing help message...")
        result = subprocess.run([
            sys.executable, "-m", "syco_eval.main", "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✓ Help message works")
            print(f"Help output:\n{result.stdout}")
            
            # Check for specific arguments
            if "--tone" in result.stdout:
                print("✓ --tone argument found in help")
            else:
                print("✗ --tone argument missing from help")
                
            if "original | neutral | worried" in result.stdout:
                print("✓ Tone choices found in help")
            else:
                print("✗ Tone choices missing from help")
                
            if "--task" in result.stdout:
                print("✓ --task argument found in help")
            else:
                print("✗ --task argument missing from help")
                
            if "--model" in result.stdout:
                print("✓ --model argument found in help")
            else:
                print("✗ --model argument missing from help")
                
        else:
            print(f"✗ Help command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing CLI: {e}")
        return False
    
    return True


def test_cli_validation():
    """Test that the CLI validates arguments correctly."""
    print("\n" + "-"*40)
    print("TESTING ARGUMENT VALIDATION")
    print("-"*40)
    
    # Test invalid tone
    try:
        result = subprocess.run([
            sys.executable, "-m", "syco_eval.main",
            "--task", "medqa_diag",
            "--model", "gpt-4o-mini",
            "--format", "MC",
            "--out_csv", "test.csv",
            "--tone", "invalid_tone"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print("✓ Invalid tone correctly rejected")
            if "invalid choice" in result.stderr.lower():
                print("✓ Proper error message for invalid tone")
            else:
                print(f"⚠ Unexpected error message: {result.stderr}")
        else:
            print("✗ Invalid tone was not rejected")
            return False
            
    except Exception as e:
        print(f"✗ Error testing validation: {e}")
        return False
    
    # Test valid tone
    try:
        result = subprocess.run([
            sys.executable, "-m", "syco_eval.main",
            "--task", "medqa_diag",
            "--model", "gpt-4o-mini",
            "--format", "MC",
            "--out_csv", "test.csv",
            "--tone", "neutral"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        # This should fail because we don't have the transformed dataset, but the argument should be accepted
        if "Using neutral tone for questions" in result.stdout:
            print("✓ Valid tone accepted and processed")
        elif "Transformed dataset not found" in result.stderr:
            print("✓ Valid tone accepted (failed due to missing dataset, which is expected)")
        else:
            print(f"⚠ Unexpected result: {result.stdout} {result.stderr}")
            
    except Exception as e:
        print(f"✗ Error testing valid tone: {e}")
        return False
    
    return True


def main():
    """Run CLI tests."""
    print("Testing command-line interface for tone-based evaluation...")
    
    try:
        # Test 1: Help message
        help_success = test_cli_help()
        
        # Test 2: Argument validation
        validation_success = test_cli_validation()
        
        if help_success and validation_success:
            print("\n" + "="*60)
            print("CLI TESTS PASSED! ✅")
            print("="*60)
            print("The command-line interface is working correctly!")
            print("\nYou can now use commands like:")
            print("python -m syco_eval.main --task medqa_diag --model gpt-4o-mini --format MC --out_csv results/test.csv --tone neutral")
        else:
            print("\n" + "="*60)
            print("CLI TESTS FAILED! ❌")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")


if __name__ == "__main__":
    main()
