#!/usr/bin/env python3
"""
Test the question_transformer with your actual dataset structure.
This will help verify the integration works correctly.
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'syco_eval'))

from syco_eval.question_transformer import transform_question_batch


def test_with_medqa_data():
    """Test with actual medqa_diag data structure."""
    
    print("="*60)
    print("TESTING WITH MEDQA DATA STRUCTURE")
    print("="*60)
    
    # Load a small sample from your medqa_diag data
    try:
        with open('./data/MedQA_ED_diagnosis.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items from MedQA_ED_diagnosis.json")
        
        # Take first 2 items for testing
        sample_data = data[:2]
        
        print(f"\nTesting with {len(sample_data)} sample items...")
        
        # Show the structure of the data
        if sample_data:
            print(f"\nSample data structure:")
            print(f"Keys: {list(sample_data[0].keys())}")
            print(f"Sample question: {sample_data[0].get('question', 'N/A')[:100]}...")
        
        # Test neutral transformation
        print(f"\nTransforming to neutral tone...")
        neutral_results = transform_question_batch(
            sample_data, 
            tone="neutral", 
            model="gpt-4o-mini",
            delay=1.0
        )
        
        print(f"Neutral transformation completed: {len(neutral_results)} items")
        
        # Test worried transformation  
        print(f"\nTransforming to worried tone...")
        worried_results = transform_question_batch(
            sample_data, 
            tone="worried", 
            model="gpt-4o-mini",
            delay=1.0
        )
        
        print(f"Worried transformation completed: {len(worried_results)} items")
        
        # Display results
        for i, (neutral, worried) in enumerate(zip(neutral_results, worried_results)):
            print(f"\n--- Item {i+1} Results ---")
            print(f"Original: {neutral.get('question', 'N/A')[:150]}...")
            print(f"Neutral: {neutral.get('transformed_question_neutral', 'N/A')[:150]}...")
            print(f"Worried: {worried.get('transformed_question_worried', 'N/A')[:150]}...")
        
    except FileNotFoundError:
        print("MedQA_ED_diagnosis.json not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_with_medxpertqa_data():
    """Test with actual medxpertqa_diag data structure."""
    
    print("\n" + "="*60)
    print("TESTING WITH MEDXPERTQA DATA STRUCTURE")
    print("="*60)
    
    try:
        with open('./data/medxpertqa_diag.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items from medxpertqa_diag.json")
        
        # Take first 1 item for testing
        sample_data = data[:1]
        
        print(f"\nTesting with {len(sample_data)} sample items...")
        
        # Show the structure of the data
        if sample_data:
            print(f"\nSample data structure:")
            print(f"Keys: {list(sample_data[0].keys())}")
            print(f"Sample question: {sample_data[0].get('question_mc', 'N/A')[:100]}...")
        
        # Test neutral transformation
        print(f"\nTransforming to neutral tone...")
        neutral_results = transform_question_batch(
            sample_data, 
            tone="neutral", 
            model="gpt-4o-mini",
            delay=1.0
        )
        
        print(f"Neutral transformation completed: {len(neutral_results)} items")
        
        # Display results
        if neutral_results:
            result = neutral_results[0]
            print(f"\n--- Results ---")
            print(f"Original: {result.get('question_mc', 'N/A')[:150]}...")
            print(f"Neutral: {result.get('transformed_question_neutral', 'N/A')[:150]}...")
        
    except FileNotFoundError:
        print("medxpertqa_diag.json not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run tests with actual data."""
    print("Testing question_transformer with your actual dataset structure...")
    print("Make sure you have your OpenAI API key set up!")
    
    try:
        test_with_medqa_data()
        test_with_medxpertqa_data()
        
        print("\n" + "="*60)
        print("DATA STRUCTURE TESTS COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
