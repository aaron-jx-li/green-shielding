#!/usr/bin/env python3
"""
Test script for the question_transformer module.
This script tests the transformation functions with sample data from your datasets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'syco_eval'))

from syco_eval.question_transformer import (
    transform_to_first_person_neutral,
    transform_to_first_person_worried,
    transform_question_batch,
    validate_transformation
)
from syco_eval.data_utils import get_dataset
import json


def test_single_transformations():
    """Test individual transformation functions with sample questions."""
    print("="*60)
    print("TESTING SINGLE TRANSFORMATIONS")
    print("="*60)
    
    # Sample questions for testing
    test_questions = [
        "A 45-year-old male presents with chest pain and shortness of breath. He has a history of hypertension and smoking. Physical examination reveals elevated blood pressure and irregular heart rhythm.",
        "A 30-year-old female comes to the emergency department with severe abdominal pain, nausea, and vomiting. She reports the pain started suddenly 2 hours ago and is getting worse.",
        "A 65-year-old patient with diabetes presents with a non-healing foot ulcer. The ulcer has been present for 3 weeks and shows signs of infection."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test Question {i} ---")
        print(f"Original: {question}")
        
        # Test neutral transformation
        print("\nTransforming to neutral tone...")
        neutral_result = transform_to_first_person_neutral(question, model="gpt-4o-mini")
        print(f"Neutral: {neutral_result}")
        
        # Test worried transformation
        print("\nTransforming to worried tone...")
        worried_result = transform_to_first_person_worried(question, model="gpt-4o-mini")
        print(f"Worried: {worried_result}")
        
        # Validate transformations
        neutral_validation = validate_transformation(question, neutral_result, "neutral")
        worried_validation = validate_transformation(question, worried_result, "worried")
        
        print(f"\nNeutral validation: {neutral_validation}")
        print(f"Worried validation: {worried_validation}")
        
        print("-" * 40)


def test_batch_transformation():
    """Test batch transformation with a small sample from your datasets."""
    print("\n" + "="*60)
    print("TESTING BATCH TRANSFORMATION")
    print("="*60)
    
    # Load a small sample from medqa_diag dataset
    try:
        print("Loading medqa_diag dataset...")
        dataset = get_dataset("medqa_diag")
        print(f"Dataset loaded with {len(dataset)} items")
        
        # Take first 2 items for testing
        sample_questions = dataset.select(range(2))
        print(f"Testing with {len(sample_questions)} sample questions")
        
        # Convert to list of dicts
        questions_list = [dict(item) for item in sample_questions]
        
        # Test neutral batch transformation
        print("\nTesting neutral batch transformation...")
        neutral_results = transform_question_batch(
            questions_list, 
            tone="neutral", 
            model="gpt-4o-mini",
            delay=1.0  # Slower for testing
        )
        
        print(f"Neutral batch results: {len(neutral_results)} items processed")
        
        # Test worried batch transformation
        print("\nTesting worried batch transformation...")
        worried_results = transform_question_batch(
            questions_list, 
            tone="worried", 
            model="gpt-4o-mini",
            delay=1.0  # Slower for testing
        )
        
        print(f"Worried batch results: {len(worried_results)} items processed")
        
        # Display results
        for i, (neutral, worried) in enumerate(zip(neutral_results, worried_results)):
            print(f"\n--- Batch Item {i+1} ---")
            print(f"Original question: {neutral.get('question', 'N/A')}")
            print(f"Neutral transformed: {neutral.get('transformed_question_neutral', 'N/A')[:200]}...")
            print(f"Worried transformed: {worried.get('transformed_question_worried', 'N/A')[:200]}...")
        
    except Exception as e:
        print(f"Error testing batch transformation: {e}")
        print("This might be due to missing dataset files or API issues")


def test_medxpertqa_sample():
    """Test with medxpertqa dataset if available."""
    print("\n" + "="*60)
    print("TESTING MEDXPERTQA SAMPLE")
    print("="*60)
    
    try:
        print("Loading medxpertqa_diag dataset...")
        dataset = get_dataset("medxpertqa_diag")
        print(f"Dataset loaded with {len(dataset)} items")
        
        # Take first 1 item for testing
        sample_question = dataset.select(range(1))[0]
        print(f"Sample question: {sample_question.get('question_mc', 'N/A')[:100]}...")
        
        # Test transformations
        question_text = sample_question.get('question_mc', '')
        if question_text:
            print("\nTransforming to neutral...")
            neutral_result = transform_to_first_person_neutral(question_text)
            print(f"Neutral: {neutral_result[:200]}...")
            
            print("\nTransforming to worried...")
            worried_result = transform_to_first_person_worried(question_text)
            print(f"Worried: {worried_result[:200]}...")
        
    except Exception as e:
        print(f"Error testing medxpertqa: {e}")


def main():
    """Run all tests."""
    print("Starting question_transformer tests...")
    print("Make sure you have your OpenAI API key set up!")
    
    try:
        # Test 1: Single transformations
        test_single_transformations()
        
        # Test 2: Batch transformation
        test_batch_transformation()
        
        # Test 3: MedXpertQA sample
        test_medxpertqa_sample()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
