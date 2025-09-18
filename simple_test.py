#!/usr/bin/env python3
"""
Simple test for question_transformer module.
Run this to test basic functionality without using your datasets.
"""

import sys
import os

path_to_syco_eval = os.path.join(os.path.dirname(__file__), 'syco_eval')
print(f"Adding {path_to_syco_eval} to sys.path")
sys.path.append(path_to_syco_eval)

from syco_eval.question_transformer import (
    transform_to_first_person_neutral,
    transform_to_first_person_worried,
    validate_transformation
)


def simple_test():
    """Test with a simple medical question."""
    
    # Simple test question
    test_question = "A 45-year-old male presents with chest pain and shortness of breath. He has a history of hypertension and smoking."
    
    print("="*60)
    print("SIMPLE TRANSFORMATION TEST")
    print("="*60)
    print(f"Original question: {test_question}")
    
    print("\n" + "-"*40)
    print("Testing NEUTRAL transformation...")
    print("-"*40)
    
    try:
        neutral_result = transform_to_first_person_neutral(test_question, model="gpt-4o-mini")
        print(f"Neutral result: {neutral_result}")
        
        # Validate
        neutral_validation = validate_transformation(test_question, neutral_result, "neutral")
        print(f"Validation: {neutral_validation}")
        
    except Exception as e:
        print(f"Error in neutral transformation: {e}")
    
    print("\n" + "-"*40)
    print("Testing WORRIED transformation...")
    print("-"*40)
    
    try:
        worried_result = transform_to_first_person_worried(test_question, model="gpt-4o-mini")
        print(f"Worried result: {worried_result}")
        
        # Validate
        worried_validation = validate_transformation(test_question, worried_result, "worried")
        print(f"Validation: {worried_validation}")
        
    except Exception as e:
        print(f"Error in worried transformation: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    simple_test()
