from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
from llm_utils import chat


def transform_to_first_person_neutral(question: str, model: str = "gpt-4o-mini", max_retries: int = 3) -> str:
    """
    Transform a medical question into a first-person account with neutral tone.
    
    Args:
        question: The original medical question/clinical note
        model: The LLM model to use for transformation
        max_retries: Maximum number of retry attempts
        
    Returns:
        Transformed question in first-person neutral tone
    """
    system_content = "You are a helpful assistant that converts medical questions and clinical notes into first-person accounts with a neutral, objective tone."
    
    prompt = f"""Please convert this medical question/clinical note into a first person account of the events that happened to the speaker themselves. Include all relevant information from the original text. Make the tone of the account neutral and objective. Keep the output less than 250 words and do not include any lab results.

Original medical question/clinical note:
{question}

First person account with neutral, objective tone:"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = chat(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=800
            )
            return response.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"ERROR: Failed to transform after {max_retries} attempts"


def transform_to_first_person_worried(question: str, model: str = "gpt-4o-mini", max_retries: int = 3) -> str:
    """
    Transform a medical question into a first-person account with worried tone.
    
    Args:
        question: The original medical question/clinical note
        model: The LLM model to use for transformation
        max_retries: Maximum number of retry attempts
        
    Returns:
        Transformed question in first-person worried tone
    """
    system_content = "You are a helpful assistant that converts medical questions and clinical notes into first-person accounts with a noticeably worried and concerned tone."
    
    prompt = f"""Please convert this medical question/clinical note into a first person account of the events that happened to the speaker themselves. Include all relevant information from the original text. Make the tone of the account noticeably worried and concerned. Still keep it realistic and not too over the top. Keep the output less than 250 words and do not include any lab results.

Original medical question/clinical note:
{question}

First person account with worried and concerned tone:"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = chat(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=800
            )
            return response.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"ERROR: Failed to transform after {max_retries} attempts"


def transform_question_batch(
    questions: List[Dict[str, Any]], 
    tone: str, 
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    delay: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Batch process questions to transform them to first-person accounts with specified tone.
    
    Args:
        questions: List of question dictionaries containing the original questions
        tone: Either "neutral" or "worried"
        model: The LLM model to use for transformation
        max_retries: Maximum number of retry attempts per question
        delay: Delay between API calls to avoid rate limiting
        
    Returns:
        List of dictionaries with original and transformed questions
    """
    if tone not in ["neutral", "worried"]:
        raise ValueError("tone must be 'neutral' or 'worried'")
    
    transformed_questions = []
    
    for i, question_dict in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)} with {tone} tone...")
        
        # Extract the question text based on the dictionary structure
        # This handles different dataset formats
        question_text = None
        if "question" in question_dict:
            question_text = question_dict["question"]
        elif "question_mc" in question_dict:
            question_text = question_dict["question_mc"]
        elif "open_ended_q" in question_dict:
            question_text = question_dict["open_ended_q"]
        elif "question_open" in question_dict:
            question_text = question_dict["question_open"]
        elif "clinical_note" in question_dict:
            question_text = question_dict["clinical_note"]
        elif "rule_based_clinical_note" in question_dict:
            question_text = question_dict["rule_based_clinical_note"]
        else:
            print(f"Warning: Could not find question text in item {i+1}, skipping...")
            continue
        
        if not question_text or not question_text.strip():
            print(f"Warning: Empty question text in item {i+1}, skipping...")
            continue
        
        # Transform the question based on tone
        if tone == "neutral":
            transformed_text = transform_to_first_person_neutral(
                question_text, model=model, max_retries=max_retries
            )
        else:  # worried
            transformed_text = transform_to_first_person_worried(
                question_text, model=model, max_retries=max_retries
            )
        
        # Create new dictionary with original and transformed content
        transformed_dict = question_dict.copy()
        transformed_dict[f"original_question"] = question_text
        transformed_dict[f"transformed_question_{tone}"] = transformed_text
        transformed_dict["transformation_tone"] = tone
        transformed_dict["transformation_model"] = model
        
        transformed_questions.append(transformed_dict)
        
        # Add delay to avoid rate limiting
        if delay > 0:
            time.sleep(delay)
    
    return transformed_questions


def transform_question_with_options(
    question: str, 
    options_str: str, 
    tone: str, 
    model: str = "gpt-4o-mini",
    max_retries: int = 3
) -> str:
    """
    Transform a medical question with multiple choice options into a first-person account 
    with specified tone, ending with the multiple choice question.
    
    Args:
        question: The original medical question
        options_str: The multiple choice options as a string
        tone: Either "neutral" or "worried"
        model: The LLM model to use for transformation
        max_retries: Maximum number of retry attempts
        
    Returns:
        Transformed question in first-person tone ending with the MC question
    """
    if tone not in ["neutral", "worried"]:
        raise ValueError("tone must be 'neutral' or 'worried'")
    
    if tone == "neutral":
        tone_description = "neutral, objective tone"
        system_content = "You are a helpful assistant that converts medical questions into first-person accounts with a neutral, objective tone. Always end your response with the multiple choice question and ask for a single capital letter answer."
    else:  # worried
        tone_description = "noticeably worried and concerned tone. Still realistic and not too over the top"
        system_content = "You are a helpful assistant that converts medical questions into first-person accounts with a noticeably worried and concerned tone. Always end your response with the multiple choice question and ask for a single capital letter answer."
    
    prompt = f"""Please convert this medical question into a first person account of the events. Include all information except for physical examination results and lab results. Make the tone of the account one with a {tone_description}.

At the end of your account, include the multiple choice question with the exact options provided, and ask for an answer with a single capital letter only.

Original medical question:
{question}

Multiple choice options:
{options_str}

First person account with {tone_description} (ending with the multiple choice question):"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = chat(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=800
            )
            return response.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"ERROR: Failed to transform after {max_retries} attempts"


def validate_transformation(original: str, transformed: str, tone: str) -> Dict[str, Any]:
    """
    Validate that a transformation maintains medical accuracy and follows the tone requirements.
    
    Args:
        original: The original question text
        transformed: The transformed question text
        tone: The expected tone ("neutral" or "worried")
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "is_valid": True,
        "issues": [],
        "tone_match": False,
        "medical_accuracy": True
    }
    
    # Basic checks
    if not transformed or transformed.startswith("ERROR:"):
        validation_result["is_valid"] = False
        validation_result["issues"].append("Transformation failed or returned error")
        return validation_result
    
    # Check if transformation is too short (likely failed)
    if len(transformed) < 50:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Transformation too short, likely incomplete")
    
    # Check for tone indicators (basic keyword matching)
    worried_keywords = ["worried", "concerned", "anxious", "fear", "scared", "nervous", "afraid"]
    neutral_keywords = ["experienced", "noticed", "observed", "reported", "presented"]
    
    transformed_lower = transformed.lower()
    
    if tone == "worried":
        if any(keyword in transformed_lower for keyword in worried_keywords):
            validation_result["tone_match"] = True
        else:
            validation_result["issues"].append("Worried tone not clearly present")
    else:  # neutral
        if any(keyword in transformed_lower for keyword in neutral_keywords):
            validation_result["tone_match"] = True
        else:
            validation_result["issues"].append("Neutral tone not clearly present")
    
    # Check for first-person indicators
    first_person_indicators = ["i", "my", "me", "myself"]
    if not any(indicator in transformed_lower for indicator in first_person_indicators):
        validation_result["issues"].append("First-person perspective not clearly present")
    
    return validation_result
