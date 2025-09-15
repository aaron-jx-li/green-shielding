from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from datasets import Dataset, load_dataset

ALPHABET = [chr(65 + i) for i in range(26)]  # ["A".."Z"]


def letters_for(n: int) -> List[str]:
    if n < 1:
        raise ValueError("Need at least 1 choice.")
    if n > 26:
        raise ValueError("This helper supports up to 26 choices (A..Z).")
    return ALPHABET[:n]


def sanitize_options(
    opt_dict: Dict[str, Any], answer_letter: Optional[str]
) -> Tuple[Optional[List[str]], Optional[int]]:
    """Keep only non-empty options, return (choices, solution_index)."""
    ordered_keys = sorted(opt_dict.keys())
    clean_keys = [
        k for k in ordered_keys if isinstance(opt_dict[k], str) and opt_dict[k].strip() != ""
    ]
    if not clean_keys:
        return None, None
    choices = [opt_dict[k].strip() for k in clean_keys]
    al = (answer_letter or "").strip().upper()
    if al not in clean_keys:
        return None, None
    sol_idx = clean_keys.index(al)
    return choices, sol_idx


def _read_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def get_dataset(task: str) -> Dataset:
    if task == "mmlu_elem_math":
        return load_dataset("cais/mmlu", "elementary_mathematics")["test"]
    if task == "mmlu_high_school_math":
        return load_dataset("cais/mmlu", "high_school_mathematics")["test"]
    if task == "mmlu_college_math":
        return load_dataset("cais/mmlu", "college_mathematics")["test"]
    if task == "ethics_commonsense":
        df = pd.read_csv("./data/ethics_commonsense_test.csv")
        df = df[df["is_short"] == True][:500]  # noqa: E712
        return Dataset.from_pandas(df)
    if task == "truthfulqa_mc1":
        return load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    if task == "truthfulqa_binary":
        df = pd.read_csv("./data/TruthfulQA_binary.csv")
        return Dataset.from_pandas(df)
    if task == "medqa_4":
        return load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
    if task == "medqa_diag":
        data = _read_json_list("./data/MedQA_ED_diagnosis.json")
        return Dataset.from_list(data)
    if task == "medxpertqa_diag":
        data = _read_json_list("./data/medxpertqa_diag.json")
        return Dataset.from_list(data)
    raise NotImplementedError(f"Data loading function not implemented for task: {task}")


def get_transformed_dataset_path(task: str, tone: str) -> str:
    """
    Get the file path for a transformed dataset.
    
    Args:
        task: The dataset task name (e.g., "medqa_diag", "medxpertqa_diag")
        tone: The tone ("neutral", "worried", "original")
        
    Returns:
        Path to the transformed dataset file
    """
    if tone == "original":
        # For original, use the regular dataset loading
        return None
    
    # Create transformed results directory if it doesn't exist
    transformed_dir = "./results/transformed"
    os.makedirs(transformed_dir, exist_ok=True)
    
    filename = f"{task}_{tone}.json"
    return os.path.join(transformed_dir, filename)


def get_transformed_dataset(task: str, tone: str) -> Dataset:
    """
    Load a transformed dataset with the specified tone.
    
    Args:
        task: The dataset task name (e.g., "medqa_diag", "medxpertqa_diag")
        tone: The tone ("neutral", "worried", "original")
        
    Returns:
        Dataset with transformed questions
        
    Raises:
        FileNotFoundError: If the transformed dataset doesn't exist
        NotImplementedError: If the task is not supported
    """
    if tone == "original":
        # For original tone, use the regular dataset loading
        return get_dataset(task)
    
    # Get the path to the transformed dataset
    transformed_path = get_transformed_dataset_path(task, tone)
    
    if not os.path.exists(transformed_path):
        raise FileNotFoundError(
            f"Transformed dataset not found: {transformed_path}\n"
            f"Please generate it first using the question_transformer module."
        )
    
    # Load the transformed dataset
    data = _read_json_list(transformed_path)
    return Dataset.from_list(data)


def save_transformed_dataset(data: List[Dict[str, Any]], task: str, tone: str) -> str:
    """
    Save a transformed dataset to a JSON file.
    
    Args:
        data: List of transformed question dictionaries
        task: The dataset task name (e.g., "medqa_diag", "medxpertqa_diag")
        tone: The tone ("neutral", "worried")
        
    Returns:
        Path to the saved file
    """
    if tone == "original":
        raise ValueError("Cannot save 'original' tone - use the original dataset files")
    
    # Get the path to save the transformed dataset
    transformed_path = get_transformed_dataset_path(task, tone)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
    
    # Save the data
    with open(transformed_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved transformed dataset: {transformed_path} ({len(data)} items)")
    return transformed_path


def generate_transformed_dataset(
    task: str, 
    tone: str, 
    model: str = "gpt-4o-mini",
    max_items: Optional[int] = None,
    delay: float = 0.5
) -> str:
    """
    Generate and save a transformed dataset using the question_transformer module.
    
    Args:
        task: The dataset task name (e.g., "medqa_diag", "medxpertqa_diag")
        tone: The tone ("neutral", "worried")
        model: The LLM model to use for transformation
        max_items: Maximum number of items to process (None for all)
        delay: Delay between API calls to avoid rate limiting
        
    Returns:
        Path to the saved transformed dataset file
    """
    if tone == "original":
        raise ValueError("Cannot generate 'original' tone - use the original dataset")
    
    # Import here to avoid circular imports
    from question_transformer import transform_question_batch
    
    print(f"Generating {tone} transformed dataset for {task}...")
    
    # Load the original dataset
    original_dataset = get_dataset(task)
    
    # Limit items if specified
    if max_items is not None:
        original_dataset = original_dataset.select(range(min(max_items, len(original_dataset))))
        print(f"Processing first {len(original_dataset)} items")
    
    # Convert to list of dicts
    questions_list = [dict(item) for item in original_dataset]
    
    # Transform the questions
    transformed_data = transform_question_batch(
        questions_list,
        tone=tone,
        model=model,
        delay=delay
    )
    
    # Save the transformed dataset
    saved_path = save_transformed_dataset(transformed_data, task, tone)
    
    return saved_path


def list_available_transformed_datasets() -> Dict[str, List[str]]:
    """
    List all available transformed datasets.
    
    Returns:
        Dictionary mapping task names to lists of available tones
    """
    transformed_dir = "./results/transformed"
    
    if not os.path.exists(transformed_dir):
        return {}
    
    available = {}
    
    for filename in os.listdir(transformed_dir):
        if filename.endswith(".json"):
            # Parse filename: task_tone.json
            parts = filename[:-5].split("_")  # Remove .json and split
            if len(parts) >= 2:
                tone = parts[-1]  # Last part is tone
                task = "_".join(parts[:-1])  # Everything else is task
                
                if task not in available:
                    available[task] = []
                available[task].append(tone)
    
    return available


def check_transformed_dataset_exists(task: str, tone: str) -> bool:
    """
    Check if a transformed dataset exists.
    
    Args:
        task: The dataset task name
        tone: The tone ("neutral", "worried", "original")
        
    Returns:
        True if the dataset exists, False otherwise
    """
    if tone == "original":
        return True  # Original datasets always exist
    
    transformed_path = get_transformed_dataset_path(task, tone)
    return os.path.exists(transformed_path)
