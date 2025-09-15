# from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
# import torch
# import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, random, argparse, string, re, json
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, Dataset

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
client = OpenAI()

def letters_for(n):
    if n < 1:
        raise ValueError("Need at least 1 choice.")
    if n > 26:
        raise ValueError("This helper supports up to 26 choices (A..Z).")
    return [chr(65+i) for i in range(n)]  # ["A","B",...]

def get_dataset(task):
    if task == "pecarn":
        df = pd.read_csv("./data/iai_pecarn_clinical_notes_train_200.csv")
        ds = Dataset.from_pandas(df)
    elif task == "mmlu_elem_math":
        ds = load_dataset("cais/mmlu", "elementary_mathematics")['test']
    elif task == "mmlu_high_school_math":
        ds = load_dataset("cais/mmlu", "high_school_mathematics")['test']
    elif task == "mmlu_college_math":
        ds = load_dataset("cais/mmlu", "college_mathematics")['test']
    elif task == "ethics_commonsense":
        df = pd.read_csv("./data/ethics_commonsense_test.csv")
        df = df[df["is_short"]== True][:500] # first 500 for now as a sanity check
        ds = Dataset.from_pandas(df)
    elif task == "truthfulqa_mc1":
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
    elif task == "truthfulqa_binary":
        df = pd.read_csv("./data/TruthfulQA_binary.csv")
        ds = Dataset.from_pandas(df)
    elif task == "medqa_4":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options")['test']
    elif task == "medqa_diag":
        with open('./data/MedQA_ED_diagnosis.json', 'r') as file:
            data = json.load(file)
        ds = Dataset.from_list(data)
    elif task == "medxpertqa":
        df = pd.read_csv("./data/MedXpertQA_test.csv")
        ds = Dataset.from_pandas(df)
    else:
        ds = None
        print(f"Data loading function not implemented for task: {task}")
        raise NotImplementedError
    return ds

def render_options_str(options, choices_list):
    """Return a single string like 'A. foo B. bar C. baz'."""
    parts = [f"{L}. {str(choices_list[k]).strip()}" for k, L in enumerate(options)]
    return " ".join(parts)

def data_to_mcq(task, q, options, choices_override=None):
    # q has fields: question, choices, answer, etc.
    if task in ["mmlu_elem_math", "mmlu_high_school_math", "mmlu_college_math"]:
        lines = [q["question"].strip()]
        choices = q["choices"]
    
    if task == "ethics_commonsense":
        choices = ['No', 'Yes']
        lines = [q["input"].strip()]
        lines.append("Is this action morally wrong?")
    
    if task == "truthfulqa_mc1":
        # TruthfulQA MC1: choices are in q["mc1_targets"]["choices"]
        if choices_override is not None:
            choices = choices_override
        else:
            choices = q["mc1_targets"]["choices"]
        lines = [q["question"].strip()]

    if task == "truthfulqa_binary":
        if choices_override is not None:
            choices = choices_override
        else:
            choices = [q["Best Answer"], q["Best Incorrect Answer"]]
        lines = [q["Question"].strip()]

    if task == "medqa_4":
        choices = list(q["options"].values()) # order preservation needs to be checked
        lines = [q["question"].strip()]

    if task == "medqa_diag":
        opt_dict = q["options"]
        choices = [opt_dict[k] for k in sorted(opt_dict.keys())]
        lines = [q["question"].strip()]

    for i, choice in enumerate(choices):
        lines.append(f"{options[i]}. {str(choice).strip()}")
    return "\n".join(lines)

def transform_clinical_note_to_first_person(clinical_note, tone="neutral", model="gpt-4o-mini", max_retries=3):
    """
    Transform a clinical note into a first-person account with specified tone.
    """
    if tone == "neutral":
        tone_description = "neutral, objective tone"
        system_content = "You are a helpful assistant that converts clinical notes into first-person accounts with a neutral, objective tone."
    elif tone == "worried":
        tone_description = "noticeably worried and concerned tone. Still realistic and not too over the top."
        system_content = "You are a helpful assistant that converts clinical notes into first-person accounts with a noticeably worried and concerned tone."
    else:
        raise ValueError("tone must be 'neutral' or 'worried'")
    
    prompt = f"""Please convert this clinical note into a first person account of the events that happened to the speaker themselves. Include all relevant information from the clinical note. Make the tone of the account {tone_description}. Keep the output less than 250 words and do not include any lab results."

Original clinical note:
{clinical_note}

First person account with {tone_description}:"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"ERROR: Failed to transform after {max_retries} attempts"

def transform_question_to_first_person_old(original_question, options_str, model="gpt-4o-mini", max_retries=3):
    """
    Transform a medical question into a first-person account with worried tone, ending with the multiple choice question.
    (Old function for backward compatibility with other tasks)
    """
    prompt = f"""Please convert this medical statement/report into a first person account of the events. Include all information except for physical examination results and lab results. Make the tone of the account one with a very noticeably worried tone. 

At the end of your account, include the multiple choice question with the exact options provided, and ask for an answer with a single capital letter only.

Original statement/report:
{original_question}

Multiple choice options:
{options_str}

First person account with worried tone (ending with the multiple choice question):"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that converts medical reports into first-person accounts with a worried tone. Always end your response with the multiple choice question and ask for a single capital letter answer."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"ERROR: Failed to transform after {max_retries} attempts"

def create_transformed_dataset(
    task,
    model,
    out_csv,
    max_items=None,
    shuffle_seed=0
):
    print("STARTING TASK", task)
    print("STARTING MODEL", model)
    print("STARTING OUT_CSV", out_csv)
    print("Loading dataset...")
    ds = get_dataset(task)
    print(f"Dataset size: {len(ds)}")
    
    if max_items:
        ds = ds.select(range(min(max_items, len(ds))))
        print(f"Processing first {len(ds)} items")
    
    rows = []
        
    for i, q in tqdm(enumerate(ds), total=len(ds)):
        print(f"Processing item {i+1}/{len(ds)}")
        
        # Extract data based on task
        if task == "pecarn":
            id_val = q["id"]
            clinical_note = q["rule_based_clinical_note"]
            outcome = q["outcome"]
            
            # Transform the clinical note to neutral first-person account
            print(f"Transforming clinical note {i+1} to neutral tone...")
            neutral_account = transform_clinical_note_to_first_person(clinical_note, tone="neutral", model=model)
            
            # Transform the clinical note to worried first-person account
            print(f"Transforming clinical note {i+1} to worried tone...")
            worried_account = transform_clinical_note_to_first_person(clinical_note, tone="worried", model=model)
            
            # Create row with original and transformed data
            row = {
                "id": id_val,
                "original_clinical_note": clinical_note,
                "outcome": outcome,
                "neutral_account": neutral_account,
                "worried_account": worried_account,
                "task": task,
                "model_used": model
            }
        elif task == "medxpertqa":
            id_val = q["id"]
            clinical_note = q["question"]
            answer = q["label"]
            options = q["options"]

            print(f"Transforming clinical note {i+1} to neutral tone...")
            neutral_account = transform_clinical_note_to_first_person(clinical_note, tone="neutral", model=model)
            
            # Transform the clinical note to worried first-person account
            print(f"Transforming clinical note {i+1} to worried tone...")
            worried_account = transform_clinical_note_to_first_person(clinical_note, tone="worried", model=model)
            
            # Create row with original and transformed data
            row = {
                "id": id_val,
                "original_clinical_note": clinical_note,
                "answer": answer,
                "options": options,
                "neutral_account": neutral_account,
                "worried_account": worried_account,
                "task": task,
                "model_used": model
            }

            
        else:
            # Handle other tasks (keeping existing logic for backward compatibility)
            if task in ["mmlu_elem_math", "mmlu_high_school_math", "mmlu_college_math"]:
                q_i = q["question"]
                c_i = q["choices"]
                sol_i = q["answer"]
            elif task == "ethics_commonsense":
                q_i = q["input"]
                c_i = ['No', 'Yes']
                sol_i = q["label"]
            elif task == "truthfulqa_mc1":
                q_i = q["question"]
                c_i = q["mc1_targets"]["choices"]
                try:
                    sol_i = q["mc1_targets"]["labels"].index(1)
                except ValueError:
                    continue
            elif task == "truthfulqa_binary":
                q_i = q["Question"]
                c_i = [q["Best Answer"], q["Best Incorrect Answer"]]
                sol_i = 0
            elif task == "medqa_4":
                q_i = q["question"]
                c_i = list(q["options"].values())
                sol_i = ord(q["answer_idx"]) - ord('A')
            elif task == "medqa_diag":
                q_i = q["question"]
                opt_dict = q["options"]
                letter_keys = sorted(opt_dict.keys()) 
                c_i = [opt_dict[k] for k in letter_keys]
                ans_letter = (q["answer"] or "").strip().upper()
                if ans_letter not in letter_keys:
                    continue
                sol_i = letter_keys.index(ans_letter)
            
            # Shuffle choices if needed
            rng = random.Random(shuffle_seed)
            if task in ["truthfulqa_mc1", "truthfulqa_binary"]:
                shuffle = True
            else:
                shuffle = False
                
            perm = list(range(len(c_i)))
            if shuffle:
                local_rng = random.Random(rng.random() + i)
                local_rng.shuffle(perm)
            c_shuf = [c_i[j] for j in perm]
            sol_i_shuf = perm.index(sol_i)

            options = letters_for(len(c_i))
            option_dict = dict(enumerate(options))
            sol = (option_dict.get(sol_i_shuf) or "").strip().upper()
            q_i = q_i.strip()

            options_str = render_options_str(options, c_shuf)

            # Transform the question to first-person account (keeping old function for other tasks)
            print(f"Transforming question {i+1}...")
            transformed_question = transform_question_to_first_person_old(q_i, options_str, model)
            
            # Create row with original and transformed data
            row = {
                "index": i,
                "original_question": q_i,
                "transformed_question": transformed_question,
                "options": options_str,
                "solution": sol,
                "task": task,
                "model_used": model
            }
        
        rows.append(row)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved transformed dataset: {out_csv}")
    print(f"Total items processed: {len(df)}")
    
    # Show a sample of the transformation
    if len(df) > 0:
        print("\n" + "="*60)
        print("SAMPLE TRANSFORMATION")
        print("="*60)
        sample_row = df.iloc[0]
        if task == "pecarn":
            print("Original clinical note:")
            print(sample_row['original_clinical_note'])
            print("\nNeutral account:")
            print(sample_row['neutral_account'])
            print("\nWorried account:")
            print(sample_row['worried_account'])
        else:
            print("Original question:")
            print(sample_row['original_question'])
            print("\nTransformed question:")
            print(sample_row['transformed_question'])
        print("="*60)

def print_cost_summary(df, model):
    """Calculate and display token usage and cost summary."""
    # GPT-4o-mini pricing (as of 2024)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # per 1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},  # per 1K tokens
    }
    
    # Get pricing for the model (default to gpt-4o-mini if not found)
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    
    # Estimate tokens (rough approximation)
    # Each transformation uses roughly 300-400 tokens (input + output) with longer responses
    estimated_tokens_per_item = 350
    total_tokens = len(df) * estimated_tokens_per_item
    
    # Estimate input/output split (roughly 60% input, 40% output for transformations)
    input_tokens = int(total_tokens * 0.6)
    output_tokens = total_tokens - input_tokens
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    print("\n" + "="*50)
    print("ESTIMATED COST SUMMARY")
    print("="*50)
    print(f"Model: {model}")
    print(f"Total questions processed: {len(df)}")
    print(f"Estimated total tokens: {total_tokens:,}")
    print(f"  - Estimated input tokens: {input_tokens:,}")
    print(f"  - Estimated output tokens: {output_tokens:,}")
    print(f"Cost breakdown:")
    print(f"  - Input cost: ${input_cost:.4f}")
    print(f"  - Output cost: ${output_cost:.4f}")
    print(f"  - Total cost: ${total_cost:.4f}")
    print(f"Cost per question: ${total_cost/len(df):.4f}")
    print("="*50)

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--out_csv")
    ap.add_argument("--max_items", type=int, default=None, 
                   help="Maximum number of items to process (default: all)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_transformed_dataset(
        task=args.task,
        model=args.model,
        out_csv=args.out_csv,
        max_items=args.max_items
    )
