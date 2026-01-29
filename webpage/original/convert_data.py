#!/usr/bin/env python3
"""
Convert CSV data to JSON format for the annotation webpage.
"""

import pandas as pd
import json
import sys
import os

def convert_csv_to_json(csv_path, output_path):
    """Convert CSV file to JSON format for the annotation interface."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Clean up the data
        df = df.dropna(subset=['question', 'default_response', 'truth'])
        
        # Convert to list of dictionaries
        data = []
        for _, row in df.iterrows():
            case = {
                'question': str(row['question']).strip(),
                'default_response': str(row['default_response']).strip(),
                'truth': str(row['truth']).strip(),
                'judge_wq5_dec': str(row.get('judge_wq5_dec', '')).strip()
            }
            data.append(case)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(data)} cases to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        return False

if __name__ == "__main__":
    # Default paths
    csv_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding2/green-shielding/results/judge/ak_review_round0/conf_correct_overall.csv"
    output_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding2/green-shielding/webpage/data.json"
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Convert the data
    success = convert_csv_to_json(csv_path, output_path)
    
    if success:
        print("Data conversion completed successfully!")
    else:
        print("Data conversion failed!")
        sys.exit(1)
