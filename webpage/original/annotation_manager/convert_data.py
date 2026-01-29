#!/usr/bin/env python3
"""
This script assumes that there arer files in a different folder with 
different category questions and their metadata. The script will then 
concatenate all the questions and their metadata into a single csv file.

The script will then also convert the concatenated data into a JSON file 
that can be used by the annotation interface.
"""

import pandas as pd
import csv
import numpy as np
import json
import sys
import csv
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
    ## assuming that the you are in webpage_local/annotation_manager foldeer
    sampled_qs_path = "../../results/judge/ak_review_round0"
    out_qs_path = "../../webpage_local/annotation_manager/ak_review_round0"
    
    ## these are the basenames of the files that contain the questions and their metadata
    sampled_qs_basenames = ['conf_correct_overall', 'conf_incorrect_overall', 'leniency_ordered_model', 'unordered_other']
    
    ## this is a dictionary that maps the basename of the file to the category of the questions
    bn_to_cat = {'conf_correct_overall': 1, 'conf_incorrect_overall': 2, 'leniency_ordered_model':3, 'unordered_other': 4}
    
    ## this is the overall dataframe that will contain all the questions and their metadata
    overall_data = pd.DataFrame()
    ## this is the loop that will concatenate all the questions and their metadata into a single dataframe
    for sampled_qs_basename in sampled_qs_basenames:
        current_fn = os.path.join(sampled_qs_path, f"{sampled_qs_basename}.csv")
        current_data = pd.read_csv(current_fn)
        current_data['category'] = bn_to_cat[sampled_qs_basename]
        current_data['expert_dec'] = -1 # -1 no answer, positive number describes some category
        overall_data = pd.concat([overall_data, current_data])

    ## this is the dataframe that will contain all the questions and their metadata
    overall_data = overall_data.drop(columns =["Unnamed: 0"])
    overall_data['Index'] = np.arange(len(overall_data))
    overall_data[['expert_dec', 'to_be_seen', 'judge_wq5_dec','category', 'Index']].to_csv(\
        os.path.join(out_qs_path, f"all_questions_metadata.csv"), index=False, quoting=csv.QUOTE_ALL)

    text_data = overall_data[['question', 'default_response', 'truth','Index']]
    text_data['Index'] = text_data['Index'].astype(int)
    with open(os.path.join(out_qs_path, f"all_questions.json"), 'w', encoding='utf-8') as f:
            json.dump(text_data.to_dict(orient='records'), f, indent=2, ensure_ascii=False)
