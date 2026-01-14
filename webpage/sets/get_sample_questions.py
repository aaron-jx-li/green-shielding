"""
This script is used to get a sample of questions from the json file.
"""
import os
import json
import pandas as pd
import random

# ===== Paths =====
data_path = "//Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding3/green-shielding/results/HCM-3k/eval_converted_gpt-4.1-mini.json"
dest_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding3/green-shielding/webpage_local_sets/annotation_manager/data/sampled_data_HCM-3k.json"
json_data = json.load(open(data_path)) # has keys "per_sample" and "summary"
sample_data = json_data['per_sample'] # has keys 'index', 'has_pxhx', 'input', 'model_response', 'reference_diagnosis', 'judge_dx_space', 'metrics'
ex_dx_space=sample_data[0]['judge_dx_space'] # has keys 'plausible_set', 'highly_likely_set'
import pdb; pdb.set_trace()
# randomly pick 150 samples from the sample_data
random.seed(1)
sampled_samples = random.sample(sample_data, 150)

# save down as new json file
new_json_data = {'per_sample': sampled_samples}
with open(dest_path, 'w') as f:
    json.dump(new_json_data, f)
