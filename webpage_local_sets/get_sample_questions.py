"""
This script is used to get a sample of questions from the json file.
"""
import os
import json
import pandas as pd
import random

# ===== Paths =====
data_path = "../results/HCM-3k/eval_converted_gpt-4.1-mini.json"
data_path = "../results/HCM-3k/merged_truth_new.json"
dest_path = "annotation_manager/data/sampled_data_HCM-3k.json"
json_data = json.load(open(data_path)) # has keys "per_sample" and "summary"
import pdb; pdb.set_trace()
# json_data[0]['ground_truth_space_majority'].keys()
# dict_keys(['plausible_set', 'highly_likely_set', 'cannot_miss_set', 'highly_likely_evidence', 'cannot_miss_evidence'])
sample_data = json_data # has keys 'index', 'has_pxhx', 'input', 'model_response', 'reference_diagnosis', 'judge_dx_space', 'metrics'

# randomly pick 150 samples from the sample_data
random.seed(1)
sampled_samples = random.sample(sample_data, 150)

# add a new index entry to every entry in the sample_data
for i in range(len(sample_data)):
    sample_data[i]['index'] = i

# save down as new json file
new_json_data = {'per_sample': sampled_samples}
with open(dest_path, 'w') as f:
    json.dump(new_json_data, f)


