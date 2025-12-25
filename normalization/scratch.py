import numpy as np
import pandas as pd
import json

with open("./results/HCM_ref-9k_new/judge_sets_gpt-4.1.json", "r") as f:
    data = json.load(f)

num_p = 0
num_h = 0

for i, item in enumerate(data):
    if item["judge_doctor_agreement"]["in_plausible_set"]:
        num_p += 1
    if item["judge_doctor_agreement"]["in_highly_likely_set"]:
        num_h += 1

print(f"num plausible: {num_p} / {len(data)}")
print(f"num highly likely: {num_h} / {len(data)}")