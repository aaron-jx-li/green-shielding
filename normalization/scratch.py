import numpy as np
import pandas as pd
import json

with open("./results/HCM-9k/eval_converted_gpt-4.1-mini_1.json", "r") as f:
    data = json.load(f)
    # data = data["per_sample"]

with open("./results/HCM-9k/out_converted_gpt-4.1-mini.json", "r") as f:
    data_out = json.load(f)
# num_p = 0
# num_h = 0

# for i, item in enumerate(data):
#     if item["judge_doctor_agreement"]["in_plausible_set"]:
#         num_p += 1
#     if item["judge_doctor_agreement"]["in_highly_likely_set"]:
#         num_h += 1

# print(f"num plausible: {num_p} / {len(data)}")
# print(f"num highly likely: {num_h} / {len(data)}")
print(len(data))
for i, item in enumerate(data["per_sample"]):
    item["input"] = data_out[i]["normalized_prompt"]
json.dump(data, open("./results/HCM-9k/eval_converted_gpt-4.1-mini.json", "w"), indent=2)