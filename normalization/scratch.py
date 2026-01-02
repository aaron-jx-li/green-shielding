import numpy as np
import pandas as pd
import json

# with open("./results/HCM-9k/eval_converted_gpt-4.1-mini_1.json", "r") as f:
#     data = json.load(f)
#     # data = data["per_sample"]

# with open("./results/HCM-9k/out_converted_gpt-4.1-mini.json", "r") as f:
#     data_out = json.load(f)

# with open("./results/HCM-9k/judge_sets_gpt-4.1.json", "r") as f:
#     data = json.load(f)

# total_p = 0
# total_h = 0
# for i, item in enumerate(data):
#     dx = item["judge_dx_space"]
#     total_p += len(dx["plausible_set"])
#     total_h += len(dx["highly_likely_set"])
# print("avg plausible diagnoses:", total_p / len(data))
# print("avg highly likely diagnoses:", total_h / len(data))

with open("./data/HCM-9k_explicit.json", "r") as f:
    data = json.load(f)
print(len(data))
high_conf = [
    x for x in data
    if x.get("_explicit_dx_ask", {}).get("confidence") == 5
]

print("Confidence = 5:", len(high_conf))

with open("./data/HCM-9k_explicit_conf5.json", "w") as f:
    json.dump(high_conf, f, indent=2, ensure_ascii=False)
