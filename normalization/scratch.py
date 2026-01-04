import json
import re
from tqdm import tqdm

def normalize_text(s: str) -> str:
    """Normalize text to improve matching robustness."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ---- paths ----
BIG_FILE = "./results/HCM_ref-9k/converted_gpt-4.1-mini.json"         # contains "input", "output", "model_response"
SMALL_FILE = "./results/HCM-3k/responses_gpt-4.1-mini.json"     # contains "raw_input", "original_output"
OUT_FILE = "./results/HCM-3k/responses_gpt-4.1-mini.json"

# ---- load files ----
with open(BIG_FILE, "r") as f:
    big_data = json.load(f)

with open(SMALL_FILE, "r") as f:
    small_data = json.load(f)

# ---- build lookup from big file ----
lookup = {}
for item in big_data:
    key = (
        normalize_text(item.get("raw_input", "")),
        normalize_text(item.get("original_output", "")),
    )
    lookup[key] = item.get("model_response")

# ---- attach model_response to small file ----
num_matched = 0
num_missing = 0

for item in tqdm(small_data):
    key = (
        normalize_text(item.get("raw_input", "")),
        normalize_text(item.get("original_output", "")),
    )
    model_resp = lookup.get(key)

    if model_resp is not None:
        item["model_response_converted"] = model_resp
        num_matched += 1
    else:
        item["model_response_converted"] = None  # or leave it out if you prefer
        num_missing += 1

# ---- save result ----
with open(OUT_FILE, "w") as f:
    json.dump(small_data, f, indent=2)

print(f"Done.")
print(f"Matched: {num_matched}")
print(f"Missing: {num_missing}")
print(f"Output written to: {OUT_FILE}")
