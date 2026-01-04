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
BIG_FILE = "./results/HCM-9k_legacy/out_converted_gpt-4.1-mini.json"         # contains "input", "output", "model_response"
SMALL_FILE = "./results/HCM-3k/responses_gpt-4.1-mini_new.json"     # contains "raw_input", "original_output"
OUT_FILE = "./results/HCM-3k/responses_gpt-4.1-mini_new.json"

# ---- load files ----
with open(BIG_FILE, "r") as f:
    big_data = json.load(f)

with open(SMALL_FILE, "r") as f:
    small_data = json.load(f)

# ---- build lookup from big file ----
lookup = {}
for item in big_data:
    key = (
        normalize_text(item.get("normalized_prompt", "")),
        normalize_text(item.get("original_output", "")),
    )
    lookup[key] = item.get("raw_input")

# ---- attach model_response to small file ----
num_matched = 0
num_missing = 0
num_diff = 0

for item in tqdm(small_data):
    key = (
        normalize_text(item.get("normalized_prompt", "")),
        normalize_text(item.get("original_output", "")),
    )
    raw_input = lookup.get(key)

    if raw_input is not None:
        item["raw_input"] = raw_input
        num_matched += 1
    else:
        # item["model_response_converted"] = None  # or leave it out if you prefer
        num_missing += 1
    if raw_input != item.get("raw_input"):
        num_diff += 1
# ---- save result ----
with open(OUT_FILE, "w") as f:
    json.dump(small_data, f, indent=2)

print(f"Done.")
print(f"Matched: {num_matched}")
print(f"Missing: {num_missing}")
print(f"Differences in raw_input: {num_diff}")
print(f"Output written to: {OUT_FILE}")
