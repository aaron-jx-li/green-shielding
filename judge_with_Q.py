from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
from tqdm import tqdm

# from syco_eval.data_utils import get_dataset, letters_for, sanitize_options
# from syco_eval.enums import QFormat, Template
from syco_eval.llm_utils import chat
# from syco_eval.prompt_builder import (
#     build_binary_messages_with_templates,
#     build_binary_prompt,
#     build_open_default_messages,
#     build_open_messages,
#     build_default_prompt,
#     build_sycophancy_mc_messages,
#     extract_letter,
#     render_options_str,
# )
from syco_eval.judge import judge
import pandas as pd

csv_name = "./results/medxpertqa_diag_gpt-5-mini_open-ended.csv"
df = pd.read_csv(csv_name)
# print(df.columns.tolist())
decisions = []
judge_raws = []
for i in tqdm(range(len(df))):
    q = df.iloc[i]['question']
    response = df.iloc[i]['default']
    truth = df.iloc[i]['solution_text']
    is_correct, judge_raw = judge(q, response, truth, model="gpt-4.1-mini", template='with_Q')
    decisions.append(is_correct)
    judge_raws.append(judge_raw)

new_df = pd.DataFrame({'question': df['question'], 'default_response': df['default'], 'truth': df['solution_text'], 'decision': decisions, 'judge_raw': judge_raws})
new_df.to_csv("./results/judge/medxpertqa_diag_gpt-5-mini_open-ended_with_Q_gpt-4.1-mini.csv", index=False)
