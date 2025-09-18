import sys
import os
import pandas as pd
# open /Users/sancheznicolas/Documents/Research/GreenTeam/green-shielding/results/transformed/medqa_diag_worried_open-ended.csv
df = pd.read_csv("/Users/sancheznicolas/Documents/Research/GreenTeam/green-shielding/results/transformed/medqa_diag_worried_open-ended.csv")

# grab the first row
first_row = df.iloc[0]
for index_name, value in first_row.items():
    print(index_name, value)