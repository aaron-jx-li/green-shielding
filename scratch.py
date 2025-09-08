import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, random, argparse, string, re, json
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, Dataset

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)

print(response.output_text)