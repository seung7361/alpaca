import pandas as pd
from transformers import GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset

import json

output = []
dataset = load_dataset("HuggingFaceH4/no_robots")['train']
for i in range(len(dataset)):
    if len(dataset[i]['messages']) > 2:
        print(dataset[i]['messages'])