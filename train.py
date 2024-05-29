import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import TextDataset
from datasets import load_dataset
from tqdm import tqdm

import pandas as pd
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("output/checkpoint-12000")

dataset = load_dataset("text", data_files="gpt4_final_dataset.json")['train']
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], return_tensors='pt', padding="longest", max_length=1024, truncation=True), batched=True, num_proc=4)

training_args = TrainingArguments(
    output_dir="./result",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=1500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=10,
    logging_first_step=True,
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("result_")