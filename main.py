import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

import pandas as pd
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import deepspeed
deepspeed.init_distributed()

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="</s>")
model = AutoModelForCausalLM.from_pretrained("output/checkpoint-12000")

dataset = json.load(open("gpt4_final_dataset.json", "r"))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

print(dataset[0])

model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, model_parameters=model.parameters(), training_data=dataset, config="ds_config.json"
)

def train():
    for epoch in range(1):
        step = 0
        for data in tqdm(train_dataloader):
            model_engine.train()
            optimizer.zero_grad()

            input_ids = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=1024).input_ids.cuda()

            outputs = model(input_ids[:, :-1].contiguous()).logits
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), input_ids[:, 1:].contiguous().view(-1))

            model_engine.backward(loss)
            model_engine.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

            step += 1
            if step % 10000 == 0:
                model_engine.save_checkpoint(f"checkpoint.pt")

if __name__ == "__main__":
    train()