import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_name = "result_"
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", pad_token="</s>")
model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()

while True:
    sentence = input("Input: ")
    sentence = f"Human: {sentence}\nAssistant: "
    input_ids = tokenizer.encode(sentence, return_tensors='pt').cuda()
    output = model.generate(input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0], skip_special_tokens=False))
