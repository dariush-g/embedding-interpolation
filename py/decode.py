import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_ids = [[73, 17697, 30987,    11,   474, 17697, 30987,    11,   474, 17697]]

for id in input_ids[0]:
    decoded = tokenizer.decode(id)
    print(f"{id}: '{decoded}'")
