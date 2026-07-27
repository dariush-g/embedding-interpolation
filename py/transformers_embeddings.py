
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)


inputs = tokenizer("jingle bells, jingle bells, jingle", return_tensors="pt")

token_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print(tokens)

for token in tokens:
    for c in token:
        print(c, ord(c))
    
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states 
# shape: (batch, sequence_length, hidden_dim)

# for i, layer in enumerate(hidden_states):
#     print(f"Layer {i}: {layer.shape} | mean activation: {layer.mean().item():.4f}")
    
    
logits = outputs.logits  
probs = torch.softmax(logits, dim=-1)

# probabilities for the last token (what comes next)
last_token_probs = probs[0, -1, :]

# top 5 predictions
top5 = torch.topk(last_token_probs, 5)
for prob, idx in zip(top5.values, top5.indices):
    print(f"{tokenizer.decode(idx)!r}: {prob.item():.4f}")

N = 8  # however many dimensions you want

# prints embeddings

print("\nLayer 0 embeddings (first N dimensions per token):")
for token, embedding in zip(tokens, hidden_states[0][0]):
    dims = embedding[:N].tolist()
    formatted = [f"{x:.3f}" for x in dims]
    print(f"{token!r}: {formatted}")