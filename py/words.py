from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch

logging.set_verbosity_error()

repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
gguf_file = "Llama-3.2-3B-Instruct-Q8_0.gguf"   

tokenizer = AutoTokenizer.from_pretrained(repo, gguf_file=gguf_file)
model = AutoModelForCausalLM.from_pretrained(repo, gguf_file=gguf_file)

# add the new word to the vocabulary
new_word = "boogeyman"
tokenizer.add_tokens([new_word])
model.resize_token_embeddings(len(tokenizer))
token_id = tokenizer.convert_tokens_to_ids(new_word)

# do print statements to verify changed tokens

def first_embed(word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return model.get_input_embeddings().weight[ids[0]].clone()

with torch.no_grad():
    monster_vec = first_embed("monster")
    spooky_vec  = first_embed("spooky")
    person_vec  = first_embed("person")

    haunting_direction = spooky_vec - person_vec
    boogeyman_vector = monster_vec + (0.3 * haunting_direction) + (0.5 * person_vec)

    model.get_input_embeddings().weight[token_id] = boogeyman_vector

model.tie_weights()



prompt = f"What is a {new_word}?"
inputs = tokenizer(prompt, return_tensors="pt")

# 5. Generate text
print("Generating completion...\n")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,       
        do_sample=True,         
        top_k=50,                
        top_p=0.95,              
        temperature=0.8,          
        pad_token_id=tokenizer.eos_token_id
    )

# 6. Decode and view the results
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("--- Result ---")
print(generated_text)
