
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch

import time

models = { "gpt2" : (0, 0), "microsoft/phi-2" : (0, 0) }
models = { "gpt2" : (0, 0) }

phrases = { 
        "jingle bells, jingle bells, jingle" : "all", 
        "the early bird catches the" : "worm",
        "roses are red, violets are" : "blue",
        "kill two birds with one" : "stone" 
        }

phrases = {
    "I love pizza, my favorite topping is the" : "ketchup"
}

phrases = {
    "I love icecream, my favorite flavor is" : "vanilla"
}

phrases = {
    "I love school because it's very" : "boring"
}

def test_models(models: dict[str, tuple[int, int]], phrases: dict[str, str]):    
    logging.set_verbosity_error() 
    logging.disable_progress_bar()

    for model_name in models:
        score = 0
        gen_time = 0
        print(model_name + ":")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        
        for (phrase, expected)in phrases.items():
            start_time = time.perf_counter()

            inputs = tokenizer(phrase, return_tensors="pt")

            token_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
                
            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states 
            # print(hidden_states)
            # shape: (batch, sequence_length, hidden_dim)

            logits = outputs.logits  
            probs = torch.softmax(logits, dim=-1)

            last_token_probs = probs[0, -1, :]

            # top 5 predictions
            largest_value = ""
            max = 0

            top5 = torch.topk(last_token_probs, 5)
            for prob, idx in zip(top5.values, top5.indices):
                # print(f"{tokenizer.decode(idx)!r}: {prob.item():.4f}")
                if prob.item() > max:
                    largest_value = tokenizer.decode(idx)
                    max = prob.item()
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            gen_time += execution_time
                    
            passed = largest_value.strip() == expected
            if passed:
                score += 1
            color = "\033[6;37;42m" if passed else "\033[6;37;41m"
            status = color + ("passed" if passed else "failed")
            print(f"{status}: {phrase:<36} chose: {largest_value.strip():<8} expected: {expected}\033[0m")
            
        print(f"score: {score}")
        models[model_name] = (score, gen_time)

    for model_name, (score, gen_time) in models.items():
        print(f"{model_name}: {score} passed in {gen_time:.2f} seconds")

if __name__ == "__main__":
    test_models(models, phrases)