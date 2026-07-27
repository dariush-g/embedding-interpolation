from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
import time

models = { "gpt2" : (0, 0) }
phrases = {
    ("roses are red, violets are", 2, "yellow") : "blue",  # (phrase, index to morph, target word) : expected
}

logging.set_verbosity_error()
logging.disable_progress_bar()

def test_models(models: dict, phrases: dict):
    for model_name in models:
        score = 0
        gen_time = 0
        print(model_name + ":")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        embedding_layer = model.get_input_embeddings()

        def get_word_embedding(W: str, leading_space=True) -> torch.Tensor:
            text = (" " + W) if leading_space else W
            ids = tokenizer.encode(text)
            if len(ids) != 1:
                toks = tokenizer.convert_ids_to_tokens(ids)
                print(f"  warning: {W!r} -> {len(ids)} tokens {toks}, using first")
            return embedding_layer.weight[ids[0]]

        for ((phrase, changed_index, target_word), expected) in phrases.items():
            start_time = time.perf_counter()

            inputs = tokenizer(phrase, return_tensors="pt")
            attn = inputs["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            print("  tokens:", list(enumerate(tokens)))

            base_embeds = embedding_layer(inputs["input_ids"]).detach()
            original_vec = base_embeds[0, changed_index, :].clone()
            target_vec = get_word_embedding(target_word)
            
            flipped_at = None
            for alpha in torch.linspace(0, 1, 51):  # 0.00, 0.02, ... 1.00
                embeds = base_embeds.clone()
                lerped = (1 - alpha) * original_vec + alpha * target_vec
                embeds[0, changed_index, :] = lerped

                with torch.no_grad():
                    logits = model(inputs_embeds=embeds, attention_mask=attn).logits

                pred = tokenizer.decode(logits[0, -1].argmax()).strip()

                if alpha == 0 and pred != expected:
                    print(f"  note: fails at alpha=0 (predicts {pred!r}, expected {expected!r})")

                if pred != expected:
                    flipped_at = alpha.item()
                    became = pred
                    break
                
                probs = torch.softmax(logits, dim=-1)

                last_token_probs = probs[0, -1, :]
                
                top5 = torch.topk(last_token_probs, 5)
                for prob, idx in zip(top5.values, top5.indices):
                    # print(f"{tokenizer.decode(idx)!r}: {prob.item():.4f}")
                    if prob.item() > max:
                        largest_value = tokenizer.decode(idx)
                        max = prob.item()
            
                

            gen_time += time.perf_counter() - start_time

            if flipped_at is None:
                color = "\033[6;37;42m"
                print(f"{color}held: {phrase:<30} never flipped, stayed {expected!r}\033[0m")
                score += 1
            else:
                color = "\033[6;37;41m"
                print(f"{color}flip: {phrase:<30} broke at alpha={flipped_at:.2f}, "
                      f"{expected!r} -> {became!r}\033[0m")
                # print(f"{last}")

        print(f"score: {score}")
        models[model_name] = (score, gen_time)

    for model_name, (score, gen_time) in models.items():
        print(f"{model_name}: {score} passed in {gen_time:.2f} seconds")

if __name__ == "__main__":
    test_models(models, phrases)