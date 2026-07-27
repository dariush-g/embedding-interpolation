from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
import time
import random

# Track (Clean Score, Corrupted Score, Total Gen Time)
models = {
    "gpt2" : (0, 0, 0),
    "microsoft/phi-2" : (0, 0, 0)
}

phrases = {
    "jingle bells, jingle bells, jingle" : "all",
    "the early bird catches the" : "worm",
    "roses are red, violets are" : "blue",
    "kill two birds with one" : "stone",
    "why did the chicken cross the" : "road"
}

def build_corruptions(phrases: dict[str, str], seed: int = 0) -> dict[str, dict]:
    """Decide the corruption for each phrase ONCE, at the word level,
    so every model gets the exact same corrupted string."""
    rng = random.Random(seed)

    # word bank to draw replacements from: every word across all phrases
    bank = sorted({w for p in phrases for w in p.split()})

    corruptions = {}
    for phrase in phrases:
        words = phrase.split()
        idx = rng.randint(0, len(words) - 2)
        original_word = words[idx]

        # pick a replacement that isn't the same word
        replacement = rng.choice(bank)
        while replacement == original_word:
            replacement = rng.choice(bank)

        corrupted_words = words.copy()
        corrupted_words[idx] = replacement
        corrupted_phrase = " ".join(corrupted_words)

        corruptions[phrase] = {
            "corrupted_phrase": corrupted_phrase,
            "word_idx": idx,
            "original_word": original_word,
            "replacement": replacement,
        }
    return corruptions

def test_models(models: dict[str, tuple[int, int, float]], phrases: dict[str, str]):
    logging.set_verbosity_error()
    logging.disable_progress_bar()

    # pick all corruptions up front, shared across every model
    corruptions = build_corruptions(phrases, seed=0)

    for phrase, c in corruptions.items():
        print(f"Mutation for \"{phrase}\": "
              f"word {c['word_idx']} ['{c['original_word']}'] -> ['{c['replacement']}']")

    for model_name in models:
        clean_score = 0
        corrupt_score = 0
        total_gen_time = 0

        print(f"\n========================================\n{model_name}:\n========================================")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

        for phrase, expected in phrases.items():
            start_time = time.perf_counter()

            corrupted_phrase = corruptions[phrase]["corrupted_phrase"]

            inputs = tokenizer(phrase, return_tensors="pt")
            corrupted_inputs = tokenizer(corrupted_phrase, return_tensors="pt")

            with torch.no_grad():
                clean_outputs = model(**inputs)
                corrupt_outputs = model(**corrupted_inputs)

            clean_probs = torch.softmax(clean_outputs.logits[0, -1, :], dim=-1)
            clean_pred = tokenizer.decode(torch.argmax(clean_probs).item()).strip()
            clean_passed = clean_pred == expected
            if clean_passed:
                clean_score += 1

            corrupt_probs = torch.softmax(corrupt_outputs.logits[0, -1, :], dim=-1)
            corrupt_pred = tokenizer.decode(torch.argmax(corrupt_probs).item()).strip()
            corrupt_passed = corrupt_pred == expected
            if corrupt_passed:
                corrupt_score += 1

            total_gen_time += time.perf_counter() - start_time

            clean_color = "\033[6;37;42m" if clean_passed else "\033[6;37;41m"
            corrupt_color = "\033[6;37;42m" if corrupt_passed else "\033[6;37;41m"
            reset = "\033[0m"

            print(f"Phrase: \"{phrase}\"  ->  Corrupted: \"{corrupted_phrase}\"")
            print(f"  {clean_color}Clean{reset}     -> Chose: {clean_pred:<8} (Expected: {expected})")
            print(f"  {corrupt_color}Corrupted{reset} -> Chose: {corrupt_pred:<8} (Expected: {expected})\n")

        models[model_name] = (clean_score, corrupt_score, total_gen_time)

    print("\n========================================")
    print("FINAL RESULTS")
    print("========================================")
    for model_name, (c_score, corr_score, gen_time) in models.items():
        print(f"{model_name}:")
        print(f"  Clean Accuracy:     {c_score}/{len(phrases)}")
        print(f"  Corrupted Accuracy: {corr_score}/{len(phrases)}")
        print(f"  Total Runtime:      {gen_time:.2f} seconds")

if __name__ == "__main__":
    test_models(models, phrases)