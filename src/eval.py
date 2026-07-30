import os
import re
import pandas as pd
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rag import retrieve_top_k, generate_answer

smooth = SmoothingFunction().method1

def token_metrics(generated: str, reference: str) -> dict:
    gen = re.findall(r"\w+", str(generated).lower())
    ref = re.findall(r"\w+", str(reference).lower())
    if not gen or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    gen_c, ref_c = Counter(gen), Counter(ref)
    overlap = sum(min(c, ref_c.get(t, 0)) for t, c in gen_c.items())
    p = overlap / len(gen)
    r = overlap / len(ref)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}

def bleu(generated: str, reference: str) -> float:
    g, r = str(generated).split(), str(reference).split()
    if not g or not r:
        return 0.0
    return float(sentence_bleu([r], g, weights=(0.5, 0.5), smoothing_function=smooth))

def evaluate(validation_csv: str, top_k: int = 3, sample_n: int = None, checkpoint_every: int = 100):
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv(validation_csv)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)

    results = []
    total = len(df)
    for i, row in df.iterrows():
        q, gold = row['question'], row['gold_answer']
        retrieved, scores = retrieve_top_k(q, k=top_k)
        gen = generate_answer(q, retrieved)
        m = token_metrics(gen, gold)
        b = bleu(gen, gold)
        results.append({"question": q, "generated": gen, "gold_answer": gold, **m, "bleu": b})

        if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
            pd.DataFrame(results).to_csv("results/evaluation_results_partial.csv", index=False)
            print(f"[{i+1}/{total}] checkpointed.")

    out = pd.DataFrame(results)
    out.to_csv("results/evaluation_results.csv", index=False)
    print(f"\nDone: {len(out)}/{total} rows evaluated.")
    print(out[["precision", "recall", "f1", "bleu"]].mean())
    return out

if __name__ == "__main__":
    evaluate("data/validation_questions.csv", top_k=3, sample_n=None)