import os
import re
import pandas as pd
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rag import generate_answer

smooth = SmoothingFunction().method1

def token_metrics(generated, reference):
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

def bleu(generated, reference):
    g, r = str(generated).split(), str(reference).split()
    if not g or not r:
        return 0.0
    return float(sentence_bleu([r], g, weights=(0.5, 0.5), smoothing_function=smooth))

cleaned = pd.read_csv("data/cleaned/indiana_reports_cleaned.csv")
text_by_uid = dict(zip(cleaned["uid"], cleaned["combined_text"]))

val = pd.read_csv("data/validation_questions.csv")

results = []
for i, row in val.iterrows():
    q, gold, uid = row["question"], row["gold_answer"], row["uid"]
    own_report = text_by_uid.get(uid, "")
    gen = generate_answer(q, [own_report])   # oracle: this row's OWN report as context
    m = token_metrics(gen, gold)
    b = bleu(gen, gold)
    results.append({"question": q, "generated": gen, "gold_answer": gold, **m, "bleu": b})
    if (i + 1) % 200 == 0:
        print(f"[{i+1}/{len(val)}]")
    if (i + 1) % 200 == 0 or (i + 1) == len(val):
        pd.DataFrame(results).to_csv("results/evaluation_oracle_partial.csv", index=False)

out = pd.DataFrame(results)
out.to_csv("results/evaluation_oracle_results.csv", index=False)
print(f"\n=== ORACLE-CONTEXT GENERATION RESULTS (n={len(out)}, real per-row variation) ===")
print(out[["precision", "recall", "f1", "bleu"]].describe())