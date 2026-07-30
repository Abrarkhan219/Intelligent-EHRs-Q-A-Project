import pandas as pd
from rag import retrieve_top_k

cleaned = pd.read_csv("data/cleaned/indiana_reports_cleaned.csv")
problems_by_uid = dict(zip(cleaned["uid"], cleaned["Problems"].fillna("").str.lower()))

val = pd.read_csv("data/validation_questions.csv")

CONDITION_KEYWORDS = {
    "Is there any cardiomegaly?": "cardiomegaly",
    "Is there any effusion?": "effusion",
    "Is there any opacity?": "opacity",
    "Is there any pneumothorax?": "pneumothorax",
    "Is there any edema?": "edema",
    "Is there any atelectasis?": "atelectasis",
    "Is the heart size normal?": "normal",
}

results = []
for i, row in val.iterrows():
    q, source_uid = row["question"], row["uid"]

    # what condition SHOULD a topically relevant document mention?
    if q in CONDITION_KEYWORDS:
        target_keyword = CONDITION_KEYWORDS[q]
    else:
        # generic questions: relevance = matching this report's own normal/abnormal status
        source_problems = problems_by_uid.get(source_uid, "")
        target_keyword = "normal" if source_problems.strip() == "normal" else None

    retrieved, scores = retrieve_top_k(q, k=3)

    if target_keyword:
        topical_hit = any(target_keyword in r.lower() for r in retrieved)
    else:
        # abnormal-report case: hit if retrieved doc is NOT a "normal" boilerplate report
        topical_hit = any("normal" not in r.lower()[:60] for r in retrieved)

    results.append({"question": q, "uid": source_uid, "topical_hit": topical_hit})
    if (i + 1) % 1000 == 0:
        print(f"[{i+1}/{len(val)}]")

out = pd.DataFrame(results)
out.to_csv("results/retrieval_topical_hitrate.csv", index=False)
print("\nOverall topical hit-rate@3:", out["topical_hit"].mean())
print("\nBy question type:")
print(out.groupby("question")["topical_hit"].agg(["mean", "count"]))