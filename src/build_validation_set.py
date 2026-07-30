import pandas as pd
import re
import os

CONDITIONS = ["cardiomegaly", "effusion", "opacity", "pneumothorax", "edema", "atelectasis"]

def clean(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\n", " ")).strip()


def build(cleaned_csv: str, outpath: str):
    df = pd.read_csv(cleaned_csv)
    rows = []

    for _, r in df.iterrows():
        problems = str(r.get('Problems', '')).lower()
        concise = clean(r['concise_answer'])
        if not concise:
            continue

        rows.append({
            "question": "What are the key findings in this chest X-ray report?",
            "gold_answer": concise,
            "uid": r['uid']
        })
        if problems.strip() == "normal":
            rows.append({
                "question": "What abnormality is present according to the report?",
                "gold_answer": "No abnormality is present. " + concise,
                "uid": r['uid']
            })
            rows.append({
                "question": "Is the heart size normal?",
                "gold_answer": "Yes, the heart size and mediastinal contour are within normal limits.",
                "uid": r['uid']
            })
        else:
            rows.append({
                "question": "What abnormality is present according to the report?",
                "gold_answer": concise,
                "uid": r['uid']
            })
            for cond in CONDITIONS:
                if cond in problems:
                    rows.append({
                        "question": f"Is there any {cond}?",
                        "gold_answer": concise,
                        "uid": r['uid']
                    })
    out_df = pd.DataFrame(rows).drop_duplicates(subset=['question', 'gold_answer', 'uid'])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    out_df.to_csv(outpath, index=False)
    print(f"Validation set built: {len(out_df)} pairs -> {outpath}")
    print(out_df['question'].value_counts())
    return out_df

if __name__ == "__main__":
    build("data/cleaned/indiana_reports_cleaned.csv", "data/validation_questions.csv")
        