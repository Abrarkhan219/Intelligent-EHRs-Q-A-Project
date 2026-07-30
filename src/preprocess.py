import pandas as pd
import re
import os

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).replace("\n", " ").strip()
    s = re.sub(r"\bXXXX\b", "", s)      
    return re.sub(r"\s+", " ", s).strip()


def preprocess_reports(inpath: str, outpath: str) -> pd.DataFrame:
    df = pd.read_csv(inpath)

    corpus_cols = ['findings', 'impression', 'indication', 'comparison']
    corpus_cols = [c for c in corpus_cols if c in df.columns]
    df['combined_text'] = df[corpus_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)


    df['concise_answer'] = df['impression'].fillna('').apply(clean_text)
    empty_mask = df['concise_answer'].str.strip() == ''
    df.loc[empty_mask, 'concise_answer'] = df.loc[empty_mask, 'findings'].fillna('').apply(clean_text)


    df = df[(df['combined_text'].str.strip() != '') & (df['concise_answer'].str.strip() != '')]

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"Saved cleaned file: {outpath} | rows: {len(df)}")
    return df


if __name__ == "__main__":
    preprocess_reports(
        "data/raw/indiana_reports.csv",
        "data/cleaned/indiana_reports_cleaned.csv"
    )

