import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def build_embeddings(clean_csv, out_emb_path, out_text_path):
    df = pd.read_csv(clean_csv)
    if 'combined_text' not in df.columns:
        raise KeyError("Column 'combined_text' not found. Run preprocess.py first.")

    texts = df['combined_text'].astype(str).tolist()
    print(f"Encoding {len(texts)} documents with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    os.makedirs(os.path.dirname(out_emb_path), exist_ok=True)
    np.save(out_emb_path, embeddings)
    with open(out_text_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    print(f"Saved embeddings {embeddings.shape} -> {out_emb_path}")
    print(f"Saved corpus texts -> {out_text_path}")

if __name__ == "__main__":
    build_embeddings(
        "data/cleaned/indiana_reports_cleaned.csv",
        "models/embeddings.npy",
        "models/texts.txt",
    )