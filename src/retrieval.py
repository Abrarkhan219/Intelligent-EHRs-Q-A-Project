import numpy as np
import faiss
import os

EMB_PATH = "models/embeddings.npy"
TEXT_PATH = "models/texts.txt"
INDEX_PATH = "models/faiss.index"

def build_index():
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError("Embeddings not found. Run embed.py first.")

    embeddings = np.load(EMB_PATH).astype("float32")
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]

    d = embeddings.shape[1]
    print(f"Embedding dimension: {d} | documents: {len(texts)}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index (cosine similarity, {index.ntotal} vectors, dim={d}) -> {INDEX_PATH}") 

if __name__ == "__main__":
    build_index()
