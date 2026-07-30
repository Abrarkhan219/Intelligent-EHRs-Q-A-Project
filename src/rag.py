import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device} | embed_model={EMBED_MODEL_NAME} | gen_model={GEN_MODEL_NAME}")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
tokenizer.truncation_side = "left"
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)

with open("models/texts.txt", "r", encoding="utf-8") as f:
    _docs = [line.strip() for line in f.readlines() if line.strip()]

def retrieve_top_k(query: str, k: int = 3, index_path: str = "models/faiss.index"):
    with torch.no_grad():
        q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    index = faiss.read_index(index_path)

    over_fetch = min(k * 5, index.ntotal)   # zyada candidates mangao
    scores, ids = index.search(q_emb, over_fetch)

    seen_prefixes = set()
    retrieved, kept_scores = [], []
    for idx, score in zip(ids[0], scores[0]):
        text = _docs[idx]
        prefix = text[:80]
        if prefix in seen_prefixes:      # near-duplicate boilerplate skip karo
            continue
        seen_prefixes.add(prefix)
        retrieved.append(text)
        kept_scores.append(score)
        if len(retrieved) == k:
            break
    return retrieved, kept_scores

def generate_answer(query: str, retrieved_texts: list[str]) -> str:
    context = "\n\n".join(t[:250] for t in retrieved_texts)
    prompt = (
        "You are a radiology assistant. Read the clinical report context and answer "
        "the question in one short, factual sentence, IN YOUR OWN WORDS. "
        "Do not copy phrases verbatim from the context. If not mentioned, say "
        "\"Information not found in the provided records.\"\n\n"
        f"--- Context ---\n{context}\n---------------\n"
        f"Question: {query}\nAnswer:"
    )
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = gen_model.generate(
            **inputs, max_length=100, num_beams=2, repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer if answer else "Information not found in the provided records."
