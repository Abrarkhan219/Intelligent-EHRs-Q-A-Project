from rag import retrieve_top_k, generate_answer

query = "What are the key findings in this chest X-ray report?"
retrieved, scores = retrieve_top_k(query, k=3)

print("=== RETRIEVED DOCS ===")
for i, (doc, score) in enumerate(zip(retrieved, scores)):
    print(f"\n--- Doc {i+1} (score={score:.3f}) ---")
    print(doc[:250])

context = "\n\n".join(t[:250] for t in retrieved)
prompt = (
    "You are a radiology assistant. Read the clinical report context and answer "
    "the question in one short, factual sentence. If not mentioned, say "
    "\"Information not found in the provided records.\"\n\n"
    f"--- Context ---\n{context}\n---------------\n"
    f"Question: {query}\nAnswer:"
)
print("\n=== FULL PROMPT SENT TO MODEL ===")
print(prompt)
print(f"\nPrompt length (characters): {len(prompt)}")

answer = generate_answer(query, retrieved)
print("\n=== GENERATED ANSWER ===")
print(answer)
