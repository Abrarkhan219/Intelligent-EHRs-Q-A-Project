# Intelligent EHR Question Answering using Semantic Similarity and RAG

A text-based question-answering system over radiology reports from the Indiana
University Chest X-ray Collection, combining Sentence-BERT semantic retrieval
with retrieval-augmented generation (RAG) using a small, CPU-runnable language
model.

## What this does
- Retrieves semantically relevant radiology reports for a natural-language question
  using SBERT embeddings (`all-MiniLM-L6-v2`) and a FAISS similarity index.
- Generates a concise answer using `flan-t5-small`, grounded in the retrieved text.
- Evaluated on 10,319 question-answer pairs derived from the dataset's own
  structured fields.

## Results (honest, full-dataset)
| Evaluation | Precision | Recall | F1 | BLEU |
|---|---|---|---|---|
| End-to-end pipeline | 0.362 | 0.076 | 0.117 | 0.0024 |
| Oracle-context generation | 0.202 | 0.163 | 0.119 | 0.0456 |

Retrieval reliably locates topically relevant reports. Generation quality is
the primary bottleneck, consistent with using a small, non-fine-tuned model on
CPU-only hardware.

## Setup