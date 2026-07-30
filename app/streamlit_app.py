import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
from rag import retrieve_top_k, generate_answer

st.set_page_config(page_title="Intelligent EHR QA", page_icon="🩺", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
    <style>
        .main { background-color: #f7f9fb; }
        .title-text {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1a3c5e;
            margin-bottom: 0;
        }
        .subtitle-text {
            color: #5a6b7a;
            font-size: 0.95rem;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }
        .answer-box {
            background-color: #eaf4ff;
            border-left: 5px solid #2b7de9;
            padding: 1.1rem 1.3rem;
            border-radius: 8px;
            font-size: 1.05rem;
            color: #16324f;
            margin-bottom: 1.5rem;
        }
        .doc-card {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .doc-score {
            display: inline-block;
            background-color: #2b7de9;
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .stButton button {
            background-color: #2b7de9;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5rem 1.6rem;
            border: none;
        }
        .stButton button:hover { background-color: #1e63c4; }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="title-text">🩺 Intelligent EHR Question Answering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Semantic Similarity (SBERT) + Retrieval-Augmented Generation '
    '(FAISS + Flan-T5-Small) over the Indiana Chest X-ray reports.</div>',
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Top-K retrieved documents", 1, 10, 3)
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Abrar Khan & Muhammad Ibrar — FYP Project")

# ---------- Input ----------
query = st.text_input("💬 Enter a question about the chest X-ray report corpus:")
go = st.button("Get Answer")

# ---------- Output ----------
if go and query.strip():
    with st.spinner("Retrieving relevant reports and generating an answer..."):
        retrieved, scores = retrieve_top_k(query, k=k)
        answer = generate_answer(query, retrieved)

    st.subheader("Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    st.subheader("Retrieved Context")
    for i, (doc, score) in enumerate(zip(retrieved, scores)):
        st.markdown(f"""
            <div class="doc-card">
                <span class="doc-score">Doc {i+1} · similarity {score:.3f}</span>
                <p style="margin-top:8px; margin-bottom:0; color:#334;">{doc[:400]}{'...' if len(doc) > 400 else ''}</p>
            </div>
        """, unsafe_allow_html=True)
elif go:
    st.warning("Please enter a question first.")