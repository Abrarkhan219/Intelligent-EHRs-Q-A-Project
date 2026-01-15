import sys
import os
# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import streamlit as st
import time
import re

from src.medical_api import get_gmeplus_data, recommend_medicine, find_nearby_store
from src.rag import retrieve_top_k, generate_answer
from src.serp_api import get_google_answer

# 🔊 Voice utilities
from src.voice_input_output import voice_to_text, text_to_speech   # <-- Corrected import


# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="Intelligent EHR QA System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS Styling -------------------
st.markdown("""
    <style>
        .main { background-color: #f8fafc; padding: 2rem; }
        h1 { color: #003366; text-align: center; font-family: 'Segoe UI', sans-serif; }
        .stButton button {
            background-color: #005b96;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
        }
        .stButton button:hover { background-color: #0074cc; }
        .response-box {
            background-color: #e6f2ff;
            border-left: 5px solid #0074cc;
            padding: 1rem;
            border-radius: 10px;
        }
        footer { text-align: center; color: #777; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Response Mode", ["Hybrid (Default)", "Dataset Only", "API Only"])
threshold = st.sidebar.slider("Dataset similarity threshold", 0.0, 1.0, 0.4)
k = st.sidebar.slider("Top K Documents", 1, 10, 3)
voice_reply = st.sidebar.checkbox("🔊 Voice Answer (Text → Speech)", value=True)

st.sidebar.info("Developed by Abrar Khan & Muhammad Ibrar — FYP Project")

# ------------------- Header -------------------
st.title("🧠 Intelligent EHR QA System")
st.markdown("""
### 🩺 Medical Question Answering using Semantic Similarity & RAG  
Ask any **patient-specific** or **general medical** question below.  
You can ask via **text OR voice**.
""")

# ------------------- Input Section -------------------
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("💬 Enter your question:")

with col2:
    if st.button("🎤 Speak"):
        with st.spinner("Listening..."):
            spoken_query = voice_to_text()
            if spoken_query:
                query = spoken_query
                st.success(f"Recognized: {query}")

# ------------------- API fallback function -------------------
def get_api_answer(query):
    try:
        keywords = re.findall(r"[A-Za-z]+", query)
        if not keywords:
            return "⚠️ Please enter a valid question."

        filtered = [w for w in keywords if len(w) > 3]
        topic = filtered[-1] if filtered else keywords[-1]
        topic = topic.title().replace(" ", "_")

        # Wikipedia first
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("extract"):
                return f"📘 Wikipedia:\n\n{data['extract']}"

        # Google fallback
        return get_google_answer(query)

    except Exception as e:
        return f"API Error: {e}"

# ------------------- Logger -------------------
def add_log(query, retrieved_docs, generated_answer):
    try:
        import json, datetime
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        with open(os.path.join(logs_dir, "answers.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.datetime.utcnow().isoformat(),
                "query": query,
                "answer": generated_answer
            }) + "\n")
    except:
        pass

# ------------------- Process Button -------------------
if st.button("🔍 Get Answer"):
    if not query or not query.strip():
        st.warning("Please enter or speak a question.")
    else:
        with st.spinner("Processing..."):
            retrieved, scores = retrieve_top_k(query, k=k)

            if mode != "API Only":
                st.markdown("### 📄 Retrieved Context")
                for i, doc in enumerate(retrieved):
                    st.write(f"**Doc {i+1} (score {scores[i]:.3f})**")
                    st.write(doc[:400] + "...")
                    st.divider()

            use_api = (mode == "API Only") or (not scores or max(scores) < threshold)

            if use_api:
                st.markdown("### 🌐 External API Answer")
                answer = get_api_answer(query)
            else:
                st.markdown("### 💡 Dataset-based Answer")
                answer = generate_answer(query, retrieved)

            # Typing animation
            placeholder = st.empty()
            typed = ""
            for ch in answer:
                typed += ch
                placeholder.markdown(f"<div class='response-box'>{typed}</div>", unsafe_allow_html=True)
                time.sleep(0.01)

            # 🔊 Voice output
            if voice_reply:
                text_to_speech(answer)

        st.success("✅ Answer generated")
        add_log(query, retrieved, answer)

# ------------------- Footer -------------------
st.markdown("""
<hr>
<footer>
Developed by <b>Abrar Khan & Muhammad Ibrar</b> | Intelligent EHR QA using RAG (FYP)
</footer>
""", unsafe_allow_html=True)