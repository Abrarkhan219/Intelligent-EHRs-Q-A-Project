import gradio as gr
import requests
from src.medical_api import recommend_medicine, find_nearby_store, get_google_answer
from src.rag import retrieve_top_k, generate_answer

# Function to get response from the dataset or API
def get_answer(query, mode="Hybrid"):
    retrieved, scores = retrieve_top_k(query, k=3)
    
    # Decision logic
    if mode == "API Only" or max(scores) < 0.4:  # Low similarity threshold
        answer = get_google_answer(query)  # Fallback to Google Search API
    else:
        answer = generate_answer(query, retrieved)  # Use dataset-based answer

    return answer

# Create Gradio interface
def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# For your medical question answering system
medical_demo = gr.Interface(fn=get_answer, inputs=["text", "radio"], outputs="text", 
                            live=True, examples=["What is the treatment for headache?", "Where can I buy Panadol?"])

medical_demo.launch()