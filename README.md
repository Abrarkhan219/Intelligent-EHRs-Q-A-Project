# Intelligent EHR Question Answering using Semantic Similarity & RAG
This project implements a medical question-answering system using Semantic Similarity and Retrieval-Augmented Generation (RAG) on Electronic Health Records (EHR) data.

## Project Overview
This project implements a medical question-answering system using **Semantic Similarity** and **Retrieval-Augmented Generation (RAG)** on **Electronic Health Records (EHR)** data. The system intelligently combines **semantic embeddings** for retrieval and **T5-based generation** to provide accurate answers to medical queries.

## Folder Structure
- **data/**: contains raw and cleaned datasets.
- **notebooks/**: contains analysis notebooks.
- **src/**: source code files for preprocessing, embedding, retrieval, and generation.
- **models/**: trained models, embeddings, and indexes.
- **app/**: simple demo interface.

## Setup Instructions
1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # For Windows

## Install dependencies:
- pip install -r requirements.txt

## Run the preprocessing and model scripts.

- How to Run

## Activate virtual environment:

- .\venv\Scripts\Activate.ps1  # For Windows


## Run Streamlit app:

- streamlit run app/streamlit_app.py

