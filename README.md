📘 Knowledge-Base Agent — TF-IDF + Extractive Summarizer
1. Working Demo

Live demo:
🔗 https://knowledge-base-agent-gdspru35fe6jdebsppn8vx.streamlit.app/

2. Repository

GitHub:
🔗 https://github.com/shreyy2276/Knowledge-Base-Agent

This repository contains:

app_tfidf.py — Streamlit UI and QA pipeline

ingest_tfidf.py — builds TF-IDF vector store

vector_store/ — embeddings, metadata, and vectorizer

papers/ — PDF files used for the agent

requirements.txt — Python dependencies

architecture.png — system architecture diagram

3. Overview

A lightweight, offline Knowledge-Base Agent that answers user questions directly from PDF documents using:

TF-IDF vector retrieval

Cosine similarity

MMR re-ranking

Extractive summarization of top sentences

It does not require external APIs or LLMs, making it fully offline and reproducible.

4. Features & Limitations
✔ Features

Offline PDF-based Q&A

TF-IDF retrieval for fast, deterministic results

Extractive summarizer (sentence scoring)

MMR-based diversity filtering

Transparent chunk citations

Streamlit interactive UI

❌ Limitations

Extractive only — cannot generate new sentences

No deep reasoning beyond provided text

Only works with readable text PDFs (no images/tables)

5. Tech Stack & Libraries

Python 3.10+

Streamlit (web UI)

scikit-learn (TF-IDF, cosine similarity)

NumPy, joblib

PyPDF for text extraction

Custom local vector-store (embeddings + metadata)

6. Setup & Run (Local)
# Clone the repo
git clone https://github.com/shreyy2276/Knowledge-Base-Agent
cd Knowledge-Base-Agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# (Optional) Rebuild vector store
python ingest_tfidf.py

# Run the app
streamlit run app_tfidf.py
