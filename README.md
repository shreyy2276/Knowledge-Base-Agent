# Knowledge-Base Agent — TF-IDF + Extractive Summarizer (Offline)

## Overview
This is a lightweight Knowledge-Base Agent that answers questions from PDF documents. It uses TF-IDF for retrieval and an extractive summarization step (sentence ranking) to produce concise answers — all **offline**, no external APIs required.

## Features
- Ingest PDFs into chunks and build TF-IDF vector store.
- Fast local retrieval using cosine similarity.
- Extractive summarization to create concise, non-repetitive answers.
- Streamlit UI for live, interactive demos.

## Files
- `ingest_tfidf.py` — ingest PDFs and save `vector_store/` (embeddings, texts, meta, vectorizer).
- `app_tfidf.py` — (improved) Streamlit app with extractive summarizer (use this for demo).
- `vector_store/` — generated artifacts after running ingestion.
- `papers/` — put input PDFs here.
- `requirements.txt` — dependencies.

## How to run
1. Create & activate venv:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate.bat
