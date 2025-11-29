# Knowledge-Base Agent — TF-IDF + Extractive Summarizer

## 1. Working Demo
**Live demo:** https://<your-app>.streamlit.app

## 2. Repository
**GitHub:** https://github.com/<your-username>/Knowledge-Base-Agent

This repository contains:
- `app_tfidf.py` — Streamlit app (Document QA agent).
- `ingest_tfidf.py` — Ingest PDFs into TF-IDF vector store.
- `vector_store/` — precomputed artifacts (embeddings.npy, texts.json, meta.json, vectorizer.joblib).
- `papers/` — source PDFs used for the demo.
- `requirements.txt` — Python packages required.

## 3. Overview
A lightweight, offline Knowledge-Base Agent that answers user questions from uploaded PDFs using TF-IDF retrieval + extractive summarization. No external LLMs required.

## 4. Features & Limitations
**Features**
- Local TF-IDF retrieval for reliable offline performance
- Extractive summarization: picks concise sentences from top chunks
- Citation of source chunks
- Fast, deterministic, and reproducible

**Limitations**
- Not generative — extracts existing sentences; may miss inference-level answers
- Quality depends on chunking & documents provided
- For more fluent answers, integrate a generative LLM with retrieved context

## 5. Tech stack & Libraries
- Python 3.10+
- Streamlit (UI)
- scikit-learn (TF-IDF)
- numpy, joblib, pypdf

## 6. Setup & Run (local)
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/Knowledge-Base-Agent.git
   cd Knowledge-Base-Agent
