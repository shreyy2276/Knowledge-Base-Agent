# app.py
import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
from numpy.linalg import norm

load_dotenv()
BLACKBOX_API_KEY = os.getenv("BLACKBOX_API_KEY")
if not BLACKBOX_API_KEY:
    st.error("Set BLACKBOX_API_KEY in .env and restart.")
    st.stop()

OUTPUT_DIR = "vector_store"
EMBED_ENDPOINT = "https://api.blackbox.ai/embeddings"
CHAT_ENDPOINT = "https://api.blackbox.ai/chat/completions"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "blackboxai"  # change if Blackbox docs recommend another model

# Helpers
def load_store():
    emb_path = Path(OUTPUT_DIR) / "embeddings.npy"
    texts_path = Path(OUTPUT_DIR) / "texts.json"
    meta_path = Path(OUTPUT_DIR) / "meta.json"
    if not emb_path.exists():
        return None
    emb = np.load(str(emb_path))
    with open(str(texts_path), "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(str(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return emb, texts, meta

def blackbox_embed(text):
    headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {BLACKBOX_API_KEY}"}
    body = {"model": EMBED_MODEL, "input": text}
    r = requests.post(EMBED_ENDPOINT, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)

def blackbox_chat_with_context(question, context_chunks):
    """Send a single prompt combining context chunks and question to Blackbox chat."""
    headers = {"Content-Type": "application/json", "x-api-key": BLACKBOX_API_KEY}
    # Compose a prompt that clearly instructs the model to use context and cite sources
    context_text = "\n\n---\n\n".join([f"[{i}] Source: {c['meta']['source']} chunk:{c['meta']['chunk_index']}\n{c['text']}" 
                                       for i,c in enumerate(context_chunks)])
    system_msg = "You are a helpful document-based question answering assistant. Use ONLY the provided CONTEXT to answer. If the answer is not present, say 'I don't know based on the provided documents.'"
    user_msg = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\n\nProvide a concise answer and then list which chunk indexes you used (e.g., [0], [2])."
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 800
    }
    r = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    # adapt if Blackbox returns a different shape
    # we expect something like data["choices"][0]["message"]["content"] or data["output"]
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    # fallback
    if "output" in data:
        return data["output"]
    return str(data)

def find_top_k(query_emb, embeddings, k=3):
    """Cosine similarity top-k retrieval."""
    # normalize
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = emb_norms.dot(query_norm)
    idx = np.argsort(-sims)[:k]
    scores = sims[idx]
    return idx, scores

# Streamlit UI
st.set_page_config(page_title="KB Agent (Blackbox)", layout="wide")
st.title("Document QA Agent — Blackbox.ai")

store = load_store()
if store is None:
    st.warning("Vector store not found. Run ingest.py first to create embeddings.")
    st.stop()

embeddings, texts, meta = store

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K chunks to retrieve", 1, 6, 3)
question = st.text_area("Ask a question about the uploaded documents", height=140)
if st.button("Get answer"):
    if not question.strip():
        st.info("Type a question first.")
    else:
        with st.spinner("Embedding query and retrieving..."):
            q_emb = blackbox_embed(question)
            idxs, scores = find_top_k(q_emb, embeddings, k=top_k)
            context_chunks = []
            for i, idx in enumerate(idxs):
                context_chunks.append({"text": texts[int(idx)], "meta": meta[int(idx)], "score": float(scores[i])})
        st.markdown("### Retrieved chunks (top results)")
        for i,c in enumerate(context_chunks):
            st.markdown(f"**Rank {i+1} — source:** {c['meta']['source']} — chunk {c['meta']['chunk_index']} — score: {c['score']:.4f}")
            st.write(c['text'][:1000])  # show truncated
            st.markdown("---")
        with st.spinner("Calling Blackbox to synthesize answer..."):
            answer = blackbox_chat_with_context(question, context_chunks)
        st.markdown("### Answer")
        st.write(answer)
