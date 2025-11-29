# app_tfidf.py
"""
Knowledge-Base Agent (TF-IDF + Extractive Summarization) — full updated file
Features:
- Lazy loads vector_store so UI appears instantly
- TF-IDF retrieval + MMR reranking for diverse chunks
- Extractive summarization (sentence scoring) with deduplication
- Sentence cleaning to avoid headings and list bullets
"""
import os
import json
import re
from pathlib import Path
import numpy as np
from numpy.linalg import norm
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq

# ---------- CONFIG ----------
OUTPUT_DIR = "vector_store"
EMBED_PATH = Path(OUTPUT_DIR) / "embeddings.npy"
TEXTS_PATH = Path(OUTPUT_DIR) / "texts.json"
META_PATH = Path(OUTPUT_DIR) / "meta.json"
VECTORIZER_PATH = Path(OUTPUT_DIR) / "vectorizer.joblib"

MAX_ANSWER_SENTENCES = 3
SENTENCE_MIN_LENGTH = 30   # ignore tiny sentences (tweak if you need short list items)
# ----------------------------

# ---------- helpers ----------
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def clean_sentence(s):
    """Remove list bullets/enumerators and collapse whitespace."""
    s = s.strip()
    s = re.sub(r'^[\s\-\*\u2022]*[\d\w]{0,3}[\.\)\-:\u2022]*\s*', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def looks_like_heading(s):
    """Heuristic to skip headings / short meta lines."""
    if len(s) < 6:
        return True
    if s.endswith(':') or '(Mandatory)' in s or 'MANDATORY' in s.upper():
        return True
    words = s.split()
    if len(words) <= 6 and sum(1 for w in words if w and w[0].isupper()) / max(1, len(words)) > 0.6:
        return True
    return False

def split_sentences(text):
    """Split text into cleaned sentences (skip very short fragments)."""
    raw = [r.strip() for r in _SENT_SPLIT_RE.split(text) if r.strip()]
    cleaned = []
    for r in raw:
        s = clean_sentence(r)
        if not s:
            continue
        if len(s) < 12:
            continue
        cleaned.append(s)
    return cleaned

def query_to_vector(vectorizer, query):
    q = vectorizer.transform([query])
    return q.toarray().astype("float32")[0]

def find_top_k_cosine(query_vec, embeddings, k=3):
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    embn = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = embn.dot(qn)
    idxs = np.argsort(-sims)[:k]
    return idxs, sims[idxs]

def sentence_is_duplicate(sent, seen_sents, overlap_threshold=0.55):
    """Return True if 'sent' is highly overlapping with any seen_sents."""
    toks = set([w.lower() for w in re.findall(r'\w+', sent) if len(w) > 2])
    if not toks:
        return True
    for s in seen_sents:
        s_toks = set([w.lower() for w in re.findall(r'\w+', s) if len(w) > 2])
        inter = len(toks & s_toks)
        ratio = inter / (1 + min(len(toks), len(s_toks)))
        if ratio > overlap_threshold:
            return True
    return False

# ---------- Extractive summarizer ----------
def extractive_summary_from_chunks(question, context_chunks, max_sentences=3):
    """
    Produces a short extractive answer from retrieved chunks.
    Returns (answer_text, contributing_chunk_indices)
    """
    sentences = []
    sent_meta = []
    for ci, chunk in enumerate(context_chunks):
        sents = split_sentences(chunk["text"])
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) < SENTENCE_MIN_LENGTH:
                continue
            if looks_like_heading(s_clean):
                continue
            s_final = clean_sentence(s_clean)
            if len(s_final) < SENTENCE_MIN_LENGTH:
                continue
            sentences.append(s_final)
            sent_meta.append({
                "chunk_index": chunk["meta"]["chunk_index"],
                "source": chunk["meta"]["source"],
                "chunk_rank": ci
            })

    if not sentences:
        # fallback: show short snippet from first chunk(s)
        snippet = " ".join([c["text"][:400] for c in context_chunks])
        return snippet[:800], [c["meta"]["chunk_index"] for c in context_chunks]

    # Vectorize sentences and question (fit on sentences for local scoring)
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=20000)
    try:
        sent_mat = v.fit_transform(sentences)
    except Exception:
        # fallback: return concatenated chunk heads
        snippet = " ".join([c["text"][:500] for c in context_chunks])
        return snippet[:800], [m["chunk_index"] for m in sent_meta]

    q_vec = v.transform([question]).toarray()[0]
    sent_arr = sent_mat.toarray()

    def cosine(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    scores = [cosine(s, q_vec) for s in sent_arr]

    # Greedy select top sentences with dedupe
    selected = []
    selected_indices = []
    used_phrases = set()
    heap = [(-scores[i], i) for i in range(len(scores))]
    heapq.heapify(heap)
    while heap and len(selected) < max_sentences:
        negscore, idx = heapq.heappop(heap)
        if scores[idx] <= 0:
            break
        sent = sentences[idx]
        # dedupe by sentence content
        if sentence_is_duplicate(sent, [sentences[i] for i in selected_indices], overlap_threshold=0.55):
            continue
        used_phrases.add(frozenset(set([w.lower() for w in re.findall(r'\w+', sent) if len(w) > 3])))
        selected.append(sent)
        selected_indices.append(idx)

    if not selected:
        top_idx = np.argsort(-np.array(scores))[:max_sentences]
        selected = [sentences[i] for i in top_idx]
        selected_indices = list(top_idx)

    contributing_chunks = set()
    for si in selected_indices:
        contributing_chunks.add(sent_meta[si]["chunk_index"])

    ordered = sorted(selected_indices, key=lambda i: (sent_meta[i]["chunk_index"], i))
    answer = " ".join([sentences[i] for i in ordered])
    if len(answer) > 1000:
        answer = answer[:1000].rsplit('.', 1)[0] + "."

    return answer.strip(), sorted(list(contributing_chunks))

# ---------- MMR reranker ----------
def mmr_rerank(query_vec, embeddings, candidate_idxs, k=3, lambda_param=0.55):
    """
    Maximal Marginal Relevance to choose diverse indices from candidate_idxs.
    Returns list of selected indices (indices into embeddings).
    """
    if len(candidate_idxs) == 0:
        return []
    selected = []
    candidate_list = list(candidate_idxs)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    qnorm = query_vec / (np.linalg.norm(query_vec) + 1e-12)

    def cos_idx(i, vec):
        return float(np.dot(emb_norm[int(i)], vec))

    # pick best by query similarity first
    best = max(candidate_list, key=lambda i: cos_idx(i, qnorm))
    selected.append(best)
    candidate_list.remove(best)

    while candidate_list and len(selected) < k:
        scores_mmr = []
        for c in candidate_list:
            sim_query = cos_idx(c, qnorm)
            sim_selected = max([float(np.dot(emb_norm[int(c)], emb_norm[int(s)])) for s in selected]) if selected else 0.0
            mmr_score = lambda_param * sim_query - (1 - lambda_param) * sim_selected
            scores_mmr.append((mmr_score, c))
        scores_mmr.sort(reverse=True, key=lambda x: x[0])
        chosen = scores_mmr[0][1]
        selected.append(chosen)
        candidate_list.remove(chosen)
    return selected

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="KB Agent (Knowledge Base)", layout="wide")
st.title("Knowledge-Base Agent — Document QA (TF-IDF + Extractive Summarizer)")

# Lazy-load vector store variables
store_loaded = False
embeddings = None
texts = None
meta = None
vectorizer = None

def load_store():
    """Load vector store on demand (returns True if OK)."""
    global store_loaded, embeddings, texts, meta, vectorizer
    if store_loaded:
        return True
    try:
        if not EMBED_PATH.exists() or not VECTORIZER_PATH.exists():
            return False
        embeddings = np.load(str(EMBED_PATH))
        with open(str(TEXTS_PATH), "r", encoding="utf-8") as f:
            texts = json.load(f)
        with open(str(META_PATH), "r", encoding="utf-8") as f:
            meta = json.load(f)
        vectorizer = joblib.load(str(VECTORIZER_PATH))
        store_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return False

# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K chunks to retrieve", 1, 6, 3)
max_sents = st.sidebar.slider("Max summary sentences", 1, 5, MAX_ANSWER_SENTENCES)
question = st.text_area("Ask a question about the documents", height=140)

if st.button("Get answer"):
    if not question.strip():
        st.info("Type a question first.")
    else:
        with st.spinner("Loading vector store (if not already loaded)..."):
            ok = load_store()
        if not ok:
            st.error("Vector store not found or failed to load. Run ingest_tfidf.py first.")
        else:
            with st.spinner("Retrieving relevant chunks (MMR rerank for diversity)..."):
                qvec = query_to_vector(vectorizer, question)
                # pull a larger candidate pool for MMR
                cand_k = max(top_k * 3, 6)
                cand_idxs, cand_scores = find_top_k_cosine(qvec, embeddings, k=cand_k)

                selected_idxs = mmr_rerank(qvec, embeddings, cand_idxs, k=top_k, lambda_param=0.55)

                # Build context chunks for display & summarization
                context_chunks = []
                for idx in selected_idxs:
                    score = float(np.dot(embeddings[int(idx)] / (np.linalg.norm(embeddings[int(idx)]) + 1e-12),
                                         qvec / (np.linalg.norm(qvec) + 1e-12)))
                    context_chunks.append({"text": texts[int(idx)], "meta": meta[int(idx)], "score": score})

            # show retrieved chunks
            st.markdown("### Retrieved chunks (top results)")
            for i, c in enumerate(context_chunks):
                st.markdown(f"**Rank {i+1} — Source:** {c['meta']['source']} — chunk {c['meta']['chunk_index']} — score: {c['score']:.4f}")
                st.write(c['text'][:900])
                st.markdown("---")

            # now summarize (extractive) with dedupe
            answer, contrib_chunks = extractive_summary_from_chunks(question, context_chunks, max_sentences=max_sents)
            st.markdown("### Agent Answer (concise)")
            st.write(answer)
            st.markdown("**Cited chunks:** " + ", ".join([f"[{i}]" for i in contrib_chunks]) if contrib_chunks else "No direct citations.")
            st.info("This answer was produced by extractive summarization (no external LLM). For stronger answers, integrate an LLM.")
