# test_debug.py
from pathlib import Path
import joblib, json
from app_tfidf import extractive_summary_from_chunks

OUT="vector_store"
emb_path = Path(OUT)/"embeddings.npy"
texts_path = Path(OUT)/"texts.json"
meta_path = Path(OUT)/"meta.json"
vec_path = Path(OUT)/"vectorizer.joblib"

print("Files exist:",
      emb_path.exists(), texts_path.exists(), meta_path.exists(), vec_path.exists())

with open(texts_path, "r", encoding="utf-8") as f:
    texts = json.load(f)
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
vec = joblib.load(str(vec_path))

# Build sample context using top 3 chunks
context_chunks = []
for i in range(min(3, len(texts))):
    context_chunks.append({"text": texts[i], "meta": meta[i], "score": 1.0})

q = "What must we submit?"
ans, contrib = extractive_summary_from_chunks(q, context_chunks, max_sentences=2)
print("\nANSWER:\n", ans)
print("\nCONTRIBUTING CHUNKS:\n", contrib)
