# ingest_tfidf.py
import os, json
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

PAPERS_DIR = "papers"
OUT = "vector_store"
os.makedirs(OUT, exist_ok=True)

def read_pdf(p):
    r = PdfReader(str(p))
    return "\n".join([pg.extract_text() or "" for pg in r.pages])

def chunk_text(text, size=500, overlap=100):
    if not text: return []
    chunks=[]
    i=0
    N=len(text)
    while i<N:
        end=i+size
        chunks.append(text[i:end].strip())
        i=max(end-overlap, end)
    return chunks

all_texts=[]
meta=[]
pdfs=[p for p in Path(PAPERS_DIR).iterdir() if p.suffix.lower()==".pdf"]
for pdf in pdfs:
    txt=read_pdf(pdf)
    ch=chunk_text(txt)
    for idx,c in enumerate(ch):
        all_texts.append(c)
        meta.append({"source":pdf.name,"chunk_index":idx})

if not all_texts:
    print("No chunks found. Put PDFs in the papers/ folder.")
    raise SystemExit(0)

# Fit TF-IDF on the document chunks
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")
X = vectorizer.fit_transform(all_texts)  # sparse matrix (N, D)

# Save dense embeddings (you can keep sparse if you want)
emb = X.toarray().astype("float32")
np.save(os.path.join(OUT,"embeddings.npy"), emb)

# Save vectorizer for query-time transforms
joblib.dump(vectorizer, os.path.join(OUT,"vectorizer.joblib"))

with open(os.path.join(OUT,"texts.json"),"w",encoding="utf-8") as f:
    json.dump(all_texts,f,ensure_ascii=False,indent=2)
with open(os.path.join(OUT,"meta.json"),"w",encoding="utf-8") as f:
    json.dump(meta,f,ensure_ascii=False,indent=2)

print("TF-IDF ingest done. chunks:", len(all_texts))
print("Saved: embeddings.npy, texts.json, meta.json, vectorizer.joblib in", OUT)
