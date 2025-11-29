# ingest.py
import os
import json
import math
import time
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import numpy as np

load_dotenv()
BLACKBOX_API_KEY = os.getenv("BLACKBOX_API_KEY")
if not BLACKBOX_API_KEY:
    raise SystemExit("Set BLACKBOX_API_KEY in .env")

# Config
PAPERS_DIR = "papers"
OUTPUT_DIR = "vector_store"
CHUNK_SIZE = 800         # characters per chunk (tune if necessary)
CHUNK_OVERLAP = 100      # overlap in characters

EMBED_ENDPOINT = "https://api.blackbox.ai/embeddings"
  # per your earlier format
EMBED_MODEL = "text-embedding-3-small"  # use this or change to supported model

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_pdf_text(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        text = p.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple character-based chunker with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)  # move forward keeping overlap
    return chunks

def blackbox_embed_batch(texts):
    """Call Blackbox embeddings API for a list of texts (iterative calls)."""
    embeddings = []
    url = EMBED_ENDPOINT
    headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {BLACKBOX_API_KEY}"}
    # We will call per text to keep simple; if Blackbox supports batch, switch to batch.
    for t in texts:
        body = {
            "model": EMBED_MODEL,
            "input": t
        }
        r = requests.post(url, json=body, headers=headers, timeout=30)
        if r.status_code != 200:
            print("Embedding call failed:", r.status_code, r.text)
            raise SystemExit("Embedding API error")
        data = r.json()
        emb = data["data"][0]["embedding"]
        embeddings.append(emb)
        time.sleep(0.05)  # small pause to avoid hammering (adjust if needed)
    return embeddings

def ingest_all():
    pdfs = [p for p in Path(PAPERS_DIR).iterdir() if p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}. Put PDFs there and run again.")
    all_texts = []
    all_meta = []
    print(f"Found {len(pdfs)} pdfs. Extracting and chunking...")
    for pdf in pdfs:
        text = read_pdf_text(str(pdf))
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            all_meta.append({"source": pdf.name, "chunk_index": i, "length": len(c)})
    print(f"Total chunks: {len(all_texts)}")

    # Create embeddings
    print("Creating embeddings via Blackbox...")
    embeddings = []
    BATCH = 16
    for i in tqdm(range(0, len(all_texts), BATCH)):
        batch_texts = all_texts[i:i+BATCH]
        batch_embs = blackbox_embed_batch(batch_texts)
        embeddings.extend(batch_embs)

    emb_array = np.array(embeddings, dtype=np.float32)
    # Save artifacts
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), emb_array)
    with open(os.path.join(OUTPUT_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print("Saved embeddings.npy, texts.json, meta.json in", OUTPUT_DIR)

if __name__ == "__main__":
    ingest_all()
