# debug_ingest.py
import os, json, time, traceback
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
import requests

load_dotenv()
print("DEBUG: starting debug_ingest.py")

try:
    print("DEBUG: current dir:", os.getcwd())

    # Load API key
    BLACKBOX_API_KEY = os.getenv("BLACKBOX_API_KEY")
    print("DEBUG: BLACKBOX_API_KEY present?:", bool(BLACKBOX_API_KEY))
    if BLACKBOX_API_KEY:
        print("DEBUG: BLACKBOX_API_KEY (first 8 chars):", BLACKBOX_API_KEY[:8] + "...")

    # List PDFs
    papers_dir = Path("papers")
    if not papers_dir.exists():
        print("DEBUG: papers/ directory NOT found")
    else:
        pdfs = [p for p in papers_dir.iterdir() if p.suffix.lower() == ".pdf"]
        print("DEBUG: found pdf count:", len(pdfs))
        for p in pdfs:
            print(" -", p.name, "size:", p.stat().st_size)
            try:
                reader = PdfReader(str(p))
                text = reader.pages[0].extract_text() or ""
                print("  first page starts:", repr(text[:400])[:400])
            except Exception as e:
                print("  PDF read error:", e)

    # Test embedding API
    if BLACKBOX_API_KEY:
        try:
            print("DEBUG: testing embedding API call (this will hit Blackbox)...")
            url = "https://api.blackbox.ai/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {BLACKBOX_API_KEY}"
            }
            body = {
                "model": "text-embedding-3-small",
                "input": "hello world"
            }
            r = requests.post(url, json=body, headers=headers, timeout=20)
            print("DEBUG: embed status:", r.status_code)
            print("DEBUG: embed response snippet:", r.text[:300].replace("\n", " "))
        except Exception as e:
            print("DEBUG: embed call exception:", e)
    else:
        print("DEBUG: skipping embed test because no API key")

    print("DEBUG: finished checks successfully")

except Exception as e:
    print("DEBUG: top-level exception:")
    traceback.print_exc()
