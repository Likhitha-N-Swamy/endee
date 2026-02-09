"""
Ingest `sample_docs.txt` into Endee:
- Split text into chunks
- Generate embeddings using sentence-transformers
- Insert vectors into Endee using JSON over HTTP

Run from repo root:
    python rag_project/ingestion/ingest.py
"""

import re
import json
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Paths
# --------------------------------------------------
RAG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = RAG_ROOT / "data"
SAMPLE_DOCS = DATA_DIR / "sample_docs.txt"
METADATA_PATH = DATA_DIR / "chunk_metadata.json"

# --------------------------------------------------
# Endee API
# --------------------------------------------------
BASE_URL = "http://localhost:8080"
INDEX_NAME = "rag_index"
INSERT_URL = f"{BASE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"

AUTH_TOKEN = None  # Set only if Endee auth is enabled

# --------------------------------------------------
# Embedding model (dim = 384, must match index)
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"


def split_into_chunks(text: str) -> list[str]:
    """
    Split text into chunks of ~2–3 sentences.
    """
    sentence_end = re.compile(r"(?<=[.!?])\s+")
    sentences = [s.strip() for s in sentence_end.split(text) if s.strip()]

    chunks = []
    i = 0
    # Take up to 3 sentences per chunk so we stay short and dense
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + 3])
        chunks.append(chunk)
        i += 3

    return chunks


def main():
    # 1. Load document
    if not SAMPLE_DOCS.exists():
        print(f"Missing file: {SAMPLE_DOCS}")
        return

    text = SAMPLE_DOCS.read_text(encoding="utf-8").strip()
    if not text:
        print("sample_docs.txt is empty")
        return

    # 2. Chunk text
    chunks = split_into_chunks(text)
    print(f"Split into {len(chunks)} chunk(s).")

    # 3. Load embedding model and encode
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # 4. Build vectors payload + local metadata
    vectors = []
    metadata = {}

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"chunk_{i}"
        vectors.append(
            {
                "id": chunk_id,
                "vector": emb.tolist(),
            }
        )
        metadata[chunk_id] = {"text": chunk}

    # 5. Save metadata locally (for RAG retrieval)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved chunk metadata to {METADATA_PATH}")

    # 6. Send vectors to Endee using JSON
    headers = {
        "Content-Type": "application/json",
    }
    if AUTH_TOKEN:
        headers["Authorization"] = AUTH_TOKEN

    print(f"Sending {len(vectors)} vectors to Endee (JSON)...")

    response = requests.post(
        INSERT_URL,
        json=vectors,
        headers=headers,
        timeout=30,
    )

    if response.status_code == 200:
        print(f"Success: Inserted {len(vectors)} vectors into '{INDEX_NAME}'.")
    else:
        print(f"Insert failed ({response.status_code}): {response.text}")


if __name__ == "__main__":
    main()
