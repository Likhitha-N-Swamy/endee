"""
Retrieve top-k chunks from Endee for a user query; map result IDs to text via chunk_metadata.json.
Run: python -m retrieval.search "your query"
  or: python retrieval/search.py
      (then enter query when prompted)
"""

import json
import sys
from pathlib import Path

import msgpack
import requests
from sentence_transformers import SentenceTransformer

# --- Paths ---
RAG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = RAG_ROOT / "data"
METADATA_PATH = DATA_DIR / "chunk_metadata.json"

# --- Endee API ---
BASE_URL = "http://localhost:8080"
INDEX_NAME = "rag_index"
SEARCH_URL = f"{BASE_URL}/api/v1/index/{INDEX_NAME}/search"
TOP_K = 3
AUTH_TOKEN = None

MODEL_NAME = "all-MiniLM-L6-v2"


def get_query() -> str:
    """Accept query from command line or interactive input."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return input("Query: ").strip()


def load_metadata() -> dict:
    """Load chunk_id -> {text, ...} from rag_project/data/chunk_metadata.json."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def retrieve_chunks(query: str) -> list[str]:
    """
    Embed query, search Endee for top-k, map result IDs to original text chunks.
    Returns list of retrieved text chunks in order of similarity.
    """
    # 1. Generate query embedding
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    # 2. POST search request to Endee
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = AUTH_TOKEN

    response = requests.post(
        SEARCH_URL,
        json={"vector": query_vector, "k": TOP_K},
        headers=headers,
        timeout=10,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Search failed: {response.text}")

    # 3. Decode MessagePack safely
    content_type = response.headers.get("Content-Type", "")

    if "application/msgpack" not in content_type:
        # Endee UI / unexpected response
        print("DEBUG: Non-msgpack response from Endee")
        print(response.text[:300])
        return []

    payload = msgpack.unpackb(response.content, raw=False)

    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, dict):
        results = payload.get("results", payload.get("dense", []))
    else:
        results = []

    # 4. Load metadata
    metadata = load_metadata()

    # 5. Map result IDs → text
    chunks = []
    for item in results:
        # Endee VectorResult is a list/tuple: id is FIRST element
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            continue

        chunk_id = item[1]

        if chunk_id in metadata and "text" in metadata[chunk_id]:
            chunks.append(metadata[chunk_id]["text"])
        else:
            chunks.append(f"[No text for id: {chunk_id}]")

    return chunks


def main():
    query = get_query()
    if not query:
        print("Error: empty query.")
        return

    try:
        chunks = retrieve_chunks(query)
    except Exception as e:
        print(f"Error: {e}")
        return

    for i, text in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(text)
        print()

    print(f"Returned {len(chunks)} chunk(s).")


if __name__ == "__main__":
    main()
