"""
Create an Endee vector index for RAG via the REST API.
Run from rag_project directory:
    python utils/create_index.py
"""

import requests

# --- Configuration ---
BASE_URL = "http://localhost:8080"
CREATE_INDEX_URL = f"{BASE_URL}/api/v1/index/create"

# Index settings: name and dimension must match your embedding model (e.g. sentence-transformers).
INDEX_NAME = "rag_index"
DIMENSION = 384
# space_type: "cosine" for normalized embeddings, "l2" for Euclidean, "ip" for inner product
SPACE_TYPE = "cosine"

# Optional: set if your Endee server uses token auth
AUTH_TOKEN = None  # e.g. "your_token_here"


def main():
    # Build the JSON body required by the /api/v1/index/create endpoint
    payload = {
        "index_name": INDEX_NAME,
        "dim": DIMENSION,
        "space_type": SPACE_TYPE,
    }

    # Prepare headers (optional auth)
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = AUTH_TOKEN

    # Send POST request to create the index
    try:
        response = requests.post(
            CREATE_INDEX_URL,
            json=payload,
            headers=headers,
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"Error: Request failed — {e}")
        return

    # Handle response and print a clear success or error message
    if response.status_code == 200:
        print(f"Success: Index '{INDEX_NAME}' created (dim={DIMENSION}, space={SPACE_TYPE}).")
    else:
        try:
            err = response.json()
            msg = err.get("error", response.text)
        except Exception:
            msg = response.text or f"HTTP {response.status_code}"
        print(f"Error: {msg}")


if __name__ == "__main__":
    main()
