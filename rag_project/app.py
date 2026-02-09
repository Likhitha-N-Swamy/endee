"""
FastAPI app exposing the RAG pipeline via GET /ask.
Run: uvicorn app:app --reload
"""

from fastapi import FastAPI, Query
from rag_pipeline import run_rag

# Initialize the FastAPI application
app = FastAPI(
    title="RAG API",
    description="Ask questions against the indexed documents using Endee vector search."
)


@app.get("/")
def root():
    """
    Root endpoint: explains how to use the API.
    """
    return {
        "message": "RAG API is running. Use GET /ask?question=Your+question",
        "example": "/ask?question=RAG"
    }


@app.get("/ask")
def ask(
    question: str = Query(..., description="The question to answer using retrieved context.")
):
    """
    Accepts a question, runs the RAG pipeline, and returns the answer in JSON.
    This endpoint NEVER crashes; errors are returned as JSON.
    """
    try:
        answer = run_rag(question)
        return {
            "question": question,
            "answer": answer
        }
    except Exception as e:
        # IMPORTANT: prevents Internal Server Error
        return {
            "error": str(e),
            "note": "Error occurred inside RAG pipeline (retrieval / Endee search)."
        }
