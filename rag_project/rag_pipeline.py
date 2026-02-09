"""
RAG pipeline: retrieve relevant chunks for a question, build a prompt, and produce an answer.
Uses a placeholder "LLM" (no real API). Run: python rag_pipeline.py "Your question?"
"""

import sys

from retrieval.search import retrieve_chunks


def build_context(chunks: list[str]) -> str:
    """Combine retrieved text chunks into a single context string (one paragraph per chunk)."""
    return "\n\n".join(chunks).strip()


def build_prompt(context: str, question: str) -> str:
    """Format the RAG prompt with context and question."""
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"


def placeholder_llm(prompt: str) -> str:
    """
    Placeholder LLM: no real model. Returns a simple rule-based answer from the prompt.
    Replace this with a call to OpenAI, Anthropic, local model, etc.
    """
    # Extract context and question from the prompt (everything after "Context:\n" and "Question:\n")
    if "Context:\n" in prompt and "\n\nQuestion:\n" in prompt:
        parts = prompt.split("\n\nQuestion:\n", 1)
        context_block = parts[0].replace("Context:\n", "", 1).strip()
        question_block = parts[1].replace("\n\nAnswer:", "", 1).strip()
        # Simple rule: summarize context in one sentence, then address the question
        first_sentence = context_block.split(". ")[0].strip()
        if not first_sentence.endswith("."):
            first_sentence += "."
        return (
            f"Based on the context: {first_sentence} "
            f"In answer to \"{question_block}\": The relevant information is in the context above."
        )
    # Fallback if prompt format is unexpected
    return "I could not generate an answer from the given prompt."


def run_rag(question: str) -> str:
    """
    Run the full RAG pipeline and return the final answer as a string.
    """
    # 1. Retrieve relevant chunks from the vector store (via Endee + chunk_metadata)
    chunks = retrieve_chunks(question)
    if not chunks:
        return "No relevant context was found for your question."

    # 2. Combine chunks into a single context string
    context = build_context(chunks)

    # 3. Build the prompt in the required format
    prompt = build_prompt(context, question)

    # 4. Generate answer using the placeholder LLM
    answer = placeholder_llm(prompt)

    # 5. Return the final answer
    return answer


def main():
    # Accept question from command line or stdin
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Question: ").strip()

    if not question:
        print("Error: empty question.")
        return

    try:
        answer = run_rag(question)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
