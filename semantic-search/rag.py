"""
RAG (Retrieval-Augmented Generation) module.

Takes retrieved document chunks from Endee and uses Groq's LLM to
synthesize a coherent, accurate answer grounded in those chunks.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_client = None


def _get_groq_client():
    """Lazy-init the Groq client so the model loads only when needed."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _build_context_block(chunks):
    """
    Format retrieved chunks into a numbered context block that the
    LLM can reference when composing its answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        similarity = chunk.get("similarity", 0)
        parts.append(
            f"[Chunk {i}] (source: {source}, relevance: {similarity:.2f})\n{text}"
        )
    return "\n\n".join(parts)


SYSTEM_PROMPT = """You are a helpful document assistant. The user will ask a question and you will be given relevant text chunks retrieved from their uploaded documents.

Rules:
- Answer ONLY using the information in the provided chunks. Do not make up facts.
- If the chunks do not contain enough information to answer, say so clearly.
- Be concise and well-structured. Use bullet points when listing items.
- When quoting specific details (names, numbers, dates), cite which chunk it came from.
- Do NOT repeat the chunks verbatim — synthesize and summarize."""


def generate_answer(query, retrieved_chunks, model=None):
    """
    Use Groq LLM to generate a grounded answer from retrieved chunks.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    retrieved_chunks : list[dict]
        Results from semantic_search(), each with 'text', 'source',
        'similarity', etc.
    model : str | None
        Override the default Groq model.

    Returns
    -------
    str
        The LLM-generated answer.
    """
    if not retrieved_chunks:
        return "No relevant document chunks were found. Please upload and process a document first."

    client = _get_groq_client()
    context = _build_context_block(retrieved_chunks)

    user_message = (
        f"Question: {query}\n\n"
        f"Retrieved document chunks:\n{context}"
    )

    response = client.chat.completions.create(
        model=model or GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content
