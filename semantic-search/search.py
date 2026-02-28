import os
from dotenv import load_dotenv
from embed import embed_single_query, get_endee_client, ensure_index_exists

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "5"))


def semantic_search(query_text, top_k=None):
    """
    Perform semantic similarity search against stored document vectors.

    Encodes the query into an embedding, then queries the Endee index
    to find the closest matching document chunks by cosine similarity.

    Returns a list of results, each containing the matched text,
    similarity score, and source document info.
    """
    if top_k is None:
        top_k = TOP_K

    query_vector = embed_single_query(query_text)

    client = get_endee_client()
    index = ensure_index_exists(client)

    try:
        raw_results = index.query(
            vector=query_vector,
            top_k=top_k,
            ef=128,
            include_vectors=False
        )
    except Exception:
        # Index might be empty or not ready yet
        return []

    if not raw_results:
        return []

    results = []
    for item in raw_results:
        meta = item.get("meta", {})
        results.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", ""),
            "similarity": round(item.get("similarity", 0.0), 4),
            "id": item.get("id", "")
        })

    return results
