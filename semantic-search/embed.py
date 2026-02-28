import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

load_dotenv()

ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
INDEX_NAME = os.getenv("INDEX_NAME", "semantic_search")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

VECTOR_DIM = 768  # all-mpnet-base-v2 output dimension

# Load the embedding model once at module level
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_endee_client():
    """Initialize Endee client with optional token authentication."""
    if ENDEE_TOKEN:
        client = Endee(token=ENDEE_TOKEN)
    else:
        client = Endee()
    client.set_base_url(ENDEE_BASE_URL)
    return client


def ensure_index_exists(client):
    """Create the search index if it doesn't already exist."""
    existing = client.list_indexes()

    # list_indexes may return a dict like {'indexes': [...]} or a plain list
    index_names = []
    if isinstance(existing, dict):
        index_list = existing.get("indexes", [])
    elif isinstance(existing, list):
        index_list = existing
    else:
        index_list = []

    for idx in index_list:
        if isinstance(idx, dict):
            index_names.append(idx.get("name", ""))
        elif isinstance(idx, str):
            index_names.append(idx)

    if INDEX_NAME not in index_names:
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            space_type="cosine",
            precision=Precision.FLOAT32
        )

    return client.get_index(name=INDEX_NAME)


def generate_embeddings(texts):
    """Convert a list of text strings into vector embeddings."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def store_chunks_in_endee(chunks, source_filename="unknown"):
    """
    Generate embeddings for text chunks and store them in Endee.

    Each chunk is stored with its text as metadata so we can retrieve
    the original content during search.
    """
    if not chunks:
        return 0

    client = get_endee_client()
    index = ensure_index_exists(client)

    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    # Endee supports max 1000 vectors per upsert, batch accordingly
    batch_size = 500
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_vectors = embeddings[i:i + batch_size]

        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            vectors_to_upsert.append({
                "id": f"{source_filename}_{chunk['id']}",
                "vector": batch_vectors[j],
                "meta": {
                    "text": chunk["text"],
                    "source": source_filename,
                    "chunk_id": chunk["id"]
                }
            })

        index.upsert(vectors_to_upsert)
        total_stored += len(vectors_to_upsert)

    return total_stored


def embed_single_query(query_text):
    """Generate embedding for a single search query."""
    model = get_model()
    embedding = model.encode([query_text])
    return embedding[0].tolist()
