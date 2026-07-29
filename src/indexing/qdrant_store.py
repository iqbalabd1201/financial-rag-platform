"""Qdrant wrapper for production vector storage.

Kept separate from src/retrieval/retriever.py (the FAISS-based version) on
purpose: reproduce_from_scratch.py stays 100% local/deterministic/free and
must not depend on an external Qdrant Cloud connection. Only the live API
(src/api/main.py) uses this module.

Search here uses Qdrant's native payload filtering (query_filter on doc_id)
instead of FAISS's pattern of over-fetching search_depth=400 global
candidates and filtering to doc_id in Python. This is strictly better, not
just equivalent: FAISS's approach can under-return results for a document
if its true top-k pages fall outside the top-400 *global* candidates (rare
at this corpus size, but a real edge case) -- Qdrant's filtered search has
no such ceiling.
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

COLLECTION_NAME = "financial_rag_pages"
VECTOR_SIZE = 384  # BGE-small-en-v1.5 output dimension


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set.")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, recreate: bool = False):
    """Create the collection if it doesn't exist. recreate=True drops and
    rebuilds it -- only intended for the one-time migration script."""
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def upsert_pages(client: QdrantClient, vectors, pages: list[dict], batch_size: int = 256):
    """vectors: array-like of shape (len(pages), VECTOR_SIZE), same order as pages.
    Payload stores doc_id + page_num only -- page text stays in pages.json,
    loaded separately by the API, to keep Qdrant payload small.
    """
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"doc_id": pages[i]["doc_id"], "page_num": pages[i]["page_num"]},
        )
        for i in range(len(pages))
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start:start + batch_size])


def retrieve_pages_qdrant(client: QdrantClient, query_embedding, doc_id: str,
                           k: int = 10) -> list[int]:
    """Same contract as src.retrieval.retriever.retrieve_pages: returns the
    top-k page numbers within doc_id, ranked by similarity. query_embedding
    is the (1, VECTOR_SIZE) array returned by embedder.embed_query.
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding[0].tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=k,
    )
    return [point.payload["page_num"] for point in results.points]
