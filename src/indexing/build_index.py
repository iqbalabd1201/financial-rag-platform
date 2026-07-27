"""FAISS index builder. IndexFlatIP (exact search) -- corpus here is small
enough (a few thousand pages) that approximate search isn't needed, and
exact search removes a source of non-reproducibility. Verified: rebuilding
the index from scratch reproduces hit@5/recall@5 to the decimal (58.3%/55.0%
both times) -- see experiments/02_chunk_vs_page_level.ipynb.
"""
import faiss


def build_flat_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
