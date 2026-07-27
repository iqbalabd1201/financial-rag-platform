"""Embedding wrapper. BGE-small chosen over BGE-M3 on cost/accuracy grounds:
BGE-M3 gave +1.7pp hit@5 (51.7% vs 50.0%) at 17x model size and ~16x
embedding time for 3061 pages (9m25s vs 35s). See results/retrieval_comparison.csv.
"""
from sentence_transformers import SentenceTransformer

Q_PREFIX = "Represent this sentence for searching relevant passages: "


def load_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda"):
    return SentenceTransformer(model_name, device=device)


def embed_pages(model, pages: list[dict], company_lookup: dict, page_header_fn,
                 batch_size: int = 32):
    """Embed pages using the page+header representation (the winning config).

    company_lookup: {doc_id: company_name}
    page_header_fn: function(text) -> section title, e.g. clean_text.page_header
    """
    texts = [
        f'{company_lookup[p["doc_id"]]} 10-K | {page_header_fn(p["text"])}\n{p["text"][:2000]}'
        for p in pages
    ]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")


def embed_query(model, query: str):
    return model.encode([Q_PREFIX + query], normalize_embeddings=True).astype("float32")
