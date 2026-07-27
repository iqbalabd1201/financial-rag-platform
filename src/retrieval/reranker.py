"""Cross-encoder reranking -- IMPLEMENTED, TESTED, AND DISABLED BY DEFAULT.

Do not enable this without re-reading docs/failure_analysis.md first.

Three separate tests, all showing reranking HURTS accuracy on this domain:
  1. Plain page-level index:        56.1% -> 36.6% hit@5 after reranking
  2. RRF fusion (bi-encoder+rerank): 56.1% -> 51.2% (still a loss)
  3. Page+header index, k=30/50:     58.3% -> 48.3%/43.3% at k=5 after reranking

Root cause (confirmed independently by two papers -- HiREC arXiv:2505.20368,
T2-RAGBench ACL 2026.eacl-long.8): bge-reranker-base is trained on prose-style
relevance judgments and is systematically biased toward narrative pages over
pages that are mostly raw tabular figures -- exactly the pages financial QA
questions most often need. The candidate ceiling (in_cand) was 76.7-85.0%,
so the reranker had plenty of correct candidates to work with; it simply
ranked them wrong.

Kept in the repo (not deleted) specifically so this negative result is
visible and reproducible, not silently discarded.
"""
from sentence_transformers import CrossEncoder


def load_reranker(model_name: str = "BAAI/bge-reranker-base", device: str = "cuda"):
    return CrossEncoder(model_name, max_length=512, device=device)


def rerank(reranker, query: str, candidates: list[tuple], page_text_lookup: dict) -> list:
    """candidates: list of (doc_id, page_num). Returns re-ordered list.

    NOT recommended for production use on this domain -- see module docstring.
    """
    pairs = [[query, page_text_lookup[(doc_id, pn)]] for doc_id, pn in candidates]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    return [c for c, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])]
