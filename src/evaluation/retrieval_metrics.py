"""Retrieval metrics: hit@k, recall@k, and candidate-ceiling (in_cand)
analysis -- the last one is what revealed the reranker was throwing away
good candidates rather than lacking them (ceiling 76.7-85.0% vs actual
36.6-51.2% after reranking).
"""


def hit_at_k(gold_pages: set, retrieved_pages: list, k: int) -> bool:
    return bool(gold_pages & set(retrieved_pages[:k]))


def recall_at_k(gold_pages: set, retrieved_pages: list, k: int) -> float:
    if not gold_pages:
        return 0.0
    overlap = gold_pages & set(retrieved_pages[:k])
    return len(overlap) / len(gold_pages)


def candidate_ceiling(gold_pages: set, all_candidates: list) -> bool:
    """Is the gold page anywhere in the full candidate pool, regardless of rank?
    Use this BEFORE blaming a reranker or ranking method for zero recall --
    if the ceiling itself is low, the problem is retrieval depth/recall,
    not ranking.
    """
    return bool(gold_pages & set(all_candidates))
