"""Unit tests for src.evaluation.retrieval_metrics.

These three functions produce the headline numbers in README.md
(58.3% hit@5, 55.0% recall@5) -- worth testing directly since a silent
bug here would silently invalidate every reported result, the same class
of failure documented for answer_metrics.py's judge-parsing bug.
"""
from src.evaluation.retrieval_metrics import hit_at_k, recall_at_k, candidate_ceiling


def test_hit_at_k_true_when_gold_page_present():
    assert hit_at_k({42}, [10, 42, 7, 3, 9], 5) is True


def test_hit_at_k_false_when_gold_page_absent():
    assert hit_at_k({99}, [10, 42, 7, 3, 9], 5) is False


def test_hit_at_k_respects_k_cutoff():
    """Gold page is retrieved but ranked outside the top-k window."""
    assert hit_at_k({9}, [10, 42, 7, 3, 9], 3) is False


def test_recall_at_k_partial_overlap():
    gold = {42, 7, 100}
    retrieved = [10, 42, 7, 3, 9]
    assert recall_at_k(gold, retrieved, 5) == 2 / 3


def test_recall_at_k_empty_gold_returns_zero():
    assert recall_at_k(set(), [1, 2, 3], 5) == 0.0


def test_candidate_ceiling_true_regardless_of_rank():
    """Ceiling check ignores rank entirely -- used to diagnose whether a
    failure is a retrieval-depth problem or a ranking problem."""
    assert candidate_ceiling({500}, [1, 2, 3, 500, 999]) is True


def test_candidate_ceiling_false_when_truly_absent():
    assert candidate_ceiling({500}, [1, 2, 3]) is False
