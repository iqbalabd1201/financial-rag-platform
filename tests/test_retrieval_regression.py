"""Regression gate: retrieval quality must not silently degrade.

Uses the FAISS bundle already committed to data/index_store/ (built once,
verified against README.md's 58.3%/55.0% baseline) -- NOT a live Qdrant
connection. Keeps this test free, deterministic, and runnable in CI with
no secrets, matching the same "keep it free and deterministic" principle
already applied to scripts/reproduce_from_scratch.py.

Only tests a subset (15 of 60 gold questions) to keep CI fast -- this is
a regression *gate*, not a full re-evaluation. The full 60-question check
is scripts/reproduce_from_scratch.py, run manually or on a schedule, not
on every push.
"""
import json
import pytest
import pandas as pd

from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.retriever import retrieve_pages
from src.evaluation.retrieval_metrics import hit_at_k, recall_at_k

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
SUBSET_SIZE = 15
# README.md baseline is 58.3%/55.0% on the full 60. A 15-question subset
# has more sampling noise, so the floor here is deliberately generous --
# it exists to catch a broken pipeline (e.g. hit@5 collapsing to near 0),
# not to police normal subset-to-subset variance.
MIN_HIT5 = 0.35


@pytest.fixture(scope="module")
def gold_subset():
    data = json.load(open(GOLD_QA_PATH))
    qa_df = pd.DataFrame(data["questions"])
    return qa_df.head(SUBSET_SIZE)


@pytest.fixture(scope="module")
def index_bundle():
    return load_index_bundle("data/index_store")


@pytest.fixture(scope="module")
def embed_model():
    return load_embedder(device="cpu")


def test_hit_at_5_does_not_regress(gold_subset, index_bundle, embed_model):
    index, pages, _ = index_bundle
    hits = []
    for _, r in gold_subset.iterrows():
        gold = {p + PAGE_OFFSET for p in r["evidence_pages"]}
        q_emb = embed_query(embed_model, r["question"])
        retrieved = retrieve_pages(q_emb, index, pages, r["doc_name"], k=5)
        hits.append(hit_at_k(gold, retrieved, 5))

    hit5 = sum(hits) / len(hits)
    assert hit5 >= MIN_HIT5, (
        f"Hit@5 on {SUBSET_SIZE}-question subset dropped to {hit5:.1%}, "
        f"below the {MIN_HIT5:.0%} regression floor. Compare against "
        f"README.md's 58.3% full-set baseline before merging."
    )
