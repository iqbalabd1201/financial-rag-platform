"""Shared fixtures for API integration tests.

The FastAPI app's lifespan normally loads the real embedder, connects to
Qdrant Cloud, and creates a real OpenAI client. For CI, all four of those
are replaced with fakes -- this tests the actual routing, doc_matcher
logic, response schema, and observability wiring, without needing network
access, API keys, or the 3061-page production index.

doc_matcher.match_document itself is NOT mocked -- it runs for real
against a small fake corpus, so this is a genuine (if small-scale)
integration test of Stage 1 matching + the API contract together.
"""
import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")

FAKE_PAGES = [
    {"doc_id": "TESTCO_2023_10K", "page_num": 1,
     "text": "Revenue for fiscal year 2023 was $500 million."},
    {"doc_id": "TESTCO_2023_10K", "page_num": 2,
     "text": "Capital expenditures totaled $42 million in FY2023."},
]
FAKE_COMPANY_LOOKUP = {"TESTCO_2023_10K": "TestCo"}


class _FakeIndex:
    def __init__(self, ntotal):
        self.ntotal = ntotal


@pytest.fixture
def client(monkeypatch):
    import src.api.main as main_module

    monkeypatch.setattr(
        main_module, "load_index_bundle",
        lambda path: (_FakeIndex(len(FAKE_PAGES)), FAKE_PAGES, FAKE_COMPANY_LOOKUP),
    )
    monkeypatch.setattr(main_module, "load_embedder", lambda device=None: object())
    monkeypatch.setattr(main_module, "embed_query", lambda model, question: [[0.0]])
    monkeypatch.setattr(main_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(
        main_module, "retrieve_pages_qdrant",
        lambda client, q_emb, doc_id, k=10: [1, 2],
    )

    def fake_generate_answer(client, question, context, usage_out=None):
        if usage_out is not None:
            usage_out.update(
                {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
            )
        return "Capital expenditures were $42 million.\n\nCALC: 42", 42.0

    monkeypatch.setattr(main_module, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(main_module, "OpenAI", lambda api_key: object())

    with TestClient(main_module.app) as test_client:
        yield test_client
