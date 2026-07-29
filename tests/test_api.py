"""Integration tests for the FastAPI app: /health, /query, /metrics.

Runs the real routing + doc_matcher + response-schema + observability
code with Qdrant/OpenAI/embedder faked out (see conftest.py) -- no
network calls, no cost, no need for the 3061-page production index.
"""


def test_health_reports_ready(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["n_documents"] == 1


def test_query_returns_correct_schema_and_answer(client):
    resp = client.post("/query", json={"question": "What was TestCo's FY2023 capex?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["doc_id"] == "TESTCO_2023_10K"
    assert body["doc_match_method"] == "string_match"
    assert body["computed_value"] == 42.0
    assert "42 million" in body["answer"]

    # Observability fields added in the roadmap's item #3
    assert body["retrieval_ms"] >= 0
    assert body["generation_ms"] >= 0
    assert body["prompt_tokens"] == 100
    assert body["completion_tokens"] == 20
    assert body["estimated_cost_usd"] > 0


def test_query_with_unidentifiable_document_returns_422(client):
    resp = client.post("/query", json={"question": "And the year before that?"})
    assert resp.status_code == 422
    assert "identify" in resp.json()["detail"].lower()


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422  # Pydantic min_length=1 validation


def test_metrics_reflect_recorded_requests(client):
    client.post("/query", json={"question": "What was TestCo's FY2023 capex?"})
    client.post("/query", json={"question": "What was TestCo's FY2023 revenue?"})

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] == 2
    assert body["total_errors"] == 0
    assert body["total_prompt_tokens"] == 200
    assert body["estimated_total_cost_usd"] > 0
