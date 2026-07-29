"""Request/response contracts for POST /query.

Deliberately minimal on input (question only, per the current decision) --
k and doc_id overrides can be added later as optional fields without
breaking existing callers, since Pydantic fields with defaults are
backward compatible.

The response includes retrieved_pages and doc_id (not just the answer
text) on purpose: for a financial-analyst-facing tool, "which pages was
this based on" is what makes an answer auditable, not an afterthought.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about a filing.")


class QueryResponse(BaseModel):
    answer: str
    doc_id: str | None
    doc_match_method: str
    retrieved_pages: list[int]
    computed_value: float | None
    retrieval_ms: float
    generation_ms: float
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    total_errors: int
    avg_retrieval_ms: float
    avg_generation_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_total_cost_usd: float


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    n_pages: int
    n_documents: int
