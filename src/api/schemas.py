from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about a filing.")


class QueryResponse(BaseModel):
    answer: str
    doc_id: str | None
    doc_match_method: str
    retrieved_pages: list[int]
    computed_value: float | None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    n_pages: int
    n_documents: int
