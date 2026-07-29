"""FastAPI wrapper around the existing retrieval + generation pipeline.

Nothing in src/retrieval, src/indexing, or src/generation is modified --
this module only wires the already-tested functions into an HTTP endpoint.
Same functions, same config (configs/pipeline_config.yaml), same behavior
as run_eval.py's loop, minus the gold-answer scoring step.

Startup cost this solves: the embedding model (BGE-small) and the FAISS
index must be loaded exactly ONCE per process, not per request -- loading
either inside the request handler would make every query take as long as
a cold model load. FastAPI's lifespan context manager runs once at
process startup and stores everything the request handler needs in
app.state.

Run locally:
    export INDEX_STORE_DIR=data/index_store   # or the Drive path used to build it
    export OPENAI_API_KEY=sk-...
    uvicorn src.api.main:app --reload
"""
import os
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from openai import OpenAI

from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.indexing.qdrant_store import get_qdrant_client, retrieve_pages_qdrant
from src.retrieval.doc_matcher import match_document
from src.generation.generate_answer import build_context, generate_answer
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse, MetricsResponse
from src.api.observability import setup_logging, log_query_event, estimate_cost_usd, MetricsTracker

INDEX_STORE_DIR = os.environ.get("INDEX_STORE_DIR", "data/index_store")
RETRIEVAL_K = 10  # matches configs/pipeline_config.yaml (top_k: 10, the final config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading index bundle (pages + company_lookup) from {INDEX_STORE_DIR} ...")
    index, pages, company_lookup = load_index_bundle(INDEX_STORE_DIR)
    # `index` (local FAISS) is kept only for the count-consistency check inside
    # load_index_bundle and for /health reporting -- actual retrieval now goes
    # through Qdrant, not this object. See src/indexing/qdrant_store.py.

    embed_device = os.environ.get("EMBEDDER_DEVICE", "cpu")
    print(f"Loading embedder (BGE-small, device={embed_device}) ...")
    embed_model = load_embedder(device=embed_device)

    print("Connecting to Qdrant Cloud ...")
    qdrant_client = get_qdrant_client()

    # Same page_text_lookup_by_doc shape as run_eval.py -- {doc_id: {page_num: text}}
    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set -- generation calls will fail.")
    openai_client = OpenAI(api_key=api_key)

    app.state.index = index
    app.state.pages = pages
    app.state.company_lookup = company_lookup
    app.state.doc_ids = list(company_lookup.keys())
    app.state.embed_model = embed_model
    app.state.qdrant_client = qdrant_client
    app.state.page_text_lookup_by_doc = page_text_lookup_by_doc
    app.state.openai_client = openai_client
    app.state.logger = setup_logging()
    app.state.metrics = MetricsTracker()

    print(f"Ready: {index.ntotal} vectors, {len(company_lookup)} documents, "
          f"retrieval via Qdrant.")
    yield
    # No teardown needed -- FAISS index, model, and Qdrant client are in-memory only.


app = FastAPI(title="Financial RAG API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    index = getattr(app.state, "index", None)
    return HealthResponse(
        status="ok" if index is not None else "not_ready",
        index_loaded=index is not None,
        n_pages=index.ntotal if index is not None else 0,
        n_documents=len(app.state.company_lookup) if index is not None else 0,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return MetricsResponse(**app.state.metrics.snapshot())


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    overall_start = time.perf_counter()
    state = app.state

    matched_doc, method = match_document(
        request.question, state.doc_ids, state.company_lookup
    )
    if matched_doc is None:
        # Unlike run_eval.py, there is no gold doc_name to fall back to in
        # production -- a genuine "couldn't identify the filing" is a 422,
        # not a silent wrong-document guess.
        state.logger.info(json.dumps({
            "event": "query", "status": "doc_match_failed",
            "question_preview": request.question[:80],
        }))
        raise HTTPException(
            status_code=422,
            detail="Could not identify which document this question refers to. "
                   "Try including the company name and fiscal year/quarter.",
        )

    retrieval_start = time.perf_counter()
    q_emb = embed_query(state.embed_model, request.question)
    retrieved_pages = retrieve_pages_qdrant(
        state.qdrant_client, q_emb, matched_doc, k=RETRIEVAL_K
    )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    context = build_context(
        state.pages, retrieved_pages,
        state.page_text_lookup_by_doc.get(matched_doc, {}),
    )

    generation_start = time.perf_counter()
    usage: dict = {}
    answer_text, computed_value = generate_answer(
        state.openai_client, request.question, context, usage_out=usage
    )
    generation_ms = (time.perf_counter() - generation_start) * 1000

    total_ms = (time.perf_counter() - overall_start) * 1000
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    cost = (estimate_cost_usd(prompt_tokens, completion_tokens)
            if prompt_tokens is not None else None)

    log_query_event(
        state.logger, question=request.question, doc_id=matched_doc,
        doc_match_method=method, retrieved_pages=retrieved_pages,
        retrieval_ms=retrieval_ms, generation_ms=generation_ms, total_ms=total_ms,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        status="ok",
    )
    state.metrics.record(
        retrieval_ms=retrieval_ms, generation_ms=generation_ms,
        prompt_tokens=prompt_tokens or 0, completion_tokens=completion_tokens or 0,
    )

    return QueryResponse(
        answer=answer_text,
        doc_id=matched_doc,
        doc_match_method=method,
        retrieved_pages=retrieved_pages,
        computed_value=computed_value,
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(generation_ms, 1),
        latency_ms=round(total_ms, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=round(cost, 6) if cost is not None else None,
    )
