import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from openai import OpenAI

from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.doc_matcher import match_document
from src.retrieval.retriever import retrieve_pages
from src.generation.generate_answer import build_context, generate_answer
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse

INDEX_STORE_DIR = os.environ.get("INDEX_STORE_DIR", "data/index_store")
RETRIEVAL_K = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading index bundle from {INDEX_STORE_DIR} ...")
    index, pages, company_lookup = load_index_bundle(INDEX_STORE_DIR)


    embed_device = os.environ.get("EMBEDDER_DEVICE", "cpu")
    print(f"Loading embedder (BGE-small, device={embed_device}) ...")
    embed_model = load_embedder(device=embed_device)

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
    app.state.page_text_lookup_by_doc = page_text_lookup_by_doc
    app.state.openai_client = openai_client

    print(f"Ready: {index.ntotal} vectors, {len(company_lookup)} documents.")
    yield


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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.perf_counter()
    state = app.state

    matched_doc, method = match_document(
        request.question, state.doc_ids, state.company_lookup
    )
    if matched_doc is None:
        raise HTTPException(
            status_code=422,
            detail="Could not identify which document this question refers to. "
                   "Try including the company name and fiscal year/quarter.",
        )

    q_emb = embed_query(state.embed_model, request.question)
    retrieved_pages = retrieve_pages(
        q_emb, state.index, state.pages, matched_doc, k=RETRIEVAL_K
    )

    context = build_context(
        state.pages, retrieved_pages,
        state.page_text_lookup_by_doc.get(matched_doc, {}),
    )
    answer_text, computed_value = generate_answer(
        state.openai_client, request.question, context
    )

    return QueryResponse(
        answer=answer_text,
        doc_id=matched_doc,
        doc_match_method=method,
        retrieved_pages=retrieved_pages,
        computed_value=computed_value,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )
