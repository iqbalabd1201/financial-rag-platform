# Financial Document RAG: Retrieval & Generation over SEC Filings

A two-stage RAG pipeline for question-answering over 10-K/10-Q filings, built and evaluated
on [FinanceBench](https://github.com/patronus-ai/financebench). This project documents a full
experimental cycle — baselines, failed methods, root-cause diagnosis, and a final configuration
chosen on accuracy **and** cost, not accuracy alone.

## Live demo

The full pipeline is deployed as a REST API:
**https://financial-rag-platform-production.up.railway.app**

```bash
curl https://financial-rag-platform-production.up.railway.app/health

curl -X POST https://financial-rag-platform-production.up.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was 3M'"'"'s FY2018 capital expenditure?"}'
```

Runs on Railway's free tier (CPU-only) — the container may take 30–60s to wake up on the
first request after a period of inactivity. Built with FastAPI, containerized with Docker,
retrieval served from Qdrant Cloud; see `src/api/`, `src/indexing/qdrant_store.py`, and
`Dockerfile`. 28 tests (unit + API integration + a retrieval-regression gate) run on every
push via GitHub Actions — see [Testing & CI](#testing--ci).

## Why this project

Most public RAG demos report a single headline accuracy number. This one instead asks: *what's
the realistic ceiling for this task, what actually moves the needle, and what doesn't?* Every
claim below is backed by a logged experiment, and several "obvious" improvements (reranking,
larger embedding models, more assertive prompting) are shown to **fail or backfire**, with the
failure mode diagnosed rather than hand-waved.

## Results at a glance

**Stage 1 — Document identification** (given a question, which filing does it concern?)

| Method | Accuracy | Cost |
|---|---|---|
| String + alias + fiscal-year matching (no ML) | **96.7%** (58/60) | ~free, milliseconds |

The 2 residual failures are questions with no company name mentioned at all (follow-up-style
phrasing) — a fundamentally unsolvable case for single-turn matching, not a bug.

**Stage 2 — Page-level retrieval** (given the right document, which page has the answer?)

| Method | Hit@5 | Recall@5 | Cost |
|---|---|---|---|
| Dense, chunk-based (BGE-small, 800/150 chunks) | 39.0% | — | low |
| Dense, page-level (BGE-small) | 50.0% | 46.7% | low |
| BGE-M3, full page, no truncation | 51.7% | 48.3% | **17x model size, 16x embed time** |
| Cross-encoder reranker (bge-reranker-base) | 36.6% | — | **hurts accuracy** |
| **Page-level + section header (BGE-small)** | **58.3%** | **55.0%** | **low — final choice** |

Reranking was tested three separate ways (plain, RRF-fused, on richer page+header
representations) and **lost to the no-rerank baseline every time**. Root cause: the reranker is
biased toward narrative prose over tables of raw figures — independently confirmed by two
published papers (HiREC, T2-RAGBench) that report the same failure on similar data.

**End-to-end generation** (retrieval → LLM answer, graded by LLM-as-judge)

| Condition | Score |
|---|---|
| Closed-book (no retrieval) | 3.3% |
| **Full pipeline (Stage 1 + Stage 2 + generation)** | **41.7%** |
| Oracle (gold evidence page given directly) | 63.3% |

The oracle ceiling is **not** 100% — a meaningful share of failures are LLM reasoning/arithmetic
errors that persist even with perfect retrieval, not retrieval failures.

## Cross-validation against Ragas

The custom floor/pipeline/ceiling ablation above uses this project's own metric
vocabulary. To make the results legible to anyone unfamiliar with that setup, the
final pipeline was also scored with [Ragas](https://github.com/explodinggradients/ragas),
the standard open-source RAG evaluation framework, on a 30-question subset:

| Ragas metric | Score | What it measures |
|---|---|---|
| Faithfulness | **76.6%** | Are the claims in the answer supported by the retrieved context? |
| Answer relevancy | **68.8%** | Is the answer actually on-topic for the question asked? |
| Context precision | **55.6%** | Of the retrieved pages, how many were relevant? |
| Context recall | **41.4%** | Of what's needed to answer, how much did retrieval surface? |

Context precision/recall are **lower** than this project's own hit@5/recall@5
(58.3%/55.0%) — expected, since Ragas scores context relevance via LLM judgment
per-page rather than exact page-ID matching against gold evidence. The two
methodologies disagree on the exact number but agree on the direction: retrieval
is the bigger bottleneck than generation, consistent with the oracle-ceiling gap
(58.3% actual vs a much higher in-candidate ceiling) documented above.

Faithfulness (76.6%) landing well above context precision (55.6%) is a useful
signal on its own: the model is largely *not* hallucinating beyond what it's
given — most of the accuracy ceiling is a retrieval problem, not a generation
one. See `scripts/run_ragas_eval.py`.

## What didn't work (and why that's useful to know)

- **Cross-encoder reranking** — consistently made retrieval worse, at every candidate depth
  tested (top-30, top-50) and on two different index representations. Confirmed against
  literature: HiREC and T2-RAGBench independently report the same pattern for financial
  tabular documents.
- **Bigger embedding model (BGE-M3)** — a marginal +1.7pp accuracy gain cost 17x the model size
  and 16x the embedding time. Rejected on cost/accuracy ratio.
- **Reasoning-based page navigation (small LLM reading a table of contents)** — collapsed to
  16.7% hit@5, far below any embedding method. Likely needs a frontier-scale model (as used by
  PageIndex/Mafin 2.5), not a 1.5B local model.
- **Fine-tuning a document-ID classifier** — tested and rejected. Simple string matching already
  hits 96.7%; the residual failures are unsolvable by *any* classifier because the question text
  itself omits the company name.
- **More assertive generation prompting** — raised aggregate accuracy but **shifted some failures
  from honest abstention to confident wrong answers** (e.g., asserting "no spin-off planned"
  when the retrieved context simply didn't mention it, rather than correctly saying "not found").
  This trade-off is documented explicitly rather than hidden behind the aggregate score — for a
  financial-analyst-facing tool, a confidently wrong answer is a worse failure mode than an
  honest "I don't know," even at equal accuracy.

## Prompt iteration log (generation stage)

Four iterations, each isolating a specific failure mode rather than blindly re-prompting:

1. **v2** — explicit metric-definition rules + delegate arithmetic to a Python `eval()` call
   instead of LLM mental math (fixes rounding/division errors LLMs make even with correct data).
2. **v3** — retrieval depth k=5→10, plus instructions for confident negative assertions and
   unit-scale sanity checks. Raised accuracy but introduced confident-wrong answers on 4 questions.
3. **v4** — added stricter "don't substitute similar-but-wrong line items" caution. Did **not**
   reduce the confident-wrong count — it just moved to different questions, while net accuracy
   dropped. Documented as a negative result.
4. **v5 (final)** — few-shot examples targeting the *specific* reasoning errors seen (confirmed
   absence vs. topic simply not discussed; comparing signed percentages correctly). Fixed most
   targeted cases; one persisted because its root cause was a sign-extraction error, not the
   comparison logic the example addressed — a reminder that the fix must match the diagnosis.

### Tracked in Weights & Biases

The original per-version numbers from initial development weren't preserved in a
structured form, so the four prompt versions were re-run fresh on a 30-question
subset (same subset used for the Ragas evaluation) with results logged to W&B —
[project dashboard](https://wandb.ai/iqbalabd1201-binus-university/financial-rag-prompt-iteration):

| Version | Accuracy | Confident-wrong count |
|---|---|---|
| v2 | 60.0% | 12 |
| v3 | 56.7% | 13 |
| v4 | 63.3% | 10 |
| **v5 (final)** | **53.3%** | **4** |

This fresh run's absolute numbers differ from the original development session
(different subset size, natural LLM variance) — treat this table as an independent
reproduction of the trade-off, not a re-statement of the original numbers. The
**trade-off itself replicates cleanly**: v5 has the lowest raw accuracy of the four
but by far the lowest confident-wrong count, confirming it was the right choice for
a tool where a confidently wrong answer is worse than an honest "not found," even
at equal accuracy. "Confident-wrong" here is a keyword-based heuristic (see
`scripts/track_generation_prompts_wandb.py`), not a hand-labeled ground truth.

## Architecture

```
Query
  │
  ▼
[Stage 1] String/alias/fiscal-year match → target document (96.7%, ~free)
  │
  ▼
[Stage 2] Page-level dense retrieval, BGE-small + section header, top-10 (58.3% hit@5)
  │
  ▼
[Generation] gpt-4o-mini, few-shot prompt (v5) + Python calculator for arithmetic
  │
  ▼
Answer + cited pages
```

Served in production behind a FastAPI wrapper (`POST /query`, `GET /health`, `GET /metrics`),
containerized with Docker. Retrieval runs against Qdrant Cloud (migrated from local FAISS;
verified identical hit@5/recall@5 before switching over — see `src/indexing/qdrant_store.py`).
Every request logs structured JSON (query, matched doc, per-stage latency, token cost) to
stdout, and `/metrics` exposes aggregate stats — see `src/api/observability.py`.

## Testing & CI

28 tests, split by what they actually verify rather than by file:

| Layer | What it checks |
|---|---|
| `tests/test_doc_matcher.py` | Stage 1 string/alias/year-disambiguation matching, plus the fallback_needed case |
| `tests/test_calculator.py` | The CALC-line extraction/eval step, with a fake OpenAI client (no cost, no network) |
| `tests/test_retrieval_metrics.py` | hit@k / recall@k / candidate_ceiling — the functions the headline README numbers depend on |
| `tests/test_parse_pdf.py` | `clean_page()`, plus a regression guard asserting `PAGE_OFFSET == 1` (a discovered fact, not a default, per the offset-bug story above) |
| `tests/test_api.py` | `POST /query`, `GET /health`, `GET /metrics` end-to-end via FastAPI's TestClient, with Qdrant/OpenAI/embedder mocked |
| `tests/test_retrieval_regression.py` | Hit@5 on a 15-question subset must not collapse — using the **committed local FAISS bundle**, not live Qdrant, so this stays free and dependency-free in CI |

GitHub Actions (`.github/workflows/ci.yml`) runs the full suite on every push, then verifies
the `Dockerfile` still builds. Actual deployment stays on Railway's own GitHub integration
rather than being duplicated in CI — this workflow's job is to catch a broken commit before
Railway does, not to redeploy.

## Repository structure

```
financial-rag-platform/
├── README.md
├── requirements.txt
├── requirements-dev.txt            # pytest, httpx -- test-only, kept separate from runtime deps
├── requirements-eval.txt           # ragas, datasets -- offline eval only, kept out of the Docker image
├── pytest.ini
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml                  # push -> pytest -> verify Dockerfile builds
│
├── configs/
│   └── pipeline_config.yaml        # k, model names, prompt version — swappable
│
├── data/
│   ├── raw_pdfs/                   # sample filings (gitignored in full)
│   ├── parsed_cache/               # cached page-parse output
│   ├── qa_gold/                    # FinanceBench subset used, with evidence
│   └── index_store/                # pre-built FAISS index + metadata (local reproducibility only)
│
├── src/
│   ├── ingestion/
│   │   ├── download_corpus.py      # clone FinanceBench, copy the PDFs this eval uses
│   │   ├── parse_pdf.py            # pdfplumber + page-offset handling
│   │   └── clean_text.py           # header stripping, section-title extraction
│   │
│   ├── indexing/
│   │   ├── chunker.py              # chunk-based vs page-level (both, for comparison)
│   │   ├── embedder.py             # BGE-small / BGE-M3, swappable via config
│   │   ├── build_index.py          # FAISS index builder (local/reproducibility path)
│   │   ├── persistence.py          # save/load the local FAISS bundle as one unit
│   │   └── qdrant_store.py         # production vector store: connect, upsert, filtered search
│   │
│   ├── retrieval/
│   │   ├── doc_matcher.py          # Stage 1: string/alias/year matching
│   │   ├── retriever.py            # Stage 2: local FAISS retrieval (used by reproduce_from_scratch.py)
│   │   └── reranker.py             # kept, DISABLED by default — see docstring for why
│   │
│   ├── generation/
│   │   ├── prompts.py              # v2–v5 prompt history, not just the final one
│   │   └── generate_answer.py      # + Python-eval calculator step, optional token-usage capture
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py    # hit@k, recall@k, candidate-ceiling analysis
│   │   ├── answer_metrics.py       # numeric tolerance match + LLM-judge (bug-fixed)
│   │   └── run_eval.py
│   │
│   └── api/
│       ├── main.py                 # FastAPI app: POST /query, GET /health, GET /metrics
│       ├── schemas.py              # Pydantic request/response contracts
│       └── observability.py        # structured logging + in-memory metrics aggregation
│
├── scripts/
│   ├── reproduce_from_scratch.py   # free, deterministic retrieval reproducibility check
│   ├── reproduce_generation.py     # optional, costs ~$0.05-0.10 in OpenAI credit
│   ├── build_and_save_index.py     # one-time: build + persist the local FAISS bundle
│   ├── migrate_index_to_qdrant.py  # one-time: upload the FAISS bundle's vectors to Qdrant Cloud
│   ├── verify_qdrant_migration.py  # confirms hit@5/recall@5 unchanged after migration
│   ├── run_ragas_eval.py           # cross-validate against Ragas (faithfulness, context precision/recall)
│   └── track_generation_prompts_wandb.py  # log v2-v5 prompt comparison to Weights & Biases
│
├── tests/                           # 28 tests — see Testing & CI below
│   ├── conftest.py
│   ├── test_doc_matcher.py
│   ├── test_calculator.py
│   ├── test_retrieval_metrics.py
│   ├── test_parse_pdf.py
│   ├── test_api.py
│   └── test_retrieval_regression.py
│
├── experiments/                     # notebooks, one per major finding
│   ├── 01_offset_bug_discovery.ipynb
│   ├── 02_chunk_vs_page_level.ipynb
│   ├── 03_reranker_ablation.ipynb
│   ├── 04_scale_to_22docs.ipynb
│   └── 05_generation_prompt_iteration.ipynb
│
├── results/
│   ├── retrieval_comparison.csv
│   ├── generation_results.csv
│   └── literature_comparison.md     # HiREC, T2-RAGBench, Kobeissi et al. — with citations
│
└── docs/
    └── failure_analysis.md          # the confident-wrong trade-off, in full
```

## Known limitations

- Evaluated on a 60-question stratified subset (22 documents) of FinanceBench, not the full
  benchmark — chosen deliberately to keep iteration cycles fast; methodology generalizes but
  absolute numbers should be read as directional, not definitive.
- LLM-as-judge grading has a known conservative bias (confirmed via manual spot-checks during
  development); treat the 41.7% end-to-end figure as a lower bound under this grading method,
  not an exact ground truth.
- A few FinanceBench gold answers themselves contain minor inconsistencies (documented during
  development, e.g. a rounding mismatch on one balance-sheet figure) — consistent with data
  quality caveats raised in prior published work on this benchmark.
- The live demo API does not fall back to a gold document name when Stage 1 document matching
  fails (unlike the evaluation scripts, which have ground truth to fall back to) — a genuine
  "couldn't identify the filing" returns an HTTP 422 with guidance, rather than guessing.
