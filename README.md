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
first request after a period of inactivity. Built with FastAPI, containerized with Docker;
see `src/api/` and `Dockerfile`.

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

Served in production behind a FastAPI wrapper (`POST /query`, `GET /health`), containerized
with Docker, with the index bundle pre-built and loaded once at startup rather than rebuilt
per request — see `src/api/main.py` and `src/indexing/persistence.py`.

## Repository structure

```
financial-rag-platform/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── configs/
│   └── pipeline_config.yaml        # k, model names, prompt version — swappable
│
├── data/
│   ├── raw_pdfs/                   # sample filings (gitignored in full)
│   ├── parsed_cache/               # cached page-parse output
│   ├── qa_gold/                    # FinanceBench subset used, with evidence
│   └── index_store/                # pre-built FAISS index + metadata, loaded by the API
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
│   │   ├── build_index.py          # FAISS index builder
│   │   └── persistence.py          # save/load the index bundle as one unit (for API serving)
│   │
│   ├── retrieval/
│   │   ├── doc_matcher.py          # Stage 1: string/alias/year matching
│   │   ├── retriever.py            # Stage 2: dense retrieval + page-header index
│   │   └── reranker.py             # kept, DISABLED by default — see docstring for why
│   │
│   ├── generation/
│   │   ├── prompts.py              # v2–v5 prompt history, not just the final one
│   │   └── generate_answer.py      # + Python-eval calculator step
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py    # hit@k, recall@k, candidate-ceiling analysis
│   │   ├── answer_metrics.py       # numeric tolerance match + LLM-judge (bug-fixed)
│   │   └── run_eval.py
│   │
│   └── api/
│       ├── main.py                 # FastAPI app: POST /query, GET /health
│       └── schemas.py              # Pydantic request/response contracts
│
├── scripts/
│   ├── reproduce_from_scratch.py   # free, deterministic retrieval reproducibility check
│   ├── reproduce_generation.py     # optional, costs ~$0.05-0.10 in OpenAI credit
│   └── build_and_save_index.py     # one-time: build + persist the index bundle for the API
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
