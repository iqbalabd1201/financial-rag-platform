# Literature comparison

| Source | Task | Metric | Result |
|---|---|---|---|
| This project | FinanceBench, 22 docs / 60 questions, page-level, in-document | Hit@5 | 58.3% |
| HiREC (arXiv 2505.20368) | FinanceBench, 150q/368 docs, open-domain | Page recall | 40.0% (dense baseline: 26.1%) |
| T2-RAGBench (ACL 2026.eacl-long.8) | FinQA/ConvFinQA/TAT-DQA | R@3 | Base RAG 39.8%, +reranker 33.0% (reranker hurts, same as here) |
| Kobeissi & Langlais 2026 (arXiv 2602.17981) | FinanceBench, 150q, BGE-M3 | Page recall@5 | Dense 34%, Oracle-Document ceiling 60% |
| MimirRAG | FinanceBench, 150q | End-to-end answer accuracy | 89.3% (multi-agent, metadata filtering, calculator tool) |

Note the task-difficulty gradient: this project's in-document page retrieval (document already
known) is an easier sub-task than the open-domain retrieval most of these papers report on --
the numbers are not directly comparable without accounting for that, which is why hit@5 here
(58.3%) sits well above HiREC's open-domain page recall (40.0%) despite using a much simpler
method (no reranker, no hierarchical retrieval, no agentic evidence curation).
