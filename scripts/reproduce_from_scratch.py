"""End-to-end reproducibility check: clone FinanceBench, parse 22 raw PDFs,
rebuild the retrieval index, and verify Stage 1 / Stage 2 metrics match the
numbers reported in README.md.

Doc names and the 60-question gold set come directly from
data/qa_gold/sample_60_stratified.json (committed to this repo) -- only
the PDF PARSING is redone from scratch here. This keeps the eval set
transparent and auditable: open that file to see exactly which questions
and evidence pages are used, no derivation logic to trust.

Run from the repo root:
    python scripts/reproduce_from_scratch.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.download_corpus import clone_financebench, copy_pdfs
from src.ingestion.parse_pdf import parse_corpus
from src.ingestion.clean_text import page_header, company_token
from src.indexing.embedder import load_embedder, embed_pages, embed_query
from src.indexing.build_index import build_flat_index
from src.retrieval.doc_matcher import match_document
from src.retrieval.retriever import retrieve_pages
from src.evaluation.retrieval_metrics import hit_at_k, recall_at_k

import pandas as pd

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
TARGETS = {"n_pages": 3061, "n_questions": 60, "stage1_acc": 0.967,
           "hit5": 0.583, "recall5": 0.550}


def load_gold_qa(path: str = GOLD_QA_PATH):
    """Doc names and questions come straight from this file -- nothing
    re-derived. Expects the format: {"docs": [...], "questions": [...]}."""
    data = json.load(open(path))
    doc_ids = data["docs"]
    qa_df = pd.DataFrame(data["questions"])
    return doc_ids, qa_df


def main():
    print("=== Step 1: load doc list + 60 gold questions from repo file ===")
    doc_ids, qa_sub = load_gold_qa()
    print(f"Documents: {len(doc_ids)} | Questions: {len(qa_sub)} "
          f"(target: {TARGETS['n_questions']})")
    print(qa_sub["question_type"].value_counts().to_dict())

    print("\n=== Step 2: clone FinanceBench, copy the PDFs listed above ===")
    fb_dir = clone_financebench()
    pdf_dir = "data/raw_pdfs"
    missing = copy_pdfs(fb_dir, pdf_dir, doc_ids)
    if missing:
        raise RuntimeError(f"Missing PDFs, check names against FinanceBench repo: {missing}")
    print(f"{len(doc_ids)}/{len(doc_ids)} PDFs copied to {pdf_dir}")

    print("\n=== Step 3: parse PDFs from scratch (no cache) ===")
    pages = parse_corpus(pdf_dir, doc_ids)
    print(f"Pages parsed: {len(pages)} (target: {TARGETS['n_pages']})")

    print("\n=== Step 4: build retrieval index (page + section header) ===")
    company_lookup = {d: company_token(d) for d in doc_ids}
    model = load_embedder()
    embeddings = embed_pages(model, pages, company_lookup, page_header)
    index = build_flat_index(embeddings)
    print(f"Index vectors: {index.ntotal}")

    print("\n=== Step 5: Stage 1 -- document identification ===")
    stage1_hits = [
        match_document(r["question"], doc_ids, company_lookup)[0] == r["doc_name"]
        for _, r in qa_sub.iterrows()
    ]
    stage1_acc = sum(stage1_hits) / len(stage1_hits)
    print(f"Stage 1 accuracy: {stage1_acc:.1%} (target: {TARGETS['stage1_acc']:.1%})")

    print("\n=== Step 6: Stage 2 -- page retrieval, hit@5 / recall@5 ===")
    hits, recalls = [], []
    for _, r in qa_sub.iterrows():
        gold = {p + PAGE_OFFSET for p in r["evidence_pages"]}
        q_emb = embed_query(model, r["question"])
        retrieved = retrieve_pages(q_emb, index, pages, r["doc_name"], k=5)
        hits.append(hit_at_k(gold, retrieved, 5))
        recalls.append(recall_at_k(gold, retrieved, 5))
    hit5, recall5 = sum(hits) / len(hits), sum(recalls) / len(recalls)
    print(f"Hit@5: {hit5:.1%} (target: {TARGETS['hit5']:.1%})")
    print(f"Recall@5: {recall5:.1%} (target: {TARGETS['recall5']:.1%})")

    print("\n" + "=" * 50)
    print("REPRODUCIBILITY SUMMARY")
    print("=" * 50)
    checks = [
        ("Pages parsed", len(pages), TARGETS["n_pages"], 0),
        ("Questions loaded", len(qa_sub), TARGETS["n_questions"], 0),
        ("Stage 1 accuracy", stage1_acc, TARGETS["stage1_acc"], 0.02),
        ("Hit@5", hit5, TARGETS["hit5"], 0.02),
        ("Recall@5", recall5, TARGETS["recall5"], 0.02),
    ]
    all_ok = True
    for name, actual, target, tol in checks:
        ok = abs(actual - target) <= tol
        all_ok &= ok
        print(f"  {name:20s}: {actual}  {'MATCH' if ok else 'MISMATCH -- investigate'}  (target {target})")
    print("\nRESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see above")


if __name__ == "__main__":
    main()
