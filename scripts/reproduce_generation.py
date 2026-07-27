"""Optional end-to-end generation check (retrieval + LLM answer).

Unlike reproduce_from_scratch.py, this step costs a small amount of OpenAI
API credit (~$0.05-0.10 total for 180 calls across floor/v5/ceiling) and
is NOT guaranteed bit-for-bit reproducible run to run, since it depends on
an external API rather than a local deterministic model. It is kept as a
separate, explicitly-invoked script for exactly that reason -- retrieval
(reproduce_from_scratch.py) stays a free, deterministic smoke test anyone
can run with zero setup; this one is opt-in.

Requires: an OpenAI API key, passed via --api-key or the OPENAI_API_KEY
environment variable.

Run from the repo root, AFTER reproduce_from_scratch.py has already run
successfully once (this script re-parses/re-embeds from scratch too, so
it can be run standalone, but expect it to take a similar amount of time
for the retrieval portion before generation starts):

    python scripts/reproduce_generation.py --api-key sk-...
    # or
    export OPENAI_API_KEY=sk-...
    python scripts/reproduce_generation.py
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from src.ingestion.download_corpus import clone_financebench, copy_pdfs
from src.ingestion.parse_pdf import parse_corpus
from src.ingestion.clean_text import page_header, company_token
from src.indexing.embedder import load_embedder, embed_pages, embed_query
from src.indexing.build_index import build_flat_index
from src.retrieval.retriever import retrieve_pages
from src.generation.prompts import PROMPT_V5_FEWSHOT
from src.generation.generate_answer import build_context, generate_answer
from src.evaluation.answer_metrics import numeric_match, llm_judge

import pandas as pd

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
RETRIEVAL_K = 10  # matches the final config: k=10 + few-shot prompt v5

# These are the numbers reported in README.md under "End-to-end generation".
# Generation results will not match exactly run-to-run (see module docstring)
# -- treat these as a sanity range, not an exact target like retrieval's.
TARGETS = {"floor": 0.033, "v5_pipeline": 0.417, "ceiling": 0.633}
TOLERANCE = 0.10  # generation is noisier than retrieval; allow +/-10pp


def setup_retrieval():
    """Re-run the same setup as reproduce_from_scratch.py so this script
    can also be run standalone. If you already ran that script in this
    session, this just repeats the (cheap, local) retrieval build.
    """
    doc_ids, qa_sub = load_gold_qa()
    fb_dir = clone_financebench()
    pdf_dir = "data/raw_pdfs"
    copy_pdfs(fb_dir, pdf_dir, doc_ids)
    pages = parse_corpus(pdf_dir, doc_ids)

    company_lookup = {d: company_token(d) for d in doc_ids}
    model = load_embedder()
    embeddings = embed_pages(model, pages, company_lookup, page_header)
    index = build_flat_index(embeddings)

    page_text_by_doc = {}
    for p in pages:
        page_text_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    return doc_ids, qa_sub, pages, model, index, page_text_by_doc


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return data["docs"], pd.DataFrame(data["questions"])


def score_condition(client, qa_sub, get_context_fn, label):
    """Run generation for one condition (floor/pipeline/ceiling) and score it."""
    correct = 0
    total = 0
    for _, r in qa_sub.iterrows():
        context = get_context_fn(r)
        text, computed = generate_answer(client, r["question"], context,
                                          system_prompt=PROMPT_V5_FEWSHOT)
        num_result = numeric_match(r["answer"], computed)
        if num_result is not None:
            correct += int(num_result)
        else:
            judge_result = llm_judge(client, r["question"], r["answer"], text)
            correct += int(bool(judge_result))
        total += 1
    score = correct / total
    print(f"{label}: {correct}/{total} = {score:.1%}")
    return score


def main(api_key: str):
    client = OpenAI(api_key=api_key)

    print("=== Setting up retrieval (same pipeline as reproduce_from_scratch.py) ===")
    doc_ids, qa_sub, pages, model, index, page_text_by_doc = setup_retrieval()

    print("\n=== Floor: no retrieval (closed-book) ===")
    floor_score = score_condition(client, qa_sub, lambda r: None, "Floor")

    print("\n=== Pipeline: k=10 retrieval + few-shot prompt (v5, final config) ===")
    def pipeline_context(r):
        q_emb = embed_query(model, r["question"])
        retrieved = retrieve_pages(q_emb, index, pages, r["doc_name"], k=RETRIEVAL_K)
        return build_context(pages, retrieved, page_text_by_doc.get(r["doc_name"], {}))
    pipeline_score = score_condition(client, qa_sub, pipeline_context, "Pipeline (v5, k=10)")

    print("\n=== Ceiling: gold evidence given directly (oracle) ===")
    ceiling_score = score_condition(
        client, qa_sub, lambda r: "\n\n".join(r["evidence_text"]), "Ceiling")

    print("\n" + "=" * 50)
    print("GENERATION REPRODUCIBILITY SUMMARY (sanity range, not exact match)")
    print("=" * 50)
    for name, actual, target in [
        ("Floor", floor_score, TARGETS["floor"]),
        ("Pipeline (v5)", pipeline_score, TARGETS["v5_pipeline"]),
        ("Ceiling", ceiling_score, TARGETS["ceiling"]),
    ]:
        ok = abs(actual - target) <= TOLERANCE
        print(f"  {name:15s}: {actual:.1%}  {'within range' if ok else 'OUTSIDE range -- investigate'}  "
              f"(reported: {target:.1%}, tolerance +/-{TOLERANCE:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit(
            "No API key found. Pass --api-key sk-... or set OPENAI_API_KEY.")
    main(args.api_key)
