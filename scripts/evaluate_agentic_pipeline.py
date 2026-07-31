"""Compare the v5 single-shot baseline against the LangGraph agentic retry
loop, on the same 30-question subset used for the Ragas/W&B evaluations.

Uses local FAISS retrieval (offline eval script, independent of the live
Qdrant/Railway API, matching reproduce_generation.py's design).

Tracks, beyond plain accuracy:
  - retry_engagement_rate: fraction of questions where the agent actually
    triggered a retry (if this is ~0%, the assess step isn't doing anything
    useful and the mechanism needs debugging, regardless of accuracy)
  - flip_to_correct / flip_to_wrong: among questions the baseline got
    wrong, how many did the agentic loop fix vs. break further -- the
    headline number a retry mechanism should be judged on, not aggregate
    accuracy alone (a mechanism that flips 3 wrong->right and 3 right->wrong
    nets to zero accuracy change but is NOT doing nothing)

Run from the repo root (no GPU needed):
    python scripts/evaluate_agentic_pipeline.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from openai import OpenAI
from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.retriever import retrieve_pages
from src.generation.generate_answer import build_context, generate_answer
from src.generation.agentic_pipeline import build_agentic_graph, run_agentic_query
from src.evaluation.answer_metrics import numeric_match, llm_judge

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
RETRIEVAL_K = 10
SUBSET_SIZE = 60


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return pd.DataFrame(data["questions"]).head(SUBSET_SIZE)


def score(client, question, gold_answer, generated_text, computed):
    num_result = numeric_match(str(gold_answer), computed)
    if num_result is not None:
        return num_result
    return llm_judge(client, question, str(gold_answer), generated_text)


def main():
    print("=== Step 1: load gold subset + local index ===")
    qa_df = load_gold_qa()
    index, pages, company_lookup = load_index_bundle("data/index_store")
    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    print("\n=== Step 2: load embedder, connect to OpenAI ===")
    model = load_embedder(device="cpu")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    agentic_graph = build_agentic_graph(client, model, index, pages, page_text_lookup_by_doc)

    baseline_correct = {}
    agentic_correct = {}
    retry_triggered_count = 0

    print("\n=== Step 3: run baseline (single-shot) + agentic (retry loop) per question ===")
    for i, (_, r) in enumerate(qa_df.iterrows()):
        # Baseline: exactly one retrieve + one generate, same as reproduce_generation.py
        q_emb = embed_query(model, r["question"])
        retrieved = retrieve_pages(q_emb, index, pages, r["doc_name"], k=RETRIEVAL_K)
        context = build_context(pages, retrieved, page_text_lookup_by_doc.get(r["doc_name"], {}))
        baseline_text, baseline_computed = generate_answer(client, r["question"], context)
        baseline_ok = score(client, r["question"], r["answer"], baseline_text, baseline_computed)
        baseline_correct[i] = bool(baseline_ok)

        # Agentic: retrieve -> generate -> assess -> maybe retry
        result = run_agentic_query(agentic_graph, r["question"], r["doc_name"])
        agentic_ok = score(client, r["question"], r["answer"], result["answer"], result["computed_value"])
        agentic_correct[i] = bool(agentic_ok)
        if result["retry_count"] > 0 and len(result["retrieved_pages"]) > RETRIEVAL_K:
            retry_triggered_count += 1

        print(f"  {i + 1}/{len(qa_df)}: baseline={'OK' if baseline_ok else 'X'} "
              f"agentic={'OK' if agentic_ok else 'X'} retries={result['retry_count']}", end="\r")
    print()

    n = len(qa_df)
    baseline_acc = sum(baseline_correct.values()) / n
    agentic_acc = sum(agentic_correct.values()) / n
    flip_to_correct = sum(1 for i in range(n) if not baseline_correct[i] and agentic_correct[i])
    flip_to_wrong = sum(1 for i in range(n) if baseline_correct[i] and not agentic_correct[i])

    print("\n" + "=" * 55)
    print(f"RESULTS on {n} questions")
    print("=" * 55)
    print(f"  Baseline (single-shot) accuracy: {baseline_acc:.1%}")
    print(f"  Agentic (retry loop) accuracy:   {agentic_acc:.1%}")
    print(f"  Retry actually triggered on:     {retry_triggered_count}/{n} questions")
    print(f"  Flipped wrong -> correct:        {flip_to_correct}")
    print(f"  Flipped correct -> wrong:        {flip_to_wrong}")


if __name__ == "__main__":
    main()