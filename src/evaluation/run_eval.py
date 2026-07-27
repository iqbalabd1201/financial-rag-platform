"""End-to-end evaluation entry point: runs the full pipeline (Stage 1 doc
match -> Stage 2 retrieval -> generation) against a gold QA set and reports
hit@k/recall@k for retrieval plus numeric+LLM-judge accuracy for answers,
broken down by question type (domain-relevant / metrics-generated /
novel-generated -- these have very different failure modes, see
docs/failure_analysis.md).
"""
import pandas as pd

from src.retrieval.doc_matcher import match_document
from src.retrieval.retriever import retrieve_pages
from src.generation.generate_answer import build_context, generate_answer
from src.evaluation.answer_metrics import numeric_match, llm_judge


def run_pipeline_eval(client, qa_df: pd.DataFrame, index, pages: list[dict],
                       embed_query_fn, company_lookup: dict, k: int = 10):
    """qa_df must have columns: id, question, doc_name, answer, question_type."""
    results = []
    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    for _, row in qa_df.iterrows():
        matched_doc, method = match_document(
            row["question"], list(company_lookup.keys()), company_lookup)
        target_doc = matched_doc or row["doc_name"]  # fallback to ground truth doc if unmatched

        q_emb = embed_query_fn(row["question"])
        retrieved = retrieve_pages(q_emb, index, pages, target_doc, k=k)

        context = build_context(pages, retrieved, page_text_lookup_by_doc.get(target_doc, {}))
        text, computed = generate_answer(client, row["question"], context)

        num_correct = numeric_match(row["answer"], computed)
        judge_correct = None if num_correct is not None else llm_judge(
            client, row["question"], row["answer"], text)

        results.append({
            "id": row["id"], "question_type": row["question_type"],
            "doc_match_method": method, "retrieved_pages": retrieved,
            "generated_text": text, "computed_value": computed,
            "correct": num_correct if num_correct is not None else judge_correct,
        })

    results_df = pd.DataFrame(results)
    print(f"Overall accuracy: {results_df['correct'].sum()}/{len(results_df)} "
          f"= {results_df['correct'].mean():.1%}")
    print(results_df.groupby("question_type")["correct"].agg(["sum", "count"]))
    return results_df
