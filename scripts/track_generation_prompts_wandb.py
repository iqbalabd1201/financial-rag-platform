"""Track the v2-v5 generation-prompt iteration in Weights & Biases.

The ORIGINAL per-version accuracy/confident-wrong numbers from the initial
development sessions were not preserved -- rather than reconstruct them from
memory (which would mean putting unverifiable numbers in a portfolio piece),
this script RE-RUNS all four prompt versions fresh, on the same 30-question
subset used for the Ragas evaluation, and logs real, reproducible numbers.

Confident-wrong detection is a HEURISTIC, not ground truth: among answers
judged incorrect, one is flagged "confident_wrong" if it does NOT contain a
hedging/abstention phrase (see HEDGE_PHRASES below). This mirrors the
distinction documented in docs/failure_analysis.md, but text-pattern
matching is approximate -- treat the resulting rate as directional, not
exact, same caveat as the LLM-judge grading itself.

Cost: 4 prompt versions x 30 questions x (1 generation + up to 1 judge call)
~= 200-240 OpenAI calls, roughly $0.15-0.30 total.

Run from the repo root:
    set WANDB_API_KEY=...
    set OPENAI_API_KEY=...
    python scripts/track_generation_prompts_wandb.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import wandb
from openai import OpenAI

from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.retriever import retrieve_pages
from src.generation.generate_answer import build_context, generate_answer
from src.generation.prompts import PROMPT_V2, PROMPT_V3, PROMPT_V4, PROMPT_V5_FEWSHOT
from src.evaluation.answer_metrics import numeric_match, llm_judge

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
RETRIEVAL_K = 10
SUBSET_SIZE = 30

PROMPT_VERSIONS = {
    "v2": PROMPT_V2,
    "v3": PROMPT_V3,
    "v4": PROMPT_V4,
    "v5_final": PROMPT_V5_FEWSHOT,
}

# Approximate signal for "the model asserted something rather than hedging" --
# see module docstring for why this is heuristic, not ground truth.
HEDGE_PHRASES = [
    "cannot be confirmed", "insufficient information", "does not discuss",
    "not mentioned", "not specified", "no information", "unable to determine",
    "not found", "does not mention",
]


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return pd.DataFrame(data["questions"]).head(SUBSET_SIZE)


def is_hedged(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in HEDGE_PHRASES)


def run_one_version(client, model, index, pages, page_text_lookup_by_doc,
                     qa_df, version_name, system_prompt):
    run = wandb.init(
        project="financial-rag-prompt-iteration",
        name=version_name,
        config={"prompt_version": version_name, "retrieval_k": RETRIEVAL_K,
                "subset_size": len(qa_df), "model": "gpt-4o-mini"},
        reinit=True,
    )
    table = wandb.Table(columns=["question", "gold_answer", "generated_answer",
                                   "correct", "confident_wrong"])

    correct_count = 0
    confident_wrong_count = 0
    scored_count = 0

    for _, r in qa_df.iterrows():
        q_emb = embed_query(model, r["question"])
        retrieved = retrieve_pages(q_emb, index, pages, r["doc_name"], k=RETRIEVAL_K)
        context = build_context(pages, retrieved, page_text_lookup_by_doc.get(r["doc_name"], {}))

        answer_text, computed = generate_answer(
            client, r["question"], context, system_prompt=system_prompt
        )

        num_result = numeric_match(str(r["answer"]), computed)
        if num_result is not None:
            correct = num_result
        else:
            correct = llm_judge(client, r["question"], str(r["answer"]), answer_text)

        confident_wrong = False
        if correct is False:
            confident_wrong = not is_hedged(answer_text)
            confident_wrong_count += int(confident_wrong)
        if correct is not None:
            scored_count += 1
            correct_count += int(correct)

        table.add_data(r["question"][:100], str(r["answer"]), answer_text[:200],
                        correct, confident_wrong)

    accuracy = correct_count / scored_count if scored_count else 0.0
    wandb.log({
        "accuracy": accuracy,
        "confident_wrong_count": confident_wrong_count,
        "scored_count": scored_count,
        "results_table": table,
    })
    print(f"  {version_name}: accuracy={accuracy:.1%}, "
          f"confident_wrong={confident_wrong_count}/{scored_count}")
    run.finish()
    return accuracy, confident_wrong_count


def main():
    print("=== Step 1: load gold subset + local index ===")
    qa_df = load_gold_qa()
    index, pages, company_lookup = load_index_bundle("data/index_store")
    print(f"Evaluating {len(qa_df)} questions across {len(PROMPT_VERSIONS)} prompt versions.")

    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    print("\n=== Step 2: load embedder, connect to OpenAI ===")
    model = load_embedder(device="cpu")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("\n=== Step 3: run + log each prompt version ===")
    summary = {}
    for version_name, prompt in PROMPT_VERSIONS.items():
        print(f"\n--- {version_name} ---")
        acc, cw = run_one_version(
            client, model, index, pages, page_text_lookup_by_doc, qa_df, version_name, prompt
        )
        summary[version_name] = {"accuracy": acc, "confident_wrong": cw}

    print("\n" + "=" * 50)
    print("SUMMARY (fresh run, 30-question subset)")
    print("=" * 50)
    for name, s in summary.items():
        print(f"  {name:10s}: accuracy={s['accuracy']:.1%}  confident_wrong={s['confident_wrong']}")
    print("\nView the comparison charts at https://wandb.ai (project: financial-rag-prompt-iteration)")


if __name__ == "__main__":
    main()