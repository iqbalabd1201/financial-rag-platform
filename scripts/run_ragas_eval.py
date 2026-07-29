"""Cross-validate the project's custom floor/pipeline/ceiling ablation against
Ragas, the standard RAG evaluation framework -- so results are legible to
anyone who doesn't know this repo's custom metric vocabulary.

Uses the SAME pipeline as reproduce_generation.py (local FAISS retrieval +
v5 few-shot generation) -- this is an offline evaluation script, not a
production integration, so it stays independent of the live Qdrant/Railway
API by design.

Cost note: each Ragas metric makes its own LLM calls per row, on top of
the generation call itself -- roughly 4-5x the cost of plain generation
eval. Defaults to a 30-question subset to keep this in the $0.20-0.40
range; pass --full for all 60 (~$0.50-1.00).

Run from the repo root:
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --full
"""
import sys
import os
import json
import argparse
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Workaround for a known ragas bug (github.com/vibrantlabsai/ragas/issues/2745):
# ragas/llms/base.py unconditionally imports ChatVertexAI from a
# langchain_community path that was moved to langchain-google-vertexai.
# We never use VertexAI in this project -- this shim just satisfies the
# import so ragas can load at all, on any ragas version affected by the bug.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI:  # placeholder, never instantiated
        pass
    _shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

import pandas as pd
from openai import OpenAI
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.retriever import retrieve_pages
from src.generation.generate_answer import generate_answer

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
RETRIEVAL_K = 10  # matches the final pipeline config
SUBSET_SIZE_DEFAULT = 30


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return pd.DataFrame(data["questions"])


def build_ragas_row(client, model, index, pages, page_text_lookup_by_doc, row):
    """Run the real pipeline for one question and shape the result the way
    Ragas expects: question / answer / contexts (list of strings) /
    ground_truth."""
    q_emb = embed_query(model, row["question"])
    retrieved_pages = retrieve_pages(q_emb, index, pages, row["doc_name"], k=RETRIEVAL_K)

    page_texts = page_text_lookup_by_doc.get(row["doc_name"], {})
    contexts = [page_texts.get(pn, "") for pn in retrieved_pages if page_texts.get(pn)]
    context_str = "\n\n".join(f"--- Page {pn} ---\n{page_texts.get(pn, '')}"
                               for pn in retrieved_pages)

    answer_text, _ = generate_answer(client, row["question"], context_str)

    return {
        "question": row["question"],
        "answer": answer_text,
        "contexts": contexts if contexts else ["(no context retrieved)"],
        "ground_truth": str(row["answer"]),
    }


def main(subset_size: int):
    print("=== Step 1: load gold questions + local index ===")
    qa_df = load_gold_qa().head(subset_size)
    index, pages, company_lookup = load_index_bundle("data/index_store")
    print(f"Evaluating {len(qa_df)} questions.")

    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    print("\n=== Step 2: load embedder, connect to OpenAI ===")
    model = load_embedder(device="cpu")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("\n=== Step 3: run pipeline for each question ===")
    rows = []
    for i, (_, row) in enumerate(qa_df.iterrows()):
        rows.append(build_ragas_row(client, model, index, pages, page_text_lookup_by_doc, row))
        print(f"  {i + 1}/{len(qa_df)} done", end="\r")
    print()

    dataset = Dataset.from_list(rows)

    print("\n=== Step 4: score with Ragas (this makes additional LLM calls) ===")
    ragas_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ragas_embeddings = OpenAIEmbeddings()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n" + "=" * 50)
    print("RAGAS RESULTS")
    print("=" * 50)
    print(result)
    print("\nCompare against this project's own ablation (README.md):")
    print("  Floor (no retrieval):    3.3%")
    print("  Full pipeline (v5):     41.7%")
    print("  Oracle (gold context):  63.3%")
    print("\nRagas's context_precision/context_recall assess retrieval quality "
          "independently of generation -- a useful cross-check against this "
          "project's hit@5/recall@5 (58.3%/55.0%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all 60 questions instead of the 30-question default")
    args = parser.parse_args()
    main(60 if args.full else SUBSET_SIZE_DEFAULT)