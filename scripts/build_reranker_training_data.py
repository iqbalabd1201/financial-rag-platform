"""Build (query, page_text, label) training pairs for the LoRA reranker
fine-tune, from the same 60-question gold set used everywhere else in
this project.

Design decisions worth being explicit about:

1. Train/held-out split is at the QUESTION level (45 train / 15 held-out),
   not the pair level. If pairs from the same question ended up in both
   train and eval, the reranker could partly memorize question-specific
   surface patterns rather than learn general relevance discrimination --
   the held-out hit@5 comparison later would then be inflated and
   dishonest. The 15 held-out questions are NEVER touched during training.

2. Negatives are HARD negatives: pages that the existing bi-encoder
   retrieval (top-30 candidates) surfaced as similar enough to retrieve,
   but that are NOT the gold evidence page. This directly targets the
   reranker's actual job -- discriminating plausible-but-wrong pages from
   the correct one -- rather than trivial negatives from unrelated
   documents, which off-the-shelf bge-reranker-base already handles fine
   and wouldn't need fine-tuning for.

3. The true gold evidence page is always included as a positive even if
   it fell outside top-30 (a candidate-ceiling miss) -- discarding it
   would silently shrink the positive class for exactly the hardest,
   most informative questions.

Output: data/qa_gold/reranker_train_pairs.jsonl (one JSON object per line:
question, page_text, label, doc_id, page_num) and
data/qa_gold/reranker_held_out_questions.json (the 15 question indices
never used for training -- read by the before/after evaluation script).

Run from the repo root (no GPU needed):
    python scripts/build_reranker_training_data.py
"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.retriever import retrieve_pages

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
CANDIDATE_DEPTH = 30  # matches the in_cand ceiling depth already measured (76.7%)
MAX_HARD_NEGATIVES_PER_QUESTION = 5
N_HELD_OUT = 15
RANDOM_SEED = 42  # fixed, so the split is reproducible on re-run


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return pd.DataFrame(data["questions"])


def main():
    print("=== Step 1: load gold questions + local index ===")
    qa_df = load_gold_qa()
    index, pages, company_lookup = load_index_bundle("data/index_store")
    print(f"Total questions: {len(qa_df)}")

    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    print("\n=== Step 2: split questions into train/held-out (question-level, not pair-level) ===")
    rng = random.Random(RANDOM_SEED)
    all_indices = list(qa_df.index)
    rng.shuffle(all_indices)
    held_out_indices = sorted(all_indices[:N_HELD_OUT])
    train_indices = sorted(all_indices[N_HELD_OUT:])
    print(f"Train: {len(train_indices)} questions | Held-out: {len(held_out_indices)} questions")

    with open("data/qa_gold/reranker_held_out_questions.json", "w") as f:
        json.dump(held_out_indices, f)

    print("\n=== Step 3: load embedder ===")
    model = load_embedder(device="cpu")

    print("\n=== Step 4: build training pairs (train questions only) ===")
    pairs = []
    for qi in train_indices:
        row = qa_df.loc[qi]
        gold_pages = {p + PAGE_OFFSET for p in row["evidence_pages"]}
        page_texts = page_text_lookup_by_doc.get(row["doc_name"], {})

        q_emb = embed_query(model, row["question"])
        candidates = retrieve_pages(q_emb, index, pages, row["doc_name"], k=CANDIDATE_DEPTH)

        # Positives: every gold page, whether or not retrieval surfaced it.
        for pn in gold_pages:
            if pn in page_texts:
                pairs.append({"question": row["question"], "page_text": page_texts[pn],
                              "label": 1, "doc_id": row["doc_name"], "page_num": pn})

        # Hard negatives: retrieved candidates that are NOT gold, capped per question.
        hard_negatives = [pn for pn in candidates if pn not in gold_pages][:MAX_HARD_NEGATIVES_PER_QUESTION]
        for pn in hard_negatives:
            if pn in page_texts:
                pairs.append({"question": row["question"], "page_text": page_texts[pn],
                              "label": 0, "doc_id": row["doc_name"], "page_num": pn})

    n_pos = sum(1 for p in pairs if p["label"] == 1)
    n_neg = sum(1 for p in pairs if p["label"] == 0)
    print(f"Built {len(pairs)} pairs: {n_pos} positive, {n_neg} negative (ratio 1:{n_neg/max(n_pos,1):.1f})")

    with open("data/qa_gold/reranker_train_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print("\nSaved:")
    print("  data/qa_gold/reranker_train_pairs.jsonl")
    print("  data/qa_gold/reranker_held_out_questions.json")
    print("\nNext: run the LoRA fine-tuning script (needs GPU -- do this in Colab).")


if __name__ == "__main__":
    main()