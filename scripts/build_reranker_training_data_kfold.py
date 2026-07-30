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
CANDIDATE_DEPTH = 30
MAX_HARD_NEGATIVES_PER_QUESTION = 5
N_FOLDS = 4
RANDOM_SEED = 42


def load_gold_qa(path=GOLD_QA_PATH):
    data = json.load(open(path))
    return pd.DataFrame(data["questions"])


def build_pairs_for_questions(qa_df, question_indices, model, index, pages, page_text_lookup_by_doc):
    pairs = []
    for qi in question_indices:
        row = qa_df.loc[qi]
        gold_pages = {p + PAGE_OFFSET for p in row["evidence_pages"]}
        page_texts = page_text_lookup_by_doc.get(row["doc_name"], {})

        q_emb = embed_query(model, row["question"])
        candidates = retrieve_pages(q_emb, index, pages, row["doc_name"], k=CANDIDATE_DEPTH)

        for pn in gold_pages:
            if pn in page_texts:
                pairs.append({"question": row["question"], "page_text": page_texts[pn],
                              "label": 1, "doc_id": row["doc_name"], "page_num": pn})

        hard_negatives = [pn for pn in candidates if pn not in gold_pages][:MAX_HARD_NEGATIVES_PER_QUESTION]
        for pn in hard_negatives:
            if pn in page_texts:
                pairs.append({"question": row["question"], "page_text": page_texts[pn],
                              "label": 0, "doc_id": row["doc_name"], "page_num": pn})
    return pairs


def main():
    print("=== Step 1: load gold questions + local index ===")
    qa_df = load_gold_qa()
    index, pages, company_lookup = load_index_bundle("data/index_store")
    print(f"Total questions: {len(qa_df)}")

    page_text_lookup_by_doc = {}
    for p in pages:
        page_text_lookup_by_doc.setdefault(p["doc_id"], {})[p["page_num"]] = p["text"]

    print(f"\n=== Step 2: split into {N_FOLDS} folds ===")
    rng = random.Random(RANDOM_SEED)
    all_indices = list(qa_df.index)
    rng.shuffle(all_indices)
    fold_size = len(all_indices) // N_FOLDS
    folds = [all_indices[i * fold_size:(i + 1) * fold_size] for i in range(N_FOLDS)]
    for i, f in enumerate(folds):
        print(f"  Fold {i}: {len(f)} held-out questions")

    print("\n=== Step 3: load embedder ===")
    model = load_embedder(device="cpu")

    print("\n=== Step 4: build train pairs per fold ===")
    for fold_i in range(N_FOLDS):
        held_out = sorted(folds[fold_i])
        train_indices = sorted(idx for idx in all_indices if idx not in held_out)

        with open(f"data/qa_gold/reranker_fold{fold_i}_held_out.json", "w") as f:
            json.dump(held_out, f)

        pairs = build_pairs_for_questions(
            qa_df, train_indices, model, index, pages, page_text_lookup_by_doc
        )
        n_pos = sum(1 for p in pairs if p["label"] == 1)
        n_neg = sum(1 for p in pairs if p["label"] == 0)
        print(f"  Fold {fold_i}: {len(pairs)} pairs ({n_pos} pos, {n_neg} neg), "
              f"{len(held_out)} held-out, {len(train_indices)} train")

        with open(f"data/qa_gold/reranker_fold{fold_i}_train_pairs.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

    print("\nSaved 4 folds' worth of train_pairs.jsonl + held_out.json to data/qa_gold/")


if __name__ == "__main__":
    main()
