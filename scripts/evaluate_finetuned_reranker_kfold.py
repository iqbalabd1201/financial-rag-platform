import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from src.indexing.persistence import load_index_bundle
from src.indexing.embedder import load_embedder, embed_query
from src.retrieval.retriever import retrieve_pages
from src.retrieval.reranker import load_reranker, load_finetuned_reranker, rerank
from src.evaluation.retrieval_metrics import hit_at_k, recall_at_k

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
CANDIDATE_DEPTH = 30
FINAL_K = 5
N_FOLDS = 4


def load_gold_qa(path=GOLD_QA_PATH):
    data = json.load(open(path))
    return pd.DataFrame(data["questions"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n=== Step 1: load gold questions + local index ===")
    qa_df = load_gold_qa()
    index, pages, company_lookup = load_index_bundle("data/index_store")
    page_text_lookup = {}
    for p in pages:
        page_text_lookup[(p["doc_id"], p["page_num"])] = p["text"]

    print("\n=== Step 2: load embedder + off-the-shelf reranker (shared across folds) ===")
    model = load_embedder(device=device)
    off_the_shelf = load_reranker(device=device)

    pooled = {"no_rerank": [], "off_the_shelf": [], "finetuned": []}
    pooled_recall = {"no_rerank": [], "off_the_shelf": [], "finetuned": []}

    for fold_i in range(N_FOLDS):
        held_out_path = f"data/qa_gold/reranker_fold{fold_i}_held_out.json"
        adapter_path = f"models/reranker_lora_adapter_fold{fold_i}"
        print(f"\n=== Fold {fold_i}: loading held-out set + fold-specific adapter ===")
        held_out_indices = json.load(open(held_out_path))
        qa_subset = qa_df.loc[held_out_indices]
        finetuned = load_finetuned_reranker(adapter_path=adapter_path, device=device)

        for _, r in qa_subset.iterrows():
            gold = {p + PAGE_OFFSET for p in r["evidence_pages"]}
            q_emb = embed_query(model, r["question"])

            direct_top5 = retrieve_pages(q_emb, index, pages, r["doc_name"], k=FINAL_K)
            pooled["no_rerank"].append(hit_at_k(gold, direct_top5, FINAL_K))
            pooled_recall["no_rerank"].append(recall_at_k(gold, direct_top5, FINAL_K))

            candidates_30 = retrieve_pages(q_emb, index, pages, r["doc_name"], k=CANDIDATE_DEPTH)
            candidate_tuples = [(r["doc_name"], pn) for pn in candidates_30]

            reranked_ots = rerank(off_the_shelf, r["question"], candidate_tuples, page_text_lookup)
            top5_ots = [pn for _, pn in reranked_ots[:FINAL_K]]
            pooled["off_the_shelf"].append(hit_at_k(gold, top5_ots, FINAL_K))
            pooled_recall["off_the_shelf"].append(recall_at_k(gold, top5_ots, FINAL_K))

            reranked_ft = rerank(finetuned, r["question"], candidate_tuples, page_text_lookup)
            top5_ft = [pn for _, pn in reranked_ft[:FINAL_K]]
            pooled["finetuned"].append(hit_at_k(gold, top5_ft, FINAL_K))
            pooled_recall["finetuned"].append(recall_at_k(gold, top5_ft, FINAL_K))

        print(f"  Fold {fold_i} done ({len(qa_subset)} questions evaluated).")

    print("\n" + "=" * 60)
    print(f"POOLED RESULTS across all {len(pooled['no_rerank'])} questions (4-fold CV, no leakage)")
    print("=" * 60)
    for name in ["no_rerank", "off_the_shelf", "finetuned"]:
        hit5 = sum(pooled[name]) / len(pooled[name])
        rec5 = sum(pooled_recall[name]) / len(pooled_recall[name])
        print(f"  {name:15s}: hit@5={hit5:.1%}  recall@5={rec5:.1%}")


if __name__ == "__main__":
    main()
