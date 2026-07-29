import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.indexing.embedder import load_embedder, embed_query
from src.indexing.qdrant_store import get_qdrant_client, retrieve_pages_qdrant
from src.evaluation.retrieval_metrics import hit_at_k, recall_at_k

PAGE_OFFSET = 1
GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"
TARGETS = {"hit5": 0.583, "recall5": 0.550}
TOLERANCE = 0.02


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return data["docs"], pd.DataFrame(data["questions"])


def main():
    print("=== Step 1: load gold questions ===")
    doc_ids, qa_sub = load_gold_qa()
    print(f"Questions: {len(qa_sub)}")

    print("\n=== Step 2: load embedder, connect to Qdrant ===")
    model = load_embedder(device="cpu")
    client = get_qdrant_client()

    print("\n=== Step 3: retrieve via Qdrant for each question ===")
    hits, recalls = [], []
    for _, r in qa_sub.iterrows():
        gold = {p + PAGE_OFFSET for p in r["evidence_pages"]}
        q_emb = embed_query(model, r["question"])
        retrieved = retrieve_pages_qdrant(client, q_emb, r["doc_name"], k=5)
        hits.append(hit_at_k(gold, retrieved, 5))
        recalls.append(recall_at_k(gold, retrieved, 5))

    hit5 = sum(hits) / len(hits)
    recall5 = sum(recalls) / len(recalls)

    print("\n" + "=" * 50)
    print("QDRANT MIGRATION VERIFICATION")
    print("=" * 50)
    for name, actual, target in [("Hit@5", hit5, TARGETS["hit5"]),
                                   ("Recall@5", recall5, TARGETS["recall5"])]:
        ok = abs(actual - target) <= TOLERANCE
        print(f"  {name:10s}: {actual:.1%}  {'MATCH' if ok else 'MISMATCH -- investigate'}  "
              f"(FAISS baseline: {target:.1%})")

    all_ok = abs(hit5 - TARGETS["hit5"]) <= TOLERANCE and abs(recall5 - TARGETS["recall5"]) <= TOLERANCE
    print("\nRESULT:", "MIGRATION VERIFIED -- safe to switch production API to Qdrant"
          if all_ok else "MISMATCH -- do NOT switch main.py to Qdrant yet")


if __name__ == "__main__":
    main()