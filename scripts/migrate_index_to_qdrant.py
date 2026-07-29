"""One-time migration: load the local FAISS index bundle (already built via
build_and_save_index.py) and upload its vectors + payload to Qdrant Cloud.

Does NOT re-embed anything -- reuses the exact vectors already in the FAISS
index via index.reconstruct(), so the migrated vectors are guaranteed
identical to the ones the retrieval numbers in README.md were measured on.

Run from the repo root:
    set QDRANT_URL=https://xxxxx.cloud.qdrant.io:6333
    set QDRANT_API_KEY=xxxxx
    python scripts/migrate_index_to_qdrant.py --index_dir data/index_store
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.indexing.persistence import load_index_bundle
from src.indexing.qdrant_store import get_qdrant_client, ensure_collection, upsert_pages


def main(index_dir: str):
    print(f"=== Step 1: load local FAISS bundle from {index_dir} ===")
    index, pages, company_lookup = load_index_bundle(index_dir)
    print(f"Loaded: {index.ntotal} vectors, {len(pages)} pages, {len(company_lookup)} docs")

    print("\n=== Step 2: extract vectors from FAISS index ===")
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype="float32")
    print(f"Extracted vectors shape: {vectors.shape}")

    print("\n=== Step 3: connect to Qdrant Cloud, create collection ===")
    client = get_qdrant_client()
    ensure_collection(client, recreate=True)
    print("Collection ready.")

    print("\n=== Step 4: upload vectors + payload ===")
    upsert_pages(client, vectors, pages)

    count = client.count(collection_name="financial_rag_pages").count
    print(f"\nUploaded. Qdrant collection now has {count} points "
          f"(expected {index.ntotal}).")
    if count != index.ntotal:
        print("WARNING: count mismatch -- do not use this collection until resolved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="data/index_store")
    args = parser.parse_args()
    main(args.index_dir)
