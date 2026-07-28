"""Save/load the built index + its accompanying metadata as a single unit.

Saves three things together, because they must stay in lockstep:
  - the FAISS index itself
  - `pages` (list[dict], SAME ORDER used to build the index)
  - `company_lookup` (dict[doc_id, company_name])

If `pages` order and index order ever drift, retrieval will return
plausible-looking but wrong pages with no error raised.
"""
import json
import os
import faiss

INDEX_FILENAME = "page_header_bge_small.faiss"
PAGES_FILENAME = "pages.json"
COMPANY_LOOKUP_FILENAME = "company_lookup.json"


def save_index_bundle(index, pages: list, company_lookup: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(save_dir, INDEX_FILENAME))
    with open(os.path.join(save_dir, PAGES_FILENAME), "w") as f:
        json.dump(pages, f)
    with open(os.path.join(save_dir, COMPANY_LOOKUP_FILENAME), "w") as f:
        json.dump(company_lookup, f)
    print(f"Saved index bundle to {save_dir}: "
          f"{index.ntotal} vectors, {len(pages)} pages, "
          f"{len(company_lookup)} companies")


def load_index_bundle(save_dir: str):
    index_path = os.path.join(save_dir, INDEX_FILENAME)
    pages_path = os.path.join(save_dir, PAGES_FILENAME)
    lookup_path = os.path.join(save_dir, COMPANY_LOOKUP_FILENAME)

    for path in (index_path, pages_path, lookup_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path} -- run save_index_bundle() first.")

    index = faiss.read_index(index_path)
    with open(pages_path) as f:
        pages = json.load(f)
    with open(lookup_path) as f:
        company_lookup = json.load(f)

    if index.ntotal != len(pages):
        raise ValueError(
            f"Index/pages mismatch: index has {index.ntotal} vectors but "
            f"pages.json has {len(pages)} entries. Do not use this bundle."
        )
    return index, pages, company_lookup
