"""Build the index using the same setup as reproduce_from_scratch.py,
then save it so the API can load it at startup instead of rebuilding.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.download_corpus import clone_financebench, copy_pdfs
from src.ingestion.parse_pdf import parse_corpus
from src.ingestion.clean_text import page_header, company_token
from src.indexing.embedder import load_embedder, embed_pages
from src.indexing.build_index import build_flat_index
from src.indexing.persistence import save_index_bundle
import pandas as pd

GOLD_QA_PATH = "data/qa_gold/sample_60_stratified.json"


def load_gold_qa(path: str = GOLD_QA_PATH):
    data = json.load(open(path))
    return data["docs"], pd.DataFrame(data["questions"])


def main(save_dir: str, pdf_dir: str = "data/raw_pdfs"):
    print("=== Step 1: load doc list ===")
    doc_ids, _ = load_gold_qa()
    print(f"Documents: {len(doc_ids)}")

    print("=== Step 2: clone FinanceBench, copy PDFs ===")
    fb_dir = clone_financebench()
    missing = copy_pdfs(fb_dir, pdf_dir, doc_ids)
    if missing:
        raise RuntimeError(f"Missing PDFs: {missing}")

    print("=== Step 3: parse PDFs ===")
    pages = parse_corpus(pdf_dir, doc_ids)
    print(f"Pages parsed: {len(pages)}")

    print("=== Step 4: embed + build index ===")
    company_lookup = {d: company_token(d) for d in doc_ids}
    model = load_embedder()
    embeddings = embed_pages(model, pages, company_lookup, page_header)
    index = build_flat_index(embeddings)
    print(f"Index vectors: {index.ntotal}")

    print(f"=== Step 5: save bundle to {save_dir} ===")
    save_index_bundle(index, pages, company_lookup, save_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--pdf_dir", default="data/raw_pdfs")
    args = parser.parse_args()
    main(args.save_dir, args.pdf_dir)
