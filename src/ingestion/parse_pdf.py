"""
PDF parsing with page-offset handling.

FinanceBench evidence pages are 0-indexed; pdfplumber pages are 1-indexed.
Verified empirically: delta = +1 uniformly across 76 evidence spans, 22 documents
(see experiments/01_offset_bug_discovery.ipynb). Do not assume this holds for a new
benchmark without re-verifying -- it was a discovered fact, not a framework default.
"""
import re
import pdfplumber

PAGE_OFFSET = 1


def clean_page(text: str) -> str:
    """Strip repeated running header that otherwise pollutes every page's embedding."""
    return re.sub(r"^\s*Table of Contents\s*\n", "", text, flags=re.I).strip()


def parse_pdf(path: str, doc_id: str, min_chars: int = 50) -> list[dict]:
    """Parse a single PDF into a list of {doc_id, page_num, text} dicts.

    Pages with < min_chars of extracted text are skipped (usually blank or
    image-only pages that pdfplumber can't extract from).
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < min_chars:
                continue
            pages.append({
                "doc_id": doc_id,
                "page_num": i + 1,  # 1-indexed to match pdfplumber
                "text": clean_page(text),
            })
    return pages


def parse_corpus(pdf_dir: str, doc_ids: list[str]) -> list[dict]:
    """Parse every document in doc_ids, found at {pdf_dir}/{doc_id}.pdf."""
    all_pages = []
    for doc_id in doc_ids:
        path = f"{pdf_dir}/{doc_id}.pdf"
        all_pages.extend(parse_pdf(path, doc_id))
    return all_pages
