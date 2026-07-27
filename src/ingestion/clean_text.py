"""Section-title extraction, used to build the page+header representation
that beat plain page-level retrieval by +8.3pp hit@5 (58.3% vs 50.0%) at
essentially zero extra cost (35s vs 9m25s for the BGE-M3 alternative that
gave a smaller +1.7pp gain -- see experiments/02_chunk_vs_page_level.ipynb).
"""
import re


def page_header(text: str) -> str:
    """Pull a short, human-readable section title from the top of a page.

    Heuristic: first non-empty line between 3 and 90 characters that isn't
    the literal "Table of Contents" running header. Falls back to the first
    line if nothing matches.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for l in lines[:3]:
        if 3 < len(l) < 90 and l.lower() != "table of contents":
            return l
    return lines[0][:90] if lines else ""


def company_token(doc_id: str) -> str:
    """Extract the company name portion from a doc_id like 'AMD_2022_10K'."""
    return re.split(r"_\d{4}|_20\d{2}", doc_id)[0]
