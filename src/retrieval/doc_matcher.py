"""Stage 1: document identification -- WITHOUT machine learning.

96.7% accuracy (58/60) from string + alias + fiscal-year matching alone.
A fine-tuned classifier was considered and explicitly rejected: the two
residual failures are questions that never mention a company name at all
(follow-up-style phrasing assuming prior context) -- unsolvable by any
single-turn classifier, so training one would be solving the wrong problem.
See docs/failure_analysis.md.
"""
import re

ALIASES = {
    "amex": "AMERICANEXPRESS",
    "jnj": "JOHNSON_JOHNSON",
    "j&j": "JOHNSON_JOHNSON",
    "jpm": "JPMORGAN",
    "mgm": "MGMRESORTS",
}


def extract_year_quarter(text: str):
    years = set(re.findall(r"(20\d{2})", text))
    quarter = re.search(r"\bQ([1-4])\b", text, re.I)
    return years, (quarter.group(1) if quarter else None)


def match_document(question: str, candidate_docs: list[str], company_lookup: dict):
    """Return (matched_doc_id, method) or (None, "fallback_needed").

    company_lookup: {doc_id: company_name} e.g. from clean_text.company_token
    """
    q_lower = question.lower()
    q_years, q_quarter = extract_year_quarter(question)

    candidates = [
        doc for doc in candidate_docs
        if company_lookup[doc].lower() in q_lower
        or company_lookup[doc].lower().replace(" ", "") in q_lower.replace(" ", "")
    ]
    if not candidates:
        for alias, target in ALIASES.items():
            if alias in q_lower:
                candidates += [d for d in candidate_docs if target in d]

    if not candidates:
        return None, "fallback_needed"
    if len(candidates) == 1:
        return candidates[0], "string_match"

    # Multiple docs for the same company (different years/quarters) -- disambiguate
    scored = []
    for doc in candidates:
        doc_years = set(re.findall(r"(20\d{2})", doc))
        year_match = len(doc_years & q_years)
        q_in_doc = 1 if (q_quarter and f"Q{q_quarter}" in doc) else 0
        scored.append((year_match + q_in_doc, doc))
    scored.sort(reverse=True)
    return scored[0][1], "string_match_year_disambiguated"
