"""Unit tests for src.retrieval.doc_matcher.match_document.

Covers the three code paths documented in the module's own docstring:
plain string match, alias resolution, and multi-candidate year/quarter
disambiguation -- plus the fallback_needed case (the 2/60 residual
failures the project explicitly accepts rather than tries to fix).
"""
from src.retrieval.doc_matcher import match_document


def test_plain_string_match():
    company_lookup = {"ADOBE_2017_10K": "Adobe"}
    doc, method = match_document(
        "What was Adobe's operating cash flow in FY2017?",
        ["ADOBE_2017_10K"], company_lookup,
    )
    assert doc == "ADOBE_2017_10K"
    assert method == "string_match"


def test_alias_resolution():
    company_lookup = {"AMERICANEXPRESS_2019_10K": "AMERICANEXPRESS"}
    doc, method = match_document(
        "What was Amex's net income in FY2019?",
        ["AMERICANEXPRESS_2019_10K"], company_lookup,
    )
    assert doc == "AMERICANEXPRESS_2019_10K"


def test_year_disambiguation_multiple_filings_same_company():
    company_lookup = {
        "3M_2018_10K": "3M",
        "3M_2019_10K": "3M",
    }
    doc, method = match_document(
        "What was 3M's FY2019 capital expenditure?",
        ["3M_2018_10K", "3M_2019_10K"], company_lookup,
    )
    assert doc == "3M_2019_10K"
    assert method == "string_match_year_disambiguated"


def test_no_company_name_returns_fallback_needed():
    """The 2/60 residual failure mode: a follow-up-style question with no
    company name at all -- documented as unsolvable by single-turn
    matching, not a bug to chase."""
    company_lookup = {"ADOBE_2017_10K": "Adobe"}
    doc, method = match_document(
        "And what about the following year?",
        ["ADOBE_2017_10K"], company_lookup,
    )
    assert doc is None
    assert method == "fallback_needed"


def test_case_insensitive_match():
    company_lookup = {"NIKE_2020_10K": "Nike"}
    doc, method = match_document(
        "what was NIKE's revenue in fy2020?",
        ["NIKE_2020_10K"], company_lookup,
    )
    assert doc == "NIKE_2020_10K"
