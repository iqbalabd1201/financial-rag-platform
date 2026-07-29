"""Unit tests for src.ingestion.parse_pdf.

PAGE_OFFSET=1 is a fact discovered empirically (see the module's own
docstring and experiments/01_offset_bug_discovery.ipynb), not a framework
default -- a regression guard here catches anyone (including future-us)
accidentally "simplifying" it back to 0 without re-verifying.
"""
from src.ingestion.parse_pdf import clean_page, PAGE_OFFSET


def test_page_offset_is_the_discovered_value():
    assert PAGE_OFFSET == 1


def test_clean_page_strips_table_of_contents_header():
    raw = "Table of Contents\nRevenue increased 12% year over year."
    assert clean_page(raw) == "Revenue increased 12% year over year."


def test_clean_page_is_case_insensitive():
    raw = "table of contents\nSome page body text."
    assert clean_page(raw) == "Some page body text."


def test_clean_page_leaves_normal_text_untouched():
    raw = "Net income for the quarter was $4.2 million."
    assert clean_page(raw) == raw


def test_clean_page_strips_surrounding_whitespace():
    raw = "   \n  Some page text.  \n\n"
    assert clean_page(raw) == "Some page text."
