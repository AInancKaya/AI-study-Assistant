"""Tests for the page splitter of the PDF fixture builder."""

from scripts.make_pdf import split_pages


def test_split_pages_skips_single_line_html_comments() -> None:
    source = "<!-- license note -->\nTitle\n---page---\n  <!-- x -->  \nSecond"

    assert split_pages(source) == ["Title", "Second"]
