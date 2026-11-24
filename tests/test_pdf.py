"""Tests for PDF text extraction (no network, no Ollama)."""

from pathlib import Path

import pytest

from study_assistant.ingest.pdf import (
    InvalidPDFError,
    ScannedPDFError,
    count_pages,
    extract_pages,
    make_doc_id,
)

FIXTURES = Path(__file__).parent / "fixtures"


def FIX(name: str) -> bytes:
    """Return the raw bytes of a fixture file under ``tests/fixtures/``."""
    return (FIXTURES / name).read_bytes()


def pages_of_fixture() -> list:
    return extract_pages(FIX("turkce_notlar.pdf"), "turkce_notlar.pdf")


def test_extract_keeps_turkish_and_pages():
    pages = pages_of_fixture()
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Işık ılık süt" in pages[1].text and "İstanbul" in pages[1].text


def test_hyphenation_joined():
    assert "öğrenmenin" in pages_of_fixture()[1].text


def test_doc_id_stable():
    assert make_doc_id(b"x") == make_doc_id(b"x") and len(make_doc_id(b"x")) == 12


def test_scanned_pdf_raises():
    with pytest.raises(ScannedPDFError):
        extract_pages(FIX("empty.pdf"), "e.pdf")


def test_not_a_pdf_raises():
    with pytest.raises(InvalidPDFError):
        extract_pages(b"hello", "x.pdf")


def test_count_pages():
    assert count_pages(FIX("turkce_notlar.pdf")) == 3
