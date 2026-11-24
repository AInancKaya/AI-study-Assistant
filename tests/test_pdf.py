"""Tests for PDF text extraction (no network, no Ollama)."""

from pathlib import Path

import pytest

from study_assistant.ingest.pdf import (
    InvalidPDFError,
    ScannedPDFError,
    count_pages,
    extract_pages,
    make_doc_id,
    normalize_text,
)
from study_assistant.models import Page

FIXTURES = Path(__file__).parent / "fixtures"


def FIX(name: str) -> bytes:
    """Return the raw bytes of a fixture file under ``tests/fixtures/``."""
    return (FIXTURES / name).read_bytes()


def pages_of_fixture() -> list[Page]:
    return extract_pages(FIX("turkce_notlar.pdf"), "turkce_notlar.pdf")


def test_extract_keeps_turkish_and_pages() -> None:
    pages = pages_of_fixture()
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Işık ılık süt" in pages[1].text and "İstanbul" in pages[1].text


def test_hyphenation_joined() -> None:
    assert "öğrenmenin" in pages_of_fixture()[1].text


def test_doc_id_stable() -> None:
    assert make_doc_id(b"x") == make_doc_id(b"x") and len(make_doc_id(b"x")) == 12


def test_scanned_pdf_raises() -> None:
    with pytest.raises(ScannedPDFError):
        extract_pages(FIX("empty.pdf"), "e.pdf")


def test_footer_only_pdf_raises() -> None:
    # Five pages of "Sayfa N": each page is under 20 non-whitespace characters,
    # but the total is over 20, so only a per-page check rejects it.
    with pytest.raises(ScannedPDFError):
        extract_pages(FIX("footer_only.pdf"), "footer_only.pdf")


def test_not_a_pdf_raises() -> None:
    with pytest.raises(InvalidPDFError):
        extract_pages(b"hello", "x.pdf")


def test_count_pages() -> None:
    assert count_pages(FIX("turkce_notlar.pdf")) == 3


def test_normalize_text_nfc_composes_cedilla() -> None:
    decomposed = "ş"  # "s" + combining cedilla (NFD form of "ş")
    assert normalize_text(decomposed) == "ş"  # precomposed "ş"


def test_normalize_text_collapses_spaces() -> None:
    assert normalize_text("a    b\t\tc  \n  d") == "a b c\nd"


def test_normalize_text_joins_line_end_hyphen() -> None:
    assert normalize_text("öğrenme-\nnin önemi") == "öğrenmenin önemi"
