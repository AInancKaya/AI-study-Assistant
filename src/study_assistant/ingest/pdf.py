"""Read PDF text layers into per-page text, Turkish-safe.

Scanned PDFs (images without a text layer) are rejected: this app does no OCR.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata

from pypdf import PdfReader

from study_assistant.models import Page

MIN_TEXT_CHARS = 20  # non-whitespace characters needed to call a PDF "has text"
_HYPHEN_BREAK = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_SPACE_RUN = re.compile(r"[ \t]+")


class ScannedPDFError(ValueError):
    """The PDF has no usable text layer (probably a scan)."""


class InvalidPDFError(ValueError):
    """The bytes are not a PDF that pypdf can read."""


def make_doc_id(data: bytes) -> str:
    """Return a stable 12-character id derived from the file content."""
    return hashlib.sha256(data).hexdigest()[:12]


def _reader(data: bytes) -> PdfReader:
    try:
        return PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise InvalidPDFError("This file is not a readable PDF.") from exc


def count_pages(data: bytes) -> int:
    """Return the number of pages in the PDF."""
    return len(_reader(data).pages)


def normalize_text(raw: str) -> str:
    """NFC-normalize, join line-end hyphenation and collapse runs of spaces."""
    text = unicodedata.normalize("NFC", raw)
    text = _HYPHEN_BREAK.sub("", text)
    lines = (_SPACE_RUN.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line).strip()


def _page_texts(data: bytes) -> list[str]:
    reader = _reader(data)
    try:
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise InvalidPDFError("Could not read the text of this PDF.") from exc


def extract_pages(data: bytes, filename: str) -> list[Page]:
    """Return the text of each page (1-based), skipping pages without text."""
    doc_id = make_doc_id(data)
    pages = [
        Page(doc_id=doc_id, filename=filename, page=number, text=text)
        for number, raw in enumerate(_page_texts(data), start=1)
        if (text := normalize_text(raw))
    ]
    if sum(len("".join(p.text.split())) for p in pages) < MIN_TEXT_CHARS:
        raise ScannedPDFError("This PDF has no text layer; scanned PDFs are not supported.")
    return pages
