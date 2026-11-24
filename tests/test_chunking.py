"""Tests for page-aware chunking. Pure Python: no network, no models."""

import re

import pytest

from study_assistant.ingest.chunking import chunk_pages
from study_assistant.models import Page


def page(n: int, text: str, doc_id: str = "doc1", filename: str = "notes.pdf") -> Page:
    return Page(doc_id=doc_id, filename=filename, page=n, text=text)


def long_pages() -> list[Page]:
    """Three dense pages: six paragraphs of eight short sentences each, Turkish letters included."""
    pages = []
    for n in range(1, 4):
        paragraphs = []
        for p in range(6):
            sentences = [
                f"Sayfa {n}, paragraf {p}, cümle {s}: Bu örnek notta ığüşöç ve İĞÜŞÖÇ geçer."
                for s in range(8)
            ]
            paragraphs.append(" ".join(sentences))
        pages.append(page(n, "\n\n".join(paragraphs)))
    return pages


def _normalized(text: str) -> str:
    """Same joining the chunker uses: paragraphs by a blank line, words by single spaces."""
    return "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip())


def _shared_overlap(before: str, after: str) -> int:
    """Length of the longest suffix of `before` that is also a prefix of `after`."""
    limit = min(len(before), len(after))
    return max(
        (k for k in range(limit + 1) if after[:k] == before[len(before) - k :]),
        default=0,
    )


def test_short_page_is_one_chunk():
    chunks = chunk_pages([page(1, "Kısa metin.")])
    assert len(chunks) == 1
    assert chunks[0].text == "Kısa metin."


def test_no_chunk_exceeds_size():
    chunks = chunk_pages(long_pages(), 500, 80)
    assert chunks
    assert all(len(c.text) <= 500 for c in chunks)


def test_chunks_never_cross_pages():
    pages = long_pages()
    normalized = {p.page: _normalized(p.text) for p in pages}
    for chunk in chunk_pages(pages, 500, 80):
        assert chunk.text in normalized[chunk.page]


def test_chunk_idx_sequential_and_ids_unique():
    chunks = chunk_pages(long_pages(), 500, 80)
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"{c.doc_id}:{c.page}:{c.chunk_idx}" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_overlap_between_neighbours():
    chunks = chunk_pages(long_pages(), 500, 80)
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:], strict=False) if a.page == b.page]
    assert pairs
    for before, after in pairs:
        assert _shared_overlap(before.text, after.text) >= 80


def test_single_huge_sentence_hard_split():
    text = "".join(chr(ord("a") + i % 26) for i in range(3000))  # no spaces or punctuation
    chunks = chunk_pages([page(1, text)], 1000, 200)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 1000 for c in chunks)
    # every 50-character window of the input survives in some chunk
    for start in range(0, len(text) - 50 + 1, 50):
        assert any(text[start : start + 50] in c.text for c in chunks)
    # neighbouring windows still share the configured overlap
    for before, after in zip(chunks, chunks[1:], strict=False):
        assert _shared_overlap(before.text, after.text) >= 200


def test_overlap_zero_means_no_shared_text():
    sentences = [f"Cümle numarası {i} burada biter." for i in range(30)]
    chunks = chunk_pages([page(1, " ".join(sentences))], 200, 0)
    assert all(len(c.text) <= 200 for c in chunks)
    for before, after in zip(chunks, chunks[1:], strict=False):
        before_sentences = set(re.split(r"(?<=[.!?…])\s+", before.text))
        after_sentences = set(re.split(r"(?<=[.!?…])\s+", after.text))
        assert before_sentences.isdisjoint(after_sentences)


def test_paragraphs_split_before_sentences():
    text = "Birinci paragraf. İkinci cümle.\n\nİkinci paragraf başı."
    chunks = chunk_pages([page(1, text)], size=len("Birinci paragraf. İkinci cümle."), overlap=0)
    assert [c.text for c in chunks] == [
        "Birinci paragraf. İkinci cümle.",
        "İkinci paragraf başı.",
    ]


def test_turkish_text_preserved():
    chunks = chunk_pages([page(1, "Notlar: ığüşöç İĞÜŞÖÇ, bitti.")])
    assert "ığüşöç İĞÜŞÖÇ" in chunks[0].text


def test_empty_pages_list_returns_empty():
    assert chunk_pages([]) == []


def test_whitespace_only_page_yields_no_chunks():
    assert chunk_pages([page(1, " \n\n \t ")]) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, -1)])
def test_invalid_size_or_overlap_raises(size: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_pages([page(1, "Metin.")], size, overlap)
