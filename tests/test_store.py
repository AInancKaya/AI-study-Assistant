"""Tests for the Chroma-backed vector store (``HashEmbedder``, one ``tmp_path`` per test)."""

from pathlib import Path

import pytest

from study_assistant.ingest.chunking import chunk_pages
from study_assistant.models import Chunk, DocumentInfo, Page
from study_assistant.store import VectorStore
from tests.fakes import HashEmbedder

BIO_PAGES = [
    "Osmanlı İmparatorluğu 1299 yılında kuruldu. Başkent Bursa, sonra Edirne oldu.",
    "Fotosentez bitkilerde gerçekleşir. Fotosentez sırasında klorofil ışığı emer. "
    "Fotosentez sonucunda glikoz ve oksijen üretilir.",
    "Bir üçgenin iç açıları toplamı 180 derecedir. Pisagor teoremi dik üçgenler içindir.",
]
CHEM_PAGES = [
    "Su molekülü iki hidrojen ve bir oksijen atomundan oluşur.",
    "Asitler ve bazlar tepkimeye girerek tuz ve su oluşturur.",
]


def _pages(doc_id: str, filename: str, texts: list[str]) -> list[Page]:
    return [
        Page(doc_id=doc_id, filename=filename, page=i, text=text)
        for i, text in enumerate(texts, start=1)
    ]


def _chunks(doc_id: str, filename: str, texts: list[str]) -> list[Chunk]:
    return chunk_pages(_pages(doc_id, filename, texts), size=60, overlap=10)


def _store(path: Path) -> VectorStore:
    return VectorStore(path / "chroma", HashEmbedder())


def _add_bio(store: VectorStore) -> list[Chunk]:
    chunks = _chunks("bio", "biyoloji.pdf", BIO_PAGES)
    store.add_document(chunks, pages=3)
    return chunks


def _add_chem(store: VectorStore) -> list[Chunk]:
    chunks = _chunks("chem", "kimya.pdf", CHEM_PAGES)
    store.add_document(chunks, pages=2)
    return chunks


def test_search_finds_relevant_chunk_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_bio(store)

    results = store.search("fotosentez", k=3)

    assert len(results) == 3
    assert results[0].chunk.page == 2
    assert "otosentez" in results[0].chunk.text
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 < results[0].score <= 1.0 + 1e-6


def test_search_filter_by_doc_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_bio(store)
    chem = _add_chem(store)

    results = store.search("oksijen su", k=50, doc_id="chem")

    assert results
    assert {r.chunk.doc_id for r in results} == {"chem"}
    assert len(results) == len(chem)


def test_search_k_larger_than_collection_is_clamped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = _add_bio(store)

    assert len(store.search("fotosentez", k=1000)) == len(chunks)
    assert store.search("fotosentez", k=0) == []


def test_search_returns_full_chunk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = _add_bio(store)

    top = store.search("Pisagor teoremi", k=1)[0].chunk

    assert top in chunks


def test_list_documents_counts_pages_and_chunks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bio = _add_bio(store)
    chem = _add_chem(store)

    assert store.list_documents() == [
        DocumentInfo(doc_id="bio", filename="biyoloji.pdf", pages=3, chunks=len(bio)),
        DocumentInfo(doc_id="chem", filename="kimya.pdf", pages=2, chunks=len(chem)),
    ]


def test_add_document_returns_info(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = _chunks("bio", "biyoloji.pdf", BIO_PAGES)

    info = store.add_document(chunks, pages=3)

    assert info == DocumentInfo(doc_id="bio", filename="biyoloji.pdf", pages=3, chunks=len(chunks))
    assert store.has_document("bio")


def test_add_document_twice_replaces(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_bio(store)
    shorter = _chunks("bio", "biyoloji.pdf", BIO_PAGES[:1])

    store.add_document(shorter, pages=1)

    assert store.list_documents() == [
        DocumentInfo(doc_id="bio", filename="biyoloji.pdf", pages=1, chunks=len(shorter))
    ]


def test_add_document_rejects_empty_and_mixed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mixed = _chunks("bio", "biyoloji.pdf", BIO_PAGES) + _chunks("chem", "kimya.pdf", CHEM_PAGES)

    with pytest.raises(ValueError):
        store.add_document([], pages=1)
    with pytest.raises(ValueError):
        store.add_document(mixed, pages=3)
    assert store.list_documents() == []


def test_delete_document(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_bio(store)
    _add_chem(store)

    assert store.delete_document("bio") is True
    assert store.has_document("bio") is False
    assert store.delete_document("bio") is False
    assert [d.doc_id for d in store.list_documents()] == ["chem"]
    assert {r.chunk.doc_id for r in store.search("fotosentez", k=10)} == {"chem"}


def test_persists_across_instances(tmp_path: Path) -> None:
    first = _store(tmp_path)
    chunks = _add_bio(first)

    second = _store(tmp_path)

    assert second.has_document("bio")
    assert second.get_chunks("bio") == chunks
    assert second.search("fotosentez", k=1)[0].chunk.page == 2


def test_get_chunks_filters_pages_and_orders(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = _add_bio(store)
    _add_chem(store)

    all_bio = store.get_chunks("bio")
    some = store.get_chunks("bio", pages=[3, 1])

    assert all_bio == chunks
    assert [c.chunk_idx for c in all_bio] == sorted(c.chunk_idx for c in all_bio)
    assert some == [c for c in chunks if c.page in (1, 3)]
    assert store.get_chunks("bio", pages=[]) == []
    assert store.get_chunks("missing") == []


def test_search_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.search("fotosentez", k=5) == []
    assert store.search("fotosentez", k=5, doc_id="bio") == []
    assert store.list_documents() == []
    assert store.has_document("bio") is False
