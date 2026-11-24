"""Tests for grounded question answering (``HashEmbedder`` store + ``FakeLLM``)."""

from pathlib import Path

import pytest

from study_assistant.ingest.chunking import chunk_pages
from study_assistant.ingest.pdf import extract_pages
from study_assistant.models import Chunk, Citation, LLMAnswer, Page
from study_assistant.prompts import ANSWER_SYSTEM, format_sources
from study_assistant.rag import NOT_FOUND_MESSAGE, ask
from study_assistant.store import VectorStore
from tests.fakes import FakeLLM, HashEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "turkce_notlar.pdf"
FILENAME = "turkce_notlar.pdf"
# Shares many words with page 2 of the fixture, so page 2 is the top hit.
QUESTION = "İstanbul'da öğrenmenin önemi nedir?"


@pytest.fixture
def notes() -> list[Chunk]:
    pages = extract_pages(FIXTURE.read_bytes(), FILENAME)
    return chunk_pages(pages)


@pytest.fixture
def store(tmp_path: Path, notes: list[Chunk]) -> VectorStore:
    vector_store = VectorStore(tmp_path / "chroma", HashEmbedder())
    vector_store.add_document(notes, pages=3)
    return vector_store


def _answer(cited: list[int], found: bool = True, text: str = "Öğrenmek önemlidir.") -> LLMAnswer:
    return LLMAnswer(answer=text, found=found, cited_sources=cited)


def _citation(chunk: Chunk) -> Citation:
    return Citation(doc_id=chunk.doc_id, filename=chunk.filename, page=chunk.page)


def test_answer_with_valid_citations(store: VectorStore) -> None:
    hits = store.search(QUESTION, k=3)
    llm = FakeLLM([_answer([1, 1, 2])])

    result = ask(QUESTION, store, llm, k=3, min_score=0.0)

    assert result.found is True
    assert result.answer == "Öğrenmek önemlidir."
    assert result.citations == [_citation(hits[0].chunk), _citation(hits[1].chunk)]
    assert result.citations[0].page == 2
    assert len(llm.calls) == 1


def test_out_of_range_source_ids_dropped(store: VectorStore) -> None:
    hits = store.search(QUESTION, k=3)
    llm = FakeLLM([_answer([1, 9, 0, -1])])

    result = ask(QUESTION, store, llm, k=3, min_score=0.0)

    assert result.citations == [_citation(hits[0].chunk)]


def test_citations_keep_first_cited_order(store: VectorStore) -> None:
    hits = store.search(QUESTION, k=3)
    llm = FakeLLM([_answer([3, 1, 3])])

    result = ask(QUESTION, store, llm, k=3, min_score=0.0)

    assert result.citations == [_citation(hits[2].chunk), _citation(hits[0].chunk)]


def test_found_without_valid_citation_keeps_answer(store: VectorStore) -> None:
    llm = FakeLLM([_answer([7])])

    result = ask(QUESTION, store, llm, k=3, min_score=0.0)

    assert result.found is True
    assert result.answer == "Öğrenmek önemlidir."
    assert result.citations == []


def test_below_min_score_skips_llm(store: VectorStore) -> None:
    llm = FakeLLM([])

    result = ask("Kuantum dolanıklığı nedir?", store, llm, k=3, min_score=0.99)

    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    assert llm.calls == []


def test_empty_store_skips_llm(tmp_path: Path) -> None:
    empty = VectorStore(tmp_path / "chroma", HashEmbedder())
    llm = FakeLLM([])

    result = ask(QUESTION, empty, llm, k=3, min_score=0.0)

    assert result.found is False
    assert llm.calls == []


def test_llm_says_not_found(store: VectorStore) -> None:
    llm = FakeLLM([_answer([1, 2], found=False, text="Bilmiyorum.")])

    result = ask(QUESTION, store, llm, k=3, min_score=0.0)

    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_blank_llm_answer_treated_as_not_found(store: VectorStore) -> None:
    llm = FakeLLM([_answer([1], text="   ")])

    result = ask(QUESTION, store, llm, k=3, min_score=0.0)

    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_doc_filter_passed_to_store(store: VectorStore, notes: list[Chunk]) -> None:
    other = Page(doc_id="other", filename="diger.pdf", page=1, text=QUESTION + " Başka not.")
    store.add_document(chunk_pages([other]), pages=1)
    notes_id = notes[0].doc_id

    filtered_llm = FakeLLM([_answer([1])])
    result = ask(QUESTION, store, filtered_llm, k=3, min_score=0.0, doc_id=notes_id)

    prompt = filtered_llm.calls[0][1]
    assert "diger.pdf" not in prompt
    assert prompt.count(f"({FILENAME}, s.") == 3
    assert all(c.doc_id == notes_id for c in result.citations)

    other_llm = FakeLLM([_answer([1])])
    result = ask(QUESTION, store, other_llm, k=3, min_score=0.0, doc_id="other")

    assert FILENAME not in other_llm.calls[0][1]
    assert result.citations == [Citation(doc_id="other", filename="diger.pdf", page=1)]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises(store: VectorStore, question: str) -> None:
    llm = FakeLLM([])

    with pytest.raises(ValueError):
        ask(question, store, llm, k=3, min_score=0.0)

    assert llm.calls == []


def test_prompt_contains_numbered_sources(store: VectorStore) -> None:
    llm = FakeLLM([_answer([1])])

    ask(f"  {QUESTION}  ", store, llm, k=3, min_score=0.0)

    system, user, schema = llm.calls[0]
    assert system == ANSWER_SYSTEM
    assert schema is LLMAnswer
    assert "[1] (turkce_notlar.pdf, s. 2)" in user
    assert "[2] (turkce_notlar.pdf, s." in user
    assert "[3] (turkce_notlar.pdf, s." in user
    assert "[4]" not in user
    assert user.rstrip().endswith(f"Question: {QUESTION}")


def test_format_sources_layout() -> None:
    chunks = [
        Chunk(id="a", doc_id="d", filename="ders.pdf", page=4, chunk_idx=0, text="Birinci."),
        Chunk(id="b", doc_id="d", filename="ders.pdf", page=7, chunk_idx=1, text="İkinci."),
    ]

    assert (
        format_sources(chunks) == "[1] (ders.pdf, s. 4)\nBirinci.\n\n[2] (ders.pdf, s. 7)\nİkinci."
    )
