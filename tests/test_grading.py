"""Tests for answer grading (``HashEmbedder`` store + ``FakeLLM``); no network."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from study_assistant.grading import build_grade_prompt, grade_answer, verdict_for
from study_assistant.models import Chunk, Citation, LLMGrade, Quiz, QuizQuestion
from study_assistant.prompts import GRADE_SYSTEM, format_sources
from study_assistant.store import VectorStore
from tests.fakes import FakeLLM, HashEmbedder

DOC_ID = "doc1"
FILENAME = "notlar.pdf"
OPTIONS = ["Mitokondri", "Kloroplast", "Çekirdek", "Ribozom"]


def _chunk(idx: int, page: int) -> Chunk:
    return Chunk(
        id=f"{DOC_ID}-{idx}",
        doc_id=DOC_ID,
        filename=FILENAME,
        page=page,
        chunk_idx=idx,
        text=f"Konu {idx}: fotosentez ve hücre solunumu, sayfa {page}.",
    )


def _mc() -> QuizQuestion:
    return QuizQuestion(
        id="q1",
        type="multiple_choice",
        question="Fotosentez nerede gerçekleşir?",
        options=OPTIONS,
        correct_option=1,
        reference_answer="Kloroplastta gerçekleşir.",
        source_pages=[1],
    )


def _open() -> QuizQuestion:
    return QuizQuestion(
        id="q2",
        type="open_ended",
        question="Hücre solunumunu açıklayın.",
        options=None,
        correct_option=None,
        reference_answer="Glikozun enerjiye dönüştürülmesidir.",
        source_pages=[2, 3],
    )


@pytest.fixture
def quiz() -> Quiz:
    return Quiz(
        id="quiz1",
        doc_id=DOC_ID,
        filename=FILENAME,
        created_at=datetime(2026, 11, 24, tzinfo=UTC),
        questions=[_mc(), _open()],
    )


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    vector_store = VectorStore(tmp_path / "chroma", HashEmbedder())
    vector_store.add_document([_chunk(0, 1), _chunk(1, 2), _chunk(2, 3)], pages=3)
    return vector_store


def test_mc_correct_and_wrong_no_llm(quiz: Quiz, store: VectorStore) -> None:
    llm = FakeLLM([])

    right = grade_answer(quiz, "q1", store, llm, selected_option=1)
    wrong = grade_answer(quiz, "q1", store, llm, selected_option=0)

    assert right.score == 100
    assert right.verdict == "correct"
    assert right.feedback == "Correct! The answer is: Kloroplast"
    assert right.correct_option == 1
    assert right.reference_answer == "Kloroplastta gerçekleşir."
    assert right.citations == [Citation(doc_id=DOC_ID, filename=FILENAME, page=1)]

    assert wrong.score == 0
    assert wrong.verdict == "incorrect"
    assert wrong.feedback == "Not quite. The correct answer is: Kloroplast"
    assert wrong.correct_option == 1
    assert llm.calls == []


def test_mc_without_selection_raises(quiz: Quiz, store: VectorStore) -> None:
    with pytest.raises(ValueError):
        grade_answer(quiz, "q1", store, FakeLLM([]))


@pytest.mark.parametrize("selected", [-1, 4])
def test_mc_selection_out_of_range_raises(quiz: Quiz, store: VectorStore, selected: int) -> None:
    with pytest.raises(ValueError):
        grade_answer(quiz, "q1", store, FakeLLM([]), selected_option=selected)


def test_open_ended_uses_llm_and_clamps(quiz: Quiz, store: VectorStore) -> None:
    llm = FakeLLM([LLMGrade(score=140, feedback="Çok iyi.")])

    grade = grade_answer(quiz, "q2", store, llm, answer="Glikoz enerjiye dönüşür.")

    assert grade.score == 100
    assert grade.verdict == "correct"
    assert grade.feedback == "Çok iyi."
    assert grade.correct_option is None
    assert grade.reference_answer == "Glikozun enerjiye dönüştürülmesidir."
    assert grade.citations == [
        Citation(doc_id=DOC_ID, filename=FILENAME, page=2),
        Citation(doc_id=DOC_ID, filename=FILENAME, page=3),
    ]
    assert len(llm.calls) == 1
    system, user, schema = llm.calls[0]
    assert system == GRADE_SYSTEM
    assert schema is LLMGrade
    assert "Question: Hücre solunumunu açıklayın." in user
    assert "Reference answer: Glikozun enerjiye dönüştürülmesidir." in user
    assert "Student answer: Glikoz enerjiye dönüşür." in user
    assert "sayfa 2" in user


def test_open_ended_negative_score_clamped_to_zero(quiz: Quiz, store: VectorStore) -> None:
    llm = FakeLLM([LLMGrade(score=-20, feedback="Yanlış.")])

    grade = grade_answer(quiz, "q2", store, llm, answer="Bilmiyorum")

    assert grade.score == 0
    assert grade.verdict == "incorrect"


def test_llm_verdict_is_ignored_computed_from_score(quiz: Quiz, store: VectorStore) -> None:
    llm = FakeLLM([LLMGrade(score=55, feedback="Kısmen doğru.")])

    grade = grade_answer(quiz, "q2", store, llm, answer="Enerji üretimi.")

    assert grade.verdict == "partially_correct"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_open_blank_answer_scores_zero_no_llm(
    quiz: Quiz, store: VectorStore, blank: str | None
) -> None:
    llm = FakeLLM([])

    grade = grade_answer(quiz, "q2", store, llm, answer=blank)

    assert grade.score == 0
    assert grade.verdict == "incorrect"
    assert grade.feedback == "No answer given."
    assert grade.reference_answer == "Glikozun enerjiye dönüştürülmesidir."
    assert llm.calls == []


def test_verdict_thresholds() -> None:
    assert verdict_for(80) == "correct"
    assert verdict_for(79) == "partially_correct"
    assert verdict_for(40) == "partially_correct"
    assert verdict_for(39) == "incorrect"
    assert verdict_for(0) == "incorrect"
    assert verdict_for(100) == "correct"


def test_unknown_question_raises(quiz: Quiz, store: VectorStore) -> None:
    with pytest.raises(KeyError):
        grade_answer(quiz, "missing", store, FakeLLM([]), selected_option=0)


def test_build_grade_prompt_uses_at_most_four_sources(store: VectorStore) -> None:
    chunks = [_chunk(i, i + 1) for i in range(6)]
    question = _open()

    prompt = build_grade_prompt(question, chunks, "cevap")

    assert prompt == (
        f"{format_sources(chunks[:4])}\n\n"
        "Question: Hücre solunumunu açıklayın.\n"
        "Reference answer: Glikozun enerjiye dönüştürülmesidir.\n"
        "Student answer: cevap"
    )
