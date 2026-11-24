"""Tests for quiz generation (``HashEmbedder`` store + ``FakeLLM``) and quiz storage."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from study_assistant.models import Chunk, LLMQuiz, LLMQuizQuestion, Quiz, QuizQuestion
from study_assistant.prompts import QUIZ_SYSTEM
from study_assistant.quiz import (
    MAX_SOURCE_CHARS,
    QuizGenerationError,
    fit_to_budget,
    generate_quiz,
    select_quiz_chunks,
)
from study_assistant.quiz_store import QuizRepository
from study_assistant.store import VectorStore
from tests.fakes import FakeLLM, HashEmbedder

DOC_ID = "doc1"
FILENAME = "notlar.pdf"


def _chunk(idx: int, page: int, text: str | None = None, doc_id: str = DOC_ID) -> Chunk:
    return Chunk(
        id=f"{doc_id}-{idx}",
        doc_id=doc_id,
        filename=FILENAME,
        page=page,
        chunk_idx=idx,
        text=text if text is not None else f"Konu {idx}: fotosentez ve hücre solunumu.",
    )


def _mc(
    question: str = "Fotosentez nerede gerçekleşir?",
    options: list[str] | None = None,
    correct: int | None = 1,
    reference: str = "Kloroplastta gerçekleşir.",
    source_ids: list[int] | None = None,
) -> LLMQuizQuestion:
    return LLMQuizQuestion(
        type="multiple_choice",
        question=question,
        options=options
        if options is not None
        else ["Mitokondri", "Kloroplast", "Çekirdek", "Ribozom"],
        correct_option=correct,
        reference_answer=reference,
        source_ids=source_ids if source_ids is not None else [1],
    )


def _open(
    question: str = "Hücre solunumunu açıklayın.",
    reference: str = "Glikozun enerjiye dönüştürülmesidir.",
    options: list[str] | None = None,
    source_ids: list[int] | None = None,
) -> LLMQuizQuestion:
    return LLMQuizQuestion(
        type="open_ended",
        question=question,
        options=options,
        correct_option=None,
        reference_answer=reference,
        source_ids=source_ids if source_ids is not None else [2],
    )


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    vector_store = VectorStore(tmp_path / "chroma", HashEmbedder())
    vector_store.add_document([_chunk(0, 1), _chunk(1, 2), _chunk(2, 3)], pages=3)
    return vector_store


@pytest.fixture
def repo(tmp_path: Path) -> QuizRepository:
    return QuizRepository(tmp_path / "nested" / "quizzes.db")


# --- select_quiz_chunks ------------------------------------------------------------


def test_select_chunks_even_spread() -> None:
    chunks = [_chunk(i, i + 1) for i in range(100)]

    selected = select_quiz_chunks(chunks, max_chunks=10)

    assert len(selected) == 10
    assert selected[0].chunk_idx == 0
    assert selected[-1].chunk_idx == 99
    indices = [c.chunk_idx for c in selected]
    assert indices == sorted(indices)
    assert indices == [round(i * 99 / 9) for i in range(10)]


def test_select_chunks_fewer_than_max() -> None:
    chunks = [_chunk(i, i + 1) for i in range(3)]

    assert select_quiz_chunks(chunks, max_chunks=10) == chunks


def test_select_chunks_edge_cases() -> None:
    chunks = [_chunk(i, i + 1) for i in range(5)]

    assert select_quiz_chunks([], max_chunks=10) == []
    assert select_quiz_chunks(chunks, max_chunks=1) == [chunks[0]]


# --- generate_quiz -----------------------------------------------------------------


def test_generate_valid_quiz_saved(store: VectorStore, repo: QuizRepository) -> None:
    reply = LLMQuiz(questions=[_mc(), _mc(), _mc(), _open(), _open()])
    llm = FakeLLM([reply])

    quiz = generate_quiz(DOC_ID, store, llm, repo, max_chunks=10)

    assert repo.get(quiz.id) == quiz
    assert len(quiz.id) == 12
    assert quiz.doc_id == DOC_ID
    assert quiz.filename == FILENAME
    assert [q.id for q in quiz.questions] == ["q1", "q2", "q3", "q4", "q5"]
    assert [q.type for q in quiz.questions] == ["multiple_choice"] * 3 + ["open_ended"] * 2
    mc, open_q = quiz.questions[0], quiz.questions[3]
    assert mc.options == ["Mitokondri", "Kloroplast", "Çekirdek", "Ribozom"]
    assert mc.correct_option == 1
    assert open_q.options is None
    assert open_q.correct_option is None
    system, user, schema = llm.calls[0]
    assert system == QUIZ_SYSTEM
    assert schema is LLMQuiz
    assert "[1] (notlar.pdf, s. 1)" in user
    assert "Create exactly 3 multiple_choice and 2 open_ended questions." in user


def test_prompt_omits_type_with_zero_count(store: VectorStore, repo: QuizRepository) -> None:
    llm = FakeLLM([LLMQuiz(questions=[_open()])])

    quiz = generate_quiz(
        DOC_ID, store, llm, repo, n_multiple_choice=0, n_open_ended=1, max_chunks=10
    )

    user = llm.calls[0][1]
    assert "Create exactly 1 open_ended questions." in user
    assert "multiple_choice" not in user
    assert len(quiz.questions) == 1


def test_invalid_questions_dropped(store: VectorStore, repo: QuizRepository) -> None:
    reply = LLMQuiz(
        questions=[
            _mc(options=["A", "B", "C"]),  # 3 options
            _mc(correct=7),  # correct_option out of range
            _mc(correct=None),  # missing correct_option
            _mc(options=["A", "b", "B", "C"]),  # duplicate options (case-insensitive)
            _mc(options=["A", "  ", "C", "D"]),  # empty option
            _mc(question="   "),  # empty question
            _open(reference=""),  # open-ended without reference answer
            _open(options=["x", "y"]),  # open-ended with options
            _mc(question="  Geçerli soru?  ", options=[" A ", "B", "C", "D"], reference=""),
            _open(options=[]),  # empty options list is fine
        ]
    )

    quiz = generate_quiz(DOC_ID, store, FakeLLM([reply]), repo, max_chunks=10)

    assert [q.type for q in quiz.questions] == ["multiple_choice", "open_ended"]
    mc = quiz.questions[0]
    assert mc.question == "Geçerli soru?"
    assert mc.options == ["A", "B", "C", "D"]
    assert mc.reference_answer == "B"  # falls back to the correct option text
    assert quiz.questions[1].options is None
    assert [q.id for q in quiz.questions] == ["q1", "q2"]


def test_extra_questions_trimmed_per_type(store: VectorStore, repo: QuizRepository) -> None:
    reply = LLMQuiz(questions=[_open("O1?"), _mc("M1?"), _mc("M2?"), _open("O2?"), _open("O3?")])

    quiz = generate_quiz(
        DOC_ID, store, FakeLLM([reply]), repo, n_multiple_choice=1, n_open_ended=2, max_chunks=10
    )

    assert [q.question for q in quiz.questions] == ["O1?", "M1?", "O2?"]


def test_source_ids_mapped_to_pages(store: VectorStore, repo: QuizRepository) -> None:
    reply = LLMQuiz(
        questions=[
            _mc(source_ids=[3, 1, 3, 9, 0]),  # out-of-range ids ignored, pages sorted unique
            _open(source_ids=[42, -1]),  # no valid id -> all pages of the selected chunks
        ]
    )

    quiz = generate_quiz(
        DOC_ID, store, FakeLLM([reply]), repo, n_multiple_choice=1, n_open_ended=1, max_chunks=10
    )

    assert quiz.questions[0].source_pages == [1, 3]
    assert quiz.questions[1].source_pages == [1, 2, 3]


def test_all_invalid_raises(store: VectorStore, repo: QuizRepository) -> None:
    reply = LLMQuiz(questions=[_mc(correct=7), _open(reference="  ")])

    with pytest.raises(QuizGenerationError):
        generate_quiz(DOC_ID, store, FakeLLM([reply]), repo, max_chunks=10)

    assert repo.list() == []


@pytest.mark.parametrize(("n_mc", "n_open"), [(0, 0), (8, 5), (-1, 2), (11, 0)])
def test_bad_counts_raise(store: VectorStore, repo: QuizRepository, n_mc: int, n_open: int) -> None:
    llm = FakeLLM([])

    with pytest.raises(ValueError):
        generate_quiz(
            DOC_ID, store, llm, repo, n_multiple_choice=n_mc, n_open_ended=n_open, max_chunks=10
        )

    assert llm.calls == []


def test_unknown_doc_raises(store: VectorStore, repo: QuizRepository) -> None:
    llm = FakeLLM([])

    with pytest.raises(KeyError):
        generate_quiz("missing", store, llm, repo, max_chunks=10)

    assert llm.calls == []


def test_long_sources_truncated_to_budget(tmp_path: Path, repo: QuizRepository) -> None:
    words = ("kelime " * 300)[:1500]
    chunks = [_chunk(i, i + 1, text=words) for i in range(10)]
    vector_store = VectorStore(tmp_path / "chroma2", HashEmbedder())
    vector_store.add_document(chunks, pages=10)
    llm = FakeLLM([LLMQuiz(questions=[_mc()])])

    generate_quiz(DOC_ID, vector_store, llm, repo, max_chunks=10)

    user = llm.calls[0][1]
    sources = user.rsplit("\n\nCreate exactly", 1)[0]
    headers = sum(len(f"[{n}] ({FILENAME}, s. {n})\n") for n in range(1, 11))
    assert len(sources) <= MAX_SOURCE_CHARS + headers + 2 * 9
    for n in range(1, 11):
        assert f"[{n}] ({FILENAME}, s. {n})" in sources
    assert sources.count("…") == 10


# --- QuizRepository ----------------------------------------------------------------


def _quiz(quiz_id: str, doc_id: str, created_at: datetime) -> Quiz:
    return Quiz(
        id=quiz_id,
        doc_id=doc_id,
        filename=FILENAME,
        created_at=created_at,
        questions=[
            QuizQuestion(
                id="q1",
                type="multiple_choice",
                question="Soru?",
                options=["a", "b", "c", "d"],
                correct_option=0,
                reference_answer="a",
                source_pages=[1, 2],
            )
        ],
    )


def test_repository_roundtrip_and_list_order(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "quizzes.db"
    repo = QuizRepository(db_path)
    base = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
    oldest = _quiz("a", "doc1", base)
    middle = _quiz("b", "doc2", base + timedelta(minutes=1))
    newest = _quiz("c", "doc1", base + timedelta(minutes=2))
    for quiz in (middle, oldest, newest):
        repo.save(quiz)

    assert db_path.exists()
    assert repo.get("a") == oldest
    assert repo.get("missing") is None
    assert repo.list() == [newest, middle, oldest]
    assert repo.list("doc1") == [newest, oldest]
    assert repo.list("nope") == []
    # A new repository on the same file sees the same data.
    assert QuizRepository(db_path).get("c") == newest


def test_fit_to_budget_short_and_unbroken_chunks() -> None:
    short = _chunk(0, 1, text="kısa metin")
    unbroken = _chunk(1, 2, text="x" * 30)

    assert fit_to_budget([short, unbroken], budget=100) == [short, unbroken]
    fitted = fit_to_budget([short, unbroken], budget=20)
    assert fitted[0].text == "kısa metin"  # already within its 10-char share
    assert fitted[1].text == "x" * 9 + "…"  # no whitespace: hard cut
