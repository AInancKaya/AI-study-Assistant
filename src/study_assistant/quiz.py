"""Quiz generation grounded in one document's chunks.

select evenly spread chunks -> fit them into the context budget -> ask the LLM for
questions (``QUIZ_SYSTEM`` + ``LLMQuiz`` schema) -> validate and trim each question ->
map source ids to pages -> save the quiz.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from study_assistant.llm import LLMClient
from study_assistant.models import Chunk, LLMQuiz, LLMQuizQuestion, Quiz, QuizQuestion
from study_assistant.prompts import QUIZ_SYSTEM, format_sources
from study_assistant.quiz_store import QuizRepository
from study_assistant.store import VectorStore

MAX_QUESTIONS_PER_TYPE = 10
MAX_QUESTIONS = 10
MAX_SOURCE_CHARS = 12000  # total chunk text sent to the LLM (8k-token window)
MC_OPTIONS = 4
ELLIPSIS = "…"


class QuizGenerationError(RuntimeError):
    """The LLM produced no valid question."""


def select_quiz_chunks(chunks: list[Chunk], max_chunks: int) -> list[Chunk]:
    """Pick up to ``max_chunks`` chunks spread evenly over the document, order kept.

    Uses indices ``round(i * (n - 1) / (m - 1))`` so the first and last chunk are
    always included.
    """
    n = len(chunks)
    m = min(max_chunks, n)
    if m <= 0:
        return []
    if m == 1:
        return [chunks[0]]
    return [chunks[round(i * (n - 1) / (m - 1))] for i in range(m)]


def generate_quiz(
    doc_id: str,
    store: VectorStore,
    llm: LLMClient,
    repo: QuizRepository,
    *,
    n_multiple_choice: int = 3,
    n_open_ended: int = 2,
    max_chunks: int,
) -> Quiz:
    """Generate, validate and save a quiz about the document ``doc_id``.

    Raises ValueError for bad question counts, KeyError for an unknown document and
    QuizGenerationError when the LLM returns no valid question.
    """
    _check_counts(n_multiple_choice, n_open_ended)
    if not store.has_document(doc_id):
        raise KeyError(doc_id)

    selected = fit_to_budget(select_quiz_chunks(store.get_chunks(doc_id), max_chunks))
    user = build_user_prompt(selected, n_multiple_choice, n_open_ended)
    reply = llm.complete_json(QUIZ_SYSTEM, user, LLMQuiz)

    questions = _valid_questions(reply.questions, selected, n_multiple_choice, n_open_ended)
    if not questions:
        raise QuizGenerationError("the model returned no valid quiz question")
    quiz = Quiz(
        id=uuid4().hex[:12],
        doc_id=doc_id,
        filename=selected[0].filename,
        created_at=datetime.now(UTC),
        questions=questions,
    )
    repo.save(quiz)
    return quiz


def fit_to_budget(chunks: list[Chunk], budget: int = MAX_SOURCE_CHARS) -> list[Chunk]:
    """Truncate every chunk to ``budget // len(chunks)`` chars if the total is too long."""
    if sum(len(c.text) for c in chunks) <= budget:
        return chunks
    limit = budget // len(chunks)
    return [c.model_copy(update={"text": _truncate(c.text, limit)}) for c in chunks]


def build_user_prompt(sources: list[Chunk], n_multiple_choice: int, n_open_ended: int) -> str:
    """Numbered sources block, then how many questions of each type to write."""
    parts = [
        f"{count} {kind}"
        for count, kind in ((n_multiple_choice, "multiple_choice"), (n_open_ended, "open_ended"))
        if count > 0
    ]
    return f"{format_sources(sources)}\n\nCreate exactly {' and '.join(parts)} questions."


def _check_counts(n_multiple_choice: int, n_open_ended: int) -> None:
    for count in (n_multiple_choice, n_open_ended):
        if not 0 <= count <= MAX_QUESTIONS_PER_TYPE:
            raise ValueError(f"question counts must be 0..{MAX_QUESTIONS_PER_TYPE}")
    if not 1 <= n_multiple_choice + n_open_ended <= MAX_QUESTIONS:
        raise ValueError(f"total question count must be 1..{MAX_QUESTIONS}")


def _truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` chars at the last whitespace, plus an ellipsis."""
    if len(text) <= limit:
        return text
    cut = text[: max(limit - len(ELLIPSIS), 0)]
    space = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
    if space > 0:
        cut = cut[:space]
    return cut.rstrip() + ELLIPSIS


def _valid_questions(
    raw: list[LLMQuizQuestion], sources: list[Chunk], n_multiple_choice: int, n_open_ended: int
) -> list[QuizQuestion]:
    """Keep valid questions in model order, at most the requested number per type."""
    remaining = {"multiple_choice": n_multiple_choice, "open_ended": n_open_ended}
    questions: list[QuizQuestion] = []
    for item in raw:
        if remaining[item.type] <= 0:
            continue
        question = _validate(item, f"q{len(questions) + 1}", sources)
        if question is not None:
            questions.append(question)
            remaining[item.type] -= 1
    return questions


def _validate(item: LLMQuizQuestion, qid: str, sources: list[Chunk]) -> QuizQuestion | None:
    text = item.question.strip()
    if not text:
        return None
    pages = _source_pages(item.source_ids, sources)
    reference = item.reference_answer.strip()
    if item.type == "open_ended":
        if item.options or not reference:
            return None
        return QuizQuestion(
            id=qid,
            type="open_ended",
            question=text,
            options=None,
            correct_option=None,
            reference_answer=reference,
            source_pages=pages,
        )
    options = _valid_options(item.options)
    correct = item.correct_option
    if options is None or correct is None or not 0 <= correct < MC_OPTIONS:
        return None
    return QuizQuestion(
        id=qid,
        type="multiple_choice",
        question=text,
        options=options,
        correct_option=correct,
        reference_answer=reference or options[correct],
        source_pages=pages,
    )


def _valid_options(options: list[str] | None) -> list[str] | None:
    """Stripped options if there are exactly 4, all non-empty and distinct; else None."""
    if options is None or len(options) != MC_OPTIONS:
        return None
    stripped = [o.strip() for o in options]
    if any(not o for o in stripped) or len({o.casefold() for o in stripped}) != MC_OPTIONS:
        return None
    return stripped


def _source_pages(source_ids: list[int], sources: list[Chunk]) -> list[int]:
    """Pages of the valid 1-based ``source_ids``; all source pages if none is valid."""
    pages = {sources[i - 1].page for i in source_ids if 1 <= i <= len(sources)}
    if not pages:
        pages = {c.page for c in sources}
    return sorted(pages)
