"""Grading of quiz answers: multiple choice is deterministic, open-ended is judged by the LLM.

Multiple-choice answers are compared with the stored ``correct_option`` (no LLM call).
Open-ended answers are sent to the LLM together with the reference answer and the source
chunks; the LLM's score is clamped to 0..100 and the verdict is always computed here, never
taken from the model.
"""

from __future__ import annotations

from study_assistant.llm import LLMClient
from study_assistant.models import (
    Chunk,
    Citation,
    Grade,
    LLMGrade,
    Quiz,
    QuizQuestion,
    Verdict,
)
from study_assistant.prompts import GRADE_SYSTEM, format_sources
from study_assistant.store import VectorStore

MAX_GRADE_CHUNKS = 4
CORRECT_THRESHOLD = 80
PARTIAL_THRESHOLD = 40
BLANK_FEEDBACK = "No answer given."


def verdict_for(score: int) -> Verdict:
    """Map a 0..100 score to a verdict: >=80 correct, >=40 partially correct, else incorrect."""
    if score >= CORRECT_THRESHOLD:
        return "correct"
    if score >= PARTIAL_THRESHOLD:
        return "partially_correct"
    return "incorrect"


def grade_answer(
    quiz: Quiz,
    question_id: str,
    store: VectorStore,
    llm: LLMClient,
    *,
    answer: str | None = None,
    selected_option: int | None = None,
) -> Grade:
    """Grade one answer to a question of ``quiz``.

    Raises KeyError for an unknown question id and ValueError for a missing or
    out-of-range multiple-choice selection.
    """
    question = _find_question(quiz, question_id)
    citations = _citations(quiz, question)
    if question.type == "multiple_choice":
        return _grade_multiple_choice(question, selected_option, citations)
    return _grade_open_ended(quiz, question, store, llm, answer, citations)


def build_grade_prompt(question: QuizQuestion, chunks: list[Chunk], answer: str) -> str:
    """User message for the grader: numbered sources, then question, reference and answer."""
    return (
        f"{format_sources(chunks[:MAX_GRADE_CHUNKS])}\n\n"
        f"Question: {question.question}\n"
        f"Reference answer: {question.reference_answer}\n"
        f"Student answer: {answer}"
    )


def _find_question(quiz: Quiz, question_id: str) -> QuizQuestion:
    for question in quiz.questions:
        if question.id == question_id:
            return question
    raise KeyError(question_id)


def _citations(quiz: Quiz, question: QuizQuestion) -> list[Citation]:
    return [
        Citation(doc_id=quiz.doc_id, filename=quiz.filename, page=page)
        for page in question.source_pages
    ]


def _grade_multiple_choice(
    question: QuizQuestion, selected_option: int | None, citations: list[Citation]
) -> Grade:
    options = question.options or []
    if selected_option is None:
        raise ValueError("a multiple-choice question needs selected_option")
    if not 0 <= selected_option < len(options):
        raise ValueError(f"selected_option must be 0..{len(options) - 1}")
    correct = question.correct_option
    correct_text = options[correct] if correct is not None else ""
    if selected_option == correct:
        score = 100
        feedback = f"Correct! The answer is: {correct_text}"
    else:
        score = 0
        feedback = f"Not quite. The correct answer is: {correct_text}"
    return Grade(
        score=score,
        verdict=verdict_for(score),
        feedback=feedback,
        citations=citations,
        correct_option=correct,
        reference_answer=question.reference_answer,
    )


def _grade_open_ended(
    quiz: Quiz,
    question: QuizQuestion,
    store: VectorStore,
    llm: LLMClient,
    answer: str | None,
    citations: list[Citation],
) -> Grade:
    text = (answer or "").strip()
    if not text:
        return Grade(
            score=0,
            verdict="incorrect",
            feedback=BLANK_FEEDBACK,
            citations=citations,
            reference_answer=question.reference_answer,
        )
    chunks = store.get_chunks(quiz.doc_id, question.source_pages)
    prompt = build_grade_prompt(question, chunks, text)
    reply = llm.complete_json(GRADE_SYSTEM, prompt, LLMGrade)
    score = max(0, min(100, reply.score))
    return Grade(
        score=score,
        verdict=verdict_for(score),
        feedback=reply.feedback,
        citations=citations,
        reference_answer=question.reference_answer,
    )
