"""Pydantic models shared by every layer of the app.

The ``LLM*`` models describe what the language model must return. They are passed to
Ollama as a JSON schema, so they carry no ids and no numeric constraints (small local
models handle plain schemas far more reliably); validation of ranges happens in code.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

# --- Documents and retrieval -------------------------------------------------------


class Page(BaseModel):
    doc_id: str
    filename: str
    page: int  # 1-based
    text: str


class Chunk(BaseModel):
    id: str
    doc_id: str
    filename: str
    page: int
    chunk_idx: int
    text: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity


class Citation(BaseModel):
    doc_id: str
    filename: str
    page: int


class Answer(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


# --- Quizzes -----------------------------------------------------------------------

QuestionType = Literal["multiple_choice", "open_ended"]


class QuizQuestionPublic(BaseModel):
    """A quiz question as shown to the student (no answer key)."""

    id: str
    type: QuestionType
    question: str
    options: list[str] | None


class QuizQuestion(BaseModel):
    id: str
    type: QuestionType
    question: str
    options: list[str] | None
    correct_option: int | None
    reference_answer: str
    source_pages: list[int]


class QuizPublic(BaseModel):
    id: str
    doc_id: str
    filename: str
    questions: list[QuizQuestionPublic]


class Quiz(BaseModel):
    id: str
    doc_id: str
    filename: str
    created_at: datetime
    questions: list[QuizQuestion]

    def public(self) -> QuizPublic:
        """Return the quiz without ``correct_option`` and ``reference_answer``."""
        return QuizPublic(
            id=self.id,
            doc_id=self.doc_id,
            filename=self.filename,
            questions=[
                QuizQuestionPublic(id=q.id, type=q.type, question=q.question, options=q.options)
                for q in self.questions
            ],
        )


Verdict = Literal["correct", "partially_correct", "incorrect"]


class Grade(BaseModel):
    score: int
    verdict: Verdict
    feedback: str
    citations: list[Citation]
    correct_option: int | None = None
    reference_answer: str


# --- LLM output schemas (no ids, no numeric constraints) ---------------------------


class LLMAnswer(BaseModel):
    answer: str
    found: bool
    cited_sources: list[int]


class LLMQuizQuestion(BaseModel):
    type: QuestionType
    question: str
    options: list[str] | None
    correct_option: int | None
    reference_answer: str
    source_ids: list[int]


class LLMQuiz(BaseModel):
    questions: list[LLMQuizQuestion]


class LLMGrade(BaseModel):
    score: int
    feedback: str
