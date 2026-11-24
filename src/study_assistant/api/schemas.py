"""Request and response bodies that exist only at the HTTP layer.

Domain responses (``DocumentInfo``, ``Answer``, ``QuizPublic``, ``Grade``) are reused
from ``study_assistant.models``.
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None  # None searches every stored document


class QuizRequest(BaseModel):
    doc_id: str
    n_multiple_choice: int = 3
    n_open_ended: int = 2


class GradeRequest(BaseModel):
    quiz_id: str
    question_id: str
    answer: str | None = None  # open-ended questions
    selected_option: int | None = None  # multiple-choice questions (0-based)


class HealthResponse(BaseModel):
    status: str
    llm_ok: bool
    llm_detail: str
    model: str
