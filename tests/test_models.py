import json
from datetime import UTC, datetime

from study_assistant.config import Settings, get_settings
from study_assistant.models import LLMAnswer, LLMGrade, LLMQuiz, Quiz, QuizPublic, QuizQuestion


def test_quiz_public_hides_answers():
    q = QuizQuestion(
        id="q1",
        type="multiple_choice",
        question="?",
        options=["a", "b", "c", "d"],
        correct_option=2,
        reference_answer="c",
        source_pages=[3],
    )
    quiz = Quiz(id="z", doc_id="d", filename="f.pdf", created_at=datetime.now(UTC), questions=[q])
    public = quiz.public()
    assert isinstance(public, QuizPublic)
    pub = public.model_dump()
    assert "correct_option" not in pub["questions"][0]
    assert "reference_answer" not in pub["questions"][0]
    assert pub["questions"][0] == {
        "id": "q1",
        "type": "multiple_choice",
        "question": "?",
        "options": ["a", "b", "c", "d"],
    }


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("STUDY_LLM_MODEL", "qwen3:8b")
    assert Settings(_env_file=None).llm_model == "qwen3:8b"


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("STUDY_LLM_MODEL", raising=False)
    s = Settings(_env_file=None)
    assert s.ollama_host == "http://localhost:11434"
    assert s.llm_model == "gemma2:9b"
    assert s.embedding_model == "intfloat/multilingual-e5-small"
    assert s.min_score == 0.75


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_llm_schemas_have_no_numeric_constraints():
    for m in (LLMAnswer, LLMQuiz, LLMGrade):
        s = json.dumps(m.model_json_schema())
        assert "minimum" not in s and "maxLength" not in s
