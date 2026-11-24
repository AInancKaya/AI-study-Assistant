"""Tests for the FastAPI backend (``HashEmbedder`` store + ``FakeLLM``, no network)."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel

from study_assistant.api.main import create_app
from study_assistant.config import Settings
from study_assistant.llm import LLMUnavailableError
from study_assistant.models import LLMAnswer, LLMQuiz, LLMQuizQuestion
from study_assistant.quiz_store import QuizRepository
from study_assistant.store import VectorStore
from tests.fakes import FakeLLM, HashEmbedder

FIXTURES = Path(__file__).parent / "fixtures"
NOTES_PDF = FIXTURES / "turkce_notlar.pdf"
SCANNED_PDF = FIXTURES / "empty.pdf"
FILENAME = "turkce_notlar.pdf"
# Shares many words with page 2 of the fixture, so page 2 is the top hit.
QUESTION = "İstanbul'da öğrenmenin önemi nedir?"
OLLAMA_DOWN = "Ollama is not running at http://localhost:11434. Start it with: ollama serve"


def _client(
    tmp_path: Path,
    responses: list[BaseModel | Exception] | None = None,
    check_error: Exception | None = None,
    **overrides: Any,
) -> TestClient:
    settings = Settings(data_dir=tmp_path, min_score=0.0, _env_file=None, **overrides)
    app = create_app(
        settings,
        store=VectorStore(tmp_path / "chroma", HashEmbedder()),
        llm=FakeLLM(responses or [], check_error=check_error),
        quizzes=QuizRepository(tmp_path / "q.db"),
    )
    return TestClient(app)


def _upload(client: TestClient, path: Path = NOTES_PDF, filename: str = FILENAME) -> Any:
    files = {"file": (filename, path.read_bytes(), "application/pdf")}
    return client.post("/documents", files=files)


def _mc_quiz() -> LLMQuiz:
    return LLMQuiz(
        questions=[
            LLMQuizQuestion(
                type="multiple_choice",
                question="Fotosentez nerede gerçekleşir?",
                options=["Mitokondri", "Kloroplast", "Çekirdek", "Ribozom"],
                correct_option=1,
                reference_answer="Kloroplastlarda gerçekleşir.",
                source_ids=[1],
            )
        ]
    )


def test_health_reports_llm_down(tmp_path: Path) -> None:
    client = _client(tmp_path, check_error=LLMUnavailableError(OLLAMA_DOWN))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_ok"] is False
    assert "ollama serve" in body["llm_detail"]
    assert body["model"] == "gemma2:9b"


def test_health_reports_llm_ok(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json()["llm_ok"] is True
    assert response.json()["llm_detail"] == "ok"


def test_upload_list_delete_flow(tmp_path: Path) -> None:
    client = _client(tmp_path)

    uploaded = _upload(client)
    assert uploaded.status_code == 201
    info = uploaded.json()
    assert info["filename"] == FILENAME
    assert info["pages"] == 3
    assert info["chunks"] >= 3

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json() == [info]

    deleted = client.delete(f"/documents/{info['doc_id']}")
    assert deleted.status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{info['doc_id']}").status_code == 404


def test_reupload_same_pdf_returns_200_same_id(tmp_path: Path) -> None:
    client = _client(tmp_path)

    first = _upload(client)
    second = _upload(client)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(client.get("/documents").json()) == 1


def test_upload_non_pdf_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _upload(client, filename="notlar.txt")

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


def test_upload_unreadable_pdf_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    files = {"file": ("bozuk.pdf", b"not really a pdf", "application/pdf")}

    response = client.post("/documents", files=files)

    assert response.status_code == 400
    assert client.get("/documents").json() == []


def test_upload_scanned_422(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _upload(client, path=SCANNED_PDF, filename="tarama.pdf")

    assert response.status_code == 422
    assert "scanned" in response.json()["detail"]
    assert client.get("/documents").json() == []


def test_upload_too_many_pages_413(tmp_path: Path) -> None:
    client = _client(tmp_path, max_pages=2)

    response = _upload(client)

    assert response.status_code == 413
    assert "2" in response.json()["detail"]
    assert client.get("/documents").json() == []


def test_ask_returns_answer_with_citations(tmp_path: Path) -> None:
    llm_reply = LLMAnswer(answer="Öğrenmek önemlidir.", found=True, cited_sources=[1])
    client = _client(tmp_path, responses=[llm_reply])
    doc_id = _upload(client).json()["doc_id"]

    response = client.post("/ask", json={"question": QUESTION, "doc_id": doc_id})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["answer"] == "Öğrenmek önemlidir."
    assert body["citations"] == [{"doc_id": doc_id, "filename": FILENAME, "page": 2}]


def test_ask_empty_question_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/ask", json={"question": "   "})

    assert response.status_code == 400


def test_ask_unknown_doc_404(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/ask", json={"question": QUESTION, "doc_id": "nope"})

    assert response.status_code == 404


def test_ask_ollama_down_503_with_hint(tmp_path: Path) -> None:
    client = _client(tmp_path, responses=[LLMUnavailableError(OLLAMA_DOWN)])
    _upload(client)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_quiz_hides_answers_then_grade(tmp_path: Path) -> None:
    client = _client(tmp_path, responses=[_mc_quiz()])
    doc_id = _upload(client).json()["doc_id"]

    created = client.post(
        "/quiz", json={"doc_id": doc_id, "n_multiple_choice": 1, "n_open_ended": 0}
    )

    assert created.status_code == 201
    quiz = created.json()
    question = quiz["questions"][0]
    assert "reference_answer" not in question
    assert "correct_option" not in question
    assert question["options"] == ["Mitokondri", "Kloroplast", "Çekirdek", "Ribozom"]

    graded = client.post(
        "/grade",
        json={"quiz_id": quiz["id"], "question_id": question["id"], "selected_option": 1},
    )
    assert graded.status_code == 200
    grade = graded.json()
    assert grade["score"] == 100
    assert grade["verdict"] == "correct"
    assert grade["correct_option"] == 1
    assert grade["reference_answer"] == "Kloroplastlarda gerçekleşir."

    missing = client.post("/grade", json={"quiz_id": quiz["id"], "question_id": question["id"]})
    assert missing.status_code == 400

    unknown = client.post("/grade", json={"quiz_id": quiz["id"], "question_id": "nope"})
    assert unknown.status_code == 404


def test_quiz_unknown_doc_404(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/quiz", json={"doc_id": "nope"})

    assert response.status_code == 404


def test_quiz_bad_counts_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    doc_id = _upload(client).json()["doc_id"]

    response = client.post(
        "/quiz", json={"doc_id": doc_id, "n_multiple_choice": 0, "n_open_ended": 0}
    )

    assert response.status_code == 400


def test_quiz_no_valid_question_502(tmp_path: Path) -> None:
    client = _client(tmp_path, responses=[LLMQuiz(questions=[])])
    doc_id = _upload(client).json()["doc_id"]

    response = client.post("/quiz", json={"doc_id": doc_id})

    assert response.status_code == 502


def test_grade_unknown_quiz_404(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/grade", json={"quiz_id": "nope", "question_id": "q1", "selected_option": 0}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Quiz not found"


def test_upload_too_large_413(tmp_path: Path) -> None:
    client = _client(tmp_path, max_upload_mb=1)
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024)
    files = {"file": ("buyuk.pdf", big, "application/pdf")}

    response = client.post("/documents", files=files)

    assert response.status_code == 413
