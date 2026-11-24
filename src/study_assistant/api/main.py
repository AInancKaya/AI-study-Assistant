"""FastAPI backend: upload notes, ask grounded questions, generate and grade quizzes.

There is no module-level app object. Run it as a factory so settings and dependencies
are built at startup, not at import time::

    uvicorn study_assistant.api.main:create_app --factory --port 8000

Endpoints are plain ``def`` functions: embedding, Chroma, SQLite and Ollama calls are
blocking, so FastAPI runs them in its threadpool instead of the event loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

import study_assistant
from study_assistant.api.schemas import AskRequest, GradeRequest, HealthResponse, QuizRequest
from study_assistant.config import Settings, get_settings
from study_assistant.embeddings import E5Embedder
from study_assistant.grading import grade_answer
from study_assistant.ingest.chunking import chunk_pages
from study_assistant.ingest.pdf import (
    InvalidPDFError,
    ScannedPDFError,
    count_pages,
    extract_pages,
    make_doc_id,
)
from study_assistant.llm import (
    LLMClient,
    LLMError,
    LLMUnavailableError,
    ModelNotFoundError,
    OllamaClient,
)
from study_assistant.models import Answer, DocumentInfo, Grade, QuizPublic
from study_assistant.quiz import QuizGenerationError, generate_quiz
from study_assistant.quiz_store import QuizRepository
from study_assistant.rag import ask
from study_assistant.store import VectorStore

API_TITLE = "AI Study Assistant API"
BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class Deps:
    """Everything the endpoints need, built once per app and kept on ``app.state``."""

    settings: Settings
    store: VectorStore
    llm: LLMClient
    quizzes: QuizRepository


def create_app(
    settings: Settings | None = None,
    *,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
    quizzes: QuizRepository | None = None,
) -> FastAPI:
    """Build the API. Dependencies not passed in are built from ``settings``."""
    settings = settings or get_settings()
    app = FastAPI(title=API_TITLE, version=getattr(study_assistant, "__version__", "0.1.0"))
    app.state.deps = Deps(
        settings=settings,
        store=store if store is not None else _default_store(settings),
        llm=llm if llm is not None else _default_llm(settings),
        quizzes=quizzes
        if quizzes is not None
        else QuizRepository(settings.data_dir / "quizzes.db"),
    )
    app.include_router(router)
    app.add_exception_handler(LLMUnavailableError, _llm_error_handler(503))
    app.add_exception_handler(ModelNotFoundError, _llm_error_handler(503))
    app.add_exception_handler(LLMError, _llm_error_handler(502))
    return app


def _default_store(settings: Settings) -> VectorStore:
    return VectorStore(settings.data_dir / "chroma", E5Embedder(settings.embedding_model))


def _default_llm(settings: Settings) -> OllamaClient:
    return OllamaClient(settings.ollama_host, settings.llm_model, settings.llm_timeout_s)


def _llm_error_handler(status_code: int) -> Callable[[Request, Exception], JSONResponse]:
    """Return a handler that turns an ``LLMError`` into ``{"detail": <message>}``."""

    def handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


def _deps(request: Request) -> Deps:
    return request.app.state.deps


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Always 200; ``llm_ok`` says whether Ollama and the model are ready."""
    deps = _deps(request)
    try:
        deps.llm.check()
    except LLMError as exc:
        llm_ok, detail = False, str(exc)
    else:
        llm_ok, detail = True, "ok"
    return HealthResponse(
        status="ok", llm_ok=llm_ok, llm_detail=detail, model=deps.settings.llm_model
    )


@router.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(
    request: Request, response: Response, file: Annotated[UploadFile, File()]
) -> DocumentInfo:
    """Index a PDF. 201 when new; 200 with the stored info when the same file exists."""
    deps = _deps(request)
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported.")
    data = file.file.read()
    page_count = _check_limits(data, deps.settings)

    doc_id = make_doc_id(data)
    if deps.store.has_document(doc_id):
        response.status_code = 200
        return _find_document(deps.store, doc_id)
    return _index_pdf(data, filename, page_count, deps)


def _check_limits(data: bytes, settings: Settings) -> int:
    """Return the page count; raise 413 if too large or too long, 400 if unreadable."""
    if len(data) > settings.max_upload_mb * BYTES_PER_MB:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    try:
        pages = count_pages(data)
    except InvalidPDFError as exc:
        raise HTTPException(400, str(exc)) from exc
    if pages > settings.max_pages:
        raise HTTPException(413, f"PDF has {pages} pages; the limit is {settings.max_pages}.")
    return pages


def _index_pdf(data: bytes, filename: str, page_count: int, deps: Deps) -> DocumentInfo:
    """Extract, chunk and store a new PDF; 422 for scans, 400 for unreadable text."""
    try:
        pages = extract_pages(data, filename)
    except ScannedPDFError as exc:
        raise HTTPException(422, str(exc)) from exc
    except InvalidPDFError as exc:
        raise HTTPException(400, str(exc)) from exc
    chunks = chunk_pages(pages, deps.settings.chunk_size, deps.settings.chunk_overlap)
    return deps.store.add_document(chunks, pages=page_count)


def _find_document(store: VectorStore, doc_id: str) -> DocumentInfo:
    for info in store.list_documents():
        if info.doc_id == doc_id:
            return info
    raise HTTPException(404, "Document not found")


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[DocumentInfo]:
    return _deps(request).store.list_documents()


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(request: Request, doc_id: str) -> Response:
    if not _deps(request).store.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


@router.post("/ask", response_model=Answer)
def ask_question(request: Request, body: AskRequest) -> Answer:
    """Answer from the notes (one document, or all when ``doc_id`` is omitted)."""
    deps = _deps(request)
    if body.doc_id is not None and not deps.store.has_document(body.doc_id):
        raise HTTPException(404, "Document not found")
    try:
        return ask(
            body.question,
            deps.store,
            deps.llm,
            k=deps.settings.top_k,
            min_score=deps.settings.min_score,
            doc_id=body.doc_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/quiz", response_model=QuizPublic, status_code=201)
def create_quiz(request: Request, body: QuizRequest) -> QuizPublic:
    """Generate and store a quiz; the response hides the answer key."""
    deps = _deps(request)
    try:
        quiz = generate_quiz(
            body.doc_id,
            deps.store,
            deps.llm,
            deps.quizzes,
            n_multiple_choice=body.n_multiple_choice,
            n_open_ended=body.n_open_ended,
            max_chunks=deps.settings.quiz_max_chunks,
        )
    except KeyError as exc:
        raise HTTPException(404, "Document not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except QuizGenerationError as exc:
        raise HTTPException(502, str(exc)) from exc
    return quiz.public()


@router.post("/grade", response_model=Grade)
def grade(request: Request, body: GradeRequest) -> Grade:
    deps = _deps(request)
    quiz = deps.quizzes.get(body.quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz not found")
    try:
        return grade_answer(
            quiz,
            body.question_id,
            deps.store,
            deps.llm,
            answer=body.answer,
            selected_option=body.selected_option,
        )
    except KeyError as exc:
        raise HTTPException(404, "Question not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
