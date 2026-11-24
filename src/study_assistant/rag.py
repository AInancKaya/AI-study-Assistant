"""Retrieval-augmented question answering grounded in the student's notes.

retrieve (``VectorStore.search``) -> drop weak hits (``min_score``) -> augment (numbered
sources in the prompt) -> generate (``LLMClient.complete_json``) -> validate citations.
"""

from __future__ import annotations

from study_assistant.llm import LLMClient
from study_assistant.models import Answer, Chunk, Citation, LLMAnswer
from study_assistant.prompts import ANSWER_SYSTEM, format_sources
from study_assistant.store import VectorStore

NOT_FOUND_MESSAGE = "I couldn't find this in your notes. / Bu bilgiyi notlarında bulamadım."


def ask(
    question: str,
    store: VectorStore,
    llm: LLMClient,
    *,
    k: int,
    min_score: float,
    doc_id: str | None = None,
) -> Answer:
    """Answer ``question`` from the stored notes, citing the pages the answer came from.

    The LLM is not called when no retrieved chunk reaches ``min_score``.
    Raises ValueError if the question is empty or whitespace.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    hits = store.search(question, k=k, doc_id=doc_id)
    sources = [hit.chunk for hit in hits if hit.score >= min_score]
    if not sources:
        return _not_found()

    reply = llm.complete_json(ANSWER_SYSTEM, build_user_prompt(question, sources), LLMAnswer)
    answer = reply.answer.strip()
    if not reply.found or not answer:
        return _not_found()
    return Answer(
        answer=answer, found=True, citations=valid_citations(reply.cited_sources, sources)
    )


def build_user_prompt(question: str, sources: list[Chunk]) -> str:
    """Numbered sources block, then the question."""
    return f"{format_sources(sources)}\n\nQuestion: {question}"


def valid_citations(cited: list[int], sources: list[Chunk]) -> list[Citation]:
    """Map 1-based source ids to unique citations in first-cited order; drop invalid ids."""
    citations: list[Citation] = []
    for source_id in cited:
        if not 1 <= source_id <= len(sources):
            continue
        chunk = sources[source_id - 1]
        citation = Citation(doc_id=chunk.doc_id, filename=chunk.filename, page=chunk.page)
        if citation not in citations:
            citations.append(citation)
    return citations


def _not_found() -> Answer:
    return Answer(answer=NOT_FOUND_MESSAGE, found=False, citations=[])
