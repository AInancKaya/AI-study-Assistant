"""Persistent vector store for chunks, backed by a local Chroma database.

Embeddings are always computed by our own ``Embedder`` and passed to Chroma explicitly,
so Chroma never loads its default embedding model. The collection uses cosine distance,
so ``score = 1 - distance`` is the cosine similarity of query and chunk.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from study_assistant.embeddings import Embedder
from study_assistant.models import Chunk, DocumentInfo, RetrievedChunk

Metadata = Mapping[str, Any]


class VectorStore:
    """Stores chunks with page metadata and finds the most similar ones to a query."""

    def __init__(self, path: Path, embedder: Embedder, collection: str = "chunks") -> None:
        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, chunks: list[Chunk], pages: int) -> DocumentInfo:
        """Index all chunks of one document, replacing any earlier copy of it.

        Raises ValueError if ``chunks`` is empty or spans more than one ``doc_id``.
        """
        doc_id = _single_doc_id(chunks)
        embeddings = self.embedder.embed_passages([c.text for c in chunks])
        self.delete_document(doc_id)
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c, pages) for c in chunks],
        )
        return DocumentInfo(
            doc_id=doc_id, filename=chunks[0].filename, pages=pages, chunks=len(chunks)
        )

    def has_document(self, doc_id: str) -> bool:
        found = self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return bool(found["ids"])

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks most similar to ``query``, best first."""
        n_results = min(k, self._collection.count())
        if n_results <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=n_results,
            where={"doc_id": doc_id} if doc_id is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        hits = [
            RetrievedChunk(chunk=_to_chunk(i, text, meta), score=1.0 - float(dist))
            for i, text, meta, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def get_chunks(self, doc_id: str, pages: list[int] | None = None) -> list[Chunk]:
        """Return a document's chunks (optionally only on ``pages``), ordered by chunk_idx."""
        if pages is not None and not pages:
            return []
        where: dict[str, Any] = {"doc_id": doc_id}
        if pages is not None:
            where = {"$and": [{"doc_id": doc_id}, {"page": {"$in": list(pages)}}]}
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        chunks = [
            _to_chunk(i, text, meta)
            for i, text, meta in zip(result["ids"], documents, metadatas, strict=True)
        ]
        return sorted(chunks, key=lambda c: c.chunk_idx)

    def list_documents(self) -> list[DocumentInfo]:
        """One entry per stored document, sorted by filename."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        docs: dict[str, DocumentInfo] = {}
        for meta in metadatas:
            doc_id = str(meta["doc_id"])
            info = docs.get(doc_id)
            if info is None:
                docs[doc_id] = DocumentInfo(
                    doc_id=doc_id,
                    filename=str(meta["filename"]),
                    pages=int(meta["doc_pages"]),
                    chunks=1,
                )
            else:
                info.chunks += 1
        return sorted(docs.values(), key=lambda d: (d.filename, d.doc_id))

    def delete_document(self, doc_id: str) -> bool:
        """Remove every chunk of ``doc_id``. Returns False if it was not stored."""
        if not self.has_document(doc_id):
            return False
        self._collection.delete(where={"doc_id": doc_id})
        return True


def _single_doc_id(chunks: list[Chunk]) -> str:
    if not chunks:
        raise ValueError("cannot add a document with no chunks")
    doc_ids = {c.doc_id for c in chunks}
    if len(doc_ids) != 1:
        raise ValueError(f"all chunks must share one doc_id, got {sorted(doc_ids)}")
    return chunks[0].doc_id


def _to_metadata(chunk: Chunk, pages: int) -> dict[str, str | int]:
    return {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page": chunk.page,
        "chunk_idx": chunk.chunk_idx,
        "doc_pages": pages,
    }


def _to_chunk(chunk_id: str, text: str | None, meta: Metadata) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=str(meta["doc_id"]),
        filename=str(meta["filename"]),
        page=int(meta["page"]),
        chunk_idx=int(meta["chunk_idx"]),
        text=text or "",
    )
